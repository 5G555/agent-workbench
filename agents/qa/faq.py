"""FAQ 精确检索：Postgres 表 + BM25 打分。

  表名 jpkb → faq_entries，字段 subject_name → subject

命中逻辑与原来一致：查询分词后与所有问题的 BM25 分数做 softmax，
最高分 ≥ threshold 才算命中，命中直接返回答案，否则交给向量 RAG。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from agents.qa.cache import load_faq_index, save_faq_index
from config import Settings, get_settings
from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS faq_entries (
    id          BIGSERIAL PRIMARY KEY,
    subject     VARCHAR(64)  NOT NULL DEFAULT '',
    question    VARCHAR(1000) NOT NULL,
    answer      VARCHAR(4000) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_faq_entries_question ON faq_entries (question);
"""


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，去掉空白与单字符标点。"""
    return [w for w in jieba.lcut(text.strip()) if w.strip()]


@dataclass
class FaqHit:
    """一次 FAQ 命中的结果。"""

    question: str
    answer: str
    score: float
    subject: str = ""


class FaqStore:
    """FAQ 表的读写 + BM25 内存索引。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.threshold = self.settings.faq_threshold  # 可在 .env 里调
        self._questions: list[str] = []
        self._subjects: list[str] = []
        self._answers: list[str] = []
        self._bm25: BM25Okapi | None = None

    # ── 连接 ────────────────────────────────────────────────
    async def _connect(self) -> Any:
        return await lease_connection()

    # ── 建表 ────────────────────────────────────────────────
    async def ensure_table(self) -> None:
        conn = await self._connect()
        try:
            await conn.execute(CREATE_TABLE_SQL)
            logger.info("faq.table_ready")
        finally:
            await conn.close()

    # ── 写入 ────────────────────────────────────────────────
    async def upsert_many(self, rows: list[tuple[str, str, str]]) -> int:
        """rows = [(subject, question, answer), ...]，按 question 去重覆盖。"""
        if not rows:
            return 0
        conn = await self._connect()
        try:
            await conn.executemany(
                """
                INSERT INTO faq_entries (subject, question, answer)
                VALUES ($1, $2, $3)
                ON CONFLICT (question) DO UPDATE
                SET subject = EXCLUDED.subject, answer = EXCLUDED.answer
                """,
                rows,
            )
            count = await conn.fetchval("SELECT count(*) FROM faq_entries")
            logger.info("faq.upserted", rows=len(rows), total=count)
            return len(rows)
        finally:
            await conn.close()

    # ── 加载 + 建索引 ───────────────────────────────────────
    async def reload(self) -> int:
        """载入 FAQ 并建 BM25 索引。

        优先从 Redis 拿「问题列表 + 分词结果」（多进程部署时省掉查库和 jieba 分词），
        未命中再回源 Postgres 并回填缓存。
        """
        cached = await load_faq_index()
        if cached is not None:
            rows, tokens = cached
            self._subjects = [r[0] for r in rows]
            self._questions = [r[1] for r in rows]
            self._answers = [r[2] for r in rows]
            self._bm25 = BM25Okapi(tokens) if tokens else None
            logger.info("faq.loaded_from_redis", questions=len(self._questions))
            return len(self._questions)

        conn = await self._connect()
        try:
            records = await conn.fetch("SELECT subject, question, answer FROM faq_entries")
        finally:
            await conn.close()

        self._questions = [r["question"] for r in records]
        self._subjects = [r["subject"] for r in records]
        self._answers = [r["answer"] for r in records]

        tokens = [tokenize(q) for q in self._questions]
        self._bm25 = BM25Okapi(tokens) if tokens else None

        # 回填 Redis，供后续进程直接复用
        rows = [
            [s, q, a]
            for s, q, a in zip(self._subjects, self._questions, self._answers, strict=False)
        ]
        await save_faq_index(rows, tokens)

        logger.info("faq.loaded_from_postgres", questions=len(self._questions))
        return len(self._questions)

    # ── 检索 ────────────────────────────────────────────────
    async def search(self, query: str, threshold: float | None = None) -> FaqHit | None:
        """命中返回 FaqHit，未达阈值返回 None（调用方转走向量检索）。

        threshold 不传时用配置里的 faq_threshold；显式传入可临时覆盖。
        """
        if not query or self._bm25 is None:
            return None

        limit = self.threshold if threshold is None else threshold
        tokens = tokenize(query)
        if not tokens:
            return None

        scores = self._bm25.get_scores(tokens)
        if len(scores) == 0:
            return None

        # softmax 归一化：把 BM25 原始分变成可比较的概率分布
        exp = np.exp(scores - np.max(scores))
        probs = exp / np.sum(exp)
        best = int(np.argmax(probs))
        best_score = float(probs[best])

        if best_score < limit:
            logger.info("faq.miss", query=query, best_score=round(best_score, 3), threshold=limit)
            return None

        logger.info(
            "faq.hit",
            query=query,
            matched=self._questions[best],
            score=round(best_score, 3),
        )
        return FaqHit(
            question=self._questions[best],
            answer=self._answers[best],
            score=best_score,
            subject=self._subjects[best],
        )


_faq_store: FaqStore | None = None
_faq_lock = asyncio.Lock()


async def get_faq_store() -> FaqStore:
    """懒加载单例：首次使用时建表并载入 BM25 索引。"""
    global _faq_store
    async with _faq_lock:
        if _faq_store is None:
            store = FaqStore()
            await store.ensure_table()
            await store.reload()
            _faq_store = store
    return _faq_store


async def reset_faq_store() -> None:
    """重新载入索引（导入新 FAQ 后调用）。"""
    global _faq_store
    if _faq_store is not None:
        await _faq_store.reload()
