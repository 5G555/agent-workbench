"""FAQ 表：多租户问答对 + BM25 内存索引。

多租户处理：
    表加 tenant_id 列，唯一键改成 (tenant_id, question)——同一句话在不同租户
    可以有不同答案，这才是正常的（A 公司说 7 天退款，B 公司说 15 天）。
    内存索引按租户各建一份，检索只在自己的那份里找，互不串。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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
    tenant_id   VARCHAR(64)  NOT NULL DEFAULT 'tenant_default',
    subject     VARCHAR(64)  NOT NULL DEFAULT '',
    question    VARCHAR(1000) NOT NULL,
    answer      VARCHAR(4000) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
-- 迁移：老表没有 tenant_id 时补上（CREATE TABLE IF NOT EXISTS 不会改已有表）
ALTER TABLE faq_entries
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default';
-- 老的全局唯一索引要让位给"租户内唯一"
DROP INDEX IF EXISTS ux_faq_entries_question;
CREATE UNIQUE INDEX IF NOT EXISTS ux_faq_entries_tenant_question
    ON faq_entries (tenant_id, question);
"""


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，去掉空白与单字符标点。"""
    return [w for w in jieba.lcut(text.strip()) if w.strip()]


def bm25_confidence(scores: Any) -> tuple[int, float]:
    """把 BM25 原始分转成 0~1 的置信度，返回 (最佳下标, 置信度)。

    为什么不能直接用 softmax 概率：
        4 条 FAQ 时，完全匹配的那条 softmax 也只有 0.845；12 条时是 0.917。
        也就是说概率会随库的大小漂移，同一个阈值在小租户里必然误判。
    这里用"相对均匀分布的提升度"做归一化：
        完全无关（分数都一样）→ 0；正确答案明显突出 → 接近 1；与库大小无关。
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        return 0, 0.0
    exp = np.exp(array - np.max(array))
    probs = exp / np.sum(exp)
    best = int(np.argmax(probs))
    n = probs.size
    if n <= 1:
        return best, 1.0
    uniform = 1.0 / n
    value = (float(probs[best]) - uniform) / (1.0 - uniform)
    return best, max(0.0, min(1.0, value))


@dataclass
class FaqHit:
    """一次 FAQ 命中的结果。"""

    question: str
    answer: str
    score: float
    subject: str = ""


@dataclass
class _TenantIndex:
    """某个租户的 BM25 索引。"""

    questions: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self.questions)


class FaqStore:
    """FAQ 表的读写 + 每租户一份 BM25 内存索引。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.threshold = self.settings.faq_threshold
        self._indexes: dict[str, _TenantIndex] = {}

    async def _connect(self) -> Any:
        return await lease_connection()

    async def ensure_table(self) -> None:
        conn = await self._connect()
        try:
            await conn.execute(CREATE_TABLE_SQL)
            logger.info("faq.table_ready")
        finally:
            await conn.close()

    # ── 写入 ────────────────────────────────────────────────
    async def upsert_many(
        self, rows: list[tuple[str, str, str]], tenant_id: str | None = None
    ) -> int:
        """rows = [(subject, question, answer), ...]，按 (租户, 问题) 去重覆盖。"""
        if not rows:
            return 0
        tenant = tenant_id or self.settings.default_tenant_id
        conn = await self._connect()
        try:
            await conn.executemany(
                """
                INSERT INTO faq_entries (tenant_id, subject, question, answer)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (tenant_id, question) DO UPDATE
                SET subject = EXCLUDED.subject, answer = EXCLUDED.answer
                """,
                [(tenant, s, q, a) for s, q, a in rows],
            )
            count = await conn.fetchval(
                "SELECT count(*) FROM faq_entries WHERE tenant_id = $1", tenant
            )
            logger.info("faq.upserted", rows=len(rows), tenant=tenant, total=count)
            return len(rows)
        finally:
            await conn.close()

    # ── 加载 + 建索引 ───────────────────────────────────────
    def _build(self, records: list[dict]) -> _TenantIndex:
        index = _TenantIndex(
            questions=[r["question"] for r in records],
            subjects=[r["subject"] for r in records],
            answers=[r["answer"] for r in records],
        )
        tokens = [tokenize(q) for q in index.questions]
        index.bm25 = BM25Okapi(tokens) if tokens else None
        return index

    async def _load_tenant(self, tenant_id: str) -> _TenantIndex:
        conn = await self._connect()
        try:
            records = await conn.fetch(
                """
                SELECT subject, question, answer FROM faq_entries
                WHERE tenant_id = $1 ORDER BY id
                """,
                tenant_id,
            )
        finally:
            await conn.close()
        return self._build([dict(r) for r in records])

    async def reload(self) -> int:
        """载入所有租户的 FAQ，各建一份索引。"""
        cached = await load_faq_index()
        if cached is not None:
            rows, tokens = cached
            grouped: dict[str, list[tuple[list[str], list[str]]]] = {}
            for row, token in zip(rows, tokens, strict=False):
                # 兼容老缓存：[subject, question, answer]（无租户）
                if len(row) == 4:
                    tenant, rest = row[0], row[1:]
                else:
                    tenant, rest = self.settings.default_tenant_id, row
                grouped.setdefault(tenant, []).append((rest, token))

            self._indexes = {}
            for tenant, entries in grouped.items():
                index = _TenantIndex()
                for (subject, question, answer), _token in entries:
                    index.subjects.append(subject)
                    index.questions.append(question)
                    index.answers.append(answer)
                index.bm25 = BM25Okapi([t for _, t in entries]) if entries else None
                self._indexes[tenant] = index
            logger.info("faq.loaded_from_redis", tenants=len(self._indexes))
            return sum(i.size for i in self._indexes.values())

        conn = await self._connect()
        try:
            records = await conn.fetch(
                "SELECT tenant_id, subject, question, answer FROM faq_entries ORDER BY id"
            )
        finally:
            await conn.close()

        self._indexes = {}
        by_tenant: dict[str, list[dict]] = {}
        for row in records:
            by_tenant.setdefault(row["tenant_id"], []).append(dict(row))
        for tenant, rows in by_tenant.items():
            self._indexes[tenant] = self._build(rows)

        # 回填 Redis（带租户列），供后续进程复用
        flat_rows: list[list[str]] = []
        flat_tokens: list[list[str]] = []
        for tenant, rows in by_tenant.items():
            for r in rows:
                flat_rows.append([tenant, r["subject"], r["question"], r["answer"]])
                flat_tokens.append(tokenize(r["question"]))
        await save_faq_index(flat_rows, flat_tokens)

        logger.info(
            "faq.loaded_from_postgres",
            tenants=len(self._indexes),
            questions=sum(i.size for i in self._indexes.values()),
        )
        return sum(i.size for i in self._indexes.values())

    async def _index_for(self, tenant_id: str) -> _TenantIndex:
        """取某租户的索引。

        没有、或者有但是空的（Redis 缓存可能是加租户之前写的），都回库重建一次。
        租户数量少、FAQ 也不多，这次回库成本可以接受。
        """
        index = self._indexes.get(tenant_id)
        if index is None or index.size == 0:
            index = await self._load_tenant(tenant_id)
            self._indexes[tenant_id] = index
            logger.info("faq.index_built", tenant=tenant_id, questions=index.size)
        return index

    # ── 检索 ────────────────────────────────────────────────
    async def search(
        self,
        query: str,
        tenant_id: str | None = None,
        threshold: float | None = None,
    ) -> FaqHit | None:
        """在指定租户的 FAQ 里找；未达阈值返回 None（调用方转向量检索）。"""
        if not query:
            return None
        tenant = tenant_id or self.settings.default_tenant_id
        index = await self._index_for(tenant)
        if index.bm25 is None or not index.questions:
            return None

        limit = self.threshold if threshold is None else threshold
        tokens = tokenize(query)
        if not tokens:
            return None

        scores = index.bm25.get_scores(tokens)
        if len(scores) == 0:
            return None

        best, best_score = bm25_confidence(scores)

        if best_score < limit:
            logger.info(
                "faq.miss",
                query=query,
                tenant=tenant,
                best_score=round(best_score, 3),
                threshold=limit,
            )
            return None

        logger.info(
            "faq.hit",
            query=query,
            tenant=tenant,
            matched=index.questions[best],
            score=round(best_score, 3),
        )
        return FaqHit(
            question=index.questions[best],
            answer=index.answers[best],
            score=best_score,
            subject=index.subjects[best],
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
