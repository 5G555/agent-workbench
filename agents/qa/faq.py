"""FAQ 表：多租户 + 部门级的问答对，配 BM25 内存索引。

权限模型与向量检索保持一致：
    tenant_id  公司级隔离
    department 部门级；'*' 表示全公司可见（任何部门都能命中）
    role       admin 在租户内不受部门限制

索引怎么建：
    BM25 需要一个"候选集合"，不能先算分再事后过滤（那样会命中用户看不到的条目）。
    所以按「租户 × 部门」建多份索引：
        __all__     某租户全部条目        （管理员用）
        __public__  只有 department='*' 的条目（未分配部门的用户用）
        具体部门     该部门 + '*' 的条目    （该部门用户用）
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

PUBLIC = "*"  # 文档/条目上的 * 表示全公司可见
SCOPE_ALL = "__all__"  # 管理员：租户内全部
SCOPE_PUBLIC = "__public__"  # 未分配部门：只看公共

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS faq_entries (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   VARCHAR(64)  NOT NULL DEFAULT 'tenant_default',
    department  VARCHAR(64)  NOT NULL DEFAULT '*',
    subject     VARCHAR(64)  NOT NULL DEFAULT '',
    question    VARCHAR(1000) NOT NULL,
    answer      VARCHAR(4000) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
-- 迁移：老表没有 tenant_id / department 时补上
-- （CREATE TABLE IF NOT EXISTS 不会改已有表结构）
ALTER TABLE faq_entries
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default';
ALTER TABLE faq_entries
    ADD COLUMN IF NOT EXISTS department VARCHAR(64) NOT NULL DEFAULT '*';
-- 唯一键逐步收窄：全局 → 租户 → 租户+部门
DROP INDEX IF EXISTS ux_faq_entries_question;
DROP INDEX IF EXISTS ux_faq_entries_tenant_question;
CREATE UNIQUE INDEX IF NOT EXISTS ux_faq_entries_scope_question
    ON faq_entries (tenant_id, department, question);
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


def resolve_scope(department: str | None, role: str | None) -> str:
    """把用户的部门/角色折算成索引范围键。"""
    if (role or "") == "admin":
        return SCOPE_ALL
    dept = (department or "").strip()
    if not dept or dept == PUBLIC:
        return SCOPE_PUBLIC
    return dept


@dataclass
class FaqHit:
    """一次 FAQ 命中的结果。"""

    question: str
    answer: str
    score: float
    subject: str = ""
    department: str = ""


@dataclass
class _Index:
    """某个范围（租户 × 部门）的 BM25 索引。"""

    questions: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self.questions)


def _dedupe_by_question(records: list[dict]) -> list[dict]:
    """同一句问题在多处出现时只保留一条：部门专属优先于全公司公共。

    为什么必须去重：
        1. BM25 里出现两条几乎一样的文本会互相稀释置信度，
           实测 14 条里出现一对重复，正确答案的归一化分数就从 0.9 掉到 0.46 → 直接判未命中
        2. 语义上也不该让用户看到两条同名答案
    """
    best: dict[str, dict] = {}
    for row in records:
        key = row["question"]
        current = best.get(key)
        if current is None:
            best[key] = row
        elif current.get("department", PUBLIC) == PUBLIC and row.get("department") != PUBLIC:
            best[key] = row  # 用部门专属的覆盖公共的
    return list(best.values())


def _build_index(records: list[dict], tokens: list[list[str]] | None = None) -> _Index:
    records = _dedupe_by_question(records)
    index = _Index(
        questions=[r["question"] for r in records],
        subjects=[r["subject"] for r in records],
        answers=[r["answer"] for r in records],
        departments=[r.get("department", PUBLIC) for r in records],
    )
    if tokens is not None:
        index.bm25 = BM25Okapi(tokens) if tokens else None
    else:
        # 记录里可能已经带了分词结果（来自 Redis 缓存），没有才现分
        cached_tokens = [r.get("_tokens") for r in records]
        if cached_tokens and all(cached_tokens):
            index.bm25 = BM25Okapi(cached_tokens) if records else None
        else:
            tokenized = [tokenize(q) for q in index.questions]
            index.bm25 = BM25Okapi(tokenized) if tokenized else None
    return index


class FaqStore:
    """FAQ 表的读写 + 每「租户 × 部门」一份 BM25 索引。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.threshold = self.settings.faq_threshold
        self._indexes: dict[tuple[str, str], _Index] = {}

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
        self,
        rows: list[tuple[str, str, str]],
        tenant_id: str | None = None,
        department: str = PUBLIC,
    ) -> int:
        """rows = [(subject, question, answer), ...]，按 (租户, 部门, 问题) 去重覆盖。"""
        if not rows:
            return 0
        tenant = tenant_id or self.settings.default_tenant_id
        conn = await self._connect()
        try:
            await conn.executemany(
                """
                INSERT INTO faq_entries (tenant_id, department, subject, question, answer)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (tenant_id, department, question) DO UPDATE
                SET subject = EXCLUDED.subject, answer = EXCLUDED.answer
                """,
                [(tenant, department, s, q, a) for s, q, a in rows],
            )
            count = await conn.fetchval(
                "SELECT count(*) FROM faq_entries WHERE tenant_id = $1 AND department = $2",
                tenant,
                department,
            )
            logger.info(
                "faq.upserted",
                rows=len(rows),
                tenant=tenant,
                department=department,
                total=count,
            )
            return len(rows)
        finally:
            await conn.close()

    # ── 加载 + 建索引 ───────────────────────────────────────
    async def _query(self, sql: str, *params: Any) -> list[dict]:
        """执行一条**固定** SQL（不接受调用方拼 where，避免注入面）。"""
        conn = await self._connect()
        try:
            records = await conn.fetch(sql, *params)
        finally:
            await conn.close()
        return [dict(r) for r in records]

    async def _fetch_all(self, tenant_id: str) -> list[dict]:
        return await self._query(
            "SELECT tenant_id, department, subject, question, answer "
            "FROM faq_entries WHERE tenant_id = $1 ORDER BY id",
            tenant_id,
        )

    async def _fetch_public(self, tenant_id: str) -> list[dict]:
        return await self._query(
            "SELECT tenant_id, department, subject, question, answer "
            "FROM faq_entries WHERE tenant_id = $1 AND department = $2 ORDER BY id",
            tenant_id,
            PUBLIC,
        )

    async def _fetch_department(self, tenant_id: str, department: str) -> list[dict]:
        """某部门能看到的条目：本部门 + 全公司公开。"""
        return await self._query(
            "SELECT tenant_id, department, subject, question, answer "
            "FROM faq_entries WHERE tenant_id = $1 AND department = ANY($2::text[]) ORDER BY id",
            tenant_id,
            [department, PUBLIC],
        )

    def _index_scopes(self, records: list[dict]) -> None:
        """按范围把记录分组建索引。

        同一批记录会被复用进多个范围：
            __all__     租户全量
            __public__  只有 '*' 的
            具体部门     该部门 + '*'
        """
        by_tenant: dict[str, list[dict]] = {}
        for row in records:
            by_tenant.setdefault(row["tenant_id"], []).append(row)

        for tenant, rows in by_tenant.items():
            self._indexes[(tenant, SCOPE_ALL)] = _build_index(rows)
            public = [r for r in rows if r.get("department", PUBLIC) == PUBLIC]
            self._indexes[(tenant, SCOPE_PUBLIC)] = _build_index(public)

            for dept in {r.get("department", PUBLIC) for r in rows} - {PUBLIC}:
                visible = [r for r in rows if r.get("department", PUBLIC) in (dept, PUBLIC)]
                self._indexes[(tenant, dept)] = _build_index(visible)

    async def reload(self) -> int:
        """载入全部 FAQ，按范围建索引（优先用 Redis 里的分词结果省掉 jieba）。"""
        cached = await load_faq_index()
        if cached is not None:
            rows, tokens = cached
            records: list[dict] = []
            for row, token in zip(rows, tokens, strict=False):
                item = self._row_to_dict(row)
                item["_tokens"] = token
                records.append(item)
            self._indexes = {}
            self._index_scopes(records)
            logger.info(
                "faq.loaded_from_redis",
                scopes=len(self._indexes),
                total=len(records),
            )
            return len(records)

        records = await self._query(
            "SELECT tenant_id, department, subject, question, answer FROM faq_entries ORDER BY id"
        )
        self._indexes = {}
        self._index_scopes(records)

        flat_rows = [
            [r["tenant_id"], r.get("department", PUBLIC), r["subject"], r["question"], r["answer"]]
            for r in records
        ]
        flat_tokens = [tokenize(r["question"]) for r in records]
        await save_faq_index(flat_rows, flat_tokens)

        logger.info("faq.loaded_from_postgres", scopes=len(self._indexes), total=len(records))
        return len(records)

    def _row_to_dict(self, row: list[str]) -> dict:
        """兼容两种缓存形状。

        老：[tenant, subject, question, answer]
        新：[tenant, department, subject, question, answer]
        """
        if len(row) == 5:
            tenant, dept, subject, question, answer = row
        else:  # 老缓存没有部门列
            tenant, subject, question, answer = row
            dept = PUBLIC
        return {
            "tenant_id": tenant,
            "department": dept,
            "subject": subject,
            "question": question,
            "answer": answer,
        }

    async def _index_for(self, tenant_id: str, department: str | None, role: str | None) -> _Index:
        """取某个范围（租户 × 部门）的索引；没有就回库建一份。

        新部门随时可能出现（用户改部门、脚本新导入），所以不能假设 reload 已经覆盖。
        """
        tenant = tenant_id or self.settings.default_tenant_id
        scope = resolve_scope(department, role)
        index = self._indexes.get((tenant, scope))
        if index is not None and index.size > 0:
            return index

        if scope == SCOPE_ALL:
            records = await self._fetch_all(tenant)
        elif scope == SCOPE_PUBLIC:
            records = await self._fetch_public(tenant)
        else:
            records = await self._fetch_department(tenant, scope)
        index = _build_index(records)
        self._indexes[(tenant, scope)] = index
        logger.info("faq.index_built", tenant=tenant, scope=scope, questions=index.size)
        return index

    # ── 检索 ────────────────────────────────────────────────
    async def search(
        self,
        query: str,
        tenant_id: str | None = None,
        department: str | None = None,
        role: str | None = None,
        threshold: float | None = None,
    ) -> FaqHit | None:
        """在「本租户 + 本部门可见」的 FAQ 里找；未达阈值返回 None。"""
        if not query:
            return None
        tenant = tenant_id or self.settings.default_tenant_id
        scope = resolve_scope(department, role)
        index = await self._index_for(tenant, department, role)
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
                scope=scope,
                best_score=round(best_score, 3),
                threshold=limit,
            )
            return None

        logger.info(
            "faq.hit",
            query=query,
            tenant=tenant,
            scope=scope,
            matched=index.questions[best],
            score=round(best_score, 3),
        )
        return FaqHit(
            question=index.questions[best],
            answer=index.answers[best],
            score=best_score,
            subject=index.subjects[best],
            department=index.departments[best],
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
