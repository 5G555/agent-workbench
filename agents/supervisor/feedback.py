"""路由反馈：记录用户对路由结果的纠正。

为什么要记：
    41 条评测集只能证明"见过的问题路由对了"。
    真实误判是长尾的——只能靠线上"用户纠正"这种信号捞出来，
    再补进规则或评测集。没有这个回路，阈值和规则就没法随使用情况调整。

记录什么（只记判断依据，不存敏感内容原文之外的东西）：
    question        问题原文（补规则要用）
    routed_agent    当时路由到哪
    corrected_agent 用户认为应该去哪
    confidence      当时的置信度
    reason          路由理由
"""

from __future__ import annotations

from typing import Any

from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS route_feedback (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(128) DEFAULT '',
    question TEXT NOT NULL,
    routed_agent VARCHAR(32) NOT NULL,
    corrected_agent VARCHAR(32) NOT NULL,
    confidence DOUBLE PRECISION DEFAULT 0,
    reason TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_route_feedback_tenant ON route_feedback (tenant_id, created_at DESC);
"""


class FeedbackStore:
    """路由纠正记录。"""

    async def ensure_table(self) -> None:
        conn = await lease_connection()
        try:
            await conn.execute(DDL)
        finally:
            await conn.close()

    async def record(
        self,
        tenant_id: str,
        user_id: str,
        question: str,
        routed_agent: str,
        corrected_agent: str,
        session_id: str = "",
        confidence: float = 0.0,
        reason: str = "",
    ) -> int:
        conn = await lease_connection()
        try:
            row_id = await conn.fetchval(
                "INSERT INTO route_feedback "
                "(tenant_id, user_id, session_id, question, routed_agent, "
                " corrected_agent, confidence, reason) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id",
                tenant_id,
                user_id,
                session_id,
                question[:1000],
                routed_agent,
                corrected_agent,
                confidence,
                reason[:500],
            )
        finally:
            await conn.close()
        logger.info(
            "route_feedback.recorded",
            routed=routed_agent,
            corrected=corrected_agent,
            question=question[:40],
        )
        return int(row_id or 0)

    async def summary(self, tenant_id: str, limit: int = 200) -> dict[str, Any]:
        conn = await lease_connection()
        try:
            rows = await conn.fetch(
                "SELECT routed_agent, corrected_agent, count(*) AS n "
                "FROM route_feedback WHERE tenant_id = $1 "
                "GROUP BY 1, 2 ORDER BY n DESC",
                tenant_id,
            )
            items = await conn.fetch(
                "SELECT question, routed_agent, corrected_agent, created_at "
                "FROM route_feedback WHERE tenant_id = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                tenant_id,
                limit,
            )
        finally:
            await conn.close()
        return {
            "pairs": [
                {
                    "from": r["routed_agent"],
                    "to": r["corrected_agent"],
                    "count": int(r["n"]),
                }
                for r in rows
            ],
            "items": [
                {
                    "question": r["question"],
                    "from": r["routed_agent"],
                    "to": r["corrected_agent"],
                    "at": r["created_at"].isoformat(),
                }
                for r in items
            ],
        }


_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
