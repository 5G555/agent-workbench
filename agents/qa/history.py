"""会话历史：把一问一答落到 Postgres。

用途有二：
  1. 前端查历史（/api/v1/qa/history）
  2. 多轮对话：把最近几条拼成 messages 注入图的初始状态

表结构与知识库解耦，独立于 faq_entries / Milvus。
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import Settings, get_settings
from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS qa_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  VARCHAR(128) NOT NULL,
    user_id     VARCHAR(128) NOT NULL DEFAULT '',
    role        VARCHAR(16)  NOT NULL,
    content     TEXT         NOT NULL,
    answer_mode VARCHAR(16)  NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_qa_messages_session ON qa_messages (session_id, id);
"""


class HistoryStore:
    """会话消息的读写。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ready = False

    async def _connect(self) -> Any:
        return await lease_connection()

    async def ensure_table(self) -> None:
        if self._ready:
            return
        conn = await self._connect()
        try:
            await conn.execute(CREATE_TABLE_SQL)
            self._ready = True
            logger.info("history.table_ready")
        finally:
            await conn.close()

    async def append(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        answer_mode: str = "",
    ) -> None:
        """追加一条消息。role: user / assistant。"""
        if not content:
            return
        await self.ensure_table()
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO qa_messages (session_id, user_id, role, content, answer_mode)
                VALUES ($1, $2, $3, $4, $5)
                """,
                session_id,
                user_id,
                role,
                content,
                answer_mode,
            )
        finally:
            await conn.close()

    async def recent(self, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        """最近 N 条（按时间正序返回），用于拼多轮上下文。"""
        await self.ensure_table()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM qa_messages
                    WHERE session_id = $1 ORDER BY id DESC LIMIT $2
                ) t ORDER BY id ASC
                """,
                session_id,
                limit,
            )
        finally:
            await conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, str]]:
        """历史接口用：最近 N 条，时间正序。"""
        await self.ensure_table()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT role, content, answer_mode, created_at FROM (
                    SELECT * FROM qa_messages WHERE session_id = $1 ORDER BY id DESC LIMIT $2
                ) t ORDER BY id ASC
                """,
                session_id,
                limit,
            )
        finally:
            await conn.close()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "answer_mode": r["answer_mode"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    async def clear(self, session_id: str) -> int:
        await self.ensure_table()
        conn = await self._connect()
        try:
            result = await conn.execute("DELETE FROM qa_messages WHERE session_id = $1", session_id)
        finally:
            await conn.close()
        # asyncpg 返回 "DELETE 3"
        return int(result.split()[-1]) if result else 0


_history_store: HistoryStore | None = None
_lock = asyncio.Lock()


def get_history_store() -> HistoryStore:
    global _history_store
    if _history_store is None:
        _history_store = HistoryStore()
    return _history_store
