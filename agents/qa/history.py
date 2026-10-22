"""会话历史：把一问一答落到 Postgres。

用途有二：
  1. 前端查历史（/api/v1/qa/history）
  2. 多轮对话：把最近几条拼成 messages 注入图的初始状态

表结构与知识库解耦，独立于 faq_entries / Milvus。
"""

from __future__ import annotations

import asyncio
import hashlib
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

-- 会话元信息：标题由首问自动生成，也可手动改名
CREATE TABLE IF NOT EXISTS qa_sessions (
    session_id VARCHAR(128) PRIMARY KEY,
    user_id    VARCHAR(128) NOT NULL DEFAULT '',
    title      VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 答案评价：同一会话里同一条答案只保留最后一次评价
CREATE TABLE IF NOT EXISTS qa_feedback (
    id          BIGSERIAL PRIMARY KEY,
    session_id  VARCHAR(128) NOT NULL,
    user_id     VARCHAR(128) NOT NULL DEFAULT '',
    rating      SMALLINT     NOT NULL,
    answer_hash VARCHAR(64)  NOT NULL,
    comment     TEXT         NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (session_id, answer_hash)
);
CREATE INDEX IF NOT EXISTS ix_qa_feedback_session ON qa_feedback (session_id);
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

    async def list_sessions(self, prefix: str, limit: int = 50) -> list[dict]:
        """列出某用户的所有会话（按最近活动倒序），带上标题。

        prefix 传 "{user_id}:"，会话键在存的时候就带了这个前缀，天然隔离。
        标题为空时（加 qa_sessions 表之前建的老会话）用首条用户消息兜底，
        免得前端只能显示一串 session_id。
        """
        await self.ensure_table()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT m.session_id,
                       max(m.created_at) AS last_at,
                       count(*)          AS messages,
                       coalesce(
                           nullif(s.title, ''),
                           left((
                               SELECT content FROM qa_messages q
                               WHERE q.session_id = m.session_id AND q.role = 'user'
                               ORDER BY q.id LIMIT 1
                           ), 24),
                           ''
                       ) AS title
                FROM qa_messages m
                LEFT JOIN qa_sessions s ON s.session_id = m.session_id
                WHERE m.session_id LIKE $1
                GROUP BY m.session_id, s.title
                ORDER BY last_at DESC
                LIMIT $2
                """,
                f"{prefix}%",
                limit,
            )
        finally:
            await conn.close()
        return [
            {
                "session_id": r["session_id"][len(prefix) :],
                "last_at": r["last_at"].isoformat(),
                "messages": int(r["messages"]),
                "title": r["title"] or "",
            }
            for r in rows
        ]

    async def touch_session(self, session_id: str, user_id: str, first_message: str) -> None:
        """会话首次出现时登记，并用首问自动生成标题（已有的标题不覆盖）。"""
        await self.ensure_table()
        title = first_message.strip().replace("\n", " ")[:24]
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO qa_sessions (session_id, user_id, title)
                VALUES ($1, $2, $3)
                ON CONFLICT (session_id) DO UPDATE
                   SET updated_at = now(),
                       title = CASE WHEN qa_sessions.title = '' THEN EXCLUDED.title
                                    ELSE qa_sessions.title END
                """,
                session_id,
                user_id,
                title,
            )
        finally:
            await conn.close()

    async def rename_session(self, session_id: str, title: str) -> bool:
        await self.ensure_table()
        conn = await self._connect()
        try:
            result = await conn.execute(
                "UPDATE qa_sessions SET title = $2, updated_at = now() WHERE session_id = $1",
                session_id,
                title.strip()[:128],
            )
        finally:
            await conn.close()
        return result.endswith("1")

    async def save_feedback(
        self, session_id: str, user_id: str, rating: int, answer: str, comment: str = ""
    ) -> None:
        """记录一条答案评价；同一会话里同一条答案重复评价时覆盖。"""
        await self.ensure_table()
        digest = hashlib.sha256(answer.strip().encode("utf-8")).hexdigest()[:64]
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO qa_feedback (session_id, user_id, rating, answer_hash, comment)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id, answer_hash)
                DO UPDATE SET rating = EXCLUDED.rating, comment = EXCLUDED.comment
                """,
                session_id,
                user_id,
                rating,
                digest,
                comment,
            )
        finally:
            await conn.close()

    async def feedback_stats(self, prefix: str) -> dict[str, int]:
        await self.ensure_table()
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT count(*) FILTER (WHERE rating > 0) AS up,
                       count(*) FILTER (WHERE rating < 0) AS down
                FROM qa_feedback WHERE session_id LIKE $1
                """,
                f"{prefix}%",
            )
        finally:
            await conn.close()
        return {"up": int(row["up"] or 0), "down": int(row["down"] or 0)}

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
