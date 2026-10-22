"""用户表与账号操作。

只做三件事：建表、建账号、按用户名查。密码只存 scrypt 哈希，明文不落库也不进日志。
"""

from __future__ import annotations

import asyncio

import asyncpg

from config import Settings, get_settings
from core.logger import get_logger
from core.security import hash_password

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash TEXT         NOT NULL,
    role          VARCHAR(16)  NOT NULL DEFAULT 'user',
    tenant_id     VARCHAR(64)  NOT NULL DEFAULT 'tenant_default',
    is_active     BOOLEAN      NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
"""


class UserStore:
    """用户表的读写。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ready = False

    async def _connect(self) -> asyncpg.Connection:
        s = self.settings
        return await asyncpg.connect(
            host=s.db_host,
            port=s.db_port,
            user=s.db_user,
            password=s.db_password,
            database=s.db_name,
        )

    async def ensure_table(self) -> None:
        if self._ready:
            return
        conn = await self._connect()
        try:
            await conn.execute(CREATE_TABLE_SQL)
            self._ready = True
            logger.info("users.table_ready")
        finally:
            await conn.close()

    async def create(
        self,
        username: str,
        password: str,
        role: str = "user",
        tenant_id: str = "tenant_default",
    ) -> dict | None:
        """建账号；用户名已存在返回 None。哈希是 CPU 密集操作，丢线程池。"""
        await self.ensure_table()
        password_hash = await asyncio.to_thread(hash_password, password)
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO users (username, password_hash, role, tenant_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (username) DO NOTHING
                RETURNING id, username, role, tenant_id
                """,
                username,
                password_hash,
                role,
                tenant_id,
            )
        finally:
            await conn.close()
        if row is None:
            logger.info("users.create_conflict", username=username)
            return None
        logger.info("users.created", username=username, role=role)
        return dict(row)

    async def get_by_username(self, username: str) -> dict | None:
        """登录用：返回含 password_hash 的完整记录。"""
        await self.ensure_table()
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, username, password_hash, role, tenant_id, is_active
                FROM users WHERE username = $1 LIMIT 1
                """,
                username,
            )
        finally:
            await conn.close()
        return dict(row) if row else None

    async def count(self) -> int:
        await self.ensure_table()
        conn = await self._connect()
        try:
            value = await conn.fetchval("SELECT count(*) FROM users")
            return int(value or 0)
        finally:
            await conn.close()


_user_store: UserStore | None = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store
