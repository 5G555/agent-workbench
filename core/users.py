"""用户表与账号操作。

只做三件事：建表、建账号、按用户名查。密码只存 scrypt 哈希，明文不落库也不进日志。
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import Settings, get_settings
from core.db import lease_connection
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
    department    VARCHAR(64)  NOT NULL DEFAULT '*',
    is_active     BOOLEAN      NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
-- 迁移：细粒度权限引入部门维度，老表补列。'*' 表示全公司可见
ALTER TABLE users ADD COLUMN IF NOT EXISTS department VARCHAR(64) NOT NULL DEFAULT '*';
"""


class UserStore:
    """用户表的读写。"""

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
            logger.info("users.table_ready")
        finally:
            await conn.close()

    async def create(
        self,
        username: str,
        password: str,
        role: str = "user",
        tenant_id: str = "tenant_default",
        department: str = "*",
    ) -> dict | None:
        """建账号；用户名已存在返回 None。哈希是 CPU 密集操作，丢线程池。"""
        await self.ensure_table()
        password_hash = await asyncio.to_thread(hash_password, password)
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO users (username, password_hash, role, tenant_id, department)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (username) DO NOTHING
                RETURNING id, username, role, tenant_id, department
                """,
                username,
                password_hash,
                role,
                tenant_id,
                department,
            )
        finally:
            await conn.close()
        if row is None:
            logger.info("users.create_conflict", username=username)
            return None
        logger.info("users.created", username=username, role=role, department=department)
        return dict(row)

    async def get_by_username(self, username: str) -> dict | None:
        """登录用：返回含 password_hash 的完整记录。"""
        await self.ensure_table()
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, username, password_hash, role, tenant_id, department, is_active
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

    async def count_admins(self, tenant_id: str) -> int:
        """某个租户里有多少管理员。

        用途：判断"能不能自己批自己"。
            只有 1 个管理员时，强制双人复核等于让删除永远无法执行；
            有 2 个以上时才要求换一个人批。
        """
        await self.ensure_table()
        conn = await self._connect()
        try:
            value = await conn.fetchval(
                "SELECT count(*) FROM users WHERE tenant_id = $1 AND role = 'admin'",
                tenant_id,
            )
            return int(value or 0)
        finally:
            await conn.close()


_user_store: UserStore | None = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store
