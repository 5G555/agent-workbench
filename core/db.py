"""Postgres 连接池。

原来每个操作都 `asyncpg.connect()` 新建连接，一次问答要建 3~4 条（FAQ 查询、
历史读写、用户校验），高并发下会直接打爆 Postgres 的 max_connections。

这里提供一个进程级共享的连接池：
    async with get_pool().acquire() as conn:
        await conn.fetch(...)

池在首次使用时创建，应用退出时由 main.py 的 lifespan 关闭。
"""

from __future__ import annotations

import asyncio

import asyncpg
from asyncpg.pool import PoolConnectionProxy

from config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_pool: asyncpg.Pool | None = None
_lock = asyncio.Lock()


async def get_pool(settings: Settings | None = None) -> asyncpg.Pool:
    """取连接池（懒加载，全进程共用一个）。"""
    global _pool
    if _pool is not None:
        return _pool

    async with _lock:
        if _pool is not None:
            return _pool
        s = settings or get_settings()
        _pool = await asyncpg.create_pool(
            host=s.db_host,
            port=s.db_port,
            user=s.db_user,
            password=s.db_password,
            database=s.db_name,
            min_size=s.db_pool_min,
            max_size=s.db_pool_max,
            command_timeout=30,
            max_inactive_connection_lifetime=300,
        )
        logger.info(
            "db.pool_created",
            min=s.db_pool_min,
            max=s.db_pool_max,
            database=s.db_name,
        )
    return _pool


async def close_pool() -> None:
    """关闭连接池（应用退出时调用）。"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db.pool_closed")


def pool_stats() -> dict[str, int]:
    """当前池的使用情况，排查"连接泄漏"时用。"""
    if _pool is None:
        return {"size": 0, "free": 0, "used": 0}
    return {
        "size": _pool.get_size(),
        "free": _pool.get_idle_size(),
        "used": _pool.get_size() - _pool.get_idle_size(),
    }


class LeasedConnection:
    """从池里借出的连接，`.close()` 时归还而不是真关。

    为什么要这层包装：现有代码写的是
        conn = await _connect()
        try: ...
        finally: await conn.close()
    如果 _connect 直接返回池连接，`close()` 会真的把连接关掉，池就废了。
    包一层后，`.close()` 变成归还，调用点一行都不用改，也不会漏还。
    """

    __slots__ = ("_conn", "_pool")

    def __init__(self, conn: PoolConnectionProxy, pool: asyncpg.Pool) -> None:
        self._conn: PoolConnectionProxy | None = conn
        self._pool = pool

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    async def close(self) -> None:
        """归还连接（重复调用安全）。"""
        conn, pool = self._conn, self._pool
        if conn is None:
            return
        self._conn = None
        await pool.release(conn)


async def lease_connection() -> LeasedConnection:
    """借一条连接；用完调用 `.close()` 归还。"""
    pool = await get_pool()
    return LeasedConnection(await pool.acquire(), pool)
