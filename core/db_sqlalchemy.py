"""SQLAlchemy 接入（第一步：与应用共存的引擎与会话工厂）。

先说明一个技术前提，避免误判：
    asyncpg 的 create_pool 有 min_size/max_size，池是**有上限**的，
    不会"无限爆炸"。真正的风险是两类：
        ① 池被占满后请求排队（表现为卡住），而不是崩溃
        ② 某条路径借了连接忘了还，泄漏到超时——这才是会"看起来爆炸"的原因
    SQLAlchemy 本身也是跑在 asyncpg 上的，它不会自动消除这两个问题；
    它的价值在于：统一会话生命周期（async with 自动归还）、连接预检、
    DDL 版本化（Alembic）、以及后续换库时的抽象层。

因此接入方式定为"逐步替换"而不是"一次性重写"：
    本文件先提供 engine + session_scope，供新代码使用；
    旧代码继续用 core/db.py 的 asyncpg 池，两边共存、各自有上限。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def build_dsn(settings: Settings) -> str:
    """SQLAlchemy 用的 async DSN。"""
    return (
        f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """全局异步引擎（懒加载）。

    池参数的含义与 asyncpg 一致，但多了两个关键项：
        pool_pre_ping   取连接前先探活，避免用到被服务端断掉的死连接
        pool_timeout    等不到连接时**报错**而不是无限等待——
                        没有它，池打满时请求会挂住，很难定位
    """
    global _engine, _session_factory
    if _engine is None:
        s = settings or get_settings()
        _engine = create_async_engine(
            build_dsn(s),
            pool_size=s.db_pool_min,
            max_overflow=max(s.db_pool_max - s.db_pool_min, 0),
            pool_timeout=5,
            pool_recycle=1800,
            pool_pre_ping=True,
            future=True,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
        logger.info(
            "sqlalchemy.engine_ready",
            pool_size=s.db_pool_min,
            max_overflow=max(s.db_pool_max - s.db_pool_min, 0),
        )
    return _engine


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """事务性会话：正常结束自动提交，异常自动回滚，退出必然归还连接。

    用法：
        async with session_scope() as session:
            rows = (await session.execute(text("SELECT 1"))).all()

    为什么统一用它：手写 try/finally 归还连接，漏一处就是一次泄漏。
    """
    get_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def sqlalchemy_pool_stats() -> dict[str, Any]:
    """池状态，供 /health 与排障使用。"""
    if _engine is None:
        return {"initialized": False}
    # 基类 Pool 没有 size/checkedout/overflow（只有 QueuePool 有），
    # 运行时是 QueuePool，类型上用 getattr 取，避免为类型检查写死实现类
    pool = _engine.pool
    return {
        "initialized": True,
        "size": getattr(pool, "size", lambda: -1)(),
        "checked_out": getattr(pool, "checkedout", lambda: -1)(),
        "overflow": getattr(pool, "overflow", lambda: -1)(),
        "timeout_seconds": 5,
    }


async def dispose_engine() -> None:
    """应用退出时释放。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("sqlalchemy.engine_disposed")
