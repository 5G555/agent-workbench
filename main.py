"""FastAPI 入口。

uvicorn main:app --reload --port 8000
python -m uvicorn main:app --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.router import api_router
from config import get_settings
from core.db import close_pool, get_pool
from core.logger import configure_logging, get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时初始化日志与连接池，退出时释放。"""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "app.startup",
        env=settings.app_env,
        host=settings.app_host,
        port=settings.app_port,
        milvus=f"{settings.milvus_host}:{settings.milvus_port}",
        cache_enabled=settings.cache_enabled,
        auth_enabled=settings.auth_enabled,
    )
    try:
        await get_pool()
    except Exception as exc:  # 数据库暂时不可用不该阻止启动，首个请求会再试
        logger.warning("db.pool_warmup_failed", error=str(exc))
    yield
    await close_pool()
    logger.info("app.shutdown")


app = FastAPI(title="Myagent", version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """前端页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", summary="健康检查", tags=["ops"])
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "env": settings.app_env,
        "milvus": f"{settings.milvus_host}:{settings.milvus_port}",
        "cache_enabled": settings.cache_enabled,
    }
