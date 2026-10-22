"""FastAPI 入口。

uvicorn main:app --reload --port 8000
python -m uvicorn main:app --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from api.router import api_router
from config import get_settings
from core.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时初始化日志，退出时留一条记录。"""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "app.startup",
        env=settings.app_env,
        host=settings.app_host,
        port=settings.app_port,
        milvus=f"{settings.milvus_host}:{settings.milvus_port}",
        cache_enabled=settings.cache_enabled,
    )
    yield
    logger.info("app.shutdown")


app = FastAPI(title="Myagent", version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api")


@app.get("/health", summary="健康检查", tags=["ops"])
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "env": settings.app_env,
        "milvus": f"{settings.milvus_host}:{settings.milvus_port}",
        "cache_enabled": settings.cache_enabled,
    }
