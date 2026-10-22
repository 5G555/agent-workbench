"""API 路由汇总。

v1 下每个 Agent 一个 router，全部挂在 /api/v1 之下。
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1 import qa

api_router = APIRouter()
api_router.include_router(qa.router, prefix="/v1/qa", tags=["qa"])
