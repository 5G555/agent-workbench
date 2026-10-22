"""API 路由汇总。

v1 下每个 Agent 一个 router，全部挂在 /api/v1 之下。
auth 不需要鉴权（它本身是签发 token 的地方），qa 下面的接口都要。
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1 import auth, qa

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
api_router.include_router(qa.router, prefix="/v1/qa", tags=["qa"])
