"""API 路由汇总。

v1 下每个 Agent 一个 router，全部挂在 /api/v1 之下。
auth 不需要鉴权（它本身是签发 token 的地方），其余接口都要。
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1 import ask, auth, qa, research

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
# ask.py 里的路径本身就是 /ask 和 /ask/stream，所以前缀只挂到 /v1
api_router.include_router(ask.router, prefix="/v1", tags=["ask"])
api_router.include_router(qa.router, prefix="/v1/qa", tags=["qa"])
api_router.include_router(research.router, prefix="/v1/research", tags=["research"])
