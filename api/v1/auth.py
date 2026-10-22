"""认证接口：注册、登录、查当前用户。

JWT 方案参考 EduAgent：登录签发 token，后续请求带 Authorization: Bearer <token>。
密码用标准库 scrypt 哈希，不引入 passlib/bcrypt 以避开版本兼容坑。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import get_current_user
from config import get_settings
from core.logger import get_logger
from core.security import create_access_token, verify_password
from core.users import get_user_store

logger = get_logger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码，至少 6 位")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    role: str
    tenant_id: str


@router.post("/register", response_model=TokenResponse, summary="注册并直接登录")
async def register(request: RegisterRequest) -> TokenResponse:
    store = get_user_store()
    user = await store.create(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    token, expires_in = create_access_token(
        user_id=str(user["id"]), role=user["role"], tenant_id=user["tenant_id"]
    )
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=str(user["id"]),
        role=user["role"],
        tenant_id=user["tenant_id"],
    )


@router.post("/login", response_model=TokenResponse, summary="登录")
async def login(request: LoginRequest) -> TokenResponse:
    store = get_user_store()
    user = await store.get_by_username(request.username)

    # 用户不存在与密码错误返回同一提示，避免暴露账号是否存在
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用")

    ok = await asyncio.to_thread(verify_password, request.password, user["password_hash"])
    if not ok:
        logger.info("auth.login_failed", username=request.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    token, expires_in = create_access_token(
        user_id=str(user["id"]), role=user["role"], tenant_id=user["tenant_id"]
    )
    logger.info("auth.login_success", user_id=str(user["id"]), username=user["username"])
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=str(user["id"]),
        role=user["role"],
        tenant_id=user["tenant_id"],
    )


@router.get("/me", summary="当前登录用户")
async def me(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
    settings = get_settings()
    return {**user, "auth_enabled": str(settings.auth_enabled).lower()}
