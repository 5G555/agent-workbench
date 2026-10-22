"""认证接口：注册、登录、刷新、登出、查当前用户。

令牌模型（参考 OAuth2 的双令牌思路）：
    access token   短效（默认 30 分钟），每次请求都带
    refresh token  长效（默认 7 天），只用来换新的 access
刷新时做**轮换**：旧的 refresh 立即拉黑，换发新的，旧 token 无法再用。
登出 = 拉黑当前 refresh token。

密码用标准库 scrypt 哈希，不引入 passlib/bcrypt 以避开版本兼容坑。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import get_current_user
from config import get_settings
from core import token_store
from core.logger import get_logger
from core.security import create_access_token, create_refresh_token, decode_token, verify_password
from core.users import get_user_store

logger = get_logger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码，至少 6 位")
    # 生产环境租户应由管理员开通，这里放开只是为了本地能建多个租户做隔离验证
    tenant_id: str = Field(default="tenant_default", max_length=64, description="所属租户")


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="登录时拿到的刷新令牌")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str = ""
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int = 0
    user_id: str
    role: str
    tenant_id: str


def _issue_tokens(user_id: str, role: str, tenant_id: str) -> TokenResponse:
    """签发一对令牌。"""
    settings = get_settings()
    access, access_ttl = create_access_token(user_id=user_id, role=role, tenant_id=tenant_id)
    refresh, _ = create_refresh_token(user_id=user_id, role=role, tenant_id=tenant_id)
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=access_ttl,
        refresh_expires_in=settings.jwt_refresh_token_expire_days * 86400,
        user_id=user_id,
        role=role,
        tenant_id=tenant_id,
    )


def _remaining_seconds(payload: dict) -> int:
    """算 token 还剩多少秒（用于设置黑名单 TTL）。"""
    exp = int(payload.get("exp", 0))
    return max(0, exp - int(datetime.now(UTC).timestamp()))


@router.post("/register", response_model=TokenResponse, summary="注册并直接登录")
async def register(request: RegisterRequest) -> TokenResponse:
    store = get_user_store()
    user = await store.create(request.username, request.password, tenant_id=request.tenant_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
    return _issue_tokens(str(user["id"]), user["role"], user["tenant_id"])


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

    logger.info("auth.login_success", user_id=str(user["id"]), username=user["username"])
    return _issue_tokens(str(user["id"]), user["role"], user["tenant_id"])


@router.post("/refresh", response_model=TokenResponse, summary="用 refresh 换新令牌（轮换）")
async def refresh(request: RefreshRequest) -> TokenResponse:
    try:
        payload = decode_token(request.refresh_token, expected_type="refresh")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="刷新令牌无效或已过期"
        ) from exc

    jti = str(payload.get("jti", ""))
    if await token_store.is_revoked(jti):
        logger.info("auth.refresh_revoked", jti=jti)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="刷新令牌已失效")

    # 轮换：先作废旧 token，再发新的
    await token_store.revoke(jti, _remaining_seconds(payload))
    logger.info("auth.refreshed", user_id=str(payload["sub"]))
    return _issue_tokens(
        str(payload["sub"]),
        str(payload.get("role", "user")),
        str(payload.get("tenant_id", "tenant_default")),
    )


@router.post("/logout", summary="登出（作废当前 refresh）")
async def logout(request: RefreshRequest) -> dict[str, object]:
    try:
        payload = decode_token(request.refresh_token, expected_type="refresh")
    except Exception:
        # 已经无效的 token 再登出一次，直接视为成功
        return {"revoked": False}
    revoked = await token_store.revoke(str(payload.get("jti", "")), _remaining_seconds(payload))
    return {"revoked": revoked}


@router.get("/me", summary="当前登录用户")
async def me(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
    settings = get_settings()
    return {**user, "auth_enabled": str(settings.auth_enabled).lower()}
