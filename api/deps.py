"""FastAPI 依赖：当前用户、限流。

鉴权默认开启；把 .env 里 AUTH_ENABLED 改成 false 可关掉（仅本地调试用）。
"""

from __future__ import annotations

import time

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agents.qa.cache import get_redis
from config import get_settings
from core.logger import get_logger
from core.security import decode_token

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, str]:
    """解析 Bearer Token，返回 {"user_id", "role", "tenant_id"}。"""
    s = get_settings()

    if not s.auth_enabled:  # 本地调试模式：不校验，直接用匿名身份
        return {"user_id": "anonymous", "role": "dev", "tenant_id": s.default_tenant_id}

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="缺少或无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not credentials.credentials:
        raise unauthorized

    try:
        payload = decode_token(credentials.credentials)
    except Exception as exc:
        logger.info("auth.invalid_token", error=str(exc))
        raise unauthorized from exc

    return {
        "user_id": str(payload["sub"]),
        "role": str(payload.get("role", "user")),
        "tenant_id": str(payload.get("tenant_id", s.default_tenant_id)),
    }


async def rate_limit(request: Request, user: dict[str, str] = Depends(get_current_user)) -> None:
    """按用户做每分钟限流。Redis 不可用时直接放行（不因限流组件故障阻断业务）。"""
    s = get_settings()
    limit = s.rate_limit_per_minute
    if limit <= 0:
        return

    client = await get_redis()
    if client is None:
        return

    window = int(time.time() // 60)
    key = f"rate:{user['user_id']}:{window}"
    try:
        used = int(await client.incr(key))
        if used == 1:
            await client.expire(key, 65)
    except Exception as exc:
        logger.warning("rate_limit.degraded", error=str(exc))
        return

    if used > limit:
        logger.info("rate_limit.exceeded", user_id=user["user_id"], used=used, limit=limit)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，每分钟上限 {limit} 次",
        )
