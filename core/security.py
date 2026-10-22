"""鉴权基础件：密码哈希与 JWT 签发/校验。

用标准库 hashlib.scrypt：零额外依赖、不会随第三方版本变化而崩。

JWT 用 PyJWT。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import jwt

from config import Settings, get_settings
from core.exceptions import AuthenticationError
from core.logger import get_logger

logger = get_logger(__name__)

# scrypt 参数：n=2^14 在普通 CPU 上约 50ms，够用且不会拖慢登录
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32


def hash_password(password: str) -> str:
    """生成 `scrypt$n$r$p$salt_b64$hash_b64` 格式的字符串。"""
    if not password:
        raise ValueError("密码不能为空")
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LEN,
    )
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    """校验密码；用 compare_digest 做常数时间比较，避免时序侧信道。"""
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(hash_b64)),
        )
        return hmac.compare_digest(digest, base64.b64decode(hash_b64))
    except Exception:
        return False


def _encode(
    user_id: str,
    token_type: str,
    role: str,
    tenant_id: str,
    expires_seconds: int,
    settings: Settings,
    department: str = "*",
) -> tuple[str, str]:
    """统一签发入口，返回 (token, jti)。"""
    now = datetime.now(UTC)
    jti = uuid4().hex
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "typ": token_type,  # access / refresh，防止两种令牌混用
        "jti": jti,  # 唯一 ID，吊销时按它拉黑
        "role": role,
        "tenant_id": tenant_id,
        "department": department,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_seconds)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm), jti


def create_access_token(
    user_id: str,
    role: str = "user",
    tenant_id: str = "tenant_default",
    settings: Settings | None = None,
    expires_minutes: int | None = None,
    department: str = "*",
) -> tuple[str, int]:
    """短效访问令牌，返回 (token, 有效期秒数)。"""
    s = settings or get_settings()
    minutes = expires_minutes or s.jwt_access_token_expire_minutes
    token, _ = _encode(user_id, "access", role, tenant_id, minutes * 60, s, department)
    return token, minutes * 60


def create_refresh_token(
    user_id: str,
    role: str = "user",
    tenant_id: str = "tenant_default",
    settings: Settings | None = None,
    expires_days: int | None = None,
    department: str = "*",
) -> tuple[str, str]:
    """长效刷新令牌，返回 (token, jti)。"""
    s = settings or get_settings()
    days = expires_days or s.jwt_refresh_token_expire_days
    return _encode(user_id, "refresh", role, tenant_id, days * 86400, s, department)


def decode_token(
    token: str,
    expected_type: str = "access",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """校验并解析 JWT；类型不符或失效都抛 AuthenticationError（调用方转 401）。"""
    s = settings or get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret_key, algorithms=[s.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("登录已过期，请重新登录") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("无效的认证凭证") from exc

    if not payload.get("sub"):
        raise AuthenticationError("令牌缺少用户标识")
    if payload.get("typ", "access") != expected_type:
        # 用 refresh 当 access 用（或反之）一律拒绝
        raise AuthenticationError("令牌类型不匹配")
    return payload
