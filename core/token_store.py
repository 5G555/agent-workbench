"""令牌吊销表（Redis）。

refresh token 是长效的（默认 7 天），一旦泄露就得能立刻作废。
做法：把 jti 写进黑名单，TTL 设成该 token 的剩余寿命——过期后自然清理，不用扫描。

Redis 不可用时降级为"不吊销"（只警告），保证登录/刷新不因缓存组件故障而中断；
生产环境应确保 Redis 高可用，这里的降级只是止损。
"""

from __future__ import annotations

from agents.qa.cache import get_redis
from core.logger import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "jwt:revoked:"


async def revoke(jti: str, ttl_seconds: int) -> bool:
    """把 jti 拉黑，ttl 设为该 token 剩余有效期。"""
    if not jti or ttl_seconds <= 0:
        return False
    client = await get_redis()
    if client is None:
        logger.warning("token.revoke_degraded", jti=jti)
        return False
    try:
        await client.set(f"{KEY_PREFIX}{jti}", "1", ex=ttl_seconds)
        logger.info("token.revoked", jti=jti, ttl=ttl_seconds)
        return True
    except Exception as exc:
        logger.warning("token.revoke_failed", jti=jti, error=str(exc))
        return False


async def is_revoked(jti: str) -> bool:
    """是否已吊销；Redis 不可用时返回 False（放行）。"""
    if not jti:
        return False
    client = await get_redis()
    if client is None:
        return False
    try:
        return bool(await client.exists(f"{KEY_PREFIX}{jti}"))
    except Exception as exc:
        logger.warning("token.check_failed", jti=jti, error=str(exc))
        return False
