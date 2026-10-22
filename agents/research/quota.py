"""联网检索的租户级日额度。

为什么需要：
    qa 的检索是本地 Milvus，成本固定；而联网搜索按次计费，一次 deep 问答最多
    打 1 次搜索 + 2 次抓页。没有上限时，一个租户写个循环就能把账单刷爆。

为什么按"天"而不是"分钟"：
    分钟级限流挡的是突发流量（已有 `rate_limit_per_minute` 在做），
    日额度挡的是成本总量，两者是不同维度。

设计取舍：
    - 计数键按租户 + 日期，Redis INCR 天然原子
    - 超额时先 INCR 再 DECR 回退，保证已用数不失真
    - Redis 不可用时**放行**并告警：额度组件故障不该让所有人用不了联网
    - 默认额度给得很宽松（500 次/天），先保证功能可用，需要收紧时改配置
"""

from __future__ import annotations

from datetime import UTC, datetime

from agents.qa.cache import get_redis
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "quota:web"
# 键保留两天，跨天自然过期，不用额外的清理任务
TTL_SECONDS = 2 * 86400


def quota_key(tenant_id: str, day: str | None = None) -> str:
    """租户当天的额度键。day 传 YYYY-MM-DD，便于测试。"""
    date_part = day or datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{KEY_PREFIX}:{tenant_id}:{date_part}"


async def consume_web_quota(tenant_id: str, limit: int | None = None) -> tuple[bool, int]:
    """消耗一次联网额度，返回 (是否放行, 剩余次数)。

    limit <= 0 表示不限量；此时不写 Redis，直接放行。
    """
    settings = get_settings()
    cap = settings.web_daily_quota_per_tenant if limit is None else limit
    if cap <= 0:
        return True, -1  # -1 表示不限

    client = await get_redis()
    if client is None:
        logger.warning("quota.degraded", tenant=tenant_id)
        return True, -1

    key = quota_key(tenant_id)
    try:
        used = int(await client.incr(key))
        if used == 1:
            await client.expire(key, TTL_SECONDS)
        if used > cap:
            # 超了就退回，避免把已用数越推越高（否则用户永远看不到恢复）
            await client.decr(key)
            logger.info("quota.exceeded", tenant=tenant_id, used=used - 1, limit=cap)
            return False, 0
        return True, max(0, cap - used)
    except Exception as exc:
        logger.warning("quota.degraded", tenant=tenant_id, error=str(exc))
        return True, -1


async def web_quota_status(tenant_id: str, limit: int | None = None) -> dict[str, int]:
    """查询当前用量（不消耗额度）。"""
    settings = get_settings()
    cap = settings.web_daily_quota_per_tenant if limit is None else limit
    if cap <= 0:
        return {"limit": 0, "used": 0, "remaining": -1}

    client = await get_redis()
    if client is None:
        return {"limit": cap, "used": 0, "remaining": -1}

    try:
        used = int(await client.get(quota_key(tenant_id)) or 0)
    except Exception as exc:
        logger.warning("quota.status_failed", tenant=tenant_id, error=str(exc))
        return {"limit": cap, "used": 0, "remaining": -1}
    return {"limit": cap, "used": used, "remaining": max(0, cap - used)}
