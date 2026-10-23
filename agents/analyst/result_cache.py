"""分析结果缓存：同样的数据 + 同样的问题，不重复跑。

key 由四部分组成：
    tenant_id + task_ids + 问题文本 + 数据指纹
数据指纹是每个任务的 (记录条数, 最大记录 id)：
    重新抓了同一任务 → 指纹变 → 自动失效（不需要手动清缓存）

为什么缓存在 Redis 而不是进程内存：
    多 worker / 重启后仍然有效；分析一次 4~8 秒，值得跨进程复用。
缓存不可用时静默跳过（降级为每次都算），不影响主流程。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agents.qa.cache import get_redis
from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

CACHE_PREFIX = "analyst:cache"
CACHE_TTL_SECONDS = 3600  # 1 小时


async def data_fingerprint(task_ids: list[int]) -> str:
    """每个任务的 (条数, 最大 id)，拼成指纹；查库失败返回空串（不缓存）。"""
    if not task_ids:
        return ""
    conn = await lease_connection()
    try:
        rows = await conn.fetch(
            "SELECT task_id, count(*) AS n, coalesce(max(id), 0) AS max_id "
            "FROM crawl_records WHERE task_id = ANY($1::bigint[]) GROUP BY task_id",
            task_ids,
        )
    except Exception as exc:
        logger.warning("analyst.fingerprint_failed", error=str(exc)[:120])
        return ""
    finally:
        await conn.close()

    parts = sorted(f"{r['task_id']}:{r['n']}:{r['max_id']}" for r in rows)
    return "|".join(parts)


def cache_key(
    tenant_id: str, task_ids: list[int], question: str, fingerprint: str, chart: str = ""
) -> str:
    raw = json.dumps(
        {
            "t": tenant_id,
            "tasks": sorted(int(t) for t in task_ids),
            "q": " ".join((question or "").split()).lower(),
            "fp": fingerprint,
            "chart": chart,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{CACHE_PREFIX}:{tenant_id}:{digest}"


async def get_cached(key: str) -> dict[str, Any] | None:
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(key)
    except Exception as exc:
        logger.warning("analyst.cache_get_failed", error=str(exc)[:120])
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    logger.info("analyst.cache_hit", key=key[-12:])
    return data if isinstance(data, dict) else None


async def put_cached(key: str, payload: dict[str, Any]) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.set(key, json.dumps(payload, ensure_ascii=False), ex=CACHE_TTL_SECONDS)
    except Exception as exc:  # 缓存写失败不影响结果
        logger.warning("analyst.cache_set_failed", error=str(exc)[:120])
