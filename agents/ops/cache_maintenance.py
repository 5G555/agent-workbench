r"""Redis 缓存的巡检与清理（脚本与 MCP 工具共用一份实现）。

为什么要抽出来：
    同一套逻辑现在有两个调用方——命令行（scripts/clean_cache.py）和
    运维 Agent 的 MCP 工具（clean_cache）。两边各写一份迟早会走样，
    尤其"哪些键算旧版本"这种规则一旦不一致，清理结果就对不上。

清理范围（只碰这几类，别的一概不碰）：
    qa:ans:*         问答答案（含 research 命名空间）
    faq:questions:*  FAQ 问题索引
    faq:tokens:*     FAQ 分词结果
    analyst:cache:*  分析结果缓存（按 租户+任务+问题+数据指纹，1 小时 TTL）

两类缓存的判定规则不同：
    QA 缓存带版本号，只删「版本不是当前版本」的残留
    分析缓存不带版本号，数据/问题变了 key 就变，所以按租户清或全清

判断依据：键名里的版本号 `:v(\d+)`，与当前版本不同即视为残留。
当前版本的缓存永远不删。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agents.qa.cache import (
    ANSWER_PREFIX,
    FAQ_QUESTIONS_KEY,
    FAQ_TOKENS_KEY,
    cache_version,
    get_redis,
)
from core.logger import get_logger

logger = get_logger(__name__)

# 两种键形状：
#   qa:ans:v27:s...:...          （无命名空间）
#   qa:ans:research:deep:v27:... （带命名空间）
#   faq:questions:v27
VERSION_PATTERN = re.compile(r":v(\d+)(?::|$)")

PATTERNS = [
    f"{ANSWER_PREFIX}:*",
    f"{FAQ_QUESTIONS_KEY}:*",
    f"{FAQ_TOKENS_KEY}:*",
]

# 分析结果缓存前缀（agents/analyst/result_cache.py 写入）
ANALYST_PREFIX = "analyst:cache"

# 允许清理的缓存类别
KIND_QA = "qa"
KIND_ANALYST = "analyst"
ALL_KINDS = (KIND_QA, KIND_ANALYST)

# 单次 DEL 的键数，避免一次命令太大会阻塞
BATCH_SIZE = 200


def key_version(key: str) -> int | None:
    """从键名里取出版本号；取不到返回 None。"""
    match = VERSION_PATTERN.search(key)
    return int(match.group(1)) if match else None


@dataclass
class PatternScan:
    """一个前缀的扫描结果。"""

    pattern: str
    stale: list[str] = field(default_factory=list)
    kept: int = 0

    @property
    def label(self) -> str:
        return self.pattern.replace("*", "")


@dataclass
class CleanupReport:
    """一次清理的汇总。"""

    version: int = 0
    applied: bool = False
    scans: list[PatternScan] = field(default_factory=list)
    deleted: int = 0
    # 删除前把键值备份到 Redis 的哪个回收区（可据此恢复）
    backup_id: str = ""

    @property
    def stale_total(self) -> int:
        return sum(len(s.stale) for s in self.scans)

    @property
    def kept_total(self) -> int:
        return sum(s.kept for s in self.scans)


async def scan(pattern: str, current: int) -> PatternScan:
    """扫一个前缀，分出旧版本与当前版本。"""
    result = PatternScan(pattern=pattern)
    client = await get_redis()
    if client is None:
        return result

    async for raw in client.scan_iter(match=pattern, count=500):
        key = raw if isinstance(raw, str) else raw.decode()
        version = key_version(key)
        if version is None or version == current:
            result.kept += 1
            continue
        result.stale.append(key)
    return result


async def scan_analyst(tenant: str | None = None) -> PatternScan:
    """扫分析结果缓存。

    与分析缓存本身的语义一致：它们没有版本号，
    数据/问题/图表类型一变 key 就变，所以"匹配到即可清"。
    tenant 为空时扫全部租户。
    """
    pattern = f"{ANALYST_PREFIX}:{tenant}:*" if tenant else f"{ANALYST_PREFIX}:*"
    result = PatternScan(pattern=pattern)
    client = await get_redis()
    if client is None:
        return result

    async for raw in client.scan_iter(match=pattern, count=500):
        key = raw if isinstance(raw, str) else raw.decode()
        result.stale.append(key)
    return result


TRASH_PREFIX = "cache:trash"
TRASH_TTL_SECONDS = 7 * 86400


async def _backup_keys(keys: list[str], version: int) -> str:
    """删除前把键值备份到 Redis 回收区，返回 backup_id（失败返回空串）。

    为什么要备份：
        缓存键虽然"能重新生成"，但重新生成意味着一次完整检索 + 一次模型调用，
        既花钱又变慢。备份成本几乎为零（同机 HSET），却能一键恢复。
    """
    client = await get_redis()
    if client is None or not keys:
        return ""

    backup_id = f"v{version}-{int(datetime.now(UTC).timestamp())}"
    mapping: dict[str, str] = {}
    for key in keys:
        try:
            value = await client.get(key)
        except Exception:
            value = None
        if value is not None:
            text_value = value if isinstance(value, str) else str(value)
            # 分析缓存里带 base64 图片（单条可到几十 KB），
            # 备份截断到 8KB：能证明删了什么，又不至于把备份撑爆
            if key.startswith(ANALYST_PREFIX) and len(text_value) > 8192:
                text_value = text_value[:8192]
            mapping[key] = text_value
    if not mapping:
        return ""

    try:
        # 注意：本项目用的 redis-py 5.3.1 上 hset(mapping=...) 会报
        # "wrong number of arguments for 'hset' command"，只能用 field/value 形式
        for field, value in mapping.items():
            await client.hset(f"{TRASH_PREFIX}:{backup_id}", field, value)  # type: ignore[misc]
        await client.expire(f"{TRASH_PREFIX}:{backup_id}", TRASH_TTL_SECONDS)  # type: ignore[misc]
    except Exception as exc:
        logger.error("cache.backup_failed", error=str(exc))
        return ""
    logger.info("cache.backed_up", backup_id=backup_id, keys=len(mapping))
    return backup_id


async def restore_backup(backup_id: str) -> tuple[bool, int]:
    """从备份恢复键，返回 (是否成功, 恢复条数)。"""
    client = await get_redis()
    if client is None:
        return False, 0

    try:
        # 同上：redis-py 的异步签名在类型上不可见
        data = await client.hgetall(f"{TRASH_PREFIX}:{backup_id}")  # type: ignore[misc]
    except Exception as exc:
        logger.error("cache.restore_failed", error=str(exc))
        return False, 0
    if not data:
        return False, 0

    mapping = {
        (k if isinstance(k, str) else k.decode()): (v if isinstance(v, str) else v.decode())
        for k, v in data.items()
    }
    await client.mset(mapping)  # type: ignore[misc]
    logger.info("cache.restored", backup_id=backup_id, keys=len(mapping))
    return True, len(mapping)


async def list_backups() -> list[str]:
    client = await get_redis()
    if client is None:
        return []
    ids: list[str] = []
    async for raw in client.scan_iter(match=f"{TRASH_PREFIX}:*", count=100):
        key = raw if isinstance(raw, str) else raw.decode()
        ids.append(key.split(":", 2)[-1])
    return ids


async def clean(
    apply: bool = False,
    kinds: tuple[str, ...] | list[str] | None = None,
    tenant: str | None = None,
) -> CleanupReport:
    """巡检并按需删除缓存键；apply=True 时**先备份再删**。

    kinds:  要处理的类别（qa / analyst），默认两类都处理
    tenant: 仅当 kinds 含 analyst 时有意义——清哪个租户的分析缓存，空表示全部
    """
    selected = tuple(kinds) if kinds else ALL_KINDS
    current = await cache_version()
    report = CleanupReport(version=current, applied=apply)
    if current == 0:
        return report

    client = await get_redis()
    if client is None:
        return report

    if KIND_QA in selected:
        for pattern in PATTERNS:
            report.scans.append(await scan(pattern, current))
    if KIND_ANALYST in selected:
        report.scans.append(await scan_analyst(tenant))

    if not apply:
        return report

    stale_keys = [key for scan_result in report.scans for key in scan_result.stale]
    if not stale_keys:
        return report

    # 先备份；备份失败就一个都不删（Fail-Closed：宁可留垃圾，也不做不可恢复的删除）
    backup_id = await _backup_keys(stale_keys, current)
    if not backup_id:
        logger.error("cache.clean_aborted", reason="备份失败", keys=len(stale_keys))
        return report
    report.backup_id = backup_id

    for i in range(0, len(stale_keys), BATCH_SIZE):
        await client.delete(*stale_keys[i : i + BATCH_SIZE])
        report.deleted += len(stale_keys[i : i + BATCH_SIZE])

    logger.info(
        "cache.clean_report",
        version=current,
        stale=report.stale_total,
        kept=report.kept_total,
        deleted=report.deleted,
        backup=backup_id,
    )
    return report


async def overview() -> dict[str, int]:
    """缓存概览：当前版本、各前缀的键数与残留数。"""
    current = await cache_version()
    summary = {"version": current, "total": 0, "stale": 0, "current": 0, "analyst": 0}
    if current == 0:
        return summary

    for pattern in PATTERNS:
        result = await scan(pattern, current)
        summary["total"] += len(result.stale) + result.kept
        summary["stale"] += len(result.stale)
        summary["current"] += result.kept

    analyst = await scan_analyst(None)
    summary["analyst"] = len(analyst.stale)
    return summary
