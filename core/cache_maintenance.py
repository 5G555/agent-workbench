r"""Redis 缓存的巡检与清理（脚本与 MCP 工具共用一份实现）。

为什么要抽出来：
    同一套逻辑现在有两个调用方——命令行（scripts/clean_cache.py）和
    运维 Agent 的 MCP 工具（clean_cache）。两边各写一份迟早会走样，
    尤其"哪些键算旧版本"这种规则一旦不一致，清理结果就对不上。

清理范围（只碰这三种，别的一概不碰）：
    qa:ans:*         问答答案（含 research 命名空间）
    faq:questions:*  FAQ 问题索引
    faq:tokens:*     FAQ 分词结果

判断依据：键名里的版本号 `:v(\d+)`，与当前版本不同即视为残留。
当前版本的缓存永远不删。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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


async def clean(apply: bool = False) -> CleanupReport:
    """巡检并按需删除旧版本键；apply=False 时只统计。"""
    current = await cache_version()
    report = CleanupReport(version=current, applied=apply)
    if current == 0:
        return report

    client = await get_redis()
    if client is None:
        return report

    for pattern in PATTERNS:
        result = await scan(pattern, current)
        report.scans.append(result)
        if apply and result.stale:
            for i in range(0, len(result.stale), BATCH_SIZE):
                await client.delete(*result.stale[i : i + BATCH_SIZE])
            report.deleted += len(result.stale)

    logger.info(
        "cache.clean_report",
        version=current,
        stale=report.stale_total,
        kept=report.kept_total,
        deleted=report.deleted,
        applied=apply,
    )
    return report


async def overview() -> dict[str, int]:
    """缓存概览：当前版本、各前缀的键数与残留数。"""
    current = await cache_version()
    summary = {"version": current, "total": 0, "stale": 0, "current": 0}
    if current == 0:
        return summary

    for pattern in PATTERNS:
        result = await scan(pattern, current)
        summary["total"] += len(result.stale) + result.kept
        summary["stale"] += len(result.stale)
        summary["current"] += result.kept
    return summary
