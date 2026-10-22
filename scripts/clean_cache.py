"""清理 Redis 里旧版本的缓存键。

为什么会有残留：
    每次 bump 缓存版本号，旧键并不会被删除——它们只是再也命不中了，
    靠 TTL（答案 24h / 联网 6h / FAQ 索引 7 天）自然过期。
    频繁调参时（改检索、改切分、改提示词都要 bump），残留会堆得比较多。

用法：
    python -m scripts.clean_cache            # 只统计，不删除（默认）
    python -m scripts.clean_cache --apply    # 真正删除

只会删除「版本号不等于当前版本」的键，当前版本的缓存不受影响。
"""

from __future__ import annotations

import argparse
import asyncio
import re

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

PATTERNS = [f"{ANSWER_PREFIX}:*", f"{FAQ_QUESTIONS_KEY}:*", f"{FAQ_TOKENS_KEY}:*"]


def key_version(key: str) -> int | None:
    """从键名里取出版本号；取不到返回 None。"""
    match = VERSION_PATTERN.search(key)
    return int(match.group(1)) if match else None


async def scan_stale(pattern: str, current: int) -> tuple[list[str], int]:
    """扫一个 pattern，返回 (待删键列表, 跳过数)。"""
    client = await get_redis()
    if client is None:
        return [], 0

    stale: list[str] = []
    kept = 0
    async for raw in client.scan_iter(match=pattern, count=500):
        key = raw if isinstance(raw, str) else raw.decode()
        version = key_version(key)
        if version is None or version == current:
            kept += 1
            continue
        stale.append(key)
    return stale, kept


async def main() -> None:
    parser = argparse.ArgumentParser(description="清理旧版本缓存键")
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只统计）")
    args = parser.parse_args()

    current = await cache_version()
    if current == 0:
        raise SystemExit("Redis 不可用或没有版本号，先确认缓存服务正常")

    client = await get_redis()
    if client is None:
        raise SystemExit("连不上 Redis")

    print(f"当前缓存版本：v{current}")
    print(f"模式：{'删除' if args.apply else '仅统计（加 --apply 才会删）'}\n")

    total_stale = 0
    total_kept = 0
    for pattern in PATTERNS:
        stale, kept = await scan_stale(pattern, current)
        total_stale += len(stale)
        total_kept += kept
        label = pattern.replace("*", "")
        print(f"  {label:22} 旧版本 {len(stale):3} 个 / 当前版本 {kept:3} 个")

        if args.apply and stale:
            # 分批删，避免一次 INCR 太多阻塞
            for i in range(0, len(stale), 200):
                await client.delete(*stale[i : i + 200])
            print(f"  {'':22} 已删除 {len(stale)} 个")

    print(f"\n合计：旧版本 {total_stale} 个，当前版本 {total_kept} 个")
    if not args.apply and total_stale:
        print("确认无误后加 --apply 执行删除")
    elif args.apply:
        logger.info("cache.cleaned", deleted=total_stale, version=current)


if __name__ == "__main__":
    asyncio.run(main())
