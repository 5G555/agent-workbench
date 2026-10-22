"""清理 Redis 里旧版本的缓存键。

为什么会有残留：
    每次 bump 缓存版本号，旧键并不会被删除——它们只是再也命不中了，
    靠 TTL（答案 24h / 联网 6h / FAQ 索引 7 天）自然过期。
    频繁调参时（改检索、改切分、改提示词都要 bump），残留会堆得比较多。

用法：
    python -m scripts.clean_cache            # 只统计，不删除（默认）
    python -m scripts.clean_cache --apply    # 真正删除

只会删除「版本号不等于当前版本」的键，当前版本的缓存不受影响。
具体实现见 core/cache_maintenance.py（MCP 工具用的是同一份逻辑）。
"""

from __future__ import annotations

import argparse
import asyncio

from core.cache_maintenance import clean, key_version  # noqa: F401 - key_version 供测试导入
from core.logger import get_logger

logger = get_logger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="清理旧版本缓存键")
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只统计）")
    args = parser.parse_args()

    report = await clean(apply=args.apply)
    if report.version == 0:
        raise SystemExit("Redis 不可用或没有版本号，先确认缓存服务正常")

    print(f"当前缓存版本：v{report.version}")
    print(f"模式：{'删除' if args.apply else '仅统计（加 --apply 才会删）'}\n")

    for scan in report.scans:
        print(f"  {scan.label:22} 旧版本 {len(scan.stale):3} 个 / 当前版本 {scan.kept:3} 个")
        if args.apply and scan.stale:
            print(f"  {'':22} 已删除 {len(scan.stale)} 个")

    print(f"\n合计：旧版本 {report.stale_total} 个，当前版本 {report.kept_total} 个")
    if not args.apply and report.stale_total:
        print("确认无误后加 --apply 执行删除")


if __name__ == "__main__":
    asyncio.run(main())
