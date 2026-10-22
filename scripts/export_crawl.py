"""把某次爬取任务导出成文件（命令行入口）。

用法：
    python -m scripts.export_crawl --task 5
    python -m scripts.export_crawl --task 5 --out D:/crawl_dump
    python -m scripts.export_crawl --list            # 看最近有哪些任务

需要带租户：默认 tenant_default，可 --tenant 指定。
"""

from __future__ import annotations

import argparse
import asyncio

from agents.crawler.exporter import DEFAULT_ROOT, export_task
from agents.crawler.store import get_crawl_store
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


async def _list_tasks(tenant: str) -> None:
    tasks = await get_crawl_store().recent_tasks(tenant, limit=10)
    if not tasks:
        print(f"租户 {tenant} 还没有爬取任务")
        return
    print(f"租户 {tenant} 最近的任务：")
    for task in tasks:
        print(
            f"  #{task['id']:<4} {task['status']:<7} "
            f"页={task['pages_fetched']:<3} 记录={task['records_extracted']:<4} "
            f"{task['goal'][:30]}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="导出爬取结果到文件")
    parser.add_argument("--task", type=int, help="要导出的任务 ID")
    parser.add_argument("--tenant", default=None, help="租户 ID")
    parser.add_argument("--out", default=None, help="导出根目录，默认 data/crawl")
    parser.add_argument("--list", action="store_true", help="只列出最近任务")
    args = parser.parse_args()

    tenant = args.tenant or get_settings().default_tenant_id

    if args.list or not args.task:
        await _list_tasks(tenant)
        if not args.task:
            print(
                "\n用法：python -m scripts.export_crawl --task <ID>"
                f"   （默认导出到 {DEFAULT_ROOT}）"
            )
        return

    result = await export_task(args.task, tenant, out_dir=args.out)
    if not result.ok:
        print(f"导出失败：{result.error}")
        return

    print(f"任务 #{result.task_id} 已导出到：{result.directory}")
    kb = result.bytes_written / 1024
    print(f"  页面 {result.pages} 个 / 结构化记录 {result.records} 条 / 合计 {kb:.1f} KB")
    print(f"  文件 {len(result.files)} 个，例如：")
    for name in result.files[:6]:
        print(f"    {name}")


if __name__ == "__main__":
    asyncio.run(main())
