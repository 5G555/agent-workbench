"""清理爬取导出目录的命令行工具。

用法：
    python -m scripts.clean_crawl                     # 只统计（默认不删）
    python -m scripts.clean_crawl --keep 3             # 只保留最近 3 个任务的导出
    python -m scripts.clean_crawl --older-than 7       # 删掉 7 天前的导出
    python -m scripts.clean_crawl --task 3             # 删某个任务的导出
    python -m scripts.clean_crawl --keep 3 --apply     # 真删
    python -m scripts.clean_crawl --older-than 30 --db --apply   # 连库里的老任务一起删

默认是"只统计"，加 --apply 才真正删除。
--db 会删数据库里的任务（级联删页面与记录），不可逆，只在明确要清历史时才加。
"""

from __future__ import annotations

import argparse
import asyncio

from agents.crawler.cleaner import (
    apply_cleanup,
    clean_db_tasks,
    clean_task,
    human_size,
    plan_cleanup,
    scan_exports,
)
from agents.crawler.exporter import DEFAULT_ROOT


def _print_exports(exports) -> None:
    if not exports:
        print(f"{DEFAULT_ROOT} 下还没有导出目录")
        return
    total = sum(item.bytes for item in exports)
    print(f"共 {len(exports)} 个导出目录，合计 {human_size(total)}：")
    for item in exports:
        print(
            f"  #{item.task_id:<4} {item.files:3} 个文件  {human_size(item.bytes):>9}  "
            f"{item.age_days:5.1f} 天前   {item.path}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="清理爬取导出的文件目录")
    parser.add_argument("--list", action="store_true", help="只列出导出目录")
    parser.add_argument("--keep", type=int, default=None, help="保留最近 N 个任务的导出")
    parser.add_argument("--older-than", type=float, default=None, help="删除 N 天前的导出")
    parser.add_argument("--task", type=int, default=None, help="只删某个任务的导出")
    parser.add_argument(
        "--db", action="store_true", help="同时删数据库里的老任务（需配 --older-than）"
    )
    parser.add_argument("--tenant", default=None, help="配合 --db 限定租户")
    parser.add_argument("--apply", action="store_true", help="真正执行删除（默认只统计）")
    args = parser.parse_args()

    exports = scan_exports()

    if args.list or (
        args.keep is None and args.older_than is None and args.task is None and not args.db
    ):
        _print_exports(exports)
        if not args.list:
            print("\n可用参数：--keep N | --older-than N | --task ID | --db | --apply")
        return

    # 单个任务
    if args.task is not None:
        if not args.apply:
            print(f"将删除任务 #{args.task} 的导出目录（加 --apply 执行）")
            return
        done, freed = clean_task(args.task)
        print(
            f"任务 #{args.task}："
            + (f"已删除，释放 {human_size(freed)}" if done else "没有导出目录")
        )
        return

    targets = plan_cleanup(exports, keep=args.keep, older_than_days=args.older_than)
    if not targets:
        print("没有需要清理的目录")
    else:
        freed = sum(item.bytes for item in targets)
        print(f"计划删除 {len(targets)} 个目录，释放约 {human_size(freed)}：")
        for item in targets:
            print(f"  #{item.task_id:<4} {human_size(item.bytes):>9}  {item.age_days:5.1f} 天前")
        if args.apply:
            dirs, freed_real = apply_cleanup(targets)
            print(f"已删除 {dirs} 个目录，实际释放 {human_size(freed_real)}")
        else:
            print("\n加 --apply 才会真正删除")

    if args.db:
        if args.older_than is None:
            print("--db 需要配合 --older-than（避免误删全部任务）")
            return
        if not args.apply:
            print(f"\n将删除 {args.older_than} 天前的数据库任务记录（级联删页面与记录）")
            return
        deleted = await clean_db_tasks(args.older_than, args.tenant)
        print(f"已从数据库删除 {deleted} 个历史任务")


if __name__ == "__main__":
    asyncio.run(main())
