"""清理爬取导出的文件目录。

为什么需要清理：
    data/crawl 下每个任务一个目录，一次导出可能几 MB。
    反复抓取 + 导出会越攒越多，而数据库里本来就有全量数据，
    导出目录只是"给人看/给别的工具用"的快照，可以随时重建。

清理只动文件，不动数据库——除非显式加 --db。
这点很重要：删库是不可逆的，删文件大不了重新导出一次。

安全约束：
    只处理 data/crawl 下**名字是纯数字**的子目录（任务 ID 的约定），
    其它任何目录/文件都不碰，避免误删。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agents.crawler.exporter import DEFAULT_ROOT
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExportDir:
    """一个可清理的导出目录。"""

    task_id: int
    path: Path
    bytes: int
    files: int
    modified: datetime

    @property
    def age_days(self) -> float:
        return (datetime.now(UTC) - self.modified.astimezone(UTC)).total_seconds() / 86400


def _dir_stats(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            count += 1
    return total, count


def scan_exports(root: Path | str | None = None) -> list[ExportDir]:
    """扫出所有导出目录，按任务号倒序（新的在前）。

    只认纯数字目录名：data/crawl 下可能还有手工放的东西，
    不该被清理脚本顺手删掉。
    """
    base = Path(root) if root else DEFAULT_ROOT
    if not base.exists():
        return []

    result: list[ExportDir] = []
    for child in base.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        total, count = _dir_stats(child)
        result.append(
            ExportDir(
                task_id=int(child.name),
                path=child,
                bytes=total,
                files=count,
                modified=datetime.fromtimestamp(child.stat().st_mtime, tz=UTC),
            )
        )
    return sorted(result, key=lambda d: d.task_id, reverse=True)


def plan_cleanup(
    exports: list[ExportDir],
    keep: int | None = None,
    older_than_days: float | None = None,
) -> list[ExportDir]:
    """算出该删哪些目录（纯函数，方便测试）。

    keep：保留最近 N 个任务目录，其余删
    older_than_days：删掉最后修改超过 N 天的
    两个条件都给时，满足任一就删（更激进的那条生效）。
    """
    targets: list[ExportDir] = []
    # 一个条件都没给时什么都不删——清理脚本宁可无功，也不能误删
    if keep is None and older_than_days is None:
        return targets

    # exports 已按任务号倒序，前 keep 个是"保留"
    protected = {d.task_id for d in exports[:keep]} if keep is not None else set()

    for item in exports:
        if item.task_id in protected:
            continue
        if older_than_days is not None and item.age_days < older_than_days:
            continue
        targets.append(item)
    return targets


def apply_cleanup(targets: list[ExportDir]) -> tuple[int, int]:
    """执行清理：**移到回收站**，不是物理删除。

    为什么改成软删除：
        删文件是终态的，而导出目录随时能从库里重建——
        没必要为了省几 MB 冒"删错且无法恢复"的风险。
        真删（purge）是单独的、需要审批的操作（见 agents/ops/trash.py）。

    返回 (移走的目录数, 释放的字节数)。
    """
    from agents.ops.trash import move_to_trash

    dirs = 0
    freed = 0
    for item in targets:
        # 根目录取该目录的父级：targets 本身就是从某个根 scan 出来的，
        # 这样测试用临时目录、线上用 data/crawl 都成立
        moved = move_to_trash(item.task_id, item.path.parent)
        if moved is None:
            logger.warning("crawl_clean.skipped", task=item.task_id)
            continue
        dirs += 1
        freed += moved.size
        logger.info("crawl_clean.moved_to_trash", task=item.task_id, trash_id=moved.trash_id)

    return dirs, freed


def clean_task(task_id: int, root: Path | str | None = None) -> tuple[bool, int]:
    """把单个任务的导出目录移进回收站，返回 (是否移了, 释放字节数)。"""
    from agents.ops.trash import move_to_trash

    moved = move_to_trash(task_id, root)
    if moved is None:
        return False, 0
    return True, moved.size


async def clean_db_tasks(older_than_days: float, tenant_id: str | None = None) -> int:
    """删数据库里的老任务（级联删页面与记录）。不可逆，务必谨慎。"""
    from agents.crawler.store import get_crawl_store

    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    return await get_crawl_store().delete_tasks_before(cutoff, tenant_id)


def human_size(num: int) -> str:
    """把字节数变成人看的大小。"""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"
