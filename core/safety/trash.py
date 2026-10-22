"""删除操作的软删除 / 回收站。

为什么默认不真删：
    物理删除是终态，没有任何补救空间。而导出目录本来就能从库里重新生成，
    没必要为了省几 MB 去冒"删错且无法恢复"的风险。
    所以删除的第一步永远是"移到回收站"，真删变成一次显式、单独的、
    需要审批的 purge 操作。

目录约定：
    data/crawl/.trash/{YYYYmmdd-HHMMSS}-task{ID}/   被删的目录原样移进来
    data/crawl/.trash/{...}/manifest.json           记原路径与时间，恢复用
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agents.crawler.exporter import DEFAULT_ROOT
from core.logger import get_logger

logger = get_logger(__name__)

TRASH_DIRNAME = ".trash"


def trash_root(root: Path | str | None = None) -> Path:
    base = Path(root) if root else DEFAULT_ROOT
    return base / TRASH_DIRNAME


@dataclass
class TrashItem:
    """回收站里的一项。"""

    trash_id: str
    task_id: int
    size: int
    files: int
    deleted_at: datetime
    origin: str

    @property
    def age_days(self) -> float:
        return (datetime.now(UTC) - self.deleted_at.astimezone(UTC)).total_seconds() / 86400

    def as_dict(self) -> dict:
        return {
            "trash_id": self.trash_id,
            "task_id": self.task_id,
            "size": self.size,
            "files": self.files,
            "age_days": round(self.age_days, 2),
            "origin": self.origin,
        }


def _dir_stats(path: Path) -> tuple[int, int]:
    total = count = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            count += 1
    return total, count


def move_to_trash(task_id: int, root: Path | str | None = None) -> TrashItem | None:
    """把某个任务的导出目录移进回收站（软删除）。"""
    base = Path(root) if root else DEFAULT_ROOT
    source = base / str(task_id)
    if not source.is_dir():
        return None

    total, count = _dir_stats(source)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    trash_id = f"{stamp}-task{task_id}"
    target = trash_root(base) / trash_id
    target.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(source), str(target))
    manifest = {
        "trash_id": trash_id,
        "task_id": task_id,
        "origin": str(source),
        "deleted_at": datetime.now(UTC).isoformat(),
        "size": total,
        "files": count,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("trash.moved", task=task_id, trash_id=trash_id, bytes=total)
    return TrashItem(
        trash_id=trash_id,
        task_id=task_id,
        size=total,
        files=count,
        deleted_at=datetime.now(UTC),
        origin=str(source),
    )


def list_trash(root: Path | str | None = None) -> list[TrashItem]:
    """列出回收站内容（新的在前）。"""
    base = trash_root(root)
    if not base.exists():
        return []

    items: list[TrashItem] = []
    for child in sorted(base.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        meta: dict = {}
        if manifest_path.exists():
            try:
                meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        total, count = _dir_stats(child)
        deleted_at = datetime.now(UTC)
        if meta.get("deleted_at"):
            with contextlib.suppress(ValueError):
                deleted_at = datetime.fromisoformat(meta["deleted_at"])
        items.append(
            TrashItem(
                trash_id=child.name,
                task_id=int(meta.get("task_id") or 0),
                size=total,
                files=count,
                deleted_at=deleted_at,
                origin=str(meta.get("origin") or ""),
            )
        )
    return items


def restore(trash_id: str, root: Path | str | None = None) -> tuple[bool, str]:
    """从回收站恢复一个目录，返回 (成功, 说明)。"""
    base = Path(root) if root else DEFAULT_ROOT
    source = trash_root(base) / trash_id
    if not source.is_dir():
        return False, f"回收站里没有 {trash_id}"

    manifest_path = source / "manifest.json"
    origin = ""
    if manifest_path.exists():
        try:
            origin = json.loads(manifest_path.read_text(encoding="utf-8")).get("origin", "")
        except json.JSONDecodeError:
            origin = ""

    target = Path(origin) if origin else base / trash_id.split("-task")[-1]
    if target.exists():
        return False, f"目标路径已存在，不覆盖：{target}"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    logger.info("trash.restored", trash_id=trash_id, to=str(target))
    return True, str(target)


def purge(older_than_days: float | None = None, root: Path | str | None = None) -> tuple[int, int]:
    """永久删除回收站内容（真删）。可只删超过 N 天的。返回 (目录数, 字节数)。"""
    base = trash_root(root)
    if not base.exists():
        return 0, 0

    dirs = freed = 0
    for item in list_trash(root):
        if older_than_days is not None and item.age_days < older_than_days:
            continue
        shutil.rmtree(base / item.trash_id, ignore_errors=True)
        dirs += 1
        freed += item.size
    if dirs:
        logger.info("trash.purged", dirs=dirs, freed=freed, older_than=older_than_days)
    return dirs, freed
