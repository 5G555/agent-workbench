"""导出目录清理的测试（纯文件系统操作，用临时目录）。"""

from __future__ import annotations

import time
from pathlib import Path

from agents.ops.cleaner import (
    ExportDir,
    apply_cleanup,
    clean_task,
    human_size,
    plan_cleanup,
    scan_exports,
)


def _make_export(root: Path, task_id: int, files: int = 2, size: int = 100) -> Path:
    task_dir = root / str(task_id)
    (task_dir / "pages").mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (task_dir / "pages" / f"000{i}-x.html").write_text("x" * size, encoding="utf-8")
    (task_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return task_dir


def test_scan_exports_only_reads_numeric_dirs(tmp_path):
    _make_export(tmp_path, 1)
    _make_export(tmp_path, 2)
    (tmp_path / "keepme").mkdir()  # 非数字目录不该被当成导出
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")

    exports = scan_exports(tmp_path)
    assert [item.task_id for item in exports] == [2, 1], "应按任务号倒序"
    assert exports[0].files == 3  # 2 个页面 + manifest
    assert exports[0].bytes > 0


def test_keep_protects_newest(tmp_path):
    for task_id in (1, 2, 3, 4):
        _make_export(tmp_path, task_id)
    exports = scan_exports(tmp_path)

    targets = plan_cleanup(exports, keep=2)
    assert {item.task_id for item in targets} == {1, 2}


def test_older_than_filters_by_age(tmp_path):
    old = _make_export(tmp_path, 1)
    _make_export(tmp_path, 2)
    # 把 task 1 的修改时间改到 10 天前
    past = time.time() - 10 * 86400
    for item in old.rglob("*"):
        import os

        os.utime(item, (past, past))
    import os

    os.utime(old, (past, past))

    exports = scan_exports(tmp_path)
    targets = plan_cleanup(exports, older_than_days=3)
    assert [item.task_id for item in targets] == [1]


def test_no_condition_keeps_everything(tmp_path):
    _make_export(tmp_path, 1)
    exports = scan_exports(tmp_path)
    assert plan_cleanup(exports) == []


def test_apply_cleanup_removes_only_targets(tmp_path, monkeypatch):
    import agents.ops.cleaner as cleaner

    _make_export(tmp_path, 1)
    _make_export(tmp_path, 2)
    exports = scan_exports(tmp_path)
    targets = plan_cleanup(exports, keep=1)

    monkeypatch.setattr(cleaner, "DEFAULT_ROOT", tmp_path)
    dirs, freed = apply_cleanup(targets)

    assert dirs == 1
    assert freed > 0
    assert (tmp_path / "2").exists(), "保留的目录不能被删"
    assert not (tmp_path / "1").exists()


def test_clean_task_removes_single_dir(tmp_path):
    _make_export(tmp_path, 3)
    ok, freed = clean_task(3, tmp_path)
    assert ok is True
    assert freed > 0
    assert not (tmp_path / "3").exists()


def test_clean_task_missing_returns_false(tmp_path):
    ok, freed = clean_task(99, tmp_path)
    assert ok is False
    assert freed == 0


def test_apply_cleanup_refuses_outside_root(tmp_path, monkeypatch):
    """名字不是数字的目录一律跳过，防止误删。"""
    import agents.ops.cleaner as cleaner

    outside = tmp_path / "important"
    outside.mkdir()
    (outside / "data.txt").write_text("x", encoding="utf-8")

    fake = ExportDir(
        task_id=0,
        path=outside,
        bytes=1,
        files=1,
        modified=cleaner.datetime.now(cleaner.UTC),
    )
    monkeypatch.setattr(cleaner, "DEFAULT_ROOT", tmp_path)
    dirs, _ = apply_cleanup([fake])
    assert dirs == 0
    assert outside.exists()


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(2048) == "2.0 KB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"
