"""回收站（软删除）的测试。"""

from __future__ import annotations

from pathlib import Path

from agents.ops import trash


def _make_export(root: Path, task_id: int, files: int = 2) -> Path:
    task_dir = root / str(task_id)
    (task_dir / "pages").mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (task_dir / "pages" / f"00{i}.html").write_text("x" * 100, encoding="utf-8")
    (task_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return task_dir


def test_move_to_trash_keeps_content(tmp_path, monkeypatch):
    monkeypatch.setattr(trash, "DEFAULT_ROOT", tmp_path)
    _make_export(tmp_path, 5)

    item = trash.move_to_trash(5)
    assert item is not None
    assert item is not None
    assert item.task_id == 5
    assert not (tmp_path / "5").exists(), "原目录应已移走"
    assert item.files > 0

    # 回收站里还留着内容，可以恢复
    items = trash.list_trash()
    assert [i.trash_id for i in items] == [item.trash_id]
    assert (trash.trash_root() / item.trash_id / "pages").is_dir()


def test_restore_puts_it_back(tmp_path, monkeypatch):
    monkeypatch.setattr(trash, "DEFAULT_ROOT", tmp_path)
    _make_export(tmp_path, 6)
    item = trash.move_to_trash(6)
    assert item is not None

    ok, detail = trash.restore(item.trash_id)
    assert ok is True
    assert (tmp_path / "6" / "pages").is_dir()
    assert str(tmp_path / "6") in detail
    assert trash.list_trash() == []


def test_restore_missing_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(trash, "DEFAULT_ROOT", tmp_path)
    ok, detail = trash.restore("nope")
    assert ok is False
    assert "没有" in detail


def test_restore_refuses_to_overwrite(tmp_path, monkeypatch):
    """原路径又出现了（比如重新导出了），恢复不能覆盖它。"""
    monkeypatch.setattr(trash, "DEFAULT_ROOT", tmp_path)
    _make_export(tmp_path, 7)
    item = trash.move_to_trash(7)
    assert item is not None
    _make_export(tmp_path, 7)  # 重新生成了一份

    ok, detail = trash.restore(item.trash_id)
    assert ok is False
    assert "已存在" in detail


def test_purge_only_old(tmp_path, monkeypatch):
    monkeypatch.setattr(trash, "DEFAULT_ROOT", tmp_path)
    _make_export(tmp_path, 8)
    item = trash.move_to_trash(8)
    assert item is not None

    # 刚删的，按 30 天阈值 purge 不应该删掉
    dirs, freed = trash.purge(older_than_days=30)
    assert dirs == 0
    assert trash.list_trash()

    # 不设阈值则清空
    dirs, freed = trash.purge()
    assert dirs == 1
    assert freed > 0
    assert not (trash.trash_root() / item.trash_id).exists()


def test_move_missing_task_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(trash, "DEFAULT_ROOT", tmp_path)
    assert trash.move_to_trash(999) is None
