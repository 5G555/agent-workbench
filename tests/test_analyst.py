"""SQL 守卫与探索统计的测试（纯逻辑，不连数据库）。"""

from __future__ import annotations

from agents.analyst.nodes import field_distribution


# ── 探索统计 ──────────────────────────────────────────────────
def test_field_distribution_counts_values():
    records = [
        {"data": {"author": "A", "tag": "x"}},
        {"data": {"author": "A", "tag": "y"}},
        {"data": {"author": "B", "tag": "x"}},
        {"data": {"author": "", "tag": "x"}},  # 空值不计
    ]
    stats = {item["field"]: item for item in field_distribution(records)}
    assert stats["author"]["distinct"] == 2
    assert stats["author"]["top"][0] == {"value": "A", "count": 2}
    assert stats["tag"]["distinct"] == 2


def test_field_distribution_handles_dirty_data():
    records = [
        {"data": "not-a-dict"},  # 脏数据
        {"data": None},
        {},  # 没有 data 字段
        {"data": {"k": "v"}},
    ]
    stats = field_distribution(records)
    assert [item["field"] for item in stats] == ["k"]


def test_field_distribution_empty():
    assert field_distribution([]) == []
