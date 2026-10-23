"""三项调整的测试：overview 透出、样式预设、多任务。"""

from __future__ import annotations

from agents.analyst._sandbox import STYLE_PRESET
from agents.analyst.frame import build_frame, frame_overview


def _records_two_tasks() -> list[dict]:
    return [
        {"data": {"Date": "2026-10-01", "Close": "100.5", "task_id": 1}},
        {"data": {"Date": "2026-10-02", "Close": "101.2", "task_id": 1}},
        {"data": {"Date": "2026-10-01", "Close": "200.5", "task_id": 2}},
        {"data": {"Date": "2026-10-02", "Close": "199.0", "task_id": 2}},
    ]


def test_overview_lists_usable_columns():
    """overview 必须点明哪些列能直接算，否则模型会对字符串列做 corr 得到 nan。"""
    frame, _ = build_frame(_records_two_tasks())
    text = frame_overview(frame)
    assert "Close_num" in text
    assert "Date_dt" in text
    assert "【重要】" in text


def test_overview_marks_task_column_for_comparison():
    frame, _ = build_frame(_records_two_tasks())
    text = frame_overview(frame)
    assert "task_id" in text


def test_style_preset_has_core_keys():
    """样式预设要覆盖尺寸/网格/字号，否则面板里两张图长相不一致。"""
    assert "figure.figsize" in STYLE_PRESET
    assert "axes.grid" in STYLE_PRESET
    assert "axes.titlesize" in STYLE_PRESET
    assert STYLE_PRESET["figure.dpi"] >= 100


def test_frame_keeps_task_id_as_usable_column():
    frame, report = build_frame(_records_two_tasks())
    # task_id 是整数，直接就是数值列（不再套 _num 后缀）
    assert "task_id" in frame.columns
    assert frame["task_id"].nunique() == 2
    assert "Close_num" in report.numeric
