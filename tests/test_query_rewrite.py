"""追问改写的测试。"""

from __future__ import annotations

from core.query_rewrite import format_history, rewrite_meta, should_rewrite

HISTORY = [
    {"role": "user", "content": "你们怎么收费"},
    {"role": "assistant", "content": "一号部门内部结算价：按人头 300 元/月"},
]


def test_rewrite_follow_up_questions() -> None:
    assert should_rewrite("那包年呢", HISTORY) is True
    assert should_rewrite("换成季度怎么算", HISTORY) is True
    assert should_rewrite("刚才说的月付是多少", HISTORY) is True


def test_do_not_rewrite_standalone_questions() -> None:
    """独立的问题不该改写——改写要多花一次模型调用。"""
    assert should_rewrite("你们怎么收费", HISTORY) is False
    assert should_rewrite("今天天气怎么样", HISTORY) is False
    assert should_rewrite("抓取 https://example.com 的表格", HISTORY) is False


def test_no_history_never_rewrite() -> None:
    assert should_rewrite("那包年呢", []) is False
    assert should_rewrite("", HISTORY) is False


def test_format_history_only_uses_user_turns() -> None:
    """不能把助手答案拼进去：实测会把价格混进问题里。"""
    text = format_history(HISTORY)
    assert "你们怎么收费" in text
    assert "300 元" not in text


def test_rewrite_meta_marks_change() -> None:
    assert rewrite_meta("那包年呢", "包年怎么收费")["rewritten"] is True
    assert rewrite_meta("你们怎么收费", "你们怎么收费")["rewritten"] is False
