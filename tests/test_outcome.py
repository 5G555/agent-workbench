"""联网检索状态分类的测试（纯逻辑，不联网）。"""

from __future__ import annotations

from agents.research.evidence import Evidence
from agents.research.outcome import (
    MESSAGES,
    SearchOutcome,
    SearchStatus,
    classify_error,
)


def test_status_classification():
    assert classify_error("搜索服务限流") is SearchStatus.RATE_LIMITED
    assert classify_error("HTTP 429 Too Many Requests") is SearchStatus.RATE_LIMITED
    assert classify_error("rate limit exceeded") is SearchStatus.RATE_LIMITED
    assert classify_error("未配置 TAVILY_API_KEY") is SearchStatus.NO_KEY
    assert classify_error("Connection refused") is SearchStatus.FAILED
    assert classify_error("") is SearchStatus.FAILED


def test_only_ok_with_items_is_usable():
    web = Evidence.from_web("t", "http://a", "c")
    assert SearchOutcome(SearchStatus.OK, [web]).usable
    # 状态是 OK 但没资料，也不能拿去作答
    assert not SearchOutcome(SearchStatus.OK, []).usable
    assert not SearchOutcome(SearchStatus.NO_RESULT, []).usable
    assert not SearchOutcome(SearchStatus.FAILED, []).usable


def test_messages_differ_between_no_result_and_failure():
    """这是这个模块存在的意义：两种"没答案"要给用户不同提示。"""
    no_result = SearchOutcome(SearchStatus.NO_RESULT)
    failed = SearchOutcome(SearchStatus.FAILED)
    limited = SearchOutcome(SearchStatus.RATE_LIMITED)
    no_key = SearchOutcome(SearchStatus.NO_KEY)

    assert "没有找到" in no_result.user_message()
    assert "不可用" in failed.user_message()
    assert "限流" in limited.user_message()
    assert "密钥" in no_key.user_message()

    assert no_result.user_message() != failed.user_message()


def test_every_status_has_message_except_ok():
    for status in SearchStatus:
        if status is SearchStatus.OK:
            assert MESSAGES[status] == ""
        else:
            assert MESSAGES[status], f"{status} 缺少用户提示"


def test_describe_includes_detail():
    outcome = SearchOutcome(SearchStatus.FAILED, detail="timeout")
    assert "failed" in outcome.describe()
    assert "timeout" in outcome.describe()
