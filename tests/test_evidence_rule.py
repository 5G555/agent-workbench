"""证据判定（L1/L2 规则）的测试。

这层是路由的地基：判错一个，后面全错。
"""

from __future__ import annotations

import pytest

from agents.supervisor.evidence import GRAY, MUST, NEVER, is_private_scope, needs_evidence


@pytest.mark.parametrize(
    "question",
    [
        "今天天气怎么样",
        "AAPL 现在多少钱",
        "最新版 Python 是几点几",
        "现在的美元汇率是多少",
        "这家公司的 CEO 是谁",
        "2026 年本月有哪些新政策",
        "昨天的比分结果",
    ],
)
def test_must_search_when_facts_are_volatile(question: str) -> None:
    assert needs_evidence(question) == MUST


@pytest.mark.parametrize(
    "question",
    [
        "K 线怎么画",
        "移动平均和 EMA 有什么区别",
        "帮我写一封道歉邮件",
        "翻译这段话",
        "什么是向量检索",
        "为什么要分块检索",
        "你能做什么",
        "这个方案有什么优缺点",
        "给我三条优化建议",
    ],
)
def test_never_search_for_method_and_generation(question: str) -> None:
    assert needs_evidence(question) == NEVER


@pytest.mark.parametrize(
    "question",
    [
        "这个接口现在还支持吗",
        "那家公司今年还有这个产品吗",
        "你上个月说的那个功能上线了吗",
    ],
)
def test_gray_when_cannot_tell(question: str) -> None:
    """灰区交给 answer-then-flag，不在这里硬判。"""
    assert needs_evidence(question) in {GRAY, MUST}


def test_method_wins_over_live_word() -> None:
    """ "怎么查股价"是方法，不能因为含"股价"就去搜。"""
    assert needs_evidence("怎么查股价") == NEVER


def test_private_scope_detected() -> None:
    assert is_private_scope("我们的报价里包含税吗")
    assert is_private_scope("公司内部流程怎么走")
    assert not is_private_scope("AAPL 现在多少钱")


def test_empty_question_is_gray() -> None:
    assert needs_evidence("") == GRAY
