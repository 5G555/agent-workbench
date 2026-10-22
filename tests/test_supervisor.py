"""Supervisor 的纯逻辑测试（不调模型）。"""

from __future__ import annotations

from agents.supervisor.router import RouteDecision, needs_escalation


def test_escalates_only_when_kb_had_no_result():
    assert needs_escalation("no_result") is True
    # 已经有依据的答案不该再花一次联网的钱
    assert needs_escalation("rag") is False
    assert needs_escalation("faq") is False
    # 闲聊本来就不该查知识库，也不该升级
    assert needs_escalation("llm_direct") is False
    assert needs_escalation(None) is False


def test_route_decision_defaults_to_qa():
    """路由失败时的兜底：走 qa，本地答不出还能升级；反过来要多花一次联网成本。"""
    decision = RouteDecision()
    assert decision.agent == "qa"
    assert 0.0 <= decision.confidence <= 1.0


def test_route_decision_rejects_unknown_agent():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RouteDecision(agent="coder")  # type: ignore[arg-type]
