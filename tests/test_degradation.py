"""降级与兜底体系的测试。

覆盖三层：重试 → Agent 级降级 → 系统级兜底。
重点是"每个 Agent 都有降级策略"这条，之前只有 qa，另外两个会直接掉到系统兜底。
"""

from __future__ import annotations

import pytest

from core import retry
from core.retry import (
    DEGRADE_AGENT_UNAVAILABLE,
    DEGRADE_SYSTEM,
    AgentFallbackHandler,
    _system_fallback_response,
    with_retry,
)

AGENTS = ["qa", "research", "crawler"]


# ── 每个 Agent 都要有专属降级 ──────────────────────────────
@pytest.mark.parametrize("agent_type", AGENTS)
async def test_every_agent_has_fallback_message(agent_type):
    result = await AgentFallbackHandler.handle(
        agent_type=agent_type,
        original_error=RuntimeError("boom"),
        retry_call=None,
        args=(),
        kwargs={},
    )
    assert result["fallback_used"] is True
    assert result["degrade_reason"] == DEGRADE_AGENT_UNAVAILABLE
    assert result["content"].strip(), f"{agent_type} 的降级提示不能为空"


async def test_unknown_agent_raises_to_system_fallback():
    """没有专属降级的 Agent 会抛回原异常，由系统兜底接手。"""
    with pytest.raises(RuntimeError):
        await AgentFallbackHandler.handle(
            agent_type="unknown",
            original_error=RuntimeError("boom"),
            retry_call=None,
            args=(),
            kwargs={},
        )


def test_system_fallback_is_agent_specific():
    qa_msg = _system_fallback_response("qa")["content"]
    web_msg = _system_fallback_response("research")["content"]
    crawl_msg = _system_fallback_response("crawler")["content"]
    assert "问答" in qa_msg
    assert "联网" in web_msg
    assert "采集" in crawl_msg
    assert len({qa_msg, web_msg, crawl_msg}) == 3, "三个 Agent 的兜底话术要能区分开"


def test_system_fallback_is_marked():
    result = _system_fallback_response("research")
    assert result["system_fallback"] is True
    assert result["degrade_reason"] == DEGRADE_SYSTEM
    assert result["structured_output"]["degraded"] is True


# ── research：deep 失败时真的用 fast 重跑 ──────────────────
async def test_research_deep_degrades_to_fast():
    seen: dict[str, str] = {}

    async def fake_retry(state, **kwargs):
        seen["depth"] = state["depth"]
        return {"messages": [], "depth": state["depth"]}

    result = await AgentFallbackHandler.handle(
        agent_type="research",
        original_error=TimeoutError("deep 超时"),
        retry_call=fake_retry,
        args=({"depth": "deep", "original_query": " q "},),
        kwargs={"config": {}},
    )
    assert seen["depth"] == "fast", "降级必须真的换成 fast 再跑一次"
    assert result["depth"] == "fast"


async def test_research_returns_message_when_fast_also_fails():
    async def failing_retry(state, **kwargs):
        raise TimeoutError("fast 也超时")

    result = await AgentFallbackHandler.handle(
        agent_type="research",
        original_error=TimeoutError("deep 超时"),
        retry_call=failing_retry,
        args=({"depth": "deep"},),
        kwargs={"config": {}},
    )
    assert result["fallback_used"] is True
    assert "联网" in result["content"]


async def test_research_fast_mode_skips_rerun():
    """本来就是 fast，没有更省的档位可退，不要再跑一遍。"""
    called = {"count": 0}

    async def fake_retry(state, **kwargs):
        called["count"] += 1
        return {}

    result = await AgentFallbackHandler.handle(
        agent_type="research",
        original_error=RuntimeError("x"),
        retry_call=fake_retry,
        args=({"depth": "fast"},),
        kwargs={},
    )
    assert called["count"] == 0
    assert result["fallback_used"] is True


# ── crawler：失败要带上入口，不能装作成功 ──────────────────
async def test_crawler_fallback_mentions_entry_url():
    result = await AgentFallbackHandler.handle(
        agent_type="crawler",
        original_error=RuntimeError("connection refused"),
        retry_call=None,
        args=({"entry_url": "https://example.com/list"},),
        kwargs={},
    )
    assert "https://example.com/list" in result["content"]
    assert result["structured_output"]["entry_url"] == "https://example.com/list"


# ── 三层串联 ──────────────────────────────────────────────
async def test_with_retry_falls_back_after_retries(monkeypatch):
    monkeypatch.setattr(retry, "MAX_RETRIES", 0)
    monkeypatch.setattr(retry, "RETRY_DELAYS", [0.0])
    calls = {"n": 0}

    @with_retry(agent_type="research")
    async def always_fail(state, config):
        calls["n"] += 1
        raise RuntimeError("down")

    result = await always_fail({"depth": "fast"}, {})
    assert calls["n"] == 1  # 重试次数被压到 0
    assert result["fallback_used"] is True
    assert "联网" in result["content"]


async def test_with_retry_reaches_system_fallback_for_unknown_agent(monkeypatch):
    monkeypatch.setattr(retry, "MAX_RETRIES", 0)
    monkeypatch.setattr(retry, "RETRY_DELAYS", [0.0])

    @with_retry(agent_type="mystery")
    async def always_fail(state, config):
        raise RuntimeError("down")

    result = await always_fail({}, {})
    assert result["system_fallback"] is True
    assert result["degrade_reason"] == DEGRADE_SYSTEM


async def test_with_retry_returns_result_on_success(monkeypatch):
    monkeypatch.setattr(retry, "MAX_RETRIES", 0)

    @with_retry(agent_type="qa")
    async def ok(state, config):
        return {"messages": ["fine"]}

    assert await ok({}, {}) == {"messages": ["fine"]}


# ── 编排器层：降级结果要能正确透出 ──────────────────────────
async def test_orchestrator_surfaces_degrade_reason(monkeypatch):
    """图执行失败时，orchestrator 必须返回可用的响应而不是抛异常。"""
    from core.orchestrator import AgentRequest, AgentType, get_orchestrator

    class BrokenGraph:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("图炸了")

    orchestrator = get_orchestrator()
    monkeypatch.setattr(orchestrator, "_get_agent_graph", lambda agent_type: BrokenGraph())
    monkeypatch.setattr(retry, "MAX_RETRIES", 0)
    monkeypatch.setattr(retry, "RETRY_DELAYS", [0.0])

    response = await orchestrator.handle(
        AgentRequest(
            user_id="1",
            tenant_id="tenant_default",
            session_id="deg-1",
            agent_type=AgentType.CRAWLER,
            user_message="抓取 https://example.com",
            context={"entry_url": "https://example.com"},
        )
    )

    assert response.success is True
    assert response.fallback_used is True
    assert response.degrade_reason == DEGRADE_AGENT_UNAVAILABLE
    assert "https://example.com" in response.content
