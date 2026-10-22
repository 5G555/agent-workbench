"""联网研究 Agent 的图。

START → cache_lookup ─┬─ 命中 → END（直接返回缓存的答案，不再调 Tavily）
                      └─ 未命中 → plan ─┬─ 有子问题 → search → synthesize ─┐
                                        └─ 空问题 ────────────────→ END   │
                                                                            ↓
                                                                      cache_write → END

缓存与 qa 共用同一个 AnswerCache，命名空间不同（research）、TTL 更短。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.research.nodes import (
    cache_lookup_node,
    cache_write_node,
    plan_node,
    search_node,
    synthesize_node,
)
from agents.research.state import ResearchState
from core.logger import get_logger

logger = get_logger(__name__)


def _route_after_cache(state: ResearchState) -> str:
    return "hit" if state.get("cache_hit") else "miss"


def _route_after_plan(state: ResearchState) -> str:
    """没拆出子问题说明输入为空，直接结束。"""
    return "search" if state.get("subqueries") else "end"


def build_research_graph(checkpointer: Any = None):
    """构建并编译研究 Agent 的图。"""
    builder = StateGraph(ResearchState)

    builder.add_node("cache_lookup", cache_lookup_node)
    builder.add_node("plan", plan_node)
    builder.add_node("search", search_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("cache_write", cache_write_node)

    builder.add_edge(START, "cache_lookup")
    builder.add_conditional_edges("cache_lookup", _route_after_cache, {"hit": END, "miss": "plan"})
    builder.add_conditional_edges("plan", _route_after_plan, {"search": "search", "end": END})
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", "cache_write")
    builder.add_edge("cache_write", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
