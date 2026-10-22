"""联网研究 Agent 的图。

START → plan ─┬─ 有子问题 → search ─┬─ 有结果 → synthesize → END
              │                     └─ 无结果 → synthesize（走 no_result 分支）→ END
              └─ 无子问题（空问题） → END

比 QA 的图简单：没有缓存、没有 FAQ、没有多策略检索。
复杂度在"外部工具"和"多步规划"上，不在分支上。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.research.nodes import plan_node, search_node, synthesize_node
from agents.research.state import ResearchState
from core.logger import get_logger

logger = get_logger(__name__)


def _route_after_plan(state: ResearchState) -> str:
    """没拆出子问题说明输入为空，直接结束。"""
    return "search" if state.get("subqueries") else "end"


def build_research_graph(checkpointer: Any = None):
    """构建并编译研究 Agent 的图。"""
    builder = StateGraph(ResearchState)

    builder.add_node("plan", plan_node)
    builder.add_node("search", search_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"search": "search", "end": END})
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
