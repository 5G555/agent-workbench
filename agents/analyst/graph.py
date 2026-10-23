"""数据分析 Agent 的图：只管"抓回来的数据"。

START → dispatch ─┬─ 有 URL → chain（先抓）→ explore ─┐
                  └─ 无 URL ────────────→ explore ─┴→ code →（失败重试一次）→ narrative → END

和运维的 metrics 图分开：
    ops/metrics  = 查系统指标（SQL 视图）
    analyst      = 分析采集数据（pandas + 绘图）
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.analyst.nodes import (
    chain_node,
    code_node,
    dispatch_node,
    explore_node,
    narrative_node,
    route_after_dispatch,
)
from agents.analyst.state import AnalystState
from core.logger import get_logger

logger = get_logger(__name__)


def _route_after_code(state: AnalystState) -> str:
    """代码失败且没重试过 → 回 code 改；否则去总结。"""
    if state.get("code_error") and not state.get("code_finished"):
        return "retry"
    return "narrative"


def build_analyst_graph(checkpointer: Any = None):
    builder = StateGraph(AnalystState)

    builder.add_node("dispatch", dispatch_node)
    builder.add_node("chain", chain_node)
    builder.add_node("explore", explore_node)
    builder.add_node("code", code_node)
    builder.add_node("narrative", narrative_node)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges(
        "dispatch", route_after_dispatch, {"chain": "chain", "explore": "explore"}
    )
    builder.add_edge("chain", "explore")
    builder.add_edge("explore", "code")
    builder.add_conditional_edges(
        "code", _route_after_code, {"retry": "code", "narrative": "narrative"}
    )
    builder.add_edge("narrative", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
