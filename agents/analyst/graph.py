"""数据分析 Agent 的图。

START → route ─┬─ A: sql ─┬─ 有结果 → describe → END
               │          └─ 报错/被拒 → 直接给答案（或重试一次）→ END
               ├─ B: explore → narrative → END
               └─ C: chain → explore → narrative → END

条件边只在需要的地方用，SQL 自愈靠节点内部的一次重试（避免图画得太绕）。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.analyst.nodes import (
    chain_node,
    code_node,
    describe_node,
    explore_node,
    narrative_node,
    route_node,
    sql_node,
)
from agents.analyst.state import PATH_CHAIN, PATH_EXPLORE, AnalystState
from core.logger import get_logger

logger = get_logger(__name__)


def _route_after_route(state: AnalystState) -> str:
    path = state.get("path")
    if path == PATH_EXPLORE:
        return "explore"
    if path == PATH_CHAIN:
        return "chain"
    return "sql"


def _route_after_sql(state: AnalystState) -> str:
    """拿到数据才去总结；被拦/报错时节点已经写好答案了。"""
    if state.get("sql_rows"):
        return "describe"
    return END


def _route_after_code(state: AnalystState) -> str:
    """代码跑失败且还没重试过 → 回 code 让它改；否则去总结。"""
    if state.get("code_error") and not state.get("code_finished"):
        return "retry"
    return "narrative"


def build_analyst_graph(checkpointer: Any = None):
    builder = StateGraph(AnalystState)

    builder.add_node("route", route_node)
    builder.add_node("sql", sql_node)
    builder.add_node("describe", describe_node)
    builder.add_node("explore", explore_node)
    builder.add_node("code", code_node)
    builder.add_node("narrative", narrative_node)
    builder.add_node("chain", chain_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _route_after_route,
        {"sql": "sql", "explore": "explore", "chain": "chain"},
    )
    builder.add_conditional_edges("sql", _route_after_sql, {"describe": "describe", END: END})
    builder.add_edge("describe", END)

    # C 抓完直接进入分析；B/C 都汇总到 narrative
    builder.add_edge("chain", "explore")
    builder.add_edge("explore", "code")
    builder.add_conditional_edges(
        "code",
        _route_after_code,
        {"retry": "code", "narrative": "narrative"},
    )
    builder.add_edge("narrative", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
