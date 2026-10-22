"""QA Agent 的图。

START → cache_lookup ─┬─ 命中 → END（直接返回答，零成本）
                      └─ 未命中 → classify ─┬─ GENERAL ─────→ direct ──┐
                                            │                            │
                                            └─ PROFESSIONAL → faq ─┬─命中→┤
                                                                   └─未中→ strategy
                                                                            → retrieve
                                                                            → generate ─┐
                                                                                        ↓
                                                                              cache_write → END
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.qa.nodes import (
    cache_lookup_node,
    cache_write_node,
    classify_node,
    direct_node,
    faq_node,
    generate_node,
    retrieve_node,
    strategy_node,
)
from agents.qa.state import QAState
from core.logger import get_logger

logger = get_logger(__name__)


def _route_after_cache(state: QAState) -> str:
    return "hit" if state.get("cache_hit") else "miss"


def _route_by_category(state: QAState) -> str:
    """通用知识直接答，专业咨询先查 FAQ。分类缺失时按专业咨询处理。"""
    return "direct" if state.get("query_type") == "GENERAL" else "retrieve_flow"


def _route_after_faq(state: QAState) -> str:
    """FAQ 命中就直接返回，未命中转向量检索。"""
    return "hit" if state.get("faq_hit") else "miss"


def build_qa_graph(checkpointer: Any = None):
    """构建并编译 QA 图。

    checkpointer 传 None 时用内存版；要落盘可换 SqliteSaver
    （需额外安装 langgraph-checkpoint-sqlite）。
    """
    builder = StateGraph(QAState)

    builder.add_node("cache_lookup", cache_lookup_node)
    builder.add_node("classify", classify_node)
    builder.add_node("direct", direct_node)
    builder.add_node("faq", faq_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("cache_write", cache_write_node)

    builder.add_edge(START, "cache_lookup")
    builder.add_conditional_edges(
        "cache_lookup",
        _route_after_cache,
        {"hit": END, "miss": "classify"},
    )
    builder.add_conditional_edges(
        "classify",
        _route_by_category,
        {"direct": "direct", "retrieve_flow": "faq"},
    )
    builder.add_conditional_edges(
        "faq",
        _route_after_faq,
        {"hit": "cache_write", "miss": "strategy"},
    )
    builder.add_edge("strategy", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "cache_write")
    builder.add_edge("direct", "cache_write")
    builder.add_edge("cache_write", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
