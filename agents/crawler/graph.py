"""爬虫 Agent 的图（编排层）。

START → init → decide ─┬─ 该抓 → fetch → extract → 回到 decide
                       └─ 该停 → finalize → END

这是一个**带循环的图**：decide 是入口，抓完一页回到 decide 重新判断。
用 conditional edge 表达循环，而不是在节点里写 while——
这样每一步都能被 LangGraph 的事件流看到，也方便以后插入人工确认。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.crawler.nodes import (
    decide_node,
    extract_node,
    fetch_node,
    finalize_node,
    init_node,
)
from agents.crawler.state import CrawlState
from core.logger import get_logger

logger = get_logger(__name__)


def _route(state: CrawlState) -> str:
    """decide 之后的走向：有 next_url 就抓，没有就收尾。"""
    return "fetch" if state.get("next_url") else "finalize"


def build_crawler_graph(checkpointer: Any = None):
    """构建并编译爬虫 Agent 的图。"""
    builder = StateGraph(CrawlState)

    builder.add_node("init", init_node)
    builder.add_node("decide", decide_node)
    builder.add_node("fetch", fetch_node)
    builder.add_node("extract", extract_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "init")
    builder.add_edge("init", "decide")
    builder.add_conditional_edges("decide", _route, {"fetch": "fetch", "finalize": "finalize"})
    builder.add_edge("fetch", "extract")
    builder.add_edge("extract", "decide")  # 循环
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
