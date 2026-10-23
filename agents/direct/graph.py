"""direct 的图（状态从 state.py 取，避免与 nodes 循环引用）。"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.direct.nodes import answer_node
from agents.direct.state import DirectState


def build_direct_graph(checkpointer: Any = None):
    """一个节点：不取数、不检索，只让模型答。"""
    builder = StateGraph(DirectState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
