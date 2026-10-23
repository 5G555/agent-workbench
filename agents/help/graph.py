"""能力说明的图：一个节点，不调模型。

为什么不画成多节点：
    这只是一段"我现在能做什么"的说明，没有取数、没有分支、没有失败恢复，
    强行拆成图只会让人误以为里面有多复杂。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.help.nodes import capability_node


class HelpState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str
    answer: str
    answer_mode: str
    structured_output: dict


def build_help_graph(checkpointer: Any = None):
    builder = StateGraph(HelpState)
    builder.add_node("capability", capability_node)
    builder.add_edge(START, "capability")
    builder.add_edge("capability", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
