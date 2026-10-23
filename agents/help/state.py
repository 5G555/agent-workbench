"""能力说明（help）的状态。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class HelpState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str
    question: str
    answer: str
    answer_mode: str
    structured_output: dict
