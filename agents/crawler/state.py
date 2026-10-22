"""编排 Agent 的状态。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CrawlState(TypedDict, total=False):
    """一次抓取任务的状态。"""

    messages: Annotated[list[BaseMessage], add_messages]

    # 身份
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str

    # 输入
    goal: str  # 自然语言任务目标
    query: str  # 兼容旧调用：没给 goal 时用它
    entry_url: str
    max_pages: int

    # 流程控制
    task_id: int
    frontier: list[str]  # 待抓队列
    visited: list[str]  # 已抓过的 URL
    next_url: str
    stop_reason: str
    error: str
    consecutive_failures: int

    # 当前页
    current_url: str
    current_html: str
    current_text: str

    # 结果
    pages: list[dict]  # [{url, title, chars, mode}]
    record_count: int
    strategies: list[str]

    # 输出
    summary: str
    answer_mode: str
    structured_output: dict
