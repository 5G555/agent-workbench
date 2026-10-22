"""数据分析 Agent 的状态。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# 三条路径（对应架构图）
PATH_SQL = "A"  # Text2SQL：精确取数
PATH_EXPLORE = "B"  # MCP 分析：探索/统计采集数据
PATH_CHAIN = "C"  # 先抓取再分析（A→B 串联）


class AnalystState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # 身份（视图层据此做租户隔离）
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str

    # 输入
    question: str

    # 路由
    path: str
    path_reason: str

    # 路径 A
    sql: str
    sql_error: str
    sql_rows: list[dict]
    sql_columns: list[str]
    sql_retried: bool

    # 路径 B / C
    task_id: int
    records: list[dict]
    field_stats: list[dict]
    crawl_summary: str

    # 输出
    answer: str
    answer_mode: str
    structured_output: dict
