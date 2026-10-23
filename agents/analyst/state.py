"""数据分析 Agent 的状态（只管采集数据的 pandas 分析）。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AnalystState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # 身份（取数时带上租户）
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str

    # 输入
    question: str
    entry_url: str  # 问题里带的网址（有就先去抓）
    task_id: int  # 指定分析哪个采集任务

    # 取数
    records: list[dict]
    crawl_summary: str

    # 预处理
    prep: dict

    # 代码执行
    code: str
    code_stdout: str
    code_result: str
    code_error: str
    code_retried: bool
    code_finished: bool
    code_ms: int
    figures: list[str]

    # 输出
    answer: str
    answer_mode: str
    structured_output: dict
