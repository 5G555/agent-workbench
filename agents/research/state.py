"""联网研究 Agent 的状态。

和 QA 的 state 结构相似，但字段含义不同：
    subqueries     —— 模型拆出来的子问题
    search_results —— 去重后的网页结果（已归一化成 dict）
    citations      —— 答案里实际引用到的来源 URL
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """图状态。字段都是可选的，节点返回增量。"""

    # 对话消息（add_messages 会自动累加）
    messages: Annotated[list[BaseMessage], add_messages]

    # 身份（与 qa 一致，缓存键需要完整权限范围）
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str

    # 本轮问题
    original_query: str
    # fast = 只用搜索摘要；deep = 允许模型打开网页正文（fetch_page，次数受限）
    depth: str

    # 检索阶段
    subqueries: list[str]  # 拆出来的子问题
    search_results: list[dict]  # [{title, url, content, score}]，已按 URL 去重
    search_status: str  # ok / no_result / rate_limited / no_key / failed
    search_detail: str  # 失败时的原始错误，供日志排查

    # 生成阶段
    answer: str
    answer_mode: str  # web / no_result
    sources: list[str]  # 结果标题（流式事件里展示）
    citations: list[str]  # 实际引用的 URL 列表
    cache_hit: bool  # 是否命中联网答案缓存
    structured_output: dict  # 给 API 层用的聚合结构
    error: str
