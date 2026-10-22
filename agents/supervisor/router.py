"""Supervisor：判断一个问题该交给哪个 Agent。

为什么不做成一张几百节点的"编排图"：只有两个 Agent 时，路由本质是一个二分类，
用一次结构化输出足够；等 Agent 变多、需要多步依赖时再升级成 plan-execute 图。

判断依据：
    内部知识库的主题清单（运行时从 Milvus 取） → 命中就是 qa
    需要公开/时效性/外部信息                      → research
拿不准时默认 qa：本地答错还能升级到 research，反过来则多花一次联网的钱。
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.qa.nodes import kb_topics
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

ROUTE_SYSTEM = "你是一个请求路由器，只输出 JSON，不解释。"

ROUTE_USER = """系统里有两种数据源，按"数据性质"分派：

A. qa（内部 RAG，自建 pipeline）：资料范围是——{topics}
   适用于：内部资料、企业私有信息、需要权限控制或必须给出官方口径的问题。
   这类问题**不能**发到公网去搜（既有泄密风险，也搜不到）。

B. research（联网搜索）：去公开互联网检索。
   适用于：公开可查、有时效性、外部事实（开源框架、行业动态、新闻、第三方产品对比）。

判断顺序：
1. 问题需要本企业内部资料、账号数据、制度流程 → qa
2. 问题是公开的、需要最新信息 → research
3. 两边都沾边时，看用户真正要的那个事实属于哪边

用户问题：{query}

只返回 JSON：{{"agent": "qa" 或 "research", "confidence": 0~1, "reason": "一句话理由"}}"""


class RouteDecision(BaseModel):
    """路由结果。"""

    agent: Literal["qa", "research"] = Field(default="qa")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="")


async def route(query: str) -> RouteDecision:
    """给问题选一个 Agent。失败时退回 qa（本地答不出还能升级）。"""
    if not query.strip():
        return RouteDecision(agent="qa", confidence=1.0, reason="空问题")

    try:
        topics = await kb_topics()
        llm = LLMFactory.get_structured_llm("qa", RouteDecision)
        decision = await llm.ainvoke(
            [
                SystemMessage(content=ROUTE_SYSTEM),
                HumanMessage(content=ROUTE_USER.format(topics=topics, query=query)),
            ]
        )
        if not isinstance(decision, RouteDecision):
            raise ValueError("路由输出类型不符")
    except Exception as exc:
        logger.warning("supervisor.route_failed", error=str(exc))
        return RouteDecision(agent="qa", confidence=0.0, reason=f"路由失败，默认 qa：{exc}")

    logger.info(
        "supervisor.routed",
        query=query,
        agent=decision.agent,
        confidence=round(decision.confidence, 2),
        reason=decision.reason,
    )
    return decision


def needs_escalation(answer_mode: str | None, sources: list[str] | None = None) -> bool:
    """qa 的答案是否"不够用"，需要转 research。

    只在 qa 真的查过知识库但没查到（no_result）时升级：
        - faq / rag：已有依据，不升级
        - llm_direct：本来就是闲聊，不该升级
    """
    return answer_mode == "no_result"
