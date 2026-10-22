"""联网研究 Agent 的节点。

流程：plan（拆子问题）→ search（并发调 MCP 工具）→ synthesize（带引用作答）

关键点：
    1. 搜索不是直接 httpx 调 Tavily，而是走 MCP 客户端 —— 工具由独立的
       MCP server 提供，Agent 只认协议不认实现。
    2. 多个子问题并发搜索，结果按 URL 去重后再喂给模型。
    3. 答案要求带 [序号] 引用，序号对应喂进去的资料顺序。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.qa.cache import get_answer_cache
from agents.research.collector import collect_multi
from agents.research.outcome import SearchOutcome, SearchStatus
from agents.research.prompts import (
    NO_RESULT_ANSWER,
    PLAN_SYSTEM,
    PLAN_USER,
    SYNTH_SYSTEM,
    SYNTH_USER,
)
from agents.research.state import ResearchState
from agents.research.tools import SearchResult
from config import get_settings
from core.llm_factory import LLMFactory
from core.logger import get_logger
from core.streaming import stream_or_invoke

logger = get_logger(__name__)

# 联网答案在同一个 AnswerCache 里的命名空间，与 qa 的键彻底分开
RESEARCH_NAMESPACE = "research"


class Plan(BaseModel):
    """模型输出的检索计划。"""

    subqueries: list[str] = Field(default_factory=list)


def _is_first_turn(state: ResearchState) -> bool:
    """只有首轮才用缓存：多轮答案依赖上下文，缓存键里没有历史。"""
    from agents.qa.nodes import _is_first_turn as qa_first_turn

    return qa_first_turn(state)  # type: ignore[arg-type]


def _extract_query(state: ResearchState) -> str:
    """从消息里取本轮问题；orchestrator 会显式写入 original_query。"""
    query = state.get("original_query")
    if query:
        return str(query)
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


async def cache_lookup_node(state: ResearchState) -> dict[str, Any]:
    """⓪ 联网答案缓存：命中就跳过 plan/search/synthesize，省一次 Tavily 调用。

    和 qa 共用同一个 AnswerCache，只是换命名空间（research）和更短的 TTL——
    公网信息会变，缓存一整天容易答过时内容。
    """
    query = _extract_query(state)
    if not query or not _is_first_turn(state):
        return {"cache_hit": False}

    hit = await get_answer_cache().get(
        query,
        str(state.get("tenant_id") or ""),
        str(state.get("department") or "*"),
        str(state.get("role") or "user"),
        namespace=RESEARCH_NAMESPACE,
    )
    if hit is None:
        return {"cache_hit": False}

    answer = str(hit.get("answer") or "")
    logger.info("research.cache_hit", query=query)
    return {
        "cache_hit": True,
        "original_query": query,
        "answer": answer,
        "answer_mode": str(hit.get("mode") or "web"),
        "sources": [str(s) for s in (hit.get("sources") or [])],
        "messages": [AIMessage(content=answer)],
        "structured_output": {
            "answer_mode": str(hit.get("mode") or "web"),
            "sources": hit.get("sources") or [],
            "citations": hit.get("citations") or [],
            "cached": True,
        },
    }


async def cache_write_node(state: ResearchState) -> dict[str, Any]:
    """④ 只缓存成功作答的结果。

    没搜到 / 搜失败都不写缓存：前者可能是资料会变，后者是服务故障，
    缓存下来会让后续几小时都误报。
    """
    if state.get("cache_hit"):
        return {}
    if str(state.get("answer_mode") or "") != "web":
        return {}
    query = str(state.get("original_query") or "")
    answer = str(state.get("answer") or "")
    if not query or not answer:
        return {}

    settings = get_settings()
    await get_answer_cache().set(
        query,
        answer,
        mode="web",
        sources=state.get("sources") or [],
        tenant_id=str(state.get("tenant_id") or ""),
        department=str(state.get("department") or "*"),
        role=str(state.get("role") or "user"),
        namespace=RESEARCH_NAMESPACE,
        ttl=settings.research_cache_ttl_seconds,
    )
    logger.info("research.cache_stored", query=query, ttl=settings.research_cache_ttl_seconds)
    return {}


async def plan_node(state: ResearchState) -> dict[str, Any]:
    """① 把问题拆成 1~N 个可独立搜索的子查询。失败就退化成原问题。"""
    query = _extract_query(state)
    if not query:
        return {"original_query": "", "subqueries": []}

    settings = get_settings()
    max_n = max(1, settings.research_max_subqueries)
    subqueries: list[str] = []

    try:
        llm = LLMFactory.get_structured_llm("research", Plan)
        result = await llm.ainvoke(
            [
                SystemMessage(content=PLAN_SYSTEM),
                HumanMessage(content=PLAN_USER.format(query=query, max_n=max_n)),
            ]
        )
        if isinstance(result, Plan):
            subqueries = [q.strip() for q in result.subqueries if q and q.strip()]
    except Exception as exc:
        logger.warning("research.plan_failed", error=str(exc))

    if not subqueries:
        subqueries = [query]
    # 去重 + 截断，防止模型刷出一堆重复子问题
    seen: set[str] = set()
    uniq = [q for q in subqueries if not (q in seen or seen.add(q))]
    subqueries = uniq[:max_n]

    logger.info("research.planned", query=query, subqueries=subqueries)
    return {"original_query": query, "subqueries": subqueries}


async def search_node(state: ResearchState) -> dict[str, Any]:
    """② 并发调 MCP 的 web_search 工具，结果去重。

    采集逻辑在 collector 里，这里只负责把结果和"状态"写进 state：
    搜失败和没搜到必须区分，否则上层只能统一说"没找到"。
    """
    subqueries = state.get("subqueries") or []
    if not subqueries:
        return {"search_results": [], "search_status": SearchStatus.NO_RESULT.value}

    max_items = get_settings().research_results_per_query * 2
    outcome = await collect_multi(
        str(state.get("original_query") or ""),
        subqueries,
        max_items,
    )

    logger.info(
        "research.searched",
        subqueries=len(subqueries),
        status=outcome.status.value,
        items=len(outcome.items),
        detail=outcome.detail,
    )
    return {
        "search_results": [e.__dict__ for e in outcome.items],
        "search_status": outcome.status.value,
        "search_detail": outcome.detail,
    }


async def synthesize_node(state: ResearchState) -> dict[str, Any]:
    """③ 依据资料作答，并要求标注 [序号]。

    没资料时按"状态"给不同答复：
        搜失败 / 限流 / 没配 key → 明确说是检索出了问题，提示稍后重试
        真的没搜到               → 说没找到，不怪服务
    两者不能混为一谈，否则用户会把"服务挂了"理解成"知识库没有"。
    """
    query = state.get("original_query") or ""
    results = state.get("search_results") or []
    status_value = str(state.get("search_status") or SearchStatus.NO_RESULT.value)
    status = (
        SearchStatus(status_value)
        if status_value in SearchStatus.__members__.values()
        else SearchStatus.NO_RESULT
    )

    if not results:
        outcome = SearchOutcome(
            status=status,
            detail=str(state.get("search_detail") or ""),
        )
        message = outcome.user_message() or NO_RESULT_ANSWER
        logger.info("research.no_result", query=query, status=status.value)
        return {
            "answer": message,
            "answer_mode": "search_failed" if status is not SearchStatus.NO_RESULT else "no_result",
            "citations": [],
            "messages": [AIMessage(content=message)],
            "structured_output": {
                "answer_mode": message
                and ("search_failed" if status is not SearchStatus.NO_RESULT else "no_result"),
                "sources": [],
                "citations": [],
                "subqueries": state.get("subqueries") or [],
                "search_status": status.value,
            },
        }

    context = "\n\n".join(
        SearchResult(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            content=str(item.get("content", "")),
        ).as_context(index)
        for index, item in enumerate(results, 1)
    )

    llm = LLMFactory.get_llm("research", temperature=0.2, streaming=True)
    answer = ""
    try:
        # 走统一流式封装：图内会自动推 token 事件，前端就能逐字显示
        answer = await stream_or_invoke(
            llm,
            [
                SystemMessage(content=SYNTH_SYSTEM),
                HumanMessage(content=SYNTH_USER.format(query=query, context=context)),
            ],
        )
    except Exception as exc:
        logger.error("research.synthesize_failed", error=str(exc))
        answer = "生成答案时出错了，请稍后重试。"

    # 把答案里出现的 [n] 映射回 URL，作为引用列表返回
    citations: list[str] = []
    for index, item in enumerate(results, 1):
        if f"[{index}]" in answer:
            url = str(item.get("url", ""))
            if url and url not in citations:
                citations.append(url)

    logger.info("research.answered", chars=len(answer), citations=len(citations))
    titles = [str(item.get("title", "")) for item in results]
    return {
        "answer": answer,
        "answer_mode": "web",
        "citations": citations,
        "sources": titles,
        "messages": [AIMessage(content=answer)],
        "structured_output": {
            "answer_mode": "web",
            "sources": titles,
            "citations": citations,
            "subqueries": state.get("subqueries") or [],
            "urls": [str(item.get("url", "")) for item in results],
        },
    }
