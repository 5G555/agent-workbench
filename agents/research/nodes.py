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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agents.qa.cache import get_answer_cache
from agents.research.collector import collect_multi
from agents.research.mcp_client import get_mcp_client, parse_mcp_payload
from agents.research.outcome import SearchOutcome, SearchStatus
from agents.research.prompts import (
    NO_RESULT_ANSWER,
    PLAN_SYSTEM,
    PLAN_USER,
    SYNTH_SYSTEM,
    SYNTH_USER,
)
from agents.research.state import ResearchState
from agents.research.time_filter import resolve_time_range
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
    # 只对"要最新信息"的问题填值；常识类问题应留空
    time_range: str = Field(default="")


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
        # 深度也要进键：deep 的答案（读过正文）不能和 fast 的摘要答案混用
        namespace=f"{RESEARCH_NAMESPACE}:{state.get('depth') or 'fast'!s}",
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
        namespace=f"{RESEARCH_NAMESPACE}:{state.get('depth') or 'fast'!s}",
        ttl=settings.research_cache_ttl_seconds,
        citations=state.get("citations") or [],
    )
    logger.info(
        "research.cache_stored",
        query=query,
        ttl=settings.research_cache_ttl_seconds,
        depth=str(state.get("depth") or "fast"),
    )
    return {}


async def plan_node(state: ResearchState) -> dict[str, Any]:
    """① 把问题拆成 1~N 个可独立搜索的子查询。失败就退化成原问题。"""
    query = _extract_query(state)
    if not query:
        return {"original_query": "", "subqueries": []}

    settings = get_settings()
    max_n = max(1, settings.research_max_subqueries)
    subqueries: list[str] = []
    model_range = ""

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
            model_range = result.time_range
    except Exception as exc:
        logger.warning("research.plan_failed", error=str(exc))

    if not subqueries:
        subqueries = [query]
    # 去重 + 截断，防止模型刷出一堆重复子问题
    seen: set[str] = set()
    uniq = [q for q in subqueries if not (q in seen or seen.add(q))]
    subqueries = uniq[:max_n]

    # 模型给了就用模型的，没给再用关键词兜底，都没有就不限时间
    time_range = resolve_time_range(model_range, query)

    logger.info(
        "research.planned",
        query=query,
        subqueries=subqueries,
        time_range=time_range or "-",
    )
    return {"original_query": query, "subqueries": subqueries, "time_range": time_range}


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
        time_range=str(state.get("time_range") or ""),
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


DEEP_HINT = """

【可用工具】
你可以调用 fetch_page(url) 打开上面任意一条资料的网页正文，获取摘要里没有的内容。

使用规则：
1. 问题涉及**具体数值、参数默认值、取值范围、配置项、版本号、操作步骤**时，
   必须先打开至少一页最相关的资料核对原文，不能只凭摘要作答。
2. 问题属于概述、对比、原理这类，摘要已足够时可直接作答。
3. 最多调用 {limit} 次，优先打开官方文档或最相关的那一条。"""


async def _emit(text: str) -> None:
    """把 token 推给前端：用 LangGraph 的 stream writer，不在图上下文时跳过。"""
    if not text:
        return
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(text)
    except Exception as exc:
        logger.debug("research.no_stream_writer", error=str(exc))


async def _synthesize_deep(llm: Any, query: str, context: str, results: list[dict]) -> str:
    """deep 模式：让模型真的打开网页正文，再作答。

    控制流：
        第 1 轮 强制调用 fetch_page（tool_choice=required）——用户既然选了 deep，
                 就要真去读原文，不能因为摘要看起来够了就跳过
        第 2+ 轮 由模型自主决定是否继续读（最多 research_max_page_fetches 页）
        收尾    不带工具再问一次，确保产出的是文本答案而不是工具调用

    为什么不让代码写死"抓前两条"：该抓哪一页取决于问题，模型比规则更懂；
    但仍然用次数上限兜住成本。
    """
    settings = get_settings()
    limit = max(0, settings.research_max_page_fetches)

    def plain_answer() -> Any:
        return stream_or_invoke(
            llm,
            [
                SystemMessage(content=SYNTH_SYSTEM),
                HumanMessage(content=SYNTH_USER.format(query=query, context=context)),
            ],
        )

    if limit == 0:
        return await plain_answer()

    page_tools = await _fetch_page_tools(limit)
    if not page_tools:
        logger.warning("research.deep_no_tool")
        return await plain_answer()

    page_tool = page_tools["fetch_page"]
    first_bound = llm.bind_tools([page_tool], tool_choice="required")
    next_bound = llm.bind_tools([page_tool])

    messages: list[Any] = [
        SystemMessage(content=SYNTH_SYSTEM + DEEP_HINT.format(limit=limit)),
        HumanMessage(content=SYNTH_USER.format(query=query, context=context)),
    ]

    used = 0
    answer = ""
    while used < limit:
        bound = first_bound if used == 0 else next_bound
        accumulated: Any = None
        async for chunk in bound.astream(messages):
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = str(chunk.content or "")
            if text:
                await _emit(text)

        if accumulated is None:
            break

        calls = getattr(accumulated, "tool_calls", None) or []
        logger.info(
            "research.deep_round",
            round=used + 1,
            tool_calls=len(calls),
            chars=len(str(accumulated.content or "")),
        )
        if not calls:
            # 模型认为资料够了，直接给了答案
            answer = str(accumulated.content or "").strip()
            break

        messages.append(accumulated)
        for call in calls:
            if used >= limit:
                break
            if str(call.get("name")) != "fetch_page":
                continue
            args = call.get("args") or {}
            url = str(args.get("url", ""))
            used += 1
            logger.info("research.deep_fetch", url=url, round=used)
            try:
                page = _extract_page_text(await page_tool.ainvoke(args))
            except Exception as exc:
                page = f"抓取失败：{exc}"
            messages.append(
                ToolMessage(
                    content=f"【网页正文】{url}\n{page}",
                    tool_call_id=str(call.get("id", "")),
                )
            )

    if not answer:
        # 收尾：不给工具，逼它输出文本
        answer = await stream_or_invoke(llm, messages)

    logger.info("research.deep_done", fetches=used, chars=len(answer))
    return answer


async def _fetch_page_tools(limit: int) -> dict[str, Any]:
    """从 MCP 服务取 fetch_page 工具。拿不到就返回空（上层退回普通模式）。"""
    try:
        client = await get_mcp_client()
        tools = await client.get_tools()
        page_tool = next((t for t in tools if t.name == "fetch_page"), None)
        return {"fetch_page": page_tool} if page_tool else {}
    except Exception as exc:
        logger.warning("research.fetch_tool_failed", error=str(exc))
        return {}


def _extract_page_text(raw: Any) -> str:
    """fetch_page 返回 {"url","text","status"} 或 content blocks，统一取正文。"""
    items = parse_mcp_payload(raw)
    if items:
        first = items[0]
        text = first.get("text") or first.get("content") or ""
        return str(text)[:4000]
    return str(raw)[:4000]


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
    deep = str(state.get("depth") or "fast") == "deep"
    try:
        if deep:
            # deep 模式：把 fetch_page 交给模型，让它自己决定要不要打开网页正文
            answer = await _synthesize_deep(llm, query, context, results)
        else:
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
