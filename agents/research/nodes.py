"""联网研究 Agent 的节点。

流程：plan（拆子问题）→ search（并发调 MCP 工具）→ synthesize（带引用作答）

关键点：
    1. 搜索不是直接 httpx 调 Tavily，而是走 MCP 客户端 —— 工具由独立的
       MCP server 提供，Agent 只认协议不认实现。
    2. 多个子问题并发搜索，结果按 URL 去重后再喂给模型。
    3. 答案要求带 [序号] 引用，序号对应喂进去的资料顺序。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.research.prompts import (
    NO_RESULT_ANSWER,
    PLAN_SYSTEM,
    PLAN_USER,
    SYNTH_SYSTEM,
    SYNTH_USER,
)
from agents.research.state import ResearchState
from agents.research.tools import SearchResult, dedup_results
from config import get_settings
from core.llm_factory import LLMFactory
from core.logger import get_logger
from core.streaming import stream_or_invoke

logger = get_logger(__name__)

_mcp_client: Any = None


class Plan(BaseModel):
    """模型输出的检索计划。"""

    subqueries: list[str] = Field(default_factory=list)


def _extract_query(state: ResearchState) -> str:
    """从消息里取本轮问题；orchestrator 会显式写入 original_query。"""
    query = state.get("original_query")
    if query:
        return str(query)
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


async def _get_mcp_client() -> Any:
    """MCP 客户端单例。懒加载，避免每次对话都建连接。"""
    global _mcp_client
    if _mcp_client is None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        settings = get_settings()
        _mcp_client = MultiServerMCPClient(
            {
                "web-search": {
                    "url": settings.web_search_mcp_url,
                    "transport": "streamable_http",
                }
            }
        )
        logger.info("research.mcp_client_ready", url=settings.web_search_mcp_url)
    return _mcp_client


def parse_mcp_payload(raw: Any) -> list[dict]:
    """把 MCP 工具返回值归一化成 list[dict]。

    FastMCP 把 list[dict] 序列化成多个 content block：
        [{"type": "text", "text": "{\"title\": ...}"}, ...]
    这里兼容三种形状：content blocks、单个 JSON 字符串、已是 list[dict]。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else [data]

    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and "type" in raw[0]:
            items: list[dict] = []
            for block in raw:
                if block.get("type") != "text":
                    continue
                try:
                    parsed = json.loads(block.get("text", ""))
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    items.append(parsed)
                elif isinstance(parsed, list):
                    items.extend(x for x in parsed if isinstance(x, dict))
            return items
        return [x for x in raw if isinstance(x, dict)]

    return []


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
    """② 并发调 MCP 的 web_search 工具，结果去重。"""
    subqueries = state.get("subqueries") or []
    if not subqueries:
        return {"search_results": []}

    settings = get_settings()
    try:
        client = await _get_mcp_client()
        tools = await client.get_tools()
        search_tool = next((t for t in tools if t.name == "web_search"), None)
        if search_tool is None:
            logger.error("research.tool_missing", tools=[t.name for t in tools])
            return {"search_results": [], "error": "MCP 服务未提供 web_search 工具"}

        async def one(q: str) -> list[SearchResult]:
            try:
                raw = await search_tool.ainvoke(
                    {"query": q, "max_results": settings.research_results_per_query}
                )
            except Exception as exc:
                logger.warning("research.search_failed", query=q, error=str(exc))
                return []
            return [
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    content=str(item.get("content", "")),
                    score=float(item.get("score", 0.0) or 0.0),
                )
                for item in parse_mcp_payload(raw)
            ]

        batches = await asyncio.gather(*[one(q) for q in subqueries])

        class _Batch:
            def __init__(self, results: list[SearchResult]) -> None:
                self.results = results

        merged = dedup_results([_Batch(b) for b in batches])  # type: ignore[arg-type]
        # 只保留前若干条，避免上下文过长
        merged = merged[: settings.research_results_per_query * 2]

        logger.info(
            "research.searched",
            subqueries=len(subqueries),
            raw=sum(len(b) for b in batches),
            unique=len(merged),
        )
        return {"search_results": [r.__dict__ for r in merged]}
    except Exception as exc:
        logger.error("research.search_node_failed", error=str(exc))
        return {"search_results": [], "error": str(exc)}


async def synthesize_node(state: ResearchState) -> dict[str, Any]:
    """③ 依据资料作答，并要求标注 [序号]。没资料时直接给固定答复，不调模型。"""
    query = state.get("original_query") or ""
    results = state.get("search_results") or []

    if not results:
        logger.info("research.no_result", query=query)
        return {
            "answer": NO_RESULT_ANSWER,
            "answer_mode": "no_result",
            "citations": [],
            "messages": [AIMessage(content=NO_RESULT_ANSWER)],
            "structured_output": {
                "answer_mode": "no_result",
                "sources": [],
                "citations": [],
                "subqueries": state.get("subqueries") or [],
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
