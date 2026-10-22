"""公网证据采集：只负责"把网页结果变成 Evidence"，不负责作答。

为什么要拆出来：
    改造前 research agent 自己跑完 plan→search→synthesize，产出一份独立答案，
    再和 qa 的答案拼在一起——这是"两套系统并联"。
    按数据性质的路由思路，联网搜索只是 RAG 的**检索增强组件**：
    它提供证据，作答仍然由统一的那条 RAG 生成链完成。
"""

from __future__ import annotations

import asyncio

from agents.research.evidence import WEB, Evidence
from agents.research.nodes import _get_mcp_client, parse_mcp_payload
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


async def collect_web_evidence(query: str, max_items: int = 4) -> list[Evidence]:
    """搜一次公网，返回归一化后的证据列表。失败返回空列表（不阻断主流程）。"""
    settings = get_settings()
    try:
        client = await _get_mcp_client()
        tools = await client.get_tools()
        search_tool = next((t for t in tools if t.name == "web_search"), None)
        if search_tool is None:
            logger.warning("evidence.tool_missing", tools=[t.name for t in tools])
            return []

        raw = await search_tool.ainvoke(
            {"query": query, "max_results": settings.research_results_per_query}
        )
        items = parse_mcp_payload(raw)

        evidence: list[Evidence] = []
        seen: set[str] = set()
        for item in items:
            url = str(item.get("url", "")).strip().rstrip("/")
            if not url or url in seen:
                continue
            seen.add(url)
            evidence.append(
                Evidence.from_web(
                    title=str(item.get("title", "")),
                    url=url,
                    content=str(item.get("content", "")),
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
            if len(evidence) >= max_items:
                break

        logger.info("evidence.web_collected", query=query, items=len(evidence))
        return evidence
    except Exception as exc:
        logger.warning("evidence.collect_failed", query=query, error=str(exc))
        return []


async def collect_multi(query: str, subqueries: list[str], max_items: int = 6) -> list[Evidence]:
    """按多个子问题并发采集，再按 URL 去重。子问题为空时退回单次搜索。"""
    queries = subqueries or [query]
    batches = await asyncio.gather(*[collect_web_evidence(q, max_items) for q in queries])

    merged: list[Evidence] = []
    seen: set[str] = set()
    for batch in batches:
        for item in batch:
            key = item.url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_items:
                return merged
    return merged


__all__ = ["WEB", "collect_multi", "collect_web_evidence"]
