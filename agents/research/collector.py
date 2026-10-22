"""公网证据采集：只负责"把网页结果变成 Evidence"，不负责作答。

为什么要拆出来：
    改造前 research agent 自己跑完 plan→search→synthesize，产出一份独立答案，
    再和 qa 的答案拼在一起——这是"两套系统并联"。
    按数据性质的路由思路，联网搜索只是 RAG 的**检索增强组件**：
    它提供证据，作答仍然由统一的那条 RAG 生成链完成。

返回 SearchOutcome 而不是裸列表：调用方需要知道"没搜到"和"搜失败"的区别。
"""

from __future__ import annotations

import asyncio

from agents.research.evidence import Evidence
from agents.research.mcp_client import get_mcp_client, parse_mcp_payload
from agents.research.outcome import SearchOutcome, SearchStatus, classify_error
from config import get_settings
from core.logger import get_logger
from core.quota import consume_web_quota

logger = get_logger(__name__)


def _to_evidence(items: list[dict], limit: int) -> list[Evidence]:
    """把 MCP 返回的原始条目归一化成 Evidence，并按 URL 去重。"""
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
        if len(evidence) >= limit:
            break
    return evidence


async def collect_web(
    query: str,
    max_items: int = 4,
    time_range: str = "",
    tenant_id: str = "",
) -> SearchOutcome:
    """搜一次公网，返回带状态的结果。

    每次真正发起搜索前先扣一次租户日额度：额度用完就不再打搜索服务，
    直接返回 quota_exceeded（提示语与 rate_limited 不同，用户能分清是谁的锅）。
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("evidence.no_key", query=query)
        return SearchOutcome(status=SearchStatus.NO_KEY)

    tenant = tenant_id or settings.default_tenant_id
    allowed, remaining = await consume_web_quota(tenant)
    if not allowed:
        return SearchOutcome(
            status=SearchStatus.QUOTA_EXCEEDED,
            detail=f"租户 {tenant} 当日额度已用完",
        )

    try:
        client = await get_mcp_client()
        tools = await client.get_tools()
        search_tool = next((t for t in tools if t.name == "web_search"), None)
        if search_tool is None:
            logger.warning("evidence.tool_missing", tools=[t.name for t in tools])
            return SearchOutcome(
                status=SearchStatus.FAILED, detail="MCP 服务未提供 web_search 工具"
            )

        args: dict[str, object] = {
            "query": query,
            "max_results": settings.research_results_per_query,
        }
        if time_range:  # 只有需要近期信息时才收窄检索窗口
            args["time_range"] = time_range
        raw = await search_tool.ainvoke(args)
        evidence = _to_evidence(parse_mcp_payload(raw), max_items)
    except Exception as exc:
        status = classify_error(str(exc))
        logger.warning("evidence.collect_failed", query=query, status=status.value, error=str(exc))
        return SearchOutcome(status=status, detail=str(exc))

    status = SearchStatus.OK if evidence else SearchStatus.NO_RESULT
    logger.info(
        "evidence.web_collected",
        query=query,
        status=status.value,
        items=len(evidence),
        time_range=time_range or "-",
        quota_remaining=remaining,
    )
    return SearchOutcome(status=status, items=evidence)


async def collect_multi(
    query: str,
    subqueries: list[str],
    max_items: int = 6,
    time_range: str = "",
    tenant_id: str = "",
) -> SearchOutcome:
    """按多个子问题并发采集，再按 URL 去重。

    状态取"最有希望的那个"：只要有一路成功就算成功；
    全部失败时取第一个错误状态，避免把"限流"说成"没找到"。
    """
    queries = subqueries or [query]
    outcomes = await asyncio.gather(*[collect_web(q, max_items, time_range) for q in queries])

    merged: list[Evidence] = []
    seen: set[str] = set()
    for outcome in outcomes:
        for item in outcome.items:
            key = item.url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_items:
                break
        if len(merged) >= max_items:
            break

    if merged:
        return SearchOutcome(status=SearchStatus.OK, items=merged)

    for outcome in outcomes:
        if outcome.status not in (SearchStatus.OK, SearchStatus.NO_RESULT):
            return SearchOutcome(status=outcome.status, detail=outcome.detail)
    return SearchOutcome(status=SearchStatus.NO_RESULT)


__all__ = ["collect_multi", "collect_web"]
