"""联网研究 Agent 的检索工具（Tavily）。

为什么单独抽一层：图节点只关心"给我结果"，不关心是哪个搜索服务。
将来换成自建 SearxNG 或 Bing，只要这个文件保持同样的返回结构即可。

Tavily 的响应字段：title / url / content / score。这里的 content 是它预先
抓取并截断的正文片段，直接当上下文用，不再二次抓网页（省时且稳定）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from config import get_settings
from core.exceptions import MyAgentBaseError
from core.logger import get_logger

logger = get_logger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class WebSearchError(MyAgentBaseError):
    """搜索服务失败（可重试）。"""

    pass


@dataclass
class SearchResult:
    """一条搜索结果。"""

    title: str
    url: str
    content: str
    score: float = 0.0

    def as_context(self, index: int) -> str:
        """拼成喂给模型的上下文片段，带序号方便引用。"""
        body = self.content.strip().replace("\n", " ")[:800]
        return f"[{index}] {self.title}\n来源：{self.url}\n正文：{body}"


@dataclass
class SearchResponse:
    """一次搜索的聚合结果。"""

    query: str
    results: list[SearchResult] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.results


async def web_search(query: str, max_results: int | None = None) -> SearchResponse:
    """调用 Tavily 搜索。失败抛 WebSearchError（交给 with_retry 处理）。"""
    settings = get_settings()
    if not settings.tavily_api_key:
        raise WebSearchError("未配置 TAVILY_API_KEY")

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results or settings.research_results_per_query,
        "search_depth": "basic",
        "include_answer": False,
    }

    try:
        async with httpx.AsyncClient(timeout=25.0, trust_env=False) as client:
            resp = await client.post(TAVILY_ENDPOINT, json=payload)
    except httpx.HTTPError as exc:
        raise WebSearchError(f"搜索请求失败：{exc}") from exc

    if resp.status_code == 429:
        raise WebSearchError("搜索服务限流")
    if resp.status_code >= 400:
        raise WebSearchError(f"搜索服务返回 {resp.status_code}")

    data = resp.json()
    results = [
        SearchResult(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            content=str(item.get("content", "")),
            score=float(item.get("score", 0.0)),
        )
        for item in data.get("results", [])
    ]
    logger.info("research.search_done", query=query, hits=len(results))
    return SearchResponse(query=query, results=results)


def dedup_results(responses: list[SearchResponse]) -> list[SearchResult]:
    """多个子查询的结果按 URL 去重，保留首次出现（分数更高的通常在前）。"""
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for response in responses:
        for item in response.results:
            key = item.url.strip().rstrip("/")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged
