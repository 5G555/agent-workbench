"""Web 搜索 MCP Server。

把 Tavily 搜索包装成一个标准 MCP 服务，任何 MCP 客户端都能接：
    - 本项目的 research agent
    - Claude Desktop / Cursor 等支持 MCP 的客户端
    - 以后的别的应用

启动（streamable-http，独立进程）：
    python -m mcp_servers.web_search_server
    # 默认监听 127.0.0.1:8765，MCP 端点为 http://127.0.0.1:8765/mcp

为什么用 streamable-http 而不是 stdio：
    stdio 模式下客户端每连一次就拉起一个子进程。FastAPI 常驻进程里反复拉进程
    既慢又容易泄漏，改成一个常驻的 HTTP 服务更合适。

工具清单：
    web_search(query, max_results)  → 搜索并返回 [{title, url, content, score}]
    fetch_page(url, max_chars)      → 抓取网页正文（可选，用于补充细节）
"""

from __future__ import annotations

import httpx
from mcp.server.fastmcp import FastMCP

from agents.research.tools import web_search as tavily_search
from config import get_settings
from core.logger import configure_logging, get_logger

logger = get_logger(__name__)

HOST = "127.0.0.1"
PORT = 8765

mcp = FastMCP("web-search", host=HOST, port=PORT)


@mcp.tool()
async def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Search the public web and return ranked results.

    Args:
        query: 搜索词，中文或英文均可。
        max_results: 返回条数，1~10。

    Returns:
        列表，每项含 title / url / content / score 四个字段。
    """
    n = max(1, min(int(max_results or 4), 10))
    response = await tavily_search(query, n)
    logger.info("mcp.web_search", query=query, hits=len(response.results))
    return [result.__dict__ for result in response.results]


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 4000) -> dict:
    """Fetch a web page and return its text content.

    Args:
        url: 完整 URL（http/https）。
        max_chars: 最多返回多少字符，默认 4000。

    Returns:
        {"url": ..., "text": ..., "status": ...}
    """
    if not url.startswith(("http://", "https://")):
        return {"url": url, "text": "", "status": 400, "error": "只支持 http/https"}

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Myagent/0.1"})
        text = resp.text
        # 粗暴去标签：够用即可，正文抽取不是这个工具的重点
        import re

        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "text": text[:max_chars], "status": resp.status_code}
    except httpx.HTTPError as exc:
        return {"url": url, "text": "", "status": 0, "error": str(exc)}


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.tavily_api_key:
        logger.warning("mcp.no_tavily_key", hint="web_search 会返回空结果")
    logger.info("mcp.server_starting", host=HOST, port=PORT, endpoint=f"http://{HOST}:{PORT}/mcp")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
