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

import os

import httpx
from mcp.server.fastmcp import FastMCP

from agents.research.tools import web_search as tavily_search
from config import get_settings
from core.logger import configure_logging, get_logger

logger = get_logger(__name__)

HOST = os.getenv("WEB_SEARCH_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("WEB_SEARCH_MCP_PORT", "8765"))

mcp = FastMCP("web-search", host=HOST, port=PORT)


@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 4,
    time_range: str = "",
    topic: str = "general",
) -> list[dict]:
    """Search the public web and return ranked results.

    Args:
        query: 搜索词，中文或英文均可。
        max_results: 返回条数，1~10。
        time_range: 只看多久以内的内容：day / week / month / year，留空表示不限。
        topic: general（通用）或 news（新闻）。

    Returns:
        列表，每项含 title / url / content / score 四个字段。
    """
    n = max(1, min(int(max_results or 4), 10))
    response = await tavily_search(
        query,
        n,
        time_range=time_range or None,
        topic=topic or "general",
    )
    logger.info(
        "mcp.web_search",
        query=query,
        hits=len(response.results),
        time_range=time_range or "-",
        topic=topic or "general",
    )
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
        text = html_to_text(resp.text)
        return {"url": url, "text": text[:max_chars], "status": resp.status_code}
    except httpx.HTTPError as exc:
        return {"url": url, "text": "", "status": 0, "error": str(exc)}


def html_to_text(html: str) -> str:
    """把 HTML 变成可读正文。

    四步，顺序不能乱：
        1. 优先只保留 <article> / <main> 里的内容 —— 文档站的侧边栏、推荐位
           常常不在 <nav> 里，整页剥离很难干净
        2. 从第一个 <h1> 开始切 —— 正文标题之前通常是推广条、面包屑
        3. 删掉脚本/样式/导航等整块标签
        4. 块级标签转换行（保留段落与列表结构），再去剩余标签
    """
    import re

    # 1. 定位主内容容器
    for pattern in (r"<article[^>]*>(.*?)</article>", r"<main[^>]*>(.*?)</main>"):
        match = re.search(pattern, html, re.S | re.I)
        if match and len(match.group(1)) > 500:
            html = match.group(1)
            break

    # 2. 从正文标题开始
    heading = re.search(r"<h1[^>]*>", html, re.I)
    if heading and heading.start() > 0:
        html = html[heading.start() :]

    # 3. 整块剔除噪声标签
    html = re.sub(
        r"<(script|style|nav|header|footer|aside|noscript|form|svg)[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.S | re.I,
    )

    # 4. 保留段落结构后去标签
    html = re.sub(r"</(p|div|h[1-6]|li|tr|section|article)>", "\n", html, flags=re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = html_module_unescape(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def html_module_unescape(text: str) -> str:
    """还原 &amp; &lt; 这类实体，否则正文里全是转义字符。"""
    import html as html_module

    return html_module.unescape(text)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.tavily_api_key:
        logger.warning("mcp.no_tavily_key", hint="web_search 会返回空结果")
    logger.info("mcp.server_starting", host=HOST, port=PORT, endpoint=f"http://{HOST}:{PORT}/mcp")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
