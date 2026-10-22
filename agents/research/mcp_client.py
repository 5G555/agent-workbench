"""MCP 客户端与返回解析。

单独抽出来是为了断开循环依赖：
    nodes（图节点）和 collector（证据采集）都要用 MCP，
    如果放在 nodes 里，collector 又被 nodes import 就成环了。
"""

from __future__ import annotations

import json
from typing import Any

from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_client: Any = None


async def get_mcp_client() -> Any:
    """MCP 客户端单例。懒加载，避免每次对话都建连接。"""
    global _client
    if _client is None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        settings = get_settings()
        _client = MultiServerMCPClient(
            {
                "web-search": {
                    "url": settings.web_search_mcp_url,
                    "transport": "streamable_http",
                }
            }
        )
        logger.info("research.mcp_client_ready", url=settings.web_search_mcp_url)
    return _client


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
