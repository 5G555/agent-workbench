"""MCP 客户端与返回解析。

支持多个 MCP 服务（按名字区分）：
    web-search  研究 Agent 用（外网搜索/抓页）
    ops         运维 Agent 用（任务查询/导出/清理/缓存）


为什么要分开而不是一个服务：
    web-search 是纯外网、无状态；ops 要连库、改文件、动 Redis。
    权限和故障面不同，拆开后一边挂了不影响另一边。

单独抽出来还为了断开循环依赖：
    nodes（图节点）和 collector（证据采集）都要用 MCP，
    如果放在 nodes 里，collector 又被 nodes import 就成环了。
"""

from __future__ import annotations

import json
from typing import Any

from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_clients: dict[str, Any] = {}


def _server_config(name: str) -> dict:
    """构造 MultiServerMCPClient 的连接配置。

    返回类型写成 dict（而不是 dict[str, dict[str, str]]）：
    langchain-mcp-adapters 期望的是 TypedDict 形式的 Connection，
    精确的嵌套类型会让 pyright 判不兼容。
    """
    settings = get_settings()
    if name == "ops":
        return {"ops": {"url": settings.ops_mcp_url, "transport": "streamable_http"}}
    return {"web-search": {"url": settings.web_search_mcp_url, "transport": "streamable_http"}}


async def get_mcp_client(name: str = "web-search") -> Any:
    """MCP 客户端（按服务名缓存）。懒加载，避免每次对话都建连接。"""
    if name not in _clients:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        _clients[name] = MultiServerMCPClient(_server_config(name))
        logger.info("mcp.client_ready", server=name, config=_server_config(name))
    return _clients[name]


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


def parse_mcp_dict(raw: Any) -> dict:
    """把 MCP 工具返回值归一化成 dict（对象类工具用，如 export/clean）。"""
    items = parse_mcp_payload(raw)
    if not items:
        return {}
    if len(items) == 1:
        return items[0]
    return {"items": items}
