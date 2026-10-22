"""爬虫能力的 MCP 服务：把采集/导出/清理包装成模型可调用的工具。

为什么要走 MCP 而不是直接调函数：
    清理、导出这类操作用户是用自然语言提的（"把 7 天前的导出清掉"），
    需要模型自己挑工具、填参数。MCP 把"能做什么"以 schema 暴露给模型，
    模型不需要知道背后是 Postgres 还是文件系统。

与 web_search 服务分开（不同端口）：
    web_search 是纯外网工具、无状态；这个服务要连数据库、会改文件，
    权限和故障影响完全不同，混在一个进程里不好控。

启动：
    python -m mcp_servers.crawler_server          # 默认 127.0.0.1:8766

安全：
    清理工具默认 dry-run（apply=False 只报告会删什么），
    模型要真删必须显式把 apply 传成 true，且删库另需 allow_db。
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from agents.crawler.cleaner import (
    apply_cleanup,
    clean_task,
    human_size,
    plan_cleanup,
    scan_exports,
)
from agents.crawler.exporter import export_task
from agents.crawler.store import get_crawl_store
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

mcp = FastMCP("crawler-tools", host="127.0.0.1", port=8766)


def _tenant(tenant_id: str = "") -> str:
    return tenant_id or get_settings().default_tenant_id


@mcp.tool()
async def crawl(
    entry_url: str,
    goal: str = "",
    max_pages: int = 3,
    export: bool = False,
    tenant_id: str = "",
) -> dict[str, Any]:
    """抓取一个网站并落库（会访问外部站点，属于有副作用的操作）。

    Args:
        entry_url: 入口 URL，例如 https://example.com/list
        goal: 任务目标，决定跟进哪些链接
        max_pages: 最多抓几页（1~20）
        export: 抓完是否直接导出成文件
        tenant_id: 租户，留空用默认租户

    Returns:
        task_id / pages / records / strategy / stop_reason / export_dir
    """
    from core.orchestrator import AgentRequest, AgentType, get_orchestrator

    request = AgentRequest(
        user_id="mcp",
        tenant_id=_tenant(tenant_id),
        session_id="mcp-crawl",
        agent_type=AgentType.CRAWLER,
        user_message=goal or entry_url,
        context={
            "entry_url": entry_url,
            "goal": goal,
            "max_pages": max(1, min(int(max_pages or 3), 20)),
        },
    )
    result = await get_orchestrator().handle(request)
    data = result.structured or {}
    task_id = int(data.get("task_id") or 0)

    export_dir = ""
    if export and task_id:
        exported = await export_task(task_id, _tenant(tenant_id))
        export_dir = exported.directory if exported.ok else ""

    logger.info("mcp.crawl", entry=entry_url, task=task_id, pages=data.get("pages"))
    return {
        "ok": result.success,
        "task_id": task_id,
        "pages": int(data.get("pages") or 0),
        "records": int(data.get("records") or 0),
        "strategy": str(data.get("strategy") or ""),
        "stop_reason": str(data.get("stop_reason") or ""),
        "export_dir": export_dir,
        "message": result.content,
    }


@mcp.tool()
async def list_tasks(limit: int = 10, tenant_id: str = "") -> list[dict[str, Any]]:
    """列出最近的爬取任务。

    Args:
        limit: 返回条数（1~50）
        tenant_id: 租户，留空用默认租户
    """
    tasks = await get_crawl_store().recent_tasks(_tenant(tenant_id), max(1, min(limit, 50)))
    return tasks


@mcp.tool()
async def task_records(task_id: int, limit: int = 20, tenant_id: str = "") -> dict[str, Any]:
    """看某个任务抽到的结构化记录。

    Args:
        task_id: 任务 ID
        limit: 返回条数（1~200）
        tenant_id: 租户
    """
    store = get_crawl_store()
    task = await store.get_task(task_id, _tenant(tenant_id))
    if task is None:
        return {"ok": False, "error": "任务不存在或不属于该租户"}
    records = await store.list_records(task_id, max(1, min(limit, 200)))
    return {
        "ok": True,
        "task_id": task_id,
        "pages": task.pages_fetched,
        "records": task.records_extracted,
        "items": [r["data"] for r in records],
    }


@mcp.tool()
async def export_files(task_id: int, tenant_id: str = "") -> dict[str, Any]:
    """把某任务的原始 HTML、正文、结构化记录导出成文件。

    Args:
        task_id: 任务 ID
        tenant_id: 租户

    Returns:
        directory / pages / records / bytes / files
    """
    result = await export_task(task_id, _tenant(tenant_id))
    logger.info("mcp.export", task=task_id, ok=result.ok)
    return {
        "ok": result.ok,
        "directory": result.directory,
        "pages": result.pages,
        "records": result.records,
        "bytes": result.bytes_written,
        "files": result.files,
        "error": result.error,
    }


@mcp.tool()
async def clean_exports(
    keep: int | None = None,
    older_than_days: float | None = None,
    task_id: int | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """清理导出目录（默认只统计，不真删）。

    只清文件，不动数据库——库里是唯一数据源，删掉的导出可以重新生成。

    Args:
        keep: 保留最近 N 个任务的导出，其余删除
        older_than_days: 删除 N 天前的导出
        task_id: 只删指定任务的导出
        apply: 是否真正删除（默认 False，只报告）
    """
    if task_id is not None:
        if not apply:
            return {"ok": True, "dry_run": True, "would_delete": [task_id]}
        done, freed = clean_task(task_id)
        return {"ok": done, "deleted": [task_id] if done else [], "freed": human_size(freed)}

    exports = scan_exports()
    targets = plan_cleanup(exports, keep=keep, older_than_days=older_than_days)
    if not targets:
        return {"ok": True, "dry_run": not apply, "message": "没有需要清理的目录", "deleted": []}

    freed = sum(item.bytes for item in targets)
    if not apply:
        logger.info("mcp.clean_dry_run", count=len(targets), freed=freed)
        return {
            "ok": True,
            "dry_run": True,
            "would_delete": [item.task_id for item in targets],
            "freed": human_size(freed),
            "hint": "确认无误后把 apply 传成 true 才会真正删除",
        }

    dirs, freed_real = apply_cleanup(targets)
    logger.info("mcp.clean_applied", dirs=dirs, freed=freed_real)
    return {
        "ok": True,
        "dry_run": False,
        "deleted": [item.task_id for item in targets],
        "dirs": dirs,
        "freed": human_size(freed_real),
    }


if __name__ == "__main__":
    logger.info("crawler.mcp_start", url="http://127.0.0.1:8766/mcp")
    mcp.run(transport="streamable-http")
