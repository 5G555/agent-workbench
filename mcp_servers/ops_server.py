"""运维能力的 MCP 服务：爬取任务管理 + 缓存巡检。

为什么叫 ops 而不是 crawler：
    这里暴露的是"运维动作"——查任务、导出、清文件、清缓存。
    采集本身（crawl）只是其中一个工具。以后再加数据备份、重建索引，
    都放这个服务里，不用再开新端口。

为什么与 web_search 分开：
    web_search 是纯外网、无状态；这个服务要连数据库、改文件、动 Redis，
    权限和故障影响完全不同，混在一个进程里不好控。

工具清单：
    crawl           抓一个网址落库（有副作用）
    list_tasks      列最近任务
    task_records    看结构化记录
    export_files    导出成文件
    clean_exports   清导出目录（默认 dry-run）
    cache_stats     缓存概览（版本、键数、残留）
    clean_cache     清旧版本缓存键（默认 dry-run）

启动：python -m mcp_servers.ops_server        # 默认 127.0.0.1:8766
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
from core.cache_maintenance import clean as clean_cache_keys
from core.cache_maintenance import overview as cache_overview
from core.logger import get_logger

logger = get_logger(__name__)

mcp = FastMCP("ops-tools", host="127.0.0.1", port=8766)


def _tenant(tenant_id: str = "") -> str:
    return tenant_id or get_settings().default_tenant_id


# ── 采集任务 ────────────────────────────────────────────────
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
    return await get_crawl_store().recent_tasks(_tenant(tenant_id), max(1, min(limit, 50)))


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


# ── 缓存 ────────────────────────────────────────────────────
@mcp.tool()
async def cache_stats() -> dict[str, Any]:
    """看 Redis 缓存概览：当前版本、键总数、旧版本残留数。

    只读，不消耗也不删除任何缓存。
    """
    data = await cache_overview()
    logger.info("mcp.cache_stats", **data)
    return {
        "ok": data["version"] > 0,
        "version": data["version"],
        "total_keys": data["total"],
        "current_version_keys": data["current"],
        "stale_keys": data["stale"],
    }


@mcp.tool()
async def clean_cache(apply: bool = False) -> dict[str, Any]:
    """清理 Redis 里旧版本的缓存键（默认只统计，不真删）。

    只删"版本号不等于当前版本"的键——它们已经再也命不中，只占内存；
    当前版本的缓存永远不碰，所以清理不会让正在用的缓存失效。

    Args:
        apply: 是否真正删除（默认 False，只报告会删多少）

    Returns:
        version / stale / kept / deleted / by_prefix
    """
    report = await clean_cache_keys(apply=apply)
    payload = {
        "ok": report.version > 0,
        "dry_run": not apply,
        "version": report.version,
        "stale": report.stale_total,
        "kept": report.kept_total,
        "deleted": report.deleted,
        "by_prefix": [
            {"prefix": scan.label, "stale": len(scan.stale), "kept": scan.kept}
            for scan in report.scans
        ],
    }
    if report.version == 0:
        payload["error"] = "Redis 不可用或没有版本号"
    elif not apply and report.stale_total:
        payload["hint"] = "确认无误后把 apply 传成 true 才会真正删除"
    logger.info("mcp.clean_cache", apply=apply, stale=report.stale_total)
    return payload


if __name__ == "__main__":
    logger.info("ops.mcp_start", url="http://127.0.0.1:8766/mcp")
    mcp.run(transport="streamable-http")
