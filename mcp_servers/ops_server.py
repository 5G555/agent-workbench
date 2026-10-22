"""运维能力的 MCP 服务：爬取任务管理 + 缓存巡检 + 破坏性操作的执行网关。

为什么叫 ops 而不是 crawler：
    这里暴露的是"运维动作"——查任务、导出、清文件、清缓存。
    采集本身（crawl）只是其中一个工具。

与 web_search 分开的原因：
    web_search 是纯外网、无状态；这个服务要连库、改文件、动 Redis。

破坏性工具（clean_exports / clean_cache / purge_trash）全部走
core/safety/gateway.execute：
    策略分级 → 人工审批（高风险）→ 幂等 → 预算熔断 → 快照 → 执行 → 验证/回滚
模型只能"提出请求"，放不放行由确定性策略决定，见 core/safety/policy.py。

启动：python -m mcp_servers.ops_server        # 默认 127.0.0.1:8766
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from agents.crawler.exporter import export_task
from agents.crawler.store import get_crawl_store
from agents.ops import trash
from agents.ops.cache_maintenance import clean as clean_cache_keys
from agents.ops.cache_maintenance import list_backups, restore_backup
from agents.ops.cache_maintenance import overview as cache_overview
from agents.ops.cleaner import human_size, plan_cleanup, scan_exports
from agents.ops.gateway import execute as gateway_execute
from agents.ops.policy import Decision, evaluate
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

mcp = FastMCP("ops-tools", host="127.0.0.1", port=8766)


def _tenant(tenant_id: str = "") -> str:
    return tenant_id or get_settings().default_tenant_id


# ── 采集任务（只读/可重生成）────────────────────────────────
@mcp.tool()
async def crawl(
    entry_url: str,
    goal: str = "",
    max_pages: int = 3,
    export: bool = False,
    tenant_id: str = "",
) -> dict[str, Any]:
    """抓取一个网站并落库（会访问外部站点）。

    Args:
        entry_url: 入口 URL，例如 https://example.com/list
        goal: 任务目标，决定跟进哪些链接
        max_pages: 最多抓几页（1~20）
        export: 抓完是否直接导出成文件
        tenant_id: 租户
    """
    # 入口 URL 也过策略：挡住 file://、云元数据地址、本机地址这类越界目标
    verdict = evaluate("crawl", {"entry_url": entry_url})
    if verdict.decision is Decision.BLOCK:
        return {"ok": False, "blocked": True, "reasons": verdict.reasons}

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
    """列出最近的爬取任务。"""
    return await get_crawl_store().recent_tasks(_tenant(tenant_id), max(1, min(limit, 50)))


@mcp.tool()
async def task_records(task_id: int, limit: int = 20, tenant_id: str = "") -> dict[str, Any]:
    """看某个任务抽到的结构化记录。"""
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
    """把某任务的原始 HTML、正文、结构化记录导出成文件（可重复生成）。"""
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


# ── 缓存（只读）──────────────────────────────────────────────
@mcp.tool()
async def cache_stats() -> dict[str, Any]:
    """看 Redis 缓存概览：当前版本、键总数、旧版本残留数。只读。"""
    data = await cache_overview()
    logger.info("mcp.cache_stats", **data)
    return {
        "ok": data["version"] > 0,
        "version": data["version"],
        "total_keys": data["total"],
        "current_version_keys": data["current"],
        "stale_keys": data["stale"],
    }


# ── 回收站 ──────────────────────────────────────────────────
@mcp.tool()
async def list_trash(limit: int = 20) -> list[dict[str, Any]]:
    """看删除回收站里有什么（软删除的导出目录）。"""
    return [item.as_dict() for item in trash.list_trash()[: max(1, min(limit, 100))]]


@mcp.tool()
async def restore_exports(trash_id: str) -> dict[str, Any]:
    """把回收站里的导出目录恢复回原位置。"""
    ok, detail = trash.restore(trash_id)
    logger.info("mcp.restore_exports", trash_id=trash_id, ok=ok)
    return {"ok": ok, "detail": detail}


@mcp.tool()
async def restore_cache(backup_id: str) -> dict[str, Any]:
    """从缓存回收区恢复被清理的键。"""
    ok, count = await restore_backup(backup_id)
    logger.info("mcp.restore_cache", backup_id=backup_id, ok=ok, keys=count)
    return {"ok": ok, "restored_keys": count, "available": await list_backups()}


# ── 破坏性操作：全部走执行网关 ───────────────────────────────
@mcp.tool()
async def clean_exports(
    keep: int | None = None,
    older_than_days: float | None = None,
    task_id: int | None = None,
    apply: bool = False,
    approval_token: str = "",
    idempotency_key: str = "",
    tenant_id: str = "",
) -> dict[str, Any]:
    """清理导出目录：默认只统计；真删需要策略放行（高风险要审批）。

    删除是**移到回收站**，不是物理删除，可用 restore_exports 恢复。

    Args:
        keep: 保留最近 N 个任务的导出
        older_than_days: 删除 N 天前的导出
        task_id: 只清理指定任务
        apply: 是否真正执行（默认 False 只报告）
        approval_token: 审批通过后拿到的票据（高风险操作必填）
        idempotency_key: 幂等键，重复请求返回首次结果
        tenant_id: 租户
    """
    args = {
        "task_id": task_id,
        "keep": keep,
        "older_than_days": older_than_days,
        "apply": bool(apply),
    }
    tenant = _tenant(tenant_id)

    if not apply:
        targets = plan_cleanup(scan_exports(), keep=keep, older_than_days=older_than_days)
        return {
            "ok": True,
            "dry_run": True,
            "would_delete": [item.task_id for item in targets],
            "freed": human_size(sum(item.bytes for item in targets)),
            "hint": "确认无误后传 apply=true（高风险可能还需要 approval_token）",
        }

    async def victims(_args: dict[str, Any]) -> list[Any]:
        if task_id is not None:
            return [task_id]
        return [
            item.task_id
            for item in plan_cleanup(scan_exports(), keep=keep, older_than_days=older_than_days)
        ]

    moved: list[str] = []

    async def runner(_args: dict[str, Any]) -> dict[str, Any]:
        if task_id is not None:
            item = trash.move_to_trash(task_id)
            if item is None:
                return {"deleted": [], "freed": 0, "message": "没有可清理的目录"}
            moved.append(item.trash_id)
            return {"deleted": [task_id], "freed": item.size, "trash_ids": moved}
        targets = plan_cleanup(scan_exports(), keep=keep, older_than_days=older_than_days)
        freed = 0
        for target in targets:
            item = trash.move_to_trash(target.task_id)
            if item is not None:
                moved.append(item.trash_id)
                freed += item.size
        return {"deleted": [t.task_id for t in targets], "freed": freed, "trash_ids": moved}

    async def verify(_args: dict[str, Any], data: dict[str, Any]) -> bool:
        # 验证：原目录已不在，且回收站里有对应条目
        for task in data.get("deleted") or []:
            if (trash.DEFAULT_ROOT / str(task)).exists():
                return False
        return True

    async def rollback(_args: dict[str, Any], _data: dict[str, Any]) -> None:
        for trash_id in moved:
            trash.restore(trash_id)

    result = await gateway_execute(
        "clean_exports",
        args,
        tenant_id=tenant,
        idempotency_key=idempotency_key or None,
        approval_token=approval_token or None,
        runner=runner,
        victims=victims,
        verify=verify,
        rollback=rollback,
        impact=f"清理 {len(await victims(args))} 个导出目录",
        reversible="移入 data/crawl/.trash，可用 restore_exports 恢复",
    )
    return result.as_dict()


@mcp.tool()
async def clean_cache(
    apply: bool = False,
    approval_token: str = "",
    idempotency_key: str = "",
) -> dict[str, Any]:
    """清理 Redis 里旧版本的缓存键：默认只统计；真删需要审批。

    删除前会先把键值备份到 cache:trash:*，可用 restore_cache 恢复；
    备份失败则一个都不删（Fail-Closed）。

    Args:
        apply: 是否真正删除（默认 False 只报告）
        approval_token: 审批票据
        idempotency_key: 幂等键
    """
    if not apply:
        report = await clean_cache_keys(apply=False)
        return {
            "ok": report.version > 0,
            "dry_run": True,
            "version": report.version,
            "stale": report.stale_total,
            "kept": report.kept_total,
            "hint": "确认无误后传 apply=true（需要 approval_token）",
        }

    backup_id = ""

    async def victims(_args: dict[str, Any]) -> list[Any]:
        report = await clean_cache_keys(apply=False)
        return [len(s.stale) for s in report.scans for _ in s.stale]

    async def runner(_args: dict[str, Any]) -> dict[str, Any]:
        nonlocal backup_id
        report = await clean_cache_keys(apply=True)
        backup_id = report.backup_id
        return {
            "version": report.version,
            "deleted": report.deleted,
            "kept": report.kept_total,
            "backup_id": report.backup_id,
        }

    async def verify(_args: dict[str, Any], data: dict[str, Any]) -> bool:
        return bool(data.get("backup_id")) or data.get("deleted", 0) == 0

    async def rollback(_args: dict[str, Any], data: dict[str, Any]) -> None:
        if data.get("backup_id"):
            await restore_backup(data["backup_id"])

    result = await gateway_execute(
        "clean_cache",
        {"apply": True},
        tenant_id=get_settings().default_tenant_id,
        idempotency_key=idempotency_key or None,
        approval_token=approval_token or None,
        runner=runner,
        victims=victims,
        verify=verify,
        rollback=rollback,
        impact="清理旧版本缓存键",
        reversible="先备份到 cache:trash:*，可用 restore_cache 恢复",
    )
    return result.as_dict()


@mcp.tool()
async def purge_trash(
    older_than_days: float | None = None,
    apply: bool = False,
    approval_token: str = "",
    idempotency_key: str = "",
) -> dict[str, Any]:
    """永久删除回收站内容（**不可恢复**）：默认只统计，真删必须审批。

    Args:
        older_than_days: 只清超过 N 天的；不传则清空全部
        apply: 是否真正删除
        approval_token: 审批票据（必填）
        idempotency_key: 幂等键
    """
    items = trash.list_trash()
    targets = [
        item for item in items if older_than_days is None or item.age_days >= older_than_days
    ]

    if not apply:
        return {
            "ok": True,
            "dry_run": True,
            "would_delete": [item.trash_id for item in targets],
            "freed": human_size(sum(item.size for item in targets)),
            "warning": "purge 是永久删除，无法恢复",
        }

    async def victims(_args: dict[str, Any]) -> list[Any]:
        return [item.trash_id for item in targets]

    async def runner(_args: dict[str, Any]) -> dict[str, Any]:
        dirs, freed = trash.purge(older_than_days=older_than_days)
        return {"purged": dirs, "freed": freed}

    result = await gateway_execute(
        "purge_trash",
        {"older_than_days": older_than_days, "apply": True},
        tenant_id=get_settings().default_tenant_id,
        idempotency_key=idempotency_key or None,
        approval_token=approval_token or None,
        runner=runner,
        victims=victims,
        impact=f"永久删除 {len(targets)} 个回收站条目",
        reversible="不可恢复",
    )
    return result.as_dict()


if __name__ == "__main__":
    logger.info("ops.mcp_start", url="http://127.0.0.1:8766/mcp")
    mcp.run(transport="streamable-http")
