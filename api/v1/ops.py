"""运维接口：缓存巡检/清理、导出目录巡检/清理、回收站与审批。

定位：给界面用的"确定按钮"，不是给模型用的。
    Agent 走 MCP 工具（mcp_servers/ops_server.py），界面走这里，
    两边最终都经过 agents/ops/gateway.py —— 同一套策略、同一套护栏。

所以这里的行为和 Agent 完全一致：
    - 只读接口随便调
    - 破坏性接口默认 dry-run
    - 真要执行：策略分级 → 高风险要审批票据 → 幂等 → 预算 → 快照 → 验证/回滚

界面上的"审批"就是调 /approvals/{id}/confirm，确认后拿票据再执行一次。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from agents.ops import approval, trash
from agents.ops.audit import get_audit_store
from agents.ops.cache_maintenance import clean as clean_cache_keys
from agents.ops.cache_maintenance import list_backups, restore_backup
from agents.ops.cache_maintenance import overview as cache_overview
from agents.ops.cleaner import human_size, plan_cleanup, scan_exports
from agents.ops.gateway import execute as gateway_execute
from agents.ops.metrics.views import METRIC_CATALOG, VIEW_CATALOG
from api.deps import get_current_user, rate_limit
from core.logger import get_logger
from core.users import get_user_store

logger = get_logger(__name__)

router = APIRouter()


class CacheCleanRequest(BaseModel):
    apply: bool = Field(default=False, description="是否真正删除，默认只统计")
    kinds: list[str] = Field(
        default_factory=lambda: ["qa", "analyst"],
        description="要清理的类别：qa（旧版本问答缓存）、analyst（分析结果缓存）",
    )
    tenant: str = Field(default="", description="只清某个租户的分析缓存，留空表示全部")
    approval_token: str = Field(default="", description="审批票据")
    idempotency_key: str = Field(default="", description="幂等键")


class ExportCleanRequest(BaseModel):
    keep: int | None = Field(default=None, ge=0, le=100, description="保留最近 N 个任务的导出")
    older_than_days: float | None = Field(default=None, ge=0, description="删除 N 天前的导出")
    task_id: int | None = Field(default=None, description="只删指定任务")
    apply: bool = Field(default=False, description="是否真正删除")
    approval_token: str = Field(default="", description="审批票据")
    idempotency_key: str = Field(default="", description="幂等键")


class PurgeRequest(BaseModel):
    older_than_days: float | None = Field(default=None, ge=0)
    apply: bool = Field(default=False)
    approval_token: str = Field(default="")
    idempotency_key: str = Field(default="")


# ── 只读 ────────────────────────────────────────────────────
@router.get("/cache", summary="缓存概览")
async def cache_stats(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    data = await cache_overview()
    return {
        "ok": data["version"] > 0,
        "version": data["version"],
        "total_keys": data["total"],
        "current_keys": data["current"],
        "stale_keys": data["stale"],
        "analyst_keys": data.get("analyst", 0),
        "backups": await list_backups(),
    }


@router.get("/exports", summary="导出目录列表")
async def exports(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    items = scan_exports()
    total = sum(item.bytes for item in items)
    return {
        "count": len(items),
        "total_bytes": total,
        "total_human": human_size(total),
        "items": [
            {
                "task_id": item.task_id,
                "files": item.files,
                "bytes": item.bytes,
                "size": human_size(item.bytes),
                "age_days": round(item.age_days, 1),
                "path": str(item.path),
            }
            for item in items
        ],
    }


@router.get("/trash", summary="回收站内容")
async def trash_list(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    items = trash.list_trash()
    total = sum(item.size for item in items)
    return {
        "count": len(items),
        "total_human": human_size(total),
        "items": [item.as_dict() for item in items],
    }


# ── 恢复（安全操作，不需要审批）─────────────────────────────
@router.post("/trash/{trash_id}/restore", summary="从回收站恢复")
async def trash_restore(
    trash_id: str,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    ok, detail = trash.restore(trash_id)
    return {"ok": ok, "detail": detail}


@router.post("/cache/backups/{backup_id}/restore", summary="从备份恢复缓存键")
async def cache_restore(
    backup_id: str,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    ok, count = await restore_backup(backup_id)
    return {"ok": ok, "restored_keys": count}


# ── 破坏性：走执行网关 ──────────────────────────────────────
@router.post("/exports/clean", summary="清理导出目录（移入回收站）")
async def exports_clean(
    request: ExportCleanRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    args = {
        "task_id": request.task_id,
        "keep": request.keep,
        "older_than_days": request.older_than_days,
        "apply": request.apply,
    }
    if not request.apply:
        targets = plan_cleanup(
            scan_exports(), keep=request.keep, older_than_days=request.older_than_days
        )
        return {
            "ok": True,
            "dry_run": True,
            "would_delete": [item.task_id for item in targets],
            "freed": human_size(sum(item.bytes for item in targets)),
            "hint": "确认后传 apply=true；高风险操作会返回 needs_approval",
        }

    moved: list[str] = []

    async def victims(_args: dict[str, Any]) -> list[Any]:
        if request.task_id is not None:
            return [request.task_id]
        return [
            item.task_id
            for item in plan_cleanup(
                scan_exports(), keep=request.keep, older_than_days=request.older_than_days
            )
        ]

    async def runner(_args: dict[str, Any]) -> dict[str, Any]:
        if request.task_id is not None:
            item = trash.move_to_trash(request.task_id)
            if item is None:
                return {"deleted": [], "freed": 0, "trash_ids": []}
            moved.append(item.trash_id)
            return {"deleted": [request.task_id], "freed": item.size, "trash_ids": moved}

        targets = plan_cleanup(
            scan_exports(), keep=request.keep, older_than_days=request.older_than_days
        )
        freed = 0
        for target in targets:
            item = trash.move_to_trash(target.task_id)
            if item is not None:
                moved.append(item.trash_id)
                freed += item.size
        return {"deleted": [t.task_id for t in targets], "freed": freed, "trash_ids": moved}

    async def rollback(_args: dict[str, Any], _data: dict[str, Any]) -> None:
        for trash_id in moved:
            trash.restore(trash_id)

    result = await gateway_execute(
        "clean_exports",
        args,
        tenant_id=user["tenant_id"],
        user_id=user["user_id"],
        idempotency_key=request.idempotency_key or None,
        approval_token=request.approval_token or None,
        runner=runner,
        victims=victims,
        rollback=rollback,
        impact=f"清理 {len(await victims(args))} 个导出目录",
        reversible="移入 data/crawl/.trash，可恢复",
    )
    return result.as_dict()


@router.post("/cache/clean", summary="清理旧版本缓存键（先备份）")
async def cache_clean(
    request: CacheCleanRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    selected = [k for k in request.kinds if k in {"qa", "analyst"}] or ["qa", "analyst"]
    tenant = request.tenant or None

    if not request.apply:
        report = await clean_cache_keys(apply=False, kinds=selected, tenant=tenant)
        return {
            "ok": report.version > 0,
            "dry_run": True,
            "kinds": selected,
            "version": report.version,
            "stale": report.stale_total,
            "kept": report.kept_total,
            "analyst_keys": sum(
                len(scan.stale) for scan in report.scans if scan.pattern.startswith("analyst:cache")
            ),
            "by_prefix": [
                {"prefix": scan.label, "stale": len(scan.stale), "kept": scan.kept}
                for scan in report.scans
            ],
            "hint": "确认后传 apply=true；需要审批票据",
        }

    async def victims(_args: dict[str, Any]) -> list[Any]:
        report = await clean_cache_keys(apply=False, kinds=selected, tenant=tenant)
        return [scan.label for scan in report.scans if scan.stale]

    async def runner(_args: dict[str, Any]) -> dict[str, Any]:
        report = await clean_cache_keys(apply=True, kinds=selected, tenant=tenant)
        return {
            "version": report.version,
            "deleted": report.deleted,
            "kept": report.kept_total,
            "backup_id": report.backup_id,
        }

    async def verify(_args: dict[str, Any], data: dict[str, Any]) -> bool:
        return bool(data.get("backup_id")) or int(data.get("deleted") or 0) == 0

    async def rollback(_args: dict[str, Any], data: dict[str, Any]) -> None:
        if data.get("backup_id"):
            await restore_backup(data["backup_id"])

    result = await gateway_execute(
        "clean_cache",
        {"apply": True, "kinds": sorted(selected), "tenant": request.tenant or ""},
        tenant_id=user["tenant_id"],
        user_id=user["user_id"],
        idempotency_key=request.idempotency_key or None,
        approval_token=request.approval_token or None,
        runner=runner,
        victims=victims,
        verify=verify,
        rollback=rollback,
        impact="清理缓存："
        + "+".join(selected)
        + ("（租户 " + request.tenant + "）" if request.tenant else ""),
        reversible="先备份到 cache:trash:*，可恢复",
    )
    return result.as_dict()


@router.post("/trash/purge", summary="永久清空回收站（需审批）")
async def trash_purge(
    request: PurgeRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    items = trash.list_trash()
    targets = [
        item
        for item in items
        if request.older_than_days is None or item.age_days >= request.older_than_days
    ]
    if not request.apply:
        return {
            "ok": True,
            "dry_run": True,
            "would_delete": [item.trash_id for item in targets],
            "freed": human_size(sum(item.size for item in targets)),
            "warning": "purge 不可恢复",
        }

    async def runner(_args: dict[str, Any]) -> dict[str, Any]:
        dirs, freed = trash.purge(older_than_days=request.older_than_days)
        return {"purged": dirs, "freed": human_size(freed)}

    result = await gateway_execute(
        "purge_trash",
        {"older_than_days": request.older_than_days, "apply": True},
        tenant_id=user["tenant_id"],
        user_id=user["user_id"],
        idempotency_key=request.idempotency_key or None,
        approval_token=request.approval_token or None,
        runner=runner,
        impact=f"永久删除 {len(targets)} 个回收站条目",
        reversible="不可恢复",
    )
    return result.as_dict()


# ── 审批 ────────────────────────────────────────────────────
@router.get("/approvals/{approval_id}", summary="查看审批票据")
async def approval_detail(
    approval_id: str,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    ticket = await approval.get_ticket(approval_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="票据不存在或已过期")
    return ticket


@router.post("/approvals/{approval_id}/confirm", summary="确认审批（需管理员）")
async def approval_confirm(
    approval_id: str,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    # 审批是权限动作：只有管理员能批
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="只有管理员可以审批删除操作")

    # 双人复核怎么落地，取决于本租户有几个管理员：
    #   只有 1 个 → 允许自批（否则没人能批，删除功能等于被锁死），票据记 self_approved
    #   2 个及以上 → 必须换一个人批
    admin_count = await get_user_store().count_admins(user["tenant_id"])
    ok, message, ticket = await approval.confirm_ticket(
        approval_id,
        user["user_id"],
        allow_self=admin_count <= 1,
    )
    return {
        "ok": ok,
        "message": message,
        "ticket": ticket,
        "admin_count": admin_count,
        "self_approved": bool(ticket and ticket.get("self_approved")),
    }


@router.get("/approvals", summary="待审批列表（管理员）")
async def approvals_list(
    status: str = Query(
        default="pending", description="pending / confirmed / used / rejected / all"
    ),
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    """审批收件箱。管理员才能看，且只看本租户的申请。"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="只有管理员可以查看审批列表")
    tickets = await approval.list_tickets(None if status == "all" else status)
    items = [t for t in tickets if t.get("tenant_id") == user["tenant_id"]]
    return {"count": len(items), "items": items}


@router.post("/approvals/{approval_id}/reject", summary="拒绝审批（管理员）")
async def approval_reject(
    approval_id: str,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="只有管理员可以审批删除操作")
    ok, message, ticket = await approval.reject_ticket(approval_id, user["user_id"])
    return {"ok": ok, "message": message, "ticket": ticket}


# ── 审计 ────────────────────────────────────────────────────
@router.get("/audit", summary="运维审计日志")
async def audit_log(
    limit: int = Query(default=50, ge=1, le=200),
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    """按租户返回最近的破坏性操作记录（含被阻断、待审批、已执行、已回滚）。"""
    store = get_audit_store()
    return {
        "stats": await store.stats(user["tenant_id"]),
        "items": await store.list_entries(user["tenant_id"], limit),
    }


# ── 系统指标（原 analyst 的 A 路径，现归运维可观测性）────────
class MetricRequest(BaseModel):
    question: str = Field(..., min_length=1, description="自然语言指标问题")
    session_id: str = Field(default="metrics", description="会话 ID")


@router.post("/metrics/ask", summary="问系统指标（自然语言 → SQL）")
async def metrics_ask(
    request: MetricRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    from core.orchestrator import AgentRequest, AgentType, get_orchestrator

    agent_request = AgentRequest(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        session_id=request.session_id,
        agent_type=AgentType.METRICS,
        user_message=request.question,
        context={
            "department": user.get("department", "*"),
            "role": user.get("role", "user"),
        },
    )
    result = await get_orchestrator().handle(agent_request)
    structured = result.structured or {}
    return {
        "success": result.success,
        "answer": result.content,
        "mode": str(structured.get("answer_mode") or ""),
        "sql": str(structured.get("sql") or ""),
        "columns": [str(c) for c in (structured.get("columns") or [])],
        "rows": [r for r in (structured.get("rows") or []) if isinstance(r, dict)],
        "row_count": int(structured.get("row_count") or 0),
        "error_msg": result.error_msg,
    }


@router.get("/metrics/catalog", summary="可查的系统指标与视图")
async def metrics_catalog(
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "views": [
            {
                "name": name,
                "desc": str(meta["desc"]),
                "columns": [str(c) for c in meta["columns"]],  # type: ignore[union-attr]
            }
            for name, meta in VIEW_CATALOG.items()
        ],
        "metrics": [
            {"name": name, "note": meta["note"], "sql": meta["sql"]}
            for name, meta in METRIC_CATALOG.items()
        ],
    }
