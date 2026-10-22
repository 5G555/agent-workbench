"""运维接口：缓存巡检/清理、导出目录巡检/清理。

定位：给界面用的"确定按钮"，不是给模型用的。
    Agent 走 MCP 工具（mcp_servers/ops_server.py），界面走这里的 REST，
    两边最终调用同一份实现（core/cache_maintenance.py、agents/crawler/cleaner.py），
    所以行为和报告口径一定一致。

共同约定：
    所有清理接口默认 dry_run（apply=false），只报告会删什么；
    只有 apply=true 才真删。前端必须显式传，避免误点。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.crawler.cleaner import (
    apply_cleanup,
    clean_task,
    human_size,
    plan_cleanup,
    scan_exports,
)
from api.deps import get_current_user, rate_limit
from core.cache_maintenance import clean as clean_cache_keys
from core.cache_maintenance import overview as cache_overview
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class CacheCleanRequest(BaseModel):
    apply: bool = Field(default=False, description="是否真正删除，默认只统计")


class ExportCleanRequest(BaseModel):
    keep: int | None = Field(default=None, ge=0, le=100, description="保留最近 N 个任务的导出")
    older_than_days: float | None = Field(default=None, ge=0, description="删除 N 天前的导出")
    task_id: int | None = Field(default=None, description="只删指定任务")
    apply: bool = Field(default=False, description="是否真正删除")


@router.get("/cache", summary="缓存概览")
async def cache_stats(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    data = await cache_overview()
    return {
        "ok": data["version"] > 0,
        "version": data["version"],
        "total_keys": data["total"],
        "current_keys": data["current"],
        "stale_keys": data["stale"],
    }


@router.post("/cache/clean", summary="清理旧版本缓存键")
async def cache_clean(
    request: CacheCleanRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    report = await clean_cache_keys(apply=request.apply)
    return {
        "ok": report.version > 0,
        "dry_run": not request.apply,
        "version": report.version,
        "stale": report.stale_total,
        "kept": report.kept_total,
        "deleted": report.deleted,
        "by_prefix": [
            {"prefix": scan.label, "stale": len(scan.stale), "kept": scan.kept}
            for scan in report.scans
        ],
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


@router.post("/exports/clean", summary="清理导出目录")
async def exports_clean(
    request: ExportCleanRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    if request.task_id is not None:
        if not request.apply:
            return {"ok": True, "dry_run": True, "would_delete": [request.task_id]}
        done, freed = clean_task(request.task_id)
        return {
            "ok": done,
            "dry_run": False,
            "deleted": [request.task_id] if done else [],
            "freed": human_size(freed),
        }

    targets = plan_cleanup(
        scan_exports(), keep=request.keep, older_than_days=request.older_than_days
    )
    freed = sum(item.bytes for item in targets)
    if not request.apply:
        return {
            "ok": True,
            "dry_run": True,
            "would_delete": [item.task_id for item in targets],
            "freed": human_size(freed),
        }

    dirs, freed_real = apply_cleanup(targets)
    logger.info("ops.exports_cleaned", dirs=dirs, freed=freed_real)
    return {
        "ok": True,
        "dry_run": False,
        "deleted": [item.task_id for item in targets],
        "dirs": dirs,
        "freed": human_size(freed_real),
    }
