"""数据分析接口。

定位：给"分析分区"面板用。和 /api/v1/ask 的区别：
    ask 只返回一段答案；这里要返回「SQL + 结果表 + 路径」，
    因为分析结果需要能核对——用户看到 SQL 才知道数字是怎么来的。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_current_user, rate_limit
from core.logger import get_logger
from core.orchestrator import AgentRequest, AgentResponse, AgentType, get_orchestrator

logger = get_logger(__name__)

router = APIRouter()


class AnalystRequest(BaseModel):
    question: str = Field(..., min_length=1, description="自然语言分析问题")
    session_id: str = Field(default="analyst", description="会话 ID（日志串联用）")
    task_id: int = Field(default=0, description="只看某个采集任务")
    task_ids: list[int] = Field(default_factory=list, description="多任务对比（最多 5 个）")


class AnalystResponse(BaseModel):
    success: bool
    answer: str = ""
    mode: str = ""
    path: str = ""
    sql: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    task_id: int = 0
    tasks: list[int] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    # 路径 B：pandas 代码分析
    code: str = ""
    stdout: str = ""
    figures: list[str] = Field(default_factory=list)  # base64 PNG
    prep: dict[str, Any] = Field(default_factory=dict)
    # 模型看到的表结构概览（排查"它为什么这么写"用）
    overview: str = ""
    chart_type: str = ""
    cached: bool = False  # 是否命中结果缓存
    error_msg: str | None = None


@router.post("/ask", response_model=AnalystResponse, summary="数据分析")
async def ask(
    request: AnalystRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> AnalystResponse:
    agent_request = AgentRequest(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        session_id=request.session_id,
        agent_type=AgentType.ANALYST,
        user_message=request.question,
        context={
            "department": user.get("department", "*"),
            "role": user.get("role", "user"),
            "task_id": request.task_id or None,
            "task_ids": request.task_ids or [],
        },
    )
    result: AgentResponse = await get_orchestrator().handle(agent_request)
    structured = result.structured or {}

    return AnalystResponse(
        success=result.success,
        answer=result.content,
        mode=str(structured.get("answer_mode") or ""),
        path=str(structured.get("path") or ""),
        sql=str(structured.get("sql") or ""),
        columns=[str(c) for c in (structured.get("columns") or [])],
        rows=[r for r in (structured.get("rows") or []) if isinstance(r, dict)],
        row_count=int(structured.get("row_count") or 0),
        task_id=int(structured.get("task_id") or 0),
        tasks=[int(t) for t in (structured.get("tasks") or [])],
        fields=[str(f) for f in (structured.get("fields") or [])],
        code=str(structured.get("code") or ""),
        stdout=str(structured.get("stdout") or ""),
        figures=[str(f) for f in (structured.get("figures") or [])],
        prep=structured.get("prep") or {},
        overview=str(structured.get("overview") or ""),
        chart_type=str(structured.get("chart_type") or ""),
        cached=bool(structured.get("cached")),
        error_msg=result.error_msg,
    )


# ── 手动分析：用户指定任务/字段/聚合/图表 ──────────────────
class ManualRequest(BaseModel):
    """手动分析的参数（不含自然语言）。"""

    task_ids: list[int] = Field(..., min_length=1, description="分析哪些采集任务")
    metrics: list[str] = Field(..., min_length=1, description="要统计的数值列")
    agg: str = Field(default="mean", description="mean/sum/count/min/max/median/std")
    group_by: str = Field(default="", description="按哪列分组，空表示整体")
    x_field: str = Field(default="", description="折线/散点的横轴")
    chart_type: str = Field(default="bar", description="none/bar/line/scatter/pie/hist/candlestick")
    title: str = Field(default="", description="图表标题")


async def _load_records(task_ids: list[int], tenant_id: str) -> list[dict]:
    """按租户取采集记录（与 analyst 图取数口径一致：带 task_id 列）。"""
    from core.db import lease_connection

    conn = await lease_connection()
    try:
        rows = await conn.fetch(
            "SELECT r.task_id, r.data FROM crawl_records r "
            "JOIN crawl_tasks t ON t.id = r.task_id "
            "WHERE t.tenant_id = $1 AND r.task_id = ANY($2::bigint[]) "
            "ORDER BY r.task_id, r.id LIMIT 2000",
            tenant_id,
            task_ids,
        )
    finally:
        await conn.close()
    # data 列是 JSON 文本，需要先解析；解析失败的记录跳过
    items: list[dict] = []
    for row in rows:
        raw = row["data"]
        try:
            payload = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            items.append({**payload, "task_id": int(row["task_id"])})
    return items


@router.get("/fields", summary="某个任务的可用字段（按用途分组）")
async def fields(
    task_id: int,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    from agents.analyst.frame import build_frame
    from agents.analyst.manual import column_roles

    records = await _load_records([task_id], user["tenant_id"])
    frame, report = build_frame([{"data": item} for item in records])
    return {
        "task_id": task_id,
        "rows": report.rows,
        "roles": column_roles(frame),
        "columns": [str(c) for c in frame.columns],
    }


@router.post("/manual", summary="手动分析（代码按模板生成，口径确定）")
async def manual(
    request: ManualRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> dict[str, Any]:
    from agents.analyst.code_runner import run_code
    from agents.analyst.frame import build_frame, frame_overview
    from agents.analyst.manual import ManualSpec, build_code

    records = await _load_records(request.task_ids, user["tenant_id"])
    if not records:
        return {"success": False, "answer": "这些任务没有可用记录。", "mode": "manual"}

    frame, report = build_frame([{"data": item} for item in records])
    columns = [str(c) for c in frame.columns]

    spec = ManualSpec(
        metrics=request.metrics,
        agg=request.agg,
        group_by=request.group_by,
        x_field=request.x_field,
        chart_type=request.chart_type,
        title=request.title,
    )
    try:
        code = build_code(spec, columns)
    except ValueError as exc:
        return {"success": False, "answer": str(exc), "mode": "manual"}

    result = await run_code(code=code, frame=frame)
    return {
        "success": result.ok,
        "mode": "manual",
        "answer": (
            f"按【{request.agg}】统计 {', '.join(request.metrics)}"
            + (f"，按 {request.group_by} 分组" if request.group_by else "")
            + ("，已完成。" if result.ok else f"，执行失败：{result.error}")
        ),
        "code": code,
        "stdout": result.stdout,
        "figures": result.figures,
        "rows": result.result,
        "prep": report.as_dict(),
        "overview": frame_overview(frame),
        "chart_type": request.chart_type,
        "error_msg": result.error or None,
    }
