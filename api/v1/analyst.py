"""数据分析接口。

定位：给"分析分区"面板用。和 /api/v1/ask 的区别：
    ask 只返回一段答案；这里要返回「SQL + 结果表 + 路径」，
    因为分析结果需要能核对——用户看到 SQL 才知道数字是怎么来的。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.analyst.schema import METRIC_CATALOG, VIEW_CATALOG
from api.deps import get_current_user, rate_limit
from core.logger import get_logger
from core.orchestrator import AgentRequest, AgentResponse, AgentType, get_orchestrator

logger = get_logger(__name__)

router = APIRouter()


class AnalystRequest(BaseModel):
    question: str = Field(..., min_length=1, description="自然语言分析问题")
    session_id: str = Field(default="analyst", description="会话 ID（日志串联用）")
    task_id: int = Field(default=0, description="只看某个采集任务（路径 B 用）")


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
    fields: list[str] = Field(default_factory=list)
    # 路径 B：pandas 代码分析
    code: str = ""
    stdout: str = ""
    figures: list[str] = Field(default_factory=list)  # base64 PNG
    prep: dict[str, Any] = Field(default_factory=dict)
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
        fields=[str(f) for f in (structured.get("fields") or [])],
        code=str(structured.get("code") or ""),
        stdout=str(structured.get("stdout") or ""),
        figures=[str(f) for f in (structured.get("figures") or [])],
        prep=structured.get("prep") or {},
        error_msg=result.error_msg,
    )


@router.get("/catalog", summary="可分析的数据与指标")
async def catalog(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    """面板里展示"能问什么"：视图清单 + 常用指标（带示例 SQL）。"""
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
