"""爬虫 Agent 的 HTTP 接口。

    POST /run                建一个抓取任务（入口 URL + 任务目标）
    GET  /tasks/{id}         看任务状态与统计
    GET  /tasks              看最近任务
    GET  /records/{id}       看某任务抽到的结构化记录
    POST /process/{id}       跑处理 Agent，出结论

采集与处理是两步：先 /run 落库，再 /process 分析。
这样处理失败时可以只重跑处理，不用重新抓一遍。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from agents.crawler.processor import aggregate, process_task
from agents.crawler.store import get_crawl_store
from api.deps import get_current_user, rate_limit
from core.logger import get_logger
from core.orchestrator import AgentRequest, AgentResponse, AgentType, get_orchestrator

logger = get_logger(__name__)

router = APIRouter()


class CrawlRequest(BaseModel):
    """抓取任务。"""

    entry_url: str = Field(..., min_length=8, description="入口 URL")
    goal: str = Field(default="", description="自然语言任务目标，决定跟进哪些链接")
    max_pages: int = Field(default=0, ge=0, le=20, description="最多抓几页，0 = 用默认值")
    session_id: str = Field(default="crawl", description="会话 ID（仅用于日志串联）")


class CrawlResponse(BaseModel):
    success: bool
    task_id: int = 0
    summary: str = ""
    pages: int = 0
    records: int = 0
    strategy: str = ""
    stop_reason: str = ""
    error_msg: str | None = None


class ProcessRequest(BaseModel):
    """处理任务。"""

    rules: str = Field(default="", description="处理规则，例如'按类别聚合，标出异常'")
    required_fields: list[str] = Field(
        default_factory=list, description="必需字段，用于 schema 校验"
    )
    aggregate_field: str = Field(default="", description="要按哪个字段做频次聚合")
    max_records: int = Field(default=200, ge=1, le=1000)


class ProcessResponse(BaseModel):
    task_id: int
    total: int = 0
    kept: int = 0
    dropped: int = 0
    fields: list[str] = Field(default_factory=list)
    summary: str = ""
    issues: list[str] = Field(default_factory=list)
    top_values: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/run", response_model=CrawlResponse, summary="抓取网页并落库")
async def run(
    request: CrawlRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> CrawlResponse:
    agent_request = AgentRequest(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        session_id=request.session_id,
        agent_type=AgentType.CRAWLER,
        user_message=request.goal or request.entry_url,
        context={
            "entry_url": request.entry_url,
            "goal": request.goal,
            "max_pages": request.max_pages or None,
            "department": user.get("department", "*"),
            "role": user.get("role", "user"),
        },
    )
    result: AgentResponse = await get_orchestrator().handle(agent_request)
    structured = result.structured or {}

    return CrawlResponse(
        success=result.success,
        task_id=int(structured.get("task_id") or 0),
        summary=result.content,
        pages=int(structured.get("pages") or 0),
        records=int(structured.get("records") or 0),
        strategy=str(structured.get("strategy") or ""),
        stop_reason=str(structured.get("stop_reason") or ""),
        error_msg=result.error_msg,
    )


@router.get("/tasks", summary="最近抓取任务")
async def recent_tasks(
    limit: int = Query(default=10, ge=1, le=50),
    user: dict[str, str] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    return await get_crawl_store().recent_tasks(user["tenant_id"], limit)


@router.get("/tasks/{task_id}", summary="任务详情")
async def task_detail(
    task_id: int,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    task = await get_crawl_store().get_task(task_id, user["tenant_id"])
    if task is None:
        return {"found": False, "task_id": task_id}
    return {
        "found": True,
        "task_id": task.id,
        "status": task.status,
        "goal": task.goal,
        "entry_url": task.entry_url,
        "pages": task.pages_fetched,
        "records": task.records_extracted,
        "strategy": task.strategy,
        "stop_reason": task.stop_reason,
        "error": task.error,
    }


@router.get("/records/{task_id}", summary="任务的结构化记录")
async def records(
    task_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    user: dict[str, str] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    task = await get_crawl_store().get_task(task_id, user["tenant_id"])
    if task is None:  # 不能查别的租户的任务
        return []
    return await get_crawl_store().list_records(task_id, limit)


@router.post("/process/{task_id}", response_model=ProcessResponse, summary="处理 Agent：分析记录")
async def process(
    task_id: int,
    request: ProcessRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> ProcessResponse:
    task = await get_crawl_store().get_task(task_id, user["tenant_id"])
    if task is None:
        return ProcessResponse(task_id=task_id, summary="任务不存在或不属于当前租户")

    result = await process_task(
        task_id,
        rules=request.rules,
        required_fields=request.required_fields,
        max_records=request.max_records,
    )

    top_values: list[dict[str, Any]] = []
    if request.aggregate_field:
        raw = await get_crawl_store().list_records(task_id, limit=request.max_records)
        cleaned, _ = __import__(
            "agents.crawler.processor", fromlist=["clean_records"]
        ).clean_records(raw)
        top_values = [
            {"value": value, "count": count}
            for value, count in aggregate(cleaned, request.aggregate_field)
        ]

    return ProcessResponse(
        task_id=task_id,
        total=result.total,
        kept=result.kept,
        dropped=result.dropped,
        fields=result.fields,
        summary=result.summary,
        issues=result.issues,
        top_values=top_values,
    )
