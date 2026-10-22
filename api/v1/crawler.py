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

from agents.crawler.exporter import export_task
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
    export: bool = Field(default=False, description="抓完直接导出为文件")
    session_id: str = Field(default="crawl", description="会话 ID（仅用于日志串联）")


class CrawlResponse(BaseModel):
    success: bool
    task_id: int = 0
    summary: str = ""
    pages: int = 0
    records: int = 0
    strategy: str = ""
    stop_reason: str = ""
    # 抓完就能导出（前端据此给"导出为文件"按钮）
    can_export: bool = False
    export_dir: str = ""
    degraded: bool = False
    degrade_reason: str = ""
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

    # 抓完顺手导出（用户事先选了"导出"才做，默认不产生文件）
    export_dir = ""
    task_id = int(structured.get("task_id") or 0)
    if request.export and task_id:
        exported = await export_task(task_id, user["tenant_id"])
        export_dir = exported.directory if exported.ok else ""

    return CrawlResponse(
        success=result.success,
        task_id=task_id,
        summary=result.content,
        pages=int(structured.get("pages") or 0),
        records=int(structured.get("records") or 0),
        strategy=str(structured.get("strategy") or ""),
        stop_reason=str(structured.get("stop_reason") or ""),
        can_export=bool(structured.get("task_id")),
        export_dir=export_dir or str(structured.get("export_dir") or ""),
        degraded=result.fallback_used,
        degrade_reason=result.degrade_reason,
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


@router.get("/pages/{task_id}", summary="任务抓取过的页面")
async def pages(
    task_id: int,
    user: dict[str, str] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    task = await get_crawl_store().get_task(task_id, user["tenant_id"])
    if task is None:
        return []
    return await get_crawl_store().list_pages(task_id)


@router.get("/page/{page_id}", summary="单页内容（含原始 HTML）")
async def page_detail(
    page_id: int,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    page = await get_crawl_store().get_page(page_id, user["tenant_id"])
    if page is None:
        return {"found": False, "page_id": page_id}
    return {"found": True, **page}


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


class ExportRequest(BaseModel):
    """导出请求。"""

    out_dir: str | None = Field(default=None, description="导出根目录，默认 data/crawl")


class ExportResponse(BaseModel):
    success: bool
    task_id: int = 0
    directory: str = ""
    pages: int = 0
    records: int = 0
    bytes_written: int = 0
    files: list[str] = Field(default_factory=list)
    error_msg: str | None = None


@router.post("/export/{task_id}", response_model=ExportResponse, summary="导出为文件")
async def export(
    task_id: int,
    request: ExportRequest | None = None,
    user: dict[str, str] = Depends(get_current_user),
) -> ExportResponse:
    """把任务下的原始 HTML、正文、结构化记录写成文件。"""
    out_dir = request.out_dir if request else None
    result = await export_task(task_id, user["tenant_id"], out_dir=out_dir)
    return ExportResponse(
        success=result.ok,
        task_id=result.task_id,
        directory=result.directory,
        pages=result.pages,
        records=result.records,
        bytes_written=result.bytes_written,
        files=result.files,
        error_msg=None if result.ok else result.error,
    )
