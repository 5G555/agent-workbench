"""统一问答入口：自动路由 + 失败升级。

    POST /api/v1/ask          自动：先路由到 qa / research，qa 查不到时升级
    POST /api/v1/ask/stream   同上，SSE 流式

也支持 mode 参数强制指定：
    mode="qa"       只走知识库
    mode="research" 只走联网
    mode="auto"     默认，交给 supervisor

和 /v1/qa、/v1/research 两个专用接口并存：
专用接口给"我要确定走哪条路"的场景，统一接口给前端默认行为。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.deps import get_current_user, rate_limit
from api.history_helpers import load_history, save_turn
from core.logger import get_logger
from core.orchestrator import (
    AgentRequest,
    AgentResponse,
    AgentType,
    ExecutionMode,  # noqa: F401 - 保留给后续 pipeline 模式使用
    get_orchestrator,
)
from core.query_rewrite import rewrite_question

logger = get_logger(__name__)

router = APIRouter()


class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    mode: Literal["auto", "qa", "research", "crawler", "analyst", "help", "direct"] = Field(
        default="auto", description="路由模式；crawler 表示确认后直接抓取"
    )
    source: str | None = Field(default=None, description="限定检索来源（仅 qa 有效）")
    depth: Literal["fast", "deep"] = Field(default="fast", description="联网检索深度")
    # 用户确认采集后传入（mode=crawler 时生效）
    entry_url: str | None = Field(default=None, description="采集入口 URL")
    goal: str | None = Field(default=None, description="采集目标")
    context: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    success: bool
    answer: str = ""
    agent: str = ""  # 实际回答的 Agent（可能因升级而变）
    answer_mode: str = ""
    sources: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False
    route: dict[str, Any] = Field(default_factory=dict)
    # 联网检索的结果状态：ok / no_result / rate_limited / no_key / failed（空串 = 没走联网）
    web_status: str = ""
    # 联网时实际使用的时间窗口（空 = 不限）
    time_range: str = ""
    # 需要用户确认才能继续时：answer_mode=confirm，confirm 里是待执行的动作
    # 采集类请求会返回 task_id，前端据此给"导出为文件"按钮
    task_id: int = 0
    need_confirm: bool = False
    confirm: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    degrade_reason: str = ""
    error_msg: str | None = None


def _build_request(request: AskRequest, user: dict[str, str]) -> AgentRequest:
    agent_type = {
        "qa": AgentType.QA,
        "research": AgentType.RESEARCH,
        "auto": AgentType.QA,  # auto 时由 handle_auto 覆盖
    }[request.mode]
    return AgentRequest(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        session_id=request.session_id,
        agent_type=agent_type,
        user_message=request.message,
        context={
            **request.context,
            "source": request.source,
            "department": user.get("department", "*"),
            "role": user.get("role", "user"),
            "depth": request.depth,
        },
    )


@router.post("/ask", response_model=AskResponse, summary="提问（自动路由 + 失败升级）")
async def ask(
    request: AskRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> AskResponse:
    agent_request = _build_request(request, user)
    history = await load_history(user["user_id"], request.session_id)
    # 追问改写："那包年呢" → "你们包年怎么收费"，否则检索会落空
    rewritten = await rewrite_question(request.message, history)
    if rewritten != request.message:
        agent_request.user_message = rewritten
    agent_request.context["history"] = history
    agent_request.context["original_message"] = request.message

    orchestrator = get_orchestrator()
    result: AgentResponse = (
        await orchestrator.handle_auto(agent_request)
        if request.mode == "auto"
        else await orchestrator.handle(agent_request)
    )

    structured = result.structured or {}
    mode = str(structured.get("answer_mode", ""))
    confirm = structured.get("confirm") or {}
    need_confirm = mode == "confirm"

    # 「要不要确认」是过程提示，不写进会话历史（否则多轮上下文里会混入一句反问）
    if result.success and result.content and not need_confirm:
        await save_turn(user["user_id"], request.session_id, request.message, result.content, mode)

    return AskResponse(
        success=result.success,
        answer=result.content,
        agent=result.agent_type.value,
        answer_mode=mode,
        sources=[str(s) for s in (structured.get("sources") or [])],
        citations=[str(c) for c in (structured.get("citations") or [])],
        escalated="escalated_from" in result.metadata,
        route=result.metadata.get("route") or {},
        web_status=str(result.metadata.get("web_status") or ""),
        time_range=str(result.metadata.get("time_range") or ""),
        task_id=int(structured.get("task_id") or 0),
        need_confirm=need_confirm,
        confirm=confirm if isinstance(confirm, dict) else {},
        fallback_used=result.fallback_used,
        degrade_reason=result.degrade_reason,
        error_msg=result.error_msg,
    )


@router.post("/ask/stream", summary="提问（自动路由 + 失败升级，SSE）")
async def ask_stream(
    request: AskRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> StreamingResponse:
    agent_request = _build_request(request, user)
    history = await load_history(user["user_id"], request.session_id)
    # 追问改写："那包年呢" → "你们包年怎么收费"，否则检索会落空
    rewritten = await rewrite_question(request.message, history)
    if rewritten != request.message:
        agent_request.user_message = rewritten
    agent_request.context["history"] = history
    agent_request.context["original_message"] = request.message

    async def events() -> AsyncIterator[str]:
        answer = ""
        mode = ""
        try:
            source = (
                get_orchestrator().stream_auto(agent_request)
                if request.mode == "auto"
                else get_orchestrator().stream(agent_request)
            )
            async for event in source:
                if event["type"] == "final":
                    answer = str(event.get("answer") or "")
                    mode = str(event.get("mode") or "")
                elif event["type"] == "done":
                    # 升级时会跑两个 Agent，最终以 research 的 done 为准
                    answer = str(event.get("answer") or "") or answer
                    mode = str(event.get("mode") or "") or mode
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            logger.error("ask.stream_failed", error=str(exc))
            payload = json.dumps(
                {"type": "error", "message": str(exc)}, ensure_ascii=False, default=str
            )
            yield f"data: {payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            if answer:
                await save_turn(user["user_id"], request.session_id, request.message, answer, mode)

    return StreamingResponse(events(), media_type="text/event-stream")


class RouteCorrection(BaseModel):
    """用户认为路由错了：记下来用于调阈值/补规则。"""

    session_id: str = Field(default="", description="会话 ID")
    question: str = Field(..., min_length=1, description="当时的问题")
    routed_agent: str = Field(..., description="实际路由到的 Agent")
    corrected_agent: str = Field(..., description="用户认为应该去的 Agent")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="当时的路由理由")


@router.post("/correct", summary="记录一次路由纠正")
async def record_correction(
    body: RouteCorrection,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    from agents.supervisor.feedback import get_feedback_store

    store = get_feedback_store()
    await store.ensure_table()
    row_id = await store.record(
        tenant_id=user["tenant_id"],
        user_id=user["user_id"],
        question=body.question,
        routed_agent=body.routed_agent,
        corrected_agent=body.corrected_agent,
        session_id=body.session_id,
        confidence=body.confidence,
        reason=body.reason,
    )
    return {"ok": True, "id": row_id}


@router.get("/corrections", summary="路由纠正统计")
async def correction_stats(
    user: dict[str, str] = Depends(get_current_user),
    limit: int = 100,
) -> dict[str, Any]:
    from agents.supervisor.feedback import get_feedback_store

    store = get_feedback_store()
    await store.ensure_table()
    return await store.summary(user["tenant_id"], limit)
