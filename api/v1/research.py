"""联网研究 Agent 的 HTTP 接口。

与 qa 的接口平行，但走的是 research 图：
    规划子问题 → 通过 MCP 调 web_search → 带 [序号] 引用作答

    POST /ask          一次性返回
    POST /ask/stream   SSE 流式

会话历史与 qa 共用（同一张表、同一个会话键），前端可以在一个会话里
先问知识库、再问联网，上下文连贯。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.deps import get_current_user, rate_limit
from api.history_helpers import load_history, save_turn
from core.logger import get_logger
from core.orchestrator import AgentRequest, AgentResponse, AgentType, get_orchestrator

logger = get_logger(__name__)

router = APIRouter()


class ResearchRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="研究问题")
    context: dict[str, Any] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    success: bool
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    error_msg: str | None = None


def _to_request(request: ResearchRequest, user: dict[str, str]) -> AgentRequest:
    return AgentRequest(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        session_id=request.session_id,
        agent_type=AgentType.RESEARCH,
        user_message=request.message,
        context={
            **request.context,
            "department": user.get("department", "*"),
            "role": user.get("role", "user"),
        },
    )


@router.post("/ask", response_model=ResearchResponse, summary="联网研究（一次性返回）")
async def ask(
    request: ResearchRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> ResearchResponse:
    agent_request = _to_request(request, user)
    agent_request.context["history"] = await load_history(user["user_id"], request.session_id)

    result: AgentResponse = await get_orchestrator().handle(agent_request)
    structured = result.structured or {}

    if result.success and result.content:
        await save_turn(
            user["user_id"],
            request.session_id,
            request.message,
            result.content,
            str(structured.get("answer_mode", "web")),
        )

    return ResearchResponse(
        success=result.success,
        answer=result.content,
        citations=[str(c) for c in (structured.get("citations") or [])],
        subqueries=[str(s) for s in (structured.get("subqueries") or [])],
        fallback_used=result.fallback_used,
        error_msg=result.error_msg,
    )


@router.post("/ask/stream", summary="联网研究（SSE 流式）")
async def ask_stream(
    request: ResearchRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> StreamingResponse:
    agent_request = _to_request(request, user)
    agent_request.context["history"] = await load_history(user["user_id"], request.session_id)

    async def events() -> AsyncIterator[str]:
        final_answer = ""
        final_mode = "web"
        try:
            async for event in get_orchestrator().stream(agent_request):
                if event["type"] == "done":
                    final_answer = str(event.get("answer") or "")
                    final_mode = str(event.get("mode") or "web")
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            logger.error("research.stream_failed", error=str(exc))
            payload = json.dumps(
                {"type": "error", "message": str(exc)}, ensure_ascii=False, default=str
            )
            yield f"data: {payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            if final_answer:
                await save_turn(
                    user["user_id"], request.session_id, request.message, final_answer, final_mode
                )

    return StreamingResponse(events(), media_type="text/event-stream")
