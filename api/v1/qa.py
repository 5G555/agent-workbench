"""QA Agent 的 HTTP 接口。

对外只暴露简化入参（用户/会话/问题），AgentType 由服务端固定，不让调用方指定。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.logger import get_logger
from core.orchestrator import AgentRequest, AgentResponse, AgentType, get_orchestrator

logger = get_logger(__name__)

router = APIRouter()


class AskRequest(BaseModel):
    """问答请求。"""

    user_id: str = Field(..., description="使用者 ID")
    session_id: str = Field(..., description="会话 ID，多轮对话靠它隔离")
    message: str = Field(..., min_length=1, description="用户问题")
    tenant_id: str = Field(default="tenant_default", description="租户 ID")
    context: dict[str, Any] = Field(default_factory=dict, description="附加上下文")


class AskResponse(BaseModel):
    """问答响应。字段与 AgentResponse 对齐，前端不用关心内部结构。"""

    success: bool
    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    fallback_used: bool = False
    error_msg: str | None = None


@router.post("/ask", response_model=AskResponse, summary="提问")
async def ask(request: AskRequest) -> AskResponse:
    agent_request = AgentRequest(
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        agent_type=AgentType.QA,
        user_message=request.message,
        context=request.context,
    )
    result: AgentResponse = await get_orchestrator().handle(agent_request)

    sources = []
    if result.structured and isinstance(result.structured.get("sources"), list):
        sources = [str(s) for s in result.structured["sources"]]

    return AskResponse(
        success=result.success,
        answer=result.content,
        sources=sources,
        structured=result.structured,
        fallback_used=result.fallback_used,
        error_msg=result.error_msg,
    )
