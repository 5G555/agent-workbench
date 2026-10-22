"""QA Agent 的 HTTP 接口。

对外只暴露简化入参，AgentType 由服务端固定。
所有接口都要求登录（Authorization: Bearer <token>），并做每分钟限流。

    POST /ask              一次性返回
    POST /ask/stream       SSE 流式
    GET  /history/{sid}    查历史；DELETE 清历史
    GET  /sources          可筛选的文档来源

注意：user_id / tenant_id 一律从 Token 取，不信请求体——否则调用方可以伪造身份。
会话在存储时统一加 "用户ID:" 前缀，避免拿到别人的会话 ID 就能读别人的历史。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.qa.history import get_history_store
from agents.qa.retrieval.vector_store import get_vector_store
from api.deps import get_current_user, rate_limit
from core.logger import get_logger
from core.orchestrator import AgentRequest, AgentResponse, AgentType, get_orchestrator

logger = get_logger(__name__)

router = APIRouter()


class AskRequest(BaseModel):
    """问答请求。身份信息来自 Token，不在这里传。"""

    session_id: str = Field(..., min_length=1, description="会话 ID，多轮对话靠它隔离")
    message: str = Field(..., min_length=1, description="用户问题")
    source: str | None = Field(default=None, description="限定检索来源（文件名）")
    context: dict[str, Any] = Field(default_factory=dict, description="附加上下文")


class AskResponse(BaseModel):
    """问答响应。"""

    success: bool
    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    fallback_used: bool = False
    error_msg: str | None = None


class HistoryMessage(BaseModel):
    role: str
    content: str
    answer_mode: str = ""
    created_at: str = ""


def _session_key(user: dict[str, str], session_id: str) -> str:
    """会话存库时带上用户前缀，防止串号。"""
    return f"{user['user_id']}:{session_id}"


def _to_agent_request(request: AskRequest, user: dict[str, str]) -> AgentRequest:
    return AgentRequest(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        session_id=request.session_id,
        agent_type=AgentType.QA,
        user_message=request.message,
        context={**request.context, "source": request.source},
    )


async def _load_history(
    user: dict[str, str], session_id: str, limit: int = 6
) -> list[dict[str, str]]:
    try:
        return await get_history_store().recent(_session_key(user, session_id), limit)
    except Exception as exc:  # 历史读不到不该让问答失败
        logger.warning("history.load_failed", session_id=session_id, error=str(exc))
        return []


async def _save_turn(
    user: dict[str, str], session_id: str, question: str, answer: str, mode: str
) -> None:
    try:
        store = get_history_store()
        key = _session_key(user, session_id)
        await store.append(key, user["user_id"], "user", question)
        await store.append(key, user["user_id"], "assistant", answer, answer_mode=mode)
    except Exception as exc:
        logger.warning("history.save_failed", session_id=session_id, error=str(exc))


@router.post("/ask", response_model=AskResponse, summary="提问（一次性返回）")
async def ask(
    request: AskRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> AskResponse:
    agent_request = _to_agent_request(request, user)
    agent_request.context["history"] = await _load_history(user, request.session_id)

    result: AgentResponse = await get_orchestrator().handle(agent_request)

    sources = []
    if result.structured and isinstance(result.structured.get("sources"), list):
        sources = [str(s) for s in result.structured["sources"]]

    if result.success and result.content:
        await _save_turn(
            user,
            request.session_id,
            request.message,
            result.content,
            str((result.structured or {}).get("answer_mode", "")),
        )

    return AskResponse(
        success=result.success,
        answer=result.content,
        sources=sources,
        structured=result.structured,
        fallback_used=result.fallback_used,
        error_msg=result.error_msg,
    )


@router.post("/ask/stream", summary="提问（SSE 流式）")
async def ask_stream(
    request: AskRequest,
    user: dict[str, str] = Depends(get_current_user),
    _: None = Depends(rate_limit),
) -> StreamingResponse:
    agent_request = _to_agent_request(request, user)
    agent_request.context["history"] = await _load_history(user, request.session_id)

    async def event_source() -> AsyncIterator[str]:
        final_answer = ""
        final_mode = ""
        try:
            async for event in get_orchestrator().stream(agent_request):
                if event["type"] == "done":
                    final_answer = str(event.get("answer") or "")
                    final_mode = str(event.get("mode") or "")
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            logger.error("qa.stream_failed", error=str(exc))
            payload = json.dumps(
                {"type": "error", "message": str(exc)}, ensure_ascii=False, default=str
            )
            yield f"data: {payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            if final_answer:
                await _save_turn(
                    user, request.session_id, request.message, final_answer, final_mode
                )

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/history/{session_id}", response_model=list[HistoryMessage], summary="查历史")
async def history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    user: dict[str, str] = Depends(get_current_user),
) -> list[HistoryMessage]:
    rows = await get_history_store().list_messages(_session_key(user, session_id), limit)
    return [HistoryMessage(**row) for row in rows]


@router.delete("/history/{session_id}", summary="清空历史")
async def clear_history(
    session_id: str,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    deleted = await get_history_store().clear(_session_key(user, session_id))
    return {"deleted": deleted}


@router.get("/sources", response_model=list[str], summary="可用的检索来源")
async def list_sources(user: dict[str, str] = Depends(get_current_user)) -> list[str]:
    try:
        store = get_vector_store()
        rows = store.client.query(
            collection_name=store.collection_name,
            filter='source != ""',
            output_fields=["source"],
            limit=1000,
        )
        return sorted({str(r["source"]) for r in rows if r.get("source")})
    except Exception as exc:
        logger.warning("qa.sources_failed", error=str(exc))
        return []
