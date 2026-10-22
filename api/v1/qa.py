"""QA Agent 的 HTTP 接口。

对外只暴露简化入参（用户/会话/问题），AgentType 由服务端固定，不让调用方指定。

三个问答入口：
    POST /ask          一次性返回
    POST /ask/stream   SSE 流式（先推节点进度，再推 token）
    GET  /history/{id} 查历史；DELETE 清历史
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.qa.history import get_history_store
from agents.qa.retrieval.vector_store import get_vector_store
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


def _to_agent_request(request: AskRequest) -> AgentRequest:
    return AgentRequest(
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        agent_type=AgentType.QA,
        user_message=request.message,
        context={**request.context, "source": request.source},
    )


async def _load_history(session_id: str, limit: int = 6) -> list[dict[str, str]]:
    try:
        return await get_history_store().recent(session_id, limit)
    except Exception as exc:  # 历史读不到不该让问答失败
        logger.warning("history.load_failed", session_id=session_id, error=str(exc))
        return []


async def _save_turn(session_id: str, user_id: str, question: str, answer: str, mode: str) -> None:
    try:
        store = get_history_store()
        await store.append(session_id, user_id, "user", question)
        await store.append(session_id, user_id, "assistant", answer, answer_mode=mode)
    except Exception as exc:
        logger.warning("history.save_failed", session_id=session_id, error=str(exc))


@router.post("/ask", response_model=AskResponse, summary="提问（一次性返回）")
async def ask(request: AskRequest) -> AskResponse:
    agent_request = _to_agent_request(request)
    agent_request.context["history"] = await _load_history(request.session_id)

    result: AgentResponse = await get_orchestrator().handle(agent_request)

    sources = []
    if result.structured and isinstance(result.structured.get("sources"), list):
        sources = [str(s) for s in result.structured["sources"]]

    if result.success and result.content:
        await _save_turn(
            request.session_id,
            request.user_id,
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
async def ask_stream(request: AskRequest) -> StreamingResponse:
    agent_request = _to_agent_request(request)
    agent_request.context["history"] = await _load_history(request.session_id)

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
                    request.session_id, request.user_id, request.message, final_answer, final_mode
                )

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/history/{session_id}", response_model=list[HistoryMessage], summary="查历史")
async def history(
    session_id: str, limit: int = Query(default=50, ge=1, le=200)
) -> list[HistoryMessage]:
    rows = await get_history_store().list_messages(session_id, limit)
    return [HistoryMessage(**row) for row in rows]


@router.delete("/history/{session_id}", summary="清空历史")
async def clear_history(session_id: str) -> dict[str, Any]:
    deleted = await get_history_store().clear(session_id)
    return {"deleted": deleted}


@router.get("/sources", response_model=list[str], summary="可用的检索来源")
async def list_sources() -> list[str]:
    """从 Milvus 里取当前所有文档来源，供前端做筛选下拉。"""
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
