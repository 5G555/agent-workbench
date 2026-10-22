"""QA Agent 的节点。
分类 → （通用知识直接答 | 专业咨询走检索三件套：选策略 → 检索 → 生成）
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.qa.cache import get_answer_cache
from agents.qa.classifier import get_classifier
from agents.qa.faq import get_faq_store
from agents.qa.prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER,
    DIRECT_SYSTEM,
    RAG_SYSTEM,
    RAG_USER,
    STRATEGY_SYSTEM,
    STRATEGY_USER,
)
from agents.qa.retrieval.strategies import ALL_STRATEGIES, DIRECT, run_strategy
from agents.qa.retrieval.vector_store import get_vector_store
from agents.qa.state import QAState
from config import get_settings
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)


class QueryCategory(BaseModel):
    """分类结果的结构化输出。"""

    category: Literal["GENERAL", "PROFESSIONAL"] = Field(description="问题类别")
    reason: str = Field(default="", description="一句话理由")


# ── 工具函数 ────────────────────────────────────────────────
def _extract_query(state: QAState) -> str:
    """优先用显式传入的 original_query，否则取最后一条用户消息。"""
    query = state.get("original_query") or ""
    if query:
        return query
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _history_text(messages: list[BaseMessage] | None, limit: int = 6) -> str:
    """把最近几轮对话拼成文本，给生成节点当上下文。"""
    if not messages:
        return "无历史对话"
    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"助手：{msg.content}")
    return "\n".join(lines) if lines else "无历史对话"


def _with_history(state: QAState, query: str) -> str:
    history = _history_text(state.get("messages"))
    if history == "无历史对话":
        return query
    return f"【历史对话】\n{history}\n\n【用户问题】\n{query}"


# 知识库主题列表的进程内缓存：(时间戳, 主题文本)
_topics_cache: tuple[float, str] | None = None
_TOPICS_TTL = 300.0


async def kb_topics() -> str:
    """从 Milvus 取当前所有文档来源，作为分类提示词的"知识库范围"。

    写死主题清单会随知识库增长而过时（新增值班手册后，"交接班要求"就会被误判成
    通用知识），所以每次从库里读，5 分钟缓存一次。
    """
    global _topics_cache
    now = time.monotonic()
    if _topics_cache is not None and now - _topics_cache[0] < _TOPICS_TTL:
        return _topics_cache[1]

    try:
        store = get_vector_store()

        def _query() -> list[str]:
            rows = store.client.query(
                collection_name=store.collection_name,
                filter='source != ""',
                output_fields=["source"],
                limit=1000,
            )
            return sorted({str(r["source"]) for r in rows if r.get("source")})

        sources = await asyncio.to_thread(_query)
    except Exception as exc:
        logger.warning("qa.kb_topics_failed", error=str(exc))
        sources = []

    text = "、".join(sources) if sources else "（暂时没有可检索的资料）"
    _topics_cache = (now, text)
    return text


# ── 节点 ────────────────────────────────────────────────────
def _is_first_turn(state: QAState) -> bool:
    """本轮是不是会话的第一问（没有任何历史）。

    只有首轮才碰缓存，因为：
      1. generate_node 会把历史消息拼进 prompt，答案因此依赖上下文；
      2. 而缓存键只有 query，不含会话信息。
    若非首轮也走缓存，「那 P2 呢」这类省略式提问会串到别的会话的答案上。
    """
    humans = [m for m in (state.get("messages") or []) if isinstance(m, HumanMessage)]
    return len(humans) <= 1


async def cache_lookup_node(state: QAState) -> dict[str, Any]:
    """⓪ 答案缓存：命中就把整条链路短路掉，连分类都不做。"""
    query = _extract_query(state)
    if not query:
        return {"cache_hit": False}

    if not _is_first_turn(state):  # 多轮上下文里的问题不做缓存读取
        return {"cache_hit": False, "original_query": query}

    hit = await get_answer_cache().get(query)
    if hit is None:
        return {"cache_hit": False, "original_query": query}

    answer = str(hit.get("answer") or "")
    mode = str(hit.get("mode") or "cache")
    sources = hit.get("sources") or []
    return {
        "cache_hit": True,
        "original_query": query,
        "answer": answer,
        "answer_mode": mode,
        "sources": sources,
        "messages": [AIMessage(content=answer)],
        "structured_output": {"answer_mode": mode, "sources": sources, "cached": True},
    }


async def cache_write_node(state: QAState) -> dict[str, Any]:
    """⑤ 回答完成后写缓存。缓存命中的轮次、以及非首轮的问答都不写。"""
    query = state.get("original_query") or ""
    answer = state.get("answer") or ""
    if not query or not answer or state.get("cache_hit"):
        return {}
    if not _is_first_turn(state):  # 答案依赖历史，缓存了会串
        logger.info("cache.skip_non_first_turn", query=query)
        return {}
    await get_answer_cache().set(
        query,
        answer,
        mode=state.get("answer_mode") or "rag",
        sources=state.get("sources") or [],
    )
    return {}


async def classify_node(state: QAState) -> dict[str, Any]:
    """① 判断问题要不要查知识库。

    优先用本地 BERT（快、免费）；模型不可用时退回 LLM 分类；两者都失败则保守走专业咨询。
    """
    query = _extract_query(state)
    if not query:
        return {"query_type": "GENERAL", "original_query": ""}

    # supervisor 已经判过"这是内部问题"时，不再让本节点的分类器推翻它：
    # 两个模型各判一次，冲突时以路由结论为准，否则会出现"路由到 qa 却被当闲聊"。
    if state.get("force_professional"):
        logger.info("qa.classified", query=query, category="PROFESSIONAL", via="supervisor")
        return {"query_type": "PROFESSIONAL", "original_query": query}

    category = ""
    classifier = get_classifier()
    if classifier is not None:
        try:
            category, confidence = await classifier.apredict(query)
            logger.info(
                "qa.classified",
                query=query,
                category=category,
                via="bert",
                confidence=round(confidence, 3),
            )
        except Exception as exc:
            logger.warning("qa.classifier_infer_failed", error=str(exc))

    if not category:
        try:
            llm = LLMFactory.get_structured_llm("qa", QueryCategory)
            result = await llm.ainvoke(
                [
                    SystemMessage(content=CLASSIFY_SYSTEM),
                    HumanMessage(
                        content=CLASSIFY_USER.format(query=query, topics=await kb_topics())
                    ),
                ]
            )
            category = result.category if isinstance(result, QueryCategory) else "PROFESSIONAL"
            logger.info("qa.classified", query=query, category=category, via="llm")
        except Exception as exc:
            logger.warning("qa.classify_failed", error=str(exc))
            category = "PROFESSIONAL"

    return {"query_type": category, "original_query": query}


async def strategy_node(state: QAState) -> dict[str, Any]:
    """② 为专业咨询挑一种检索策略。失败或返回未知值时退回直接检索。"""
    query = state.get("original_query", "")
    llm = LLMFactory.get_llm("qa", temperature=0)
    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=STRATEGY_SYSTEM),
                HumanMessage(content=STRATEGY_USER.format(query=query)),
            ]
        )
        raw = str(resp.content).strip()
        strategy = next((s for s in ALL_STRATEGIES if s in raw), DIRECT)
    except Exception as exc:
        logger.warning("qa.strategy_failed", error=str(exc))
        strategy = DIRECT

    logger.info("qa.strategy_selected", query=query, strategy=strategy)
    return {"retrieval_strategy": strategy}


async def faq_node(state: QAState) -> dict[str, Any]:
    """② FAQ 精确命中：命中就直接出答案，未命中交给后面的向量检索。

    FAQ 走的是"人工写好的问答表 + BM25"，比向量检索快且不会胡编；
    但只覆盖已被整理进表里的问题，所以分数不达标就必须让位给 RAG。
    """
    query = state.get("original_query", "")
    store = await get_faq_store()
    hit = await store.search(query)

    if hit is None:
        return {"faq_hit": False, "faq_score": 0.0}

    logger.info("qa.faq_answered", query=query, score=round(hit.score, 3), matched=hit.question)
    return {
        "faq_hit": True,
        "faq_score": hit.score,
        "answer": hit.answer,
        "answer_mode": "faq",
        "sources": [f"FAQ:{hit.question}"],
        "messages": [AIMessage(content=hit.answer)],
        "structured_output": {
            "answer_mode": "faq",
            "sources": [hit.question],
            "score": round(hit.score, 4),
        },
    }


async def retrieve_node(state: QAState) -> dict[str, Any]:
    """③ 按选定策略检索，把命中结果写进 ranked_chunks。"""
    query = state.get("original_query", "")
    strategy = state.get("retrieval_strategy") or DIRECT
    source_filter = state.get("source") or None  # 前端可传 source 限定学科/来源

    store = get_vector_store()
    docs = await run_strategy(store, strategy, query, source_filter=source_filter)

    # 精排分数过低的一律丢掉：检索"总能"返回几条，但那几条可能跟问题毫无关系，
    # 硬塞给模型只会得到一段编造的答案。阈值见 config.retrieval_min_score。
    threshold = get_settings().retrieval_min_score
    kept = [d for d in docs if float(d.metadata.get("rerank_score", 0.0)) >= threshold]
    dropped = len(docs) - len(kept)

    chunks = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "parent_id": doc.metadata.get("parent_id"),
            "score": round(float(doc.metadata.get("rerank_score", 0.0)), 4),
        }
        for doc in kept
    ]
    sources = sorted({str(c["source"]) for c in chunks if c["source"]})
    confidence = round(float(kept[0].metadata.get("rerank_score", 0.0)), 4) if kept else 0.0
    logger.info(
        "qa.retrieved",
        query=query,
        strategy=strategy,
        chunks=len(chunks),
        dropped=dropped,
        confidence=confidence,
    )
    return {
        "ranked_chunks": chunks,
        "sources": sources,
        "confidence": confidence,
        "is_high_confidence": confidence >= 0.6,
    }


async def _stream_or_invoke(llm: Any, messages: list[Any]) -> str:
    """有流式写入器就边生成边推送 token，否则一次性返回。

    流式能力由 LangGraph 的 custom stream 提供：调用方用
    `astream(stream_mode="custom")` 迭代时，这里 write 出去的内容会实时到达。
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        writer = None

    if writer is None:
        resp = await llm.ainvoke(messages)
        return str(resp.content).strip()

    parts: list[str] = []
    async for chunk in llm.astream(messages):
        text = str(chunk.content or "")
        if text:
            parts.append(text)
            writer(text)
    return "".join(parts).strip()


async def generate_node(state: QAState) -> dict[str, Any]:
    """④ 基于检索结果生成答案。

    检索为空时不再"对着占位符硬答"——那会让模型编一段看似合理的通用答案。
    直接返回 no_result，交给上层决定是换策略还是升级到联网研究。
    """
    query = state.get("original_query", "")
    chunks = state.get("ranked_chunks") or []

    if not chunks:
        message = "知识库中没找到与这个问题相关的内容。"
        logger.info("qa.no_result", query=query, strategy=state.get("retrieval_strategy"))
        return {
            "answer": message,
            "answer_mode": "no_result",
            "messages": [AIMessage(content=message)],
            "structured_output": {"answer_mode": "no_result", "sources": [], "no_result": True},
        }

    context = "\n\n".join(str(c["content"]) for c in chunks)
    llm = LLMFactory.get_llm("qa", temperature=0.2)
    answer = await _stream_or_invoke(
        llm,
        [
            SystemMessage(content=RAG_SYSTEM),
            HumanMessage(
                content=RAG_USER.format(
                    context=context,
                    history=_history_text(state.get("messages")),
                    question=query,
                )
            ),
        ],
    )
    logger.info("qa.generated", mode="rag", chars=len(answer), sources=state.get("sources"))
    return {
        "answer": answer,
        "answer_mode": "rag",
        "messages": [AIMessage(content=answer)],
        "structured_output": {"answer_mode": "rag", "sources": state.get("sources", [])},
    }


async def direct_node(state: QAState) -> dict[str, Any]:
    """通用知识分支：不检索，直接回答。"""
    query = state.get("original_query", "")
    llm = LLMFactory.get_llm("qa", temperature=0.3)
    answer = await _stream_or_invoke(
        llm,
        [SystemMessage(content=DIRECT_SYSTEM), HumanMessage(content=_with_history(state, query))],
    )
    logger.info("qa.generated", mode="llm_direct", chars=len(answer))
    return {
        "answer": answer,
        "answer_mode": "llm_direct",
        "messages": [AIMessage(content=answer)],
        "structured_output": {"answer_mode": "llm_direct", "sources": []},
    }
