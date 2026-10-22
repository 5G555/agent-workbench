"""四种检索增强策略。
直接检索      —— 原句直接查
假设问题检索  —— 先让模型编一段"假设答案"，用答案去查（HyDE）
子查询检索    —— 拆成多个子问题分别查，按内容去重
回溯问题检索  —— 先把问题简化成基础问题再查
"""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from agents.qa.prompts import BACKTRACK_USER, HYDE_USER, SUBQUERY_USER
from agents.qa.retrieval.vector_store import RetrievalScope, VectorStore
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

DIRECT = "直接检索"
HYDE = "假设问题检索"
SUBQUERY = "子查询检索"
BACKTRACK = "回溯问题检索"
ALL_STRATEGIES = (DIRECT, HYDE, SUBQUERY, BACKTRACK)


async def _ask_llm(prompt: str) -> str:
    """一次性短问答，用于生成假设答案 / 子查询 / 回溯问题。"""
    llm = LLMFactory.get_llm("qa", temperature=0.3)
    resp = await llm.ainvoke(
        [
            SystemMessage(content="你只输出要求的内容本身。"),
            HumanMessage(content=prompt),
        ]
    )
    return str(resp.content).strip()


async def retrieve_direct(
    store: VectorStore, query: str, scope: RetrievalScope, source_filter: str | None = None
) -> list[Document]:
    return await _search(store, query, scope, source_filter)


async def retrieve_hyde(
    store: VectorStore, query: str, scope: RetrievalScope, source_filter: str | None = None
) -> list[Document]:
    """先生成假设答案，再用假设答案去检索（抽象问题常用）。"""
    hypo = await _ask_llm(HYDE_USER.format(query=query))
    if not hypo:
        return await _search(store, query, scope, source_filter)
    logger.info("strategy.hyde_generated", chars=len(hypo))
    return await _search(store, hypo, scope, source_filter)


async def retrieve_subqueries(
    store: VectorStore, query: str, scope: RetrievalScope, source_filter: str | None = None
) -> list[Document]:
    """拆成多个子问题分别检索，按父块内容去重。"""
    raw = await _ask_llm(SUBQUERY_USER.format(query=query))
    subqueries = [line.strip() for line in raw.splitlines() if line.strip()]
    if not subqueries:
        return await _search(store, query, scope, source_filter)

    logger.info("strategy.subqueries", count=len(subqueries), queries=subqueries)
    docs: list[Document] = []
    for sub in subqueries[:4]:  # 最多 4 个，防止调用量失控
        docs.extend(await _search(store, sub, scope, source_filter))

    unique: dict[str, Document] = {}
    for doc in docs:
        unique.setdefault(doc.page_content, doc)
    return list(unique.values())


async def retrieve_backtrack(
    store: VectorStore, query: str, scope: RetrievalScope, source_filter: str | None = None
) -> list[Document]:
    """先把问题简化成基础问题再检索。"""
    simplified = await _ask_llm(BACKTRACK_USER.format(query=query))
    if not simplified:
        return await _search(store, query, scope, source_filter)
    logger.info("strategy.backtracked", simplified=simplified)
    return await _search(store, simplified, scope, source_filter)


async def _search(
    store: VectorStore, query: str, scope: RetrievalScope, source_filter: str | None = None
) -> list[Document]:
    """向量检索是同步阻塞的（torch + Milvus 客户端），丢到线程池执行，避免卡住事件循环。"""
    return await asyncio.to_thread(
        store.hybrid_search_with_rerank, query, None, source_filter, scope
    )


async def run_strategy(
    store: VectorStore,
    strategy: str,
    query: str,
    scope: RetrievalScope,
    source_filter: str | None = None,
) -> list[Document]:
    """按策略名分派。未知策略退回直接检索。"""
    handlers = {
        DIRECT: retrieve_direct,
        HYDE: retrieve_hyde,
        SUBQUERY: retrieve_subqueries,
        BACKTRACK: retrieve_backtrack,
    }
    handler = handlers.get(strategy, retrieve_direct)
    return await handler(store, query, scope, source_filter)
