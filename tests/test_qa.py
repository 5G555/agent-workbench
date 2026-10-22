"""QA Agent 的关键行为单测。

只测纯逻辑与分支，不碰模型、数据库、Milvus、Redis——那些用脚本手工验证。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agents.qa import nodes
from agents.qa.cache import answer_key, normalize
from agents.qa.faq import tokenize
from agents.qa.graph import _route_after_cache, _route_after_faq, _route_by_category
from agents.qa.splitter import chunk_text, split_document
from agents.qa.state import QAState


class FakeCache:
    """记录调用的假缓存，避免测试依赖 Redis。"""

    def __init__(self) -> None:
        self.stored: list[tuple] = []
        self.lookups: list[str] = []
        self.tenants: list[str] = []

    async def get(self, query: str, tenant_id: str = ""):
        self.lookups.append(query)
        self.tenants.append(tenant_id)
        return None

    async def set(self, query: str, answer: str, mode: str, sources=None, tenant_id: str = ""):
        self.stored.append((query, answer, mode))
        self.tenants.append(tenant_id)


# ── 缓存与多轮上下文 ───────────────────────────────────────
def test_first_turn_detection():
    assert nodes._is_first_turn({"messages": [HumanMessage(content="q1")]}) is True
    assert (
        nodes._is_first_turn(
            {
                "messages": [
                    HumanMessage(content="q1"),
                    AIMessage(content="a1"),
                    HumanMessage(content="q2"),
                ]
            }
        )
        is False
    )


async def test_cache_write_skips_multi_turn(monkeypatch):
    """多轮里的答案依赖历史，不能进只按 query 索引的缓存。"""
    fake = FakeCache()
    monkeypatch.setattr(nodes, "get_answer_cache", lambda: fake)
    state = QAState(
        messages=[
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
        ],
        original_query="q2",
        answer="基于上文补全的回答",
        answer_mode="rag",
    )
    assert await nodes.cache_write_node(state) == {}
    assert fake.stored == []


async def test_cache_write_on_first_turn(monkeypatch):
    fake = FakeCache()
    monkeypatch.setattr(nodes, "get_answer_cache", lambda: fake)
    state = QAState(
        messages=[HumanMessage(content="q1")],
        original_query="q1",
        answer="答案",
        answer_mode="rag",
        sources=["doc1"],
    )
    await nodes.cache_write_node(state)
    assert fake.stored == [("q1", "答案", "rag")]


async def test_cache_lookup_skips_multi_turn(monkeypatch):
    fake = FakeCache()
    monkeypatch.setattr(nodes, "get_answer_cache", lambda: fake)
    state = QAState(
        messages=[
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="那P2呢"),
        ],
    )
    result = await nodes.cache_lookup_node(state)
    assert result["cache_hit"] is False
    assert fake.lookups == []  # 根本没去查缓存


def test_cache_key_is_version_scoped_and_normalized():
    assert normalize("  你们 怎么  收费 ") == "你们怎么收费"
    assert answer_key(1, "你们怎么收费") == answer_key(1, " 你们怎么收费 ")
    assert answer_key(1, "x") != answer_key(2, "x")


# ── 图的路由分支 ───────────────────────────────────────────
def test_route_by_category():
    assert _route_by_category({"query_type": "GENERAL"}) == "direct"
    assert _route_by_category({"query_type": "PROFESSIONAL"}) == "retrieve_flow"
    assert _route_by_category({}) == "retrieve_flow"  # 分类缺失时保守走检索


def test_route_after_faq_and_cache():
    assert _route_after_faq({"faq_hit": True}) == "hit"
    assert _route_after_faq({}) == "miss"
    assert _route_after_cache({"cache_hit": True}) == "hit"
    assert _route_after_cache({}) == "miss"


# ── 切分 ───────────────────────────────────────────────────
def test_chunk_text_respects_size():
    text = "这是第一句话。这是第二句话！这是第三句话？" * 20
    chunks = chunk_text(text, 60, 10)
    assert chunks
    assert all(len(c) <= 60 + 10 for c in chunks)  # 允许 overlap 带来的一点溢出
    assert "".join(chunks).count("这是第一句话") >= 10


def test_split_document_metadata():
    docs = split_document(
        "第一段内容。\n\n第二段内容。",
        source="demo",
    )
    assert docs
    for doc in docs:
        assert doc.metadata["source"] == "demo"
        assert doc.metadata["parent_id"].startswith("demo_p")
        assert doc.page_content in doc.metadata["parent_content"]


# ── FAQ 分词 ───────────────────────────────────────────────
def test_tokenize_drops_blank():
    tokens = tokenize("  你们 怎么 收费 ")
    assert "" not in tokens
    assert "收费" in tokens
