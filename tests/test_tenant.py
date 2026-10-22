"""多租户隔离的纯逻辑测试。"""

from __future__ import annotations

from agents.qa.cache import answer_key
from agents.qa.faq import bm25_confidence
from agents.qa.retrieval.vector_store import VectorStore


# ── 置信度归一化（与库大小无关）──────────────────────────
def test_confidence_is_size_independent():
    """同样"完全命中"，大库和小库都应该给出高置信度。"""
    big = [5.2, 1.0, 0.2] + [0.0] * 9  # 12 条
    small = [2.79, 0.0, 0.0, 0.0]  # 4 条
    _, big_conf = bm25_confidence(big)
    _, small_conf = bm25_confidence(small)
    assert big_conf > 0.85
    assert small_conf > 0.70  # 旧实现这里只有 0.845，被 0.85 阈值挡掉
    assert abs(big_conf - small_conf) < 0.2


def test_confidence_zero_when_all_equal():
    _, conf = bm25_confidence([0.0, 0.0, 0.0, 0.0])
    assert conf == 0.0


def test_confidence_low_for_weak_overlap():
    _, conf = bm25_confidence([1.0, 0.9, 0.9, 0.9])
    assert conf < 0.1


def test_confidence_handles_empty():
    assert bm25_confidence([]) == (0, 0.0)


# ── 缓存键 ────────────────────────────────────────────────
def test_cache_key_scoped_by_tenant():
    a = answer_key(1, "你们怎么收费", "tenant_a")
    b = answer_key(1, "你们怎么收费", "tenant_b")
    assert a != b, "不同租户的相同问题不能共用缓存"


def test_cache_key_stable_for_same_tenant():
    assert answer_key(1, "你们怎么收费", "t1") == answer_key(1, " 你们 怎么收费 ", "t1")


def test_cache_key_without_tenant_keeps_old_shape():
    """不传租户时保持旧格式，脚本/测试仍可用。"""
    assert ":t" not in answer_key(1, "q")


# ── Milvus 过滤表达式 ─────────────────────────────────────
def test_filter_always_contains_tenant():
    expr = VectorStore.build_filter("tenant_a", None)
    assert expr == 'tenant_id == "tenant_a"'


def test_filter_combines_tenant_and_source():
    expr = VectorStore.build_filter("tenant_a", "01_产品概览")
    assert 'tenant_id == "tenant_a"' in expr
    assert 'source == "01_产品概览"' in expr
    assert " and " in expr


def test_filter_empty_when_no_conditions():
    assert VectorStore.build_filter(None, None) == ""
