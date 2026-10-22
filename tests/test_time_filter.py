"""时间范围推断的纯逻辑测试。"""

from __future__ import annotations

from agents.research.time_filter import (
    DAY,
    MONTH,
    VALID_RANGES,
    WEEK,
    YEAR,
    infer_time_range,
    normalize_time_range,
    resolve_time_range,
)


def test_detects_recent_keywords():
    assert infer_time_range("今天的新闻") == DAY
    assert infer_time_range("本周有什么更新") == WEEK
    assert infer_time_range("最近一个月的动态") == MONTH
    assert infer_time_range("今年的发布会") == YEAR


def test_detects_english_keywords():
    assert infer_time_range("today's release notes") == DAY
    assert infer_time_range("what's the latest version") == YEAR
    assert infer_time_range("recent changes") == YEAR


def test_tighter_range_wins():
    """同时出现"今天"和"今年"时取更紧的那个。"""
    assert infer_time_range("今天的最新消息") == DAY


def test_common_questions_are_not_restricted():
    """常识/原理类问题不该被加时间限制，否则会漏掉好资料。"""
    for q in ["什么是 RAG", "HNSW 的原理", "BGE-M3 和 reranker 的区别", "Milvus 支持哪些索引"]:
        assert infer_time_range(q) == "", f"不该限制时间：{q}"


def test_normalize_rejects_unknown_values():
    assert normalize_time_range("week") == WEEK
    assert normalize_time_range("WEEK") == WEEK
    assert normalize_time_range("  month ") == MONTH
    assert normalize_time_range("fortnight") == ""
    assert normalize_time_range(None) == ""
    assert normalize_time_range("") == ""


def test_all_values_match_tavily_contract():
    assert {"", DAY, WEEK, MONTH, YEAR} == VALID_RANGES


def test_model_value_has_priority():
    """模型判断优先于关键词：模型说 week，就不该被"今年"覆盖成 year。"""
    assert resolve_time_range("week", "今年的变化") == WEEK
    # 模型没给（空）时才用关键词兜底
    assert resolve_time_range("", "今年的变化") == YEAR
    assert resolve_time_range(None, "什么是 RAG") == ""
    # 模型给了非法值，视为没给，继续兜底
    assert resolve_time_range("someday", "本周更新") == WEEK
