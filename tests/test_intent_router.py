"""意图路由的纯逻辑测试（不调模型）。"""

from __future__ import annotations

from agents.supervisor.local_router import should_use_local
from agents.supervisor.router import (
    RouteDecision,
    detect_url,
    implies_crawl,
    mentions_crawl,
    resolve_confirmation,
)


# ── URL 抽取 ──────────────────────────────────────────────
def test_detect_url_finds_http_link():
    assert detect_url("帮我抓 https://example.com/list 的数据") == "https://example.com/list"


def test_detect_url_strips_trailing_punctuation():
    assert detect_url("抓取 https://example.com/a。") == "https://example.com/a"


def test_detect_url_returns_empty_when_none():
    assert detect_url("你们怎么收费") == ""


# ── 关键词判断 ────────────────────────────────────────────
def test_mentions_crawl_keywords():
    assert mentions_crawl("帮我抓取这个页面")
    assert mentions_crawl("爬取商品数据")
    assert mentions_crawl("采集几个网站")
    assert not mentions_crawl("这个功能怎么用")


def test_implies_crawl_without_explicit_words():
    assert implies_crawl("我想分析一下竞品官网的价格")
    assert implies_crawl("看看同行网站怎么做的")
    assert not implies_crawl("你们怎么收费")


# ── 确认规则 ──────────────────────────────────────────────
def test_crawler_with_url_and_keyword_runs_directly():
    decision = RouteDecision(agent="crawler", confidence=0.98)
    resolved = resolve_confirmation(decision, "帮我抓取 https://example.com/list 的数据")
    assert resolved.need_confirm is False
    assert resolved.entry_url == "https://example.com/list"


def test_crawler_without_url_needs_confirm():
    decision = RouteDecision(agent="crawler", confidence=0.95)
    resolved = resolve_confirmation(decision, "帮我抓取竞品的价格")
    assert resolved.need_confirm is True
    assert "网址" in resolved.confirm_question


def test_crawler_with_url_but_no_keyword_needs_confirm():
    """有 URL 但没明说要抓，先问一句更稳妥（抓取有副作用）。"""
    decision = RouteDecision(agent="crawler", confidence=0.8)
    resolved = resolve_confirmation(decision, "https://example.com/list 这个页面")
    assert resolved.need_confirm is True
    assert resolved.entry_url == "https://example.com/list"


def test_qa_with_implicit_crawl_asks_for_confirmation():
    """模型判成 qa，但用户想拿别人的页面数据 → 值得确认一次。"""
    decision = RouteDecision(agent="qa", confidence=0.7)
    resolved = resolve_confirmation(decision, "我想分析一下竞品官网的价格")
    assert resolved.need_confirm is True


def test_qa_plain_question_never_asks():
    decision = RouteDecision(agent="qa", confidence=0.9)
    resolved = resolve_confirmation(decision, "你们怎么收费")
    assert resolved.need_confirm is False
    assert resolved.confirm_question == ""


# ── 本地模型置信度门槛 ────────────────────────────────────
def test_should_use_local_threshold_boundary():
    assert should_use_local(0.70, 0.70) is True  # 等于阈值也采信
    assert should_use_local(0.69, 0.70) is False
    assert should_use_local(0.99, 0.70) is True
