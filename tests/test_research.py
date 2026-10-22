"""研究 Agent 的纯逻辑测试（不联网、不调模型）。"""

from __future__ import annotations

import json

from agents.research.nodes import parse_mcp_payload
from agents.research.tools import SearchResponse, SearchResult, dedup_results


def test_parse_content_blocks():
    """FastMCP 会把 list[dict] 序列化成多个 text block。"""
    raw = [
        {"type": "text", "text": json.dumps({"title": "A", "url": "http://a", "content": "x"})},
        {"type": "text", "text": json.dumps({"title": "B", "url": "http://b", "content": "y"})},
    ]
    items = parse_mcp_payload(raw)
    assert [i["title"] for i in items] == ["A", "B"]


def test_parse_json_string():
    items = parse_mcp_payload(json.dumps([{"title": "A", "url": "http://a"}]))
    assert items[0]["url"] == "http://a"


def test_parse_plain_list_of_dicts():
    items = parse_mcp_payload([{"title": "A", "url": "http://a"}])
    assert len(items) == 1


def test_parse_bad_input_returns_empty():
    assert parse_mcp_payload(None) == []
    assert parse_mcp_payload("not json") == []
    assert parse_mcp_payload([{"type": "text", "text": "not json"}]) == []


def test_dedup_by_url_keeps_first():
    first = SearchResponse(
        query="q1",
        results=[
            SearchResult(title="A", url="http://a/", content="1"),
            SearchResult(title="B", url="http://b", content="2"),
        ],
    )
    second = SearchResponse(
        query="q2",
        results=[
            SearchResult(title="A again", url="http://a", content="3"),  # 同 URL（结尾斜杠差异）
            SearchResult(title="C", url="http://c", content="4"),
        ],
    )
    merged = dedup_results([first, second])
    assert [m.title for m in merged] == ["A", "B", "C"]


def test_dedup_skips_empty_url():
    response = SearchResponse(query="q", results=[SearchResult(title="X", url="", content="...")])
    assert dedup_results([response]) == []


def test_context_contains_index_and_url():
    item = SearchResult(title="标题", url="http://x", content="正文")
    text = item.as_context(3)
    assert text.startswith("[3] 标题")
    assert "http://x" in text
    assert "正文" in text
