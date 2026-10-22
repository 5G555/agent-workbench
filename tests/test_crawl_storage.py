"""原始 HTML 落库的测试。"""

from __future__ import annotations

from agents.crawler.store import truncate_html


def test_short_html_kept_as_is():
    html = "<html><body>hi</body></html>"
    assert truncate_html(html, 1000) == html


def test_long_html_truncated_with_marker():
    html = "x" * 5000
    result = truncate_html(html, 1000)
    assert result.startswith("x" * 1000)
    assert "truncated" in result
    assert len(result) < 1100


def test_empty_html_stays_empty():
    assert truncate_html("", 1000) == ""


def test_zero_limit_disables_html_storage():
    """配成 0 表示不存 HTML，避免有人以为"存了但很小"。"""
    assert truncate_html("<html>abc</html>", 0) == ""


def test_default_limit_comes_from_settings():
    from config import get_settings

    cap = get_settings().crawl_max_html_chars
    assert cap > 0, "默认应当保存 HTML"
    # 略超上限的 HTML 会被截断
    assert "truncated" in truncate_html("y" * (cap + 10))
