"""爬虫工具层的测试（纯逻辑，不联网）。"""

from __future__ import annotations

from agents.crawler.tools import (
    extract_blocks,
    extract_records,
    extract_tables,
    find_links,
    html_to_text,
    paginate,
)

QUOTES_HTML = """
<html><body>
  <div class="quote">
    <span class="text">quote one</span>
    <small class="author">Author A</small>
  </div>
  <div class="quote">
    <span class="text">quote two</span>
    <small class="author">Author B</small>
  </div>
  <div class="quote">
    <span class="text">quote three</span>
    <small class="author">Author C</small>
  </div>
</body></html>
"""

TABLE_HTML = """
<table>
  <tr><th>名称</th><th>价格</th></tr>
  <tr><td>基础版</td><td>99</td></tr>
  <tr><td>专业版</td><td>299</td></tr>
</table>
"""


def test_html_to_text_drops_scripts_and_nav():
    html = """
    <html><body>
      <nav>导航菜单</nav>
      <script>var a = 1;</script>
      <h1>正文标题</h1>
      <p>第一段内容</p>
    </body></html>
    """
    text = html_to_text(html)
    assert "导航菜单" not in text
    assert "var a" not in text
    assert "第一段内容" in text


def test_extract_tables_uses_header_row():
    records = extract_tables(TABLE_HTML)
    assert len(records) == 2
    assert records[0].data["名称"] == "基础版"
    assert records[0].data["价格"] == "99"


def test_extract_blocks_pulls_multiple_fields():
    """重复出现的同结构区块要抽出多字段记录，而不是拍平成段落。"""
    records = extract_blocks(QUOTES_HTML)
    assert len(records) == 3
    assert records[0].data["text"] == "quote one"
    assert records[0].data["author"] == "Author A"


def test_extract_blocks_requires_at_least_two_fields():
    """只有一个字段的容器不算结构化，避免把纯文本叶子节点当记录。"""
    html = """
    <html><body>
      <span class="only">a</span><span class="only">b</span><span class="only">c</span>
    </body></html>
    """
    assert extract_blocks(html) == []


def test_extract_records_prefers_table_over_blocks():
    html = TABLE_HTML + QUOTES_HTML
    records, strategy = extract_records(html)
    assert strategy == "table"
    assert records


def test_extract_records_falls_back_to_blocks():
    records, strategy = extract_records(QUOTES_HTML)
    assert strategy == "block"
    assert len(records) == 3


def test_extract_records_falls_back_to_paragraph():
    records, strategy = extract_records("<html><body><p>x</p></body></html>", "纯文本段落" * 20)
    assert strategy == "paragraph"
    assert records


def test_find_links_keeps_same_host_only():
    html = """
    <a href="/docs/a">内链</a>
    <a href="https://other.com/b">外链</a>
    <a href="/static/logo.png">图片</a>
    <a href="/docs/a#section">重复</a>
    """
    links = find_links(html, "https://example.com/page")
    assert "https://example.com/docs/a" in links
    assert all("other.com" not in link for link in links)
    assert all(not link.endswith(".png") for link in links)
    assert len(links) == len(set(links)), "同一链接不应重复出现"


def test_paginate_prefers_rel_next():
    html = '<a rel="next" href="/page/2/">下一页</a>'
    assert paginate(html, "https://example.com/page/1/") == "https://example.com/page/2/"


def test_paginate_from_link_text():
    html = '<a href="/list?page=3">下一页</a>'
    assert paginate(html, "https://example.com/list?page=2") == "https://example.com/list?page=3"


def test_paginate_from_query_param_when_no_link():
    """页面里没有下一页链接时，按 URL 里的 page 参数自增。"""
    assert (
        paginate("<p>no links</p>", "https://example.com/list?page=4&sort=asc")
        == "https://example.com/list?page=5&sort=asc"
    )


def test_paginate_returns_none_when_no_hint():
    assert paginate("<p>nothing</p>", "https://example.com/about") is None
