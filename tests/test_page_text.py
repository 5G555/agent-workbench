"""网页正文提取的测试（纯函数，不联网）。"""

from __future__ import annotations

from mcp_servers.web_search_server import html_to_text


def test_prefers_main_over_body():
    """侧边栏推荐位常常不在 <nav> 里，优先只取 <main> 才干净。"""
    html = """
    <body>
      <div class="promo">不要出现-推广条</div>
      <main>
        <h1>HNSW</h1>
        <p>HNSW 是一种基于图的索引算法。</p>
      </main>
    </body>
    """
    text = html_to_text(html)
    assert "不要出现-推广条" not in text
    assert "基于图的索引算法" in text


def test_prefers_article_when_present():
    html = """
    <body>
      <div>外层噪声</div>
      <article><h1>标题</h1><p>正文段落</p></article>
    </body>
    """
    text = html_to_text(html)
    assert "正文段落" in text
    assert "外层噪声" not in text


def test_cuts_before_first_h1():
    """h1 之前通常是面包屑和推广，切掉。"""
    html = "<main><div>面包屑导航</div><h1>真正的标题</h1><p>内容</p></main>"
    text = html_to_text(html)
    assert text.startswith("真正的标题")
    assert "面包屑导航" not in text


def test_removes_script_and_style_blocks():
    html = "<main><h1>T</h1><script>var a=1;</script><style>.a{}</style><p>可见</p></main>"
    text = html_to_text(html)
    assert "var a=1" not in text
    assert ".a{}" not in text
    assert "可见" in text


def test_keeps_paragraph_breaks():
    html = "<main><h1>T</h1><p>第一段</p><p>第二段</p></main>"
    text = html_to_text(html)
    assert "第一段" in text and "第二段" in text
    assert "\n" in text, "段落之间应保留换行，否则表格/列表会糊成一团"


def test_unescapes_html_entities():
    html = "<main><h1>T</h1><p>a &amp; b &lt;tag&gt;</p></main>"
    text = html_to_text(html)
    assert "a & b <tag>" in text


def test_short_main_is_ignored():
    """有些页面的 <main> 只是空壳，太短就不采用，回退到整页。"""
    html = "<main>x</main><body><h1>标题</h1><p>正文在这里</p></body>"
    text = html_to_text(html)
    assert "正文在这里" in text
