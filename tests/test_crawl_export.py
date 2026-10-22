"""导出功能的测试：文件名安全 + 路径穿越防护。"""

from __future__ import annotations

import pytest

from agents.crawler.exporter import page_filename, safe_name


def test_safe_name_keeps_chinese_and_alnum():
    assert safe_name("产品价格表 2026") == "产品价格表-2026"


def test_safe_name_replaces_dangerous_chars():
    """路径分隔符必须被替换，否则标题里带 ../ 就能写到目录外。"""
    assert "/" not in safe_name("../../etc/passwd")
    assert "\\" not in safe_name("..\\..\\windows")
    assert safe_name("../../etc/passwd").startswith("..") is False or "-" in safe_name(
        "../../etc/passwd"
    )


def test_safe_name_falls_back_when_empty():
    assert safe_name("") == "page"
    assert safe_name("///") == "page"
    assert safe_name("   ") == "page"


def test_safe_name_truncates_long_titles():
    name = safe_name("长" * 200)
    assert len(name) <= 40


def test_page_filename_is_zero_padded():
    name = page_filename(7, "标题", "https://example.com", ".html")
    assert name.startswith("0007-")
    assert name.endswith(".html")


def test_page_filename_uses_domain_when_title_missing():
    name = page_filename(3, "", "https://quotes.toscrape.com/page/2/", ".txt")
    # 点号会被替换成连字符，域名仍可辨认
    assert "quotes-toscrape-com" in name
    assert name.startswith("0003-")


@pytest.mark.parametrize("suffix", [".html", ".txt"])
def test_page_filename_has_no_path_separators(suffix):
    name = page_filename(1, "a/b/c", "https://example.com/x", suffix)
    assert "/" not in name
