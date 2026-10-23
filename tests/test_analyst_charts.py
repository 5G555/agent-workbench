"""图表类型识别 + 结果缓存的测试。"""

from __future__ import annotations

import pytest

from agents.analyst.charts import detect_chart_type
from agents.analyst.result_cache import cache_key


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("画出这只股票的 K 线图", "candlestick"),
        ("来一张蜡烛图看看", "candlestick"),
        ("用柱状图展示每周成交量", "bar"),
        ("画个饼图看看分布", "pie"),
        ("帮我画个散点图", "scatter"),
        ("画个直方图", "hist"),
        ("画折线图看走势", "line"),
        ("做个热力图看相关性", "heatmap"),
        ("不要图，只给数字", "none"),
        ("算一下平均成交量", ""),  # 没提图表类型
    ],
)
def test_detect_chart_type(text, expected):
    assert detect_chart_type(text) == expected


def test_candlestick_wins_over_line():
    """同时出现"走势"和"K线"时要给 K 线——具体优先。"""
    assert detect_chart_type("画个 K 线图的走势") == "candlestick"


def test_cache_key_stable_and_sensitive():
    base = cache_key("tenant_default", [9], "画 K 线", "fp1", "candlestick")
    same = cache_key("tenant_default", [9], "画  K  线", "fp1", "candlestick")
    assert base == same  # 空白归一化

    assert base != cache_key("tenant_default", [9], "画 K 线", "fp2", "candlestick")  # 数据变了
    assert base != cache_key("tenant_default", [10], "画 K 线", "fp1", "candlestick")  # 任务变了
    assert base != cache_key("tenant_demo", [9], "画 K 线", "fp1", "candlestick")  # 租户隔离
    assert base != cache_key("tenant_default", [9], "画 柱状图", "fp1", "bar")  # 图表类型不同
    assert base.startswith("analyst:cache:tenant_default:")


def test_cache_key_ignores_task_order():
    assert cache_key("t", [9, 10], "q", "fp") == cache_key("t", [10, 9], "q", "fp")
