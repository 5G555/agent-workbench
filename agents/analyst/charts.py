"""图表类型识别：用户说"画 K 线"就画 K 线，不靠模型自己猜。

为什么用关键词而不是让模型判断：
    "画个走势图"和"画个 K 线图"对模型来说都是自然语言，
    但它有时会都给折线——而用户明确说了 K 线，画错就是白跑一趟。
    关键词是确定性的，命中就按用户说的来。
"""

from __future__ import annotations

# 顺序有讲究：先匹配更具体的（K线 > 柱状 > 折线）
CHART_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("candlestick", ("k线", "k 线", "kline", "蜡烛图", "阴阳线", "日k")),
    ("heatmap", ("热力图", "相关性矩阵", "相关矩阵")),
    ("box", ("箱线图", "箱型图", "箱形图")),
    ("pie", ("饼图", "占比图", "饼状")),
    ("hist", ("直方图", "分布图", "频数图")),
    ("scatter", ("散点图", "散点图", "相关图")),
    ("bar", ("柱状图", "柱形图", "条形图", "柱图")),
    ("area", ("面积图",)),
    ("line", ("折线图", "走势图", "趋势图", "曲线图")),
    ("none", ("不要图", "不用画图", "纯数字", "只给数字")),
)

CHART_LABELS = {
    "candlestick": "K 线图",
    "bar": "柱状图",
    "line": "折线图",
    "scatter": "散点图",
    "pie": "饼图",
    "hist": "直方图",
    "box": "箱线图",
    "heatmap": "热力图",
    "area": "面积图",
    "none": "不画图",
}


def detect_chart_type(text: str) -> str:
    """从问题里识别图表类型；识别不到返回空串（交给模型自己决定）。"""
    lowered = (text or "").lower()
    for chart, keywords in CHART_KEYWORDS:
        if any(word in lowered for word in keywords):
            return chart
    return ""


def chart_hint(chart: str) -> str:
    """给模型的一句话指令。"""
    if chart == "candlestick":
        return (
            "【图表类型】用户要 K 线图。请调用 kline(df, 'Date_dt', 'Open_num', "
            "'High_num', 'Low_num', 'Close_num')，不要自己用折线代替。"
        )
    if chart == "bar":
        return "【图表类型】用户要柱状图：用 plt.bar / plt.barh。"
    if chart == "line":
        return "【图表类型】用户要折线图：用 plt.plot。"
    if chart == "scatter":
        return "【图表类型】用户要散点图：用 plt.scatter。"
    if chart == "pie":
        return (
            "【图表类型】用户要饼图：用 plt.pie 并显示百分比。"
            "注意：饼图需要有限的分类维度。如果数据里只有连续数值或单一分类"
            "（比如只有一个 task_id），不要硬画成 100% 的一片，"
            "改用直方图（plt.hist）并在 result 里说明为什么换图。"
        )
    if chart == "hist":
        return "【图表类型】用户要直方图：用 plt.hist，bins 取 10~20。"
    if chart == "box":
        return "【图表类型】用户要箱线图：用 plt.boxplot。"
    if chart == "heatmap":
        return "【图表类型】用户要热力图：用 plt.imshow(corr, cmap='Blues') 并加 colorbar。"
    if chart == "area":
        return "【图表类型】用户要面积图：用 plt.fill_between。"
    if chart == "none":
        return "【图表类型】用户不要图，只输出数字结论，不要调用 plt。"
    return ""
