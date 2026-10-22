"""判断一个问题要不要限制检索时间范围。

为什么要单独做：
    Tavily 的 `time_range` 收窄检索窗口后，能显著提升"最新版本/最近动态"这类
    问题的时效性；但反过来，对"什么是 RAG"这种常识问题一旦加上时间限制，
    反而会把好资料过滤掉。所以规则必须保守：**只有问题里明确提到时间才限制**。

    判断顺序：先让模型在拆子问题时给一个值（它更懂语义），
    模型没给再用这里的词表兜底；两边都没有就不限时间。
"""

from __future__ import annotations

# 结果值必须与 Tavily 的 time_range 取值一致
DAY = "day"
WEEK = "week"
MONTH = "month"
YEAR = "year"

VALID_RANGES = {"", DAY, WEEK, MONTH, YEAR}

# 从"最紧"到"最松"的顺序很重要：一个问题里同时出现"今天"和"今年"时，取今天
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (DAY, ("今天", "今日", "刚刚", "today", "right now")),
    (WEEK, ("本周", "这周", "上周", "这一周", "this week", "past week", "last week")),
    (MONTH, ("本月", "这个月", "上个月", "近一个月", "最近一个月", "this month", "past month")),
    (
        YEAR,
        (
            "今年",
            "今年内",
            "本年度",
            "去年",
            "最近一年",
            "latest",
            "recent",
            "recently",
            "current",
            "this year",
        ),
    ),
]


def infer_time_range(question: str) -> str:
    """按关键词推断时间范围；判断不出来返回空串（不限时间）。"""
    text = (question or "").lower()
    if not text:
        return ""
    for value, keywords in _KEYWORDS:
        if any(k in text for k in keywords):
            return value
    return ""


def normalize_time_range(value: str | None) -> str:
    """把模型/调用方给的值收敛到合法集合，非法值一律当"不限"。"""
    candidate = (value or "").strip().lower()
    return candidate if candidate in VALID_RANGES else ""


def resolve_time_range(model_value: str | None, question: str) -> str:
    """优先用模型判断，其次用关键词兜底，都没有则不限。"""
    explicit = normalize_time_range(model_value)
    if explicit:
        return explicit
    return infer_time_range(question)
