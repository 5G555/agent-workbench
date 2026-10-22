"""把采集记录变成 DataFrame，并做基础预处理。

为什么预处理要单独一层、不能让模型自己写：
    爬下来的结构化记录字段类型全是字符串（JSON 文本里抽出来的），
    "价格 12.5"、"涨跌 3.2%"、"日期 2026-10-22" 都得转成数字/时间。
    每个问题都让模型重写一遍转换逻辑，既慢又容易一次一个口径。

这里做的是**保守**的自动预处理：
    - 能整列转成数字的转数字（保留原始列，另存 <列名>_num 避免丢信息）
    - 能整列解析成日期的转 datetime
    - 去掉全空列、去重、空字符串当缺失值
转不动的列原样保留，交给模型决定怎么处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PERCENT_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*%\s*$")
NUMBER_PATTERN = re.compile(r"^\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)\s*$")
DATE_PATTERN = re.compile(r"^\s*\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2})?")


@dataclass
class PrepReport:
    """预处理做了什么，给用户看。"""

    rows: int = 0
    columns: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    dropped_empty: list[str] = field(default_factory=list)
    deduped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "numeric": self.numeric,
            "datetime_columns": self.datetime,
            "dropped_empty": self.dropped_empty,
            "deduped": self.deduped,
        }


def _to_number(value: str) -> float | None:
    """字符串 → 数字；处理千分位、百分号、货币符号。"""
    text = value.strip().replace(",", "").replace("￥", "").replace("$", "")
    percent = PERCENT_PATTERN.match(text)
    if percent:
        return float(percent.group(1))
    number = NUMBER_PATTERN.match(text)
    return float(number.group(1)) if number else None


def build_frame(records: list[dict[str, Any]], dedupe: bool = True) -> tuple[Any, PrepReport]:
    """records → (DataFrame, 预处理报告)。

    records 的形状：[{"data": {"字段": "值", ...}}, ...]（crawl_records 展开后的样子）
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for record in records:
        data = record.get("data") if isinstance(record, dict) else None
        if isinstance(data, dict):
            rows.append({str(k): v for k, v in data.items()})
        elif isinstance(record, dict):
            rows.append({str(k): v for k, v in record.items()})

    frame = pd.DataFrame(rows)
    report = PrepReport(rows=len(frame))

    if frame.empty:
        return frame, report

    # 空字符串当缺失
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    # 去掉全空列
    empty_cols = [str(c) for c in frame.columns if bool(frame[c].isna().all())]
    if empty_cols:
        frame = frame.drop(columns=empty_cols)
        report.dropped_empty = empty_cols

    if dedupe:
        before = len(frame)
        frame = frame.drop_duplicates()
        report.deduped = before - len(frame)

    # 逐列尝试数字 / 日期
    for column in list(frame.columns):
        series = frame[column]
        # 已经是数值/时间的列跳过；注意 pandas 3 里字符串列不再是 object dtype，
        # 不能用 dtype == object 判断
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        text = series.dropna().astype(str)
        if text.empty:
            continue

        # 用纯 Python 判断"整列能不能转"：pandas 的 .all()/.mean() 返回类型
        # 在类型检查里是 Series|bool，容易误报，也不如显式判断清楚
        values = [str(v) for v in text]
        if values and all(_to_number(v) is not None for v in values):
            frame[f"{column}_num"] = series.map(
                lambda v: _to_number(str(v)) if pd.notna(v) else None
            ).astype("float64")
            report.numeric.append(f"{column}_num")
            continue

        date_hits = sum(1 for v in values if DATE_PATTERN.match(v))
        if values and date_hits / len(values) > 0.9:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if float(parsed.notna().sum()) / max(len(parsed), 1) > 0.9:
                frame[f"{column}_dt"] = parsed
                report.datetime.append(f"{column}_dt")

    report.rows = len(frame)
    report.columns = [str(c) for c in frame.columns]
    return frame, report


def frame_overview(frame: Any, max_rows: int = 8) -> str:
    """给模型看的表结构 + 几行样例 + 数值列统计。"""
    if frame is None or getattr(frame, "empty", True):
        return "（没有可用数据）"

    lines = [f"数据形状：{frame.shape[0]} 行 × {frame.shape[1]} 列", "列与类型："]
    for column in frame.columns:
        lines.append(f"  - {column}: {frame[column].dtype}")
    lines.append("")
    lines.append("前几行：")
    lines.append(frame.head(max_rows).to_string(index=False))

    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("")
        lines.append("数值列统计：")
        lines.append(numeric.describe().to_string())
    return "\n".join(lines)
