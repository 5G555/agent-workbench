"""手动分析：用户自己选任务/字段/聚合/图表，代码按模板生成。

为什么要有手动模式：
    自然语言分析适合探索，但同一问题两次可能得出不同口径。
    做报表、对账这类场景，用户需要"我指定口径，你给我确定结果"。

为什么代码不用模型写：
    口径既然是人定的，代码就该是确定性的。模板生成保证
    同样的 spec 永远生成同一份代码，结果可复现、可审计。
    模型只负责最后把数字讲成人话。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AGG_FUNCS = {
    "mean": "mean",
    "sum": "sum",
    "count": "count",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
}

CHART_TYPES = ("none", "bar", "line", "scatter", "pie", "hist", "candlestick")
OHLC = ("Open_num", "High_num", "Low_num", "Close_num")


@dataclass
class ManualSpec:
    """手动分析的参数。"""

    metrics: list[str] = field(default_factory=list)
    agg: str = "mean"
    group_by: str = ""
    x_field: str = ""
    chart_type: str = "bar"
    title: str = ""


def build_code(spec: ManualSpec, columns: list[str]) -> str:
    """按 spec 生成确定的 pandas 代码（与自由生成走同一个沙箱）。"""
    agg = AGG_FUNCS.get(spec.agg, "mean")
    metrics = [m for m in spec.metrics if m in columns]
    if not metrics:
        raise ValueError("请选择至少一个数值字段")
    group = spec.group_by if spec.group_by in columns else ""
    chart = spec.chart_type if spec.chart_type in CHART_TYPES else "none"
    title = spec.title or f"{agg} {','.join(metrics)}"
    x = spec.x_field if spec.x_field in columns else group

    lines: list[str] = ["d = df.copy()"]
    if group:
        lines += [
            f"g = d.groupby({group!r}, dropna=False)[{metrics!r}].{agg}()",
            "g = g.reset_index()",
        ]
    else:
        lines += [f"g = pd.DataFrame([d[{metrics!r}].{agg}()])"]
    lines += ["print(g.to_string(index=False))"]

    # 不分组又要画折线/散点：直接画原始序列，而不是画一行聚合结果
    if not group and chart in {"line", "scatter"} and x:
        lines += [f"raw = d.sort_values({x!r})", "fig, ax = plt.subplots()"]
        if chart == "line":
            lines += [
                f"ax.plot(raw[{x!r}], raw[{metric!r}], marker='o', label={metric!r})"
                for metric in metrics
            ]
            lines.append("ax.legend()")
        else:
            lines.append(f"ax.scatter(raw[{x!r}], raw[{metrics[0]!r}])")
        lines += [f"ax.set_title({title!r})", f"ax.set_xlabel({x!r})"]
    elif chart == "bar" and group:
        lines += [
            f"ax = g.set_index({group!r})[{metrics!r}].plot(kind='bar')",
            f"ax.set_title({title!r})",
            "plt.xticks(rotation=40, ha='right')",
        ]
    elif chart == "line" and x:
        lines += ["fig, ax = plt.subplots()"]
        lines += [
            f"ax.plot(g[{x!r}], g[{metric!r}], marker='o', label={metric!r})" for metric in metrics
        ]
        lines += [f"ax.set_title({title!r})", "ax.legend()"]
    elif chart == "scatter" and x:
        lines += [
            f"ax = g.plot.scatter(x={x!r}, y={metrics[0]!r})",
            f"ax.set_title({title!r})",
        ]
    elif chart == "pie":
        target = f"g.set_index({group!r})[{metrics[0]!r}]" if group else f"g[{metrics[0]!r}]"
        lines += [
            f"ax = {target}.plot(kind='pie', autopct='%1.1f%%')",
            f"ax.set_title({title!r})",
            "ax.set_ylabel('')",
        ]
    elif chart == "hist":
        lines += [
            "fig, ax = plt.subplots()",
            f"ax.hist(d[{metrics[0]!r}].dropna(), bins=20)",
            f"ax.set_title({title!r})",
        ]
    elif chart == "candlestick":
        if all(col in columns for col in OHLC):
            date_col = spec.x_field if spec.x_field in columns else "Date_dt"
            lines.append(
                f"kline(d, {date_col!r}, 'Open_num', 'High_num', 'Low_num', 'Close_num', {title!r})"
            )
        else:
            lines.append("print('数据里没有开高低收四列，跳过 K 线')")

    lines.append("result = g.to_dict('records')[:20]")
    return "\n".join(lines)


def column_roles(frame: Any) -> dict[str, list[str]]:
    """把列按用途分组，供前端下拉框。"""
    import pandas as pd

    roles: dict[str, list[str]] = {"numeric": [], "datetime": [], "categorical": []}
    for column in frame.columns:
        series = frame[column]
        name = str(column)
        if pd.api.types.is_numeric_dtype(series):
            roles["numeric"].append(name)
        elif pd.api.types.is_datetime64_any_dtype(series):
            roles["datetime"].append(name)
        else:
            roles["categorical"].append(name)
    return roles
