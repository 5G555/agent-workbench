"""数据沙箱：在独立子进程里执行 pandas 代码。

为什么必须用子进程，而不是 exec 在当前进程里跑：
    模型写的代码可能死循环、可能吃光内存、可能抛段错误。
    进程内 exec 一旦卡住就只能杀掉整个 Web 服务。
    子进程可以超时强杀，崩溃也不影响主进程。

为什么还限制内置函数：
    子进程只是"能杀"，不代表安全。这里再收窄一层：
    不提供 open / __import__ / eval / exec / globals 等，
    只给计算和绘图需要的东西（pd / np / plt / math / datetime / re）。

这个沙箱面向"本机演示"，不是生产级隔离：
    真正的隔离要用容器 + 只读文件系统 + 无网络。
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import sys
from typing import Any

ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "True": True,
    "False": False,
    "None": None,
}


# 统一的图表样式：颜色、字号、网格、线宽
# 为什么在沙箱里强制：图表样式不该由模型每次随手决定，
# 否则同一个面板里两张图配色/字号完全不同，看起来像两个人做的。
STYLE_PRESET = {
    "figure.figsize": (8.6, 4.6),
    "figure.dpi": 110,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "axes.labelcolor": "#475569",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.9,
    "lines.markersize": 4.2,
    "font.size": 10,
}

COLOR_CYCLE = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c", "#319795"]


def build_kline(frame: Any, plt: Any, np: Any) -> Any:
    """返回一个 kline(df, date_col, open_col, high_col, low_col, close_col) 函数。

    matplotlib 没有蜡烛图，这里用「影线 + 实体」手画：
        影线 = 当日的最高/最低价（细竖线）
        实体 = 开盘到收盘（矩形），涨用红、跌用绿（国内习惯）
    放在沙箱里而不是让模型每次现写，是为了保证每张 K 线长一个样。
    """

    def kline(
        df: Any,
        date_col: str = "Date_dt",
        open_col: str = "Open_num",
        high_col: str = "High_num",
        low_col: str = "Low_num",
        close_col: str = "Close_num",
        title: str = "Candlestick",
    ) -> Any:
        cols = [date_col, open_col, high_col, low_col, close_col]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"缺少列 {missing}，可用列：{list(df.columns)}")

        data = df.dropna(subset=[open_col, high_col, low_col, close_col]).copy()
        data = data.sort_values(date_col)
        if data.empty:
            raise ValueError("没有可用的 OHLC 数据（开高低收有缺失）")

        x = np.arange(len(data))
        opens = data[open_col].to_numpy(dtype=float)
        highs = data[high_col].to_numpy(dtype=float)
        lows = data[low_col].to_numpy(dtype=float)
        closes = data[close_col].to_numpy(dtype=float)

        _figure, axis = plt.subplots()
        axis.vlines(x, lows, highs, color="#94a3b8", linewidth=0.9)
        rising = closes >= opens
        colors = np.where(rising, "#e11d48", "#059669")  # 红涨绿跌
        axis.bar(
            x,
            np.abs(closes - opens) + (highs.max() - lows.min()) * 0.0008,
            bottom=np.minimum(opens, closes),
            width=0.62,
            color=colors,
            edgecolor=colors,
        )

        step = max(1, len(x) // 8)
        ticks = x[::step]
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            [str(v)[:10] for v in data[date_col].iloc[::step]], rotation=40, ha="right"
        )
        axis.set_title(title)
        axis.set_ylabel("Price")
        return axis

    return kline


def main() -> None:
    import matplotlib
    from cycler import cycler

    matplotlib.use("Agg")  # 无界面后端，才能在没有显示器的环境出图
    import matplotlib.pyplot as plt

    # 注入统一样式（模型自己设的 figsize / 颜色仍会覆盖，但不设时就是这套）
    plt.rcParams.update(STYLE_PRESET)  # type: ignore[arg-type]
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)
    import numpy as np
    import pandas as pd

    payload = json.loads(sys.stdin.read())
    code = str(payload.get("code") or "")

    if payload.get("frame_b64"):
        # 父进程预处理好的 DataFrame：dtype（含 _num / _dt）完整保留
        frame = pickle.loads(base64.b64decode(payload["frame_b64"]))  # noqa: S301
    else:
        frame = pd.DataFrame(payload.get("records") or [])
    namespace = {
        "__builtins__": ALLOWED_BUILTINS,
        "df": frame,
        "data": frame,
        "pd": pd,
        "np": np,
        "plt": plt,
        "math": __import__("math"),
        "statistics": __import__("statistics"),
        "datetime": __import__("datetime").datetime,
        "timedelta": __import__("datetime").timedelta,
        "re": __import__("re"),
        "json": json,
        "kline": build_kline(frame, plt, np),
    }

    buffer = io.StringIO()
    payload_out: dict = {"ok": True, "stdout": "", "figures": [], "error": "", "result": ""}
    try:
        import contextlib

        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<analyst>", "exec"), namespace)  # noqa: S102
        result = namespace.get("result")
        if result is not None:
            payload_out["result"] = str(result)[:2000]
    except Exception as exc:  # 用户代码的错，原样带回
        payload_out["ok"] = False
        payload_out["error"] = f"{type(exc).__name__}: {exc}"

    # 收集图片（每张 figure 一张 PNG）
    try:
        for number in plt.get_fignums():
            figure = plt.figure(number)
            raw = io.BytesIO()
            figure.savefig(raw, format="png", dpi=110, bbox_inches="tight")
            payload_out["figures"].append(base64.b64encode(raw.getvalue()).decode())
        plt.close("all")
    except Exception as exc:
        payload_out["figures"] = []
        payload_out["error"] = (payload_out["error"] or "") + f" 绘图失败：{exc}"

    payload_out["stdout"] = buffer.getvalue()[:4000]
    sys.stdout.write(json.dumps(payload_out, ensure_ascii=False))


if __name__ == "__main__":
    main()
