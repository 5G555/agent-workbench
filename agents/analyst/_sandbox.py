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


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # 无界面后端，才能在没有显示器的环境出图
    import matplotlib.pyplot as plt
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
