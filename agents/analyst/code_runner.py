"""在子进程里跑 pandas 代码，带超时与输出收集。"""

from __future__ import annotations

import asyncio
import base64
import json
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS = 25

# 出现在代码里就直接拒绝（子进程 + 受限内置已经是兜底，这里是第一道闸）
FORBIDDEN_SNIPPETS = (
    "import ",
    "__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "input(",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "requests",
)


@dataclass
class CodeResult:
    ok: bool
    stdout: str = ""
    figures: list[str] = field(default_factory=list)  # base64 PNG
    result: str = ""
    error: str = ""
    elapsed_ms: int = 0


# 这些模块沙箱里已经注入好了，模型再写 import 也无害，直接剥掉
def sanitize_code(code: str) -> tuple[str, str]:
    """剥掉白名单 import，返回 (处理后的代码, 拒绝原因)。"""
    raw = code or ""
    if not raw.strip():
        return "", "代码为空"

    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if re.match(
            r"^(import|from)\s+(pandas|numpy|matplotlib|matplotlib\.pyplot|math|"
            r"statistics|datetime|re|json)\b",
            stripped,
            re.I,
        ):
            continue  # 沙箱已注入，丢掉这一行
        lines.append(line)
    cleaned = "\n".join(lines).strip()

    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in cleaned:
            return "", f"代码里不允许出现 {snippet!r}（沙箱只做计算和绘图）"
    if len(cleaned) > 6000:
        return "", "代码过长（上限 6000 字符）"
    return cleaned, ""


def screen_code(code: str) -> str:
    """静态检查，返回拒绝原因（空串表示通过）。"""
    _cleaned, reason = sanitize_code(code)
    return reason


async def run_code(
    records: list[dict] | None = None,
    code: str = "",
    frame: object | None = None,
) -> CodeResult:
    """执行一段 pandas 代码，返回 stdout / 图片 / 结果。

    两种入参：
        frame   —— 预处理好的 DataFrame（推荐）。用 pickle 传给沙箱，
                   这样 _num / _dt 的 dtype 完整保留。
        records —— 原始记录列表，沙箱内自己 pd.DataFrame()。

    为什么优先用 frame：
        预处理已经把 "1,234.50" 转成浮点、把 "Oct 22, 2026" 转成时间，
        只传原始记录等于把预处理白做了——模型在沙箱里看到的仍是字符串。
    """
    started = time.perf_counter()
    cleaned, reason = sanitize_code(code)
    if reason:
        return CodeResult(ok=False, error=reason, elapsed_ms=0)
    code = cleaned

    payload_obj: dict[str, object] = {"code": code}
    if frame is not None:
        # pickle 的是我们自己构造的 DataFrame，不含外部输入，反序列化风险可控
        payload_obj["frame_b64"] = base64.b64encode(pickle.dumps(frame)).decode("ascii")
    else:
        payload_obj["records"] = records or []
    payload = json.dumps(payload_obj, ensure_ascii=False)
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "agents.analyst._sandbox",
            cwd=str(PROJECT_ROOT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload.encode("utf-8")),
                timeout=TIMEOUT_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("analyst.code_timeout", seconds=TIMEOUT_SECONDS)
            return CodeResult(
                ok=False,
                error=f"代码执行超过 {TIMEOUT_SECONDS} 秒，已终止（可能死循环或数据太大）",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
    except Exception as exc:
        logger.error("analyst.code_spawn_failed", error=str(exc))
        return CodeResult(ok=False, error=f"沙箱启动失败：{exc}")

    elapsed = int((time.perf_counter() - started) * 1000)
    if process.returncode != 0:
        detail = (stderr or b"").decode("utf-8", "ignore")[-800:]
        logger.warning("analyst.code_crashed", code=process.returncode)
        return CodeResult(ok=False, error=f"沙箱异常退出：{detail}", elapsed_ms=elapsed)

    try:
        data = json.loads((stdout or b"").decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        return CodeResult(
            ok=False,
            error="沙箱输出无法解析",
            elapsed_ms=elapsed,
        )

    logger.info(
        "analyst.code_ok",
        ok=data.get("ok"),
        figures=len(data.get("figures") or []),
        ms=elapsed,
    )
    return CodeResult(
        ok=bool(data.get("ok")),
        stdout=str(data.get("stdout") or ""),
        figures=list(data.get("figures") or [])[:4],
        result=str(data.get("result") or ""),
        error=str(data.get("error") or ""),
        elapsed_ms=elapsed,
    )
