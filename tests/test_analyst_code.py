"""pandas 代码分析与绘图的测试。"""

from __future__ import annotations

import pytest

from agents.analyst.code_runner import run_code, sanitize_code
from agents.analyst.frame import PrepReport, _to_number, build_frame


# ── 预处理 ────────────────────────────────────────────────────
def test_to_number_handles_formats():
    assert _to_number("12.5") == 12.5
    assert _to_number("1,234.50") == 1234.5
    assert _to_number("3.2%") == 3.2
    assert _to_number("￥88") == 88.0
    assert _to_number("abc") is None
    assert _to_number("") is None


def test_build_frame_coerces_numeric_and_date():
    records = [
        {"data": {"date": "2026-10-01", "close": "12.5", "vol": "1000"}},
        {"data": {"date": "2026-10-02", "close": "13.1", "vol": "1500"}},
        {"data": {"date": "2026-10-03", "close": "12.8", "vol": "900"}},
    ]
    frame, report = build_frame(records)
    assert "close_num" in frame.columns
    assert "date_dt" in frame.columns
    assert isinstance(report, PrepReport)
    assert report.rows == 3
    assert "close_num" in report.numeric


def test_build_frame_dedupes_and_drops_empty():
    records = [
        {"data": {"a": "1", "empty": ""}},
        {"data": {"a": "1", "empty": ""}},
        {"data": {"a": "2", "empty": ""}},
    ]
    frame, report = build_frame(records)
    assert report.deduped == 1
    assert "empty" in report.dropped_empty
    assert len(frame) == 2


def test_build_frame_keeps_plain_dicts():
    """records 直接是字段字典（不带 data 包一层）也要能处理。"""
    frame, report = build_frame([{"x": "1"}, {"x": "2"}])
    assert report.rows == 2
    assert "x_num" in frame.columns


# ── 代码清洗 ──────────────────────────────────────────────────
def test_sanitize_strips_whitelisted_imports():
    code = "import pandas as pd\nimport matplotlib.pyplot as plt\ndf.head()"
    cleaned, reason = sanitize_code(code)
    assert reason == ""
    assert "import" not in cleaned
    assert "df.head()" in cleaned


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndf.head()",
        "open('/etc/passwd')",
        "df.__class__",
        "import subprocess",
        "eval('1+1')",
        "",
    ],
)
def test_sanitize_rejects_dangerous_code(code):
    _cleaned, reason = sanitize_code(code)
    assert reason


# ── 沙箱执行 ──────────────────────────────────────────────────
async def test_run_code_computes_and_plots():
    records = [
        {"close": "12.5"},
        {"close": "13.1"},
        {"close": "12.8"},
    ]
    code = """
d = df.copy()
d['close'] = d['close'].astype(float)
d['ma2'] = d['close'].rolling(2).mean()
print(round(float(d['close'].mean()), 2))
plt.figure(figsize=(3, 2))
plt.plot(d['close'])
result = len(d)
"""
    result = await run_code(records, code)
    assert result.ok is True
    assert "12.8" in result.stdout
    assert result.result == "3"
    assert len(result.figures) == 1
    assert result.figures[0].startswith("iVBORw0KGgo")  # PNG magic


async def test_run_code_reports_python_error():
    result = await run_code([{"a": "1"}], "df['missing'].mean()")
    assert result.ok is False
    assert "KeyError" in result.error


async def test_run_code_blocks_import():
    result = await run_code([{"a": "1"}], "import os\nprint(os.listdir('.'))")
    assert result.ok is False
    assert "不允许" in result.error
