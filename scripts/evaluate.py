"""检索与回答质量评估。

跑一遍固定的评估集，输出四个指标：
    命中方式   answer_mode 是否符合预期（faq / rag / direct）
    来源命中   回答引用的来源里是否包含期望文档
    关键词     回答里是否出现期望的关键词（验证"答对了内容"而不只是"引对了文档"）
    耗时       端到端时延

用法：
    python -m scripts.evaluate              # 跑 data/eval_set.jsonl
    python -m scripts.evaluate --limit 5    # 只跑前 5 条
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

# 评估必须关掉缓存，否则第二次跑会全部命中缓存，指标失真
os.environ["CACHE_ENABLED"] = "false"

from langchain_core.messages import HumanMessage

from agents.qa.graph import build_qa_graph

EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_set.jsonl"


def load_cases(limit: int | None = None) -> list[dict]:
    if not EVAL_FILE.exists():
        raise SystemExit(f"评估集不存在：{EVAL_FILE}")
    cases = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[:limit] if limit else cases


async def run_case(app, case: dict, index: int) -> dict:
    question = case["question"]
    started = time.perf_counter()
    result: dict = {}
    async for chunk in app.astream(
        {
            "messages": [HumanMessage(content=question)],
            "user_id": "eval",
            "session_id": f"eval{index}",
        },
        {"configurable": {"thread_id": f"eval-{index}-{time.time_ns()}"}},
        stream_mode="updates",
    ):
        for _, update in chunk.items():
            if isinstance(update, dict) and update.get("answer"):
                result = update
    elapsed = time.perf_counter() - started

    mode = str(result.get("answer_mode") or "")
    answer = str(result.get("answer") or "")
    structured = result.get("structured_output") or {}
    sources = [str(s) for s in (result.get("sources") or structured.get("sources") or [])]

    expect_mode = case.get("expect_mode")
    mode_ok = (mode == expect_mode) if expect_mode else True

    expect_sources = case.get("expect_sources") or []
    if not expect_sources:
        source_ok = None
    elif mode == "faq":
        # FAQ 是人工整理的问答对，命中即认为答对了，不参与文档来源比对
        source_ok = True
    else:
        source_ok = any(exp in " ".join(sources) for exp in expect_sources)

    keywords = case.get("expect_keywords") or []
    keyword_ok = any(k in answer for k in keywords) if keywords else None

    return {
        "question": question,
        "mode": mode,
        "mode_ok": mode_ok,
        "sources": sources,
        "source_ok": source_ok,
        "keyword_ok": keyword_ok,
        "latency": elapsed,
        "answer": answer,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.limit)
    app = build_qa_graph()

    print(f"评估集 {len(cases)} 条，缓存已关闭\n")
    rows = []
    for i, case in enumerate(cases, 1):
        row = await run_case(app, case, i)
        rows.append(row)
        marks = []
        marks.append("模式✓" if row["mode_ok"] else f"模式✗({row['mode']})")
        if row["source_ok"] is not None:
            marks.append("来源✓" if row["source_ok"] else "来源✗")
        if row["keyword_ok"] is not None:
            marks.append("关键词✓" if row["keyword_ok"] else "关键词✗")
        print(f"[{i:02d}] {' '.join(marks):32} {row['latency']:5.2f}s  {row['question']}")
        if not row["mode_ok"] or row["source_ok"] is False or row["keyword_ok"] is False:
            print(f"      答: {row['answer'][:80]}")
            print(f"      源: {row['sources']}")

    def ratio(values: list[bool]) -> str:
        if not values:
            return "  -  "
        return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})"

    mode_r = ratio([r["mode_ok"] for r in rows])
    source_r = ratio([r["source_ok"] for r in rows if r["source_ok"] is not None])
    keyword_r = ratio([r["keyword_ok"] for r in rows if r["keyword_ok"] is not None])
    avg_latency = sum(r["latency"] for r in rows) / max(1, len(rows))

    print(f"\n{'=' * 60}")
    print(f"命中方式准确率 : {mode_r}")
    print(f"来源命中率     : {source_r}")
    print(f"关键词命中率   : {keyword_r}")
    print(f"平均耗时       : {avg_latency:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
