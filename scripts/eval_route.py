"""路由语义匹配度评测。

用法：
    python -m scripts.eval_route              # 全部
    python -m scripts.eval_route --verbose    # 逐条打印

输出：
    总体匹配度、分类目匹配度、混淆矩阵、错误清单。

为什么要有这个集：
    改路由规则很容易"修一个坏两个"（这轮就发生了三次：
    "北京今天气温多少度"被拉进 analyst、"K 线怎么画"被图表词拉进 analyst、
    "帮我画个流程图"被私有词"流程"拉进 qa）。没有标注集就只能靠感觉。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from agents.supervisor.router import route

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "route_eval.jsonl"


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


async def evaluate(verbose: bool = False) -> int:
    cases = load_cases()
    by_cat_total: Counter[str] = Counter()
    by_cat_hit: Counter[str] = Counter()
    confusion: dict[tuple[str, str]] = defaultdict(int)
    misses: list[tuple[dict, str, str]] = []

    for case in cases:
        resolved = await route(case["q"])
        actual = resolved.agent
        expect = case["expect"]
        by_cat_total[case["cat"]] += 1
        hit = actual == expect
        if hit:
            by_cat_hit[case["cat"]] += 1
        else:
            misses.append((case, actual, resolved.reason))
            confusion[(expect, actual)] += 1
        if verbose:
            flag = "✓" if hit else "✗"
            print(f"{flag} [{case['cat']:6}] {case['q'][:30]:32} → {actual:9} (期望 {expect})")

    total = len(cases)
    hit_total = sum(by_cat_hit.values())
    print()
    print(f"总体匹配度：{hit_total}/{total} = {hit_total / total * 100:.1f}%")
    print()
    print(f"{'类目':10} {'命中':>6} {'匹配度':>8}")
    for cat in by_cat_total:
        rate = by_cat_hit[cat] / by_cat_total[cat] * 100
        print(f"{cat:10} {by_cat_hit[cat]:>3}/{by_cat_total[cat]:<3} {rate:>7.1f}%")

    if misses:
        print()
        print("未命中：")
        for case, actual, reason in misses:
            print(f"  [{case['cat']}] {case['q']}")
            print(f"      期望 {case['expect']} → 实际 {actual}（{reason}）")

    if confusion:
        print()
        print("混淆（期望 → 实际）：")
        for (expect, actual), count in sorted(confusion.items(), key=lambda kv: -kv[1]):
            print(f"  {expect:9} → {actual:9} × {count}")

    return hit_total


def main() -> None:
    parser = argparse.ArgumentParser(description="路由语义匹配度评测")
    parser.add_argument("--verbose", "-v", action="store_true", help="逐条打印")
    args = parser.parse_args()
    asyncio.run(evaluate(verbose=args.verbose))


if __name__ == "__main__":
    main()
