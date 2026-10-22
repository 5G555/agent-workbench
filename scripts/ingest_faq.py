"""把 FAQ 数据导入 Postgres 的 faq_entries 表。

python -m scripts.ingest_faq                   # 用 data/test_faq.csv
python -m scripts.ingest_faq <csv路径>
python -m scripts.ingest_faq --query "你们怎么收费"   # 只测 BM25 命中
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

from agents.qa.cache import bump_cache_version
from agents.qa.faq import FaqStore, get_faq_store

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "test_faq.csv"


def read_csv(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if question and answer:
                rows.append(((row.get("subject") or "").strip(), question, answer))
    return rows


async def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--query":
        query = args[1] if len(args) > 1 else "你们怎么收费"
        store = await get_faq_store()
        hit = await store.search(query)
        if hit:
            print(f"命中！score={hit.score:.3f}  question={hit.question}")
            print(f"答案: {hit.answer}")
        else:
            print("未命中（分数低于阈值，会转走向量检索）")
        return

    path = Path(args[0]) if args and not args[0].startswith("-") else CSV_PATH
    if not path.exists():
        raise SystemExit(f"CSV 不存在: {path}")

    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"{path} 里没有有效数据")

    store = FaqStore()
    await store.ensure_table()
    await store.upsert_many(rows)
    total = await store.reload()
    print(f"导入 {len(rows)} 条，表中现有 {total} 条，BM25 索引已重建")

    # FAQ 变了，旧答案缓存必须失效
    version = await bump_cache_version()
    print(f"缓存版本号已提升到 {version}，旧缓存失效")


if __name__ == "__main__":
    asyncio.run(main())
