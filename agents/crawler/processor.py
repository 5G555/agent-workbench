"""处理 Agent：独立于采集的一步，读结构化记录产出最终结果。

为什么单独做而不是塞进采集流程：
    采集是"多抓多得"的重复劳动，处理是"读一批数据做一次计算"。
    两者的触发时机、耗时、失败重试策略都不同——抓取失败了要重抓，
    处理失败了只要重跑处理即可，不该把整个抓取任务重来。

职责（按架构图）：清洗 → 聚合 → schema 校验 → 语义分析 → 输出

可以当库调用（API 里），也可以当独立任务跑：
    python -m scripts.process_crawl --task 12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.crawler.store import get_crawl_store
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

SUMMARIZE_SYSTEM = """你是一个数据分析助手。给你从网页抽取的结构化记录，请输出结论。

要求：
1. 只依据记录里的内容，不要补充外部知识。
2. 先给结论，再给支撑（可以引用记录中的具体字段值）。
3. 如果记录质量差（字段缺失、内容重复、明显是噪声），先指出问题再给结论。
4. 用中文，简洁，不要复述全部记录。"""

SUMMARIZE_USER = """【处理规则】
{rules}

【记录（共 {total} 条，展示前 {shown} 条）】
{records}

请按规则输出结论。"""


class ProcessResult(BaseModel):
    """处理 Agent 的产出。"""

    total: int = 0
    kept: int = 0
    dropped: int = 0
    fields: list[str] = field(default_factory=list)
    summary: str = ""
    issues: list[str] = Field(default_factory=list)


@dataclass
class _Stats:
    """清洗阶段的统计。"""

    total: int = 0
    kept: int = 0
    dropped: int = 0
    fields: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def clean_records(
    records: list[dict[str, Any]], min_field_len: int = 2
) -> tuple[list[dict[str, Any]], _Stats]:
    """清洗：去空值、去完全重复、去字段过短的记录。"""
    stats = _Stats(total=len(records))
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    fields: set[str] = set()

    for record in records:
        data = record.get("data") or {}
        cleaned = {
            str(k).strip(): str(v).strip()
            for k, v in data.items()
            if v is not None and len(str(v).strip()) >= min_field_len
        }
        if not cleaned:
            stats.dropped += 1
            continue
        signature = "|".join(f"{k}={v}" for k, v in sorted(cleaned.items()))
        if signature in seen:
            stats.dropped += 1
            continue
        seen.add(signature)
        fields.update(cleaned.keys())
        kept.append({"page_url": record.get("page_url", ""), "data": cleaned})

    stats.kept = len(kept)
    stats.fields = sorted(fields)
    if stats.total and stats.dropped / stats.total > 0.3:
        stats.issues.append(f"丢弃比例偏高（{stats.dropped}/{stats.total}），检查抽取策略")
    if not stats.fields:
        stats.issues.append("没有抽到任何字段，可能需要调整抽取规则")
    return kept, stats


def validate_schema(records: list[dict[str, Any]], required: list[str]) -> list[str]:
    """schema 校验：缺字段的记录占比过高就报警。"""
    if not required or not records:
        return []
    missing = sum(1 for r in records if any(f not in r["data"] for f in required))
    if missing == 0:
        return []
    ratio = missing / len(records)
    if ratio > 0.5:
        return [f"超过一半记录缺少必需字段 {required}（{missing}/{len(records)}）"]
    return []


def aggregate(
    records: list[dict[str, Any]], field_name: str, top: int = 10
) -> list[tuple[str, int]]:
    """按某个字段做频次聚合，返回前 N 项。"""
    counts: dict[str, int] = {}
    for record in records:
        value = str(record["data"].get(field_name, "")).strip()
        if value:
            counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top]


async def process_task(
    task_id: int,
    rules: str = "",
    required_fields: list[str] | None = None,
    max_records: int = 200,
) -> ProcessResult:
    """读某任务的结构化记录 → 清洗 → 校验 → 语义分析 → 结果。"""
    store = get_crawl_store()
    raw = await store.list_records(task_id, limit=max_records)
    if not raw:
        return ProcessResult(total=0, summary="该任务没有结构化记录，先确认抓取是否成功。")

    records, stats = clean_records(raw)
    issues = list(stats.issues)
    issues.extend(validate_schema(records, required_fields or []))

    if not records:
        return ProcessResult(
            total=stats.total,
            dropped=stats.dropped,
            issues=issues,
            summary="记录清洗后为空，无法分析。",
        )

    shown = records[:40]
    listing = "\n".join(
        f"{i}. {', '.join(f'{k}: {v}' for k, v in r['data'].items())}"
        for i, r in enumerate(shown, 1)
    )
    summary = ""
    try:
        llm = LLMFactory.get_llm("crawler", temperature=0.2, streaming=True)
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUMMARIZE_SYSTEM),
                HumanMessage(
                    content=SUMMARIZE_USER.format(
                        rules=rules or "对记录做归纳，指出主要类别与异常",
                        total=len(records),
                        shown=len(shown),
                        records=listing,
                    )
                ),
            ]
        )
        summary = str(response.content).strip()
    except Exception as exc:
        logger.error("crawler.process_llm_failed", error=str(exc))
        summary = f"（模型分析失败：{exc}）原始统计：{len(records)} 条记录，字段 {stats.fields}"

    logger.info(
        "crawler.processed",
        task=task_id,
        total=stats.total,
        kept=stats.kept,
        dropped=stats.dropped,
    )
    return ProcessResult(
        total=stats.total,
        kept=stats.kept,
        dropped=stats.dropped,
        fields=stats.fields,
        summary=summary,
        issues=issues,
    )
