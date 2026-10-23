"""数据分析 Agent 的节点：只看"抓回来的数据"，用 pandas 算、用 matplotlib 画。

不再包含"查系统指标"的 SQL 路径——那部分已归运维（agents/ops/metrics）。

流程：
    dispatch →（问题里有 URL）→ chain（先抓）→ explore → code → narrative
             →（没有 URL）──────────────→ explore → code → narrative
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.analyst.code_runner import run_code
from agents.analyst.frame import build_frame, frame_overview
from agents.analyst.state import AnalystState
from agents.research.mcp_client import get_mcp_client, parse_mcp_dict, parse_mcp_payload
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

CODE_SYSTEM = """你是数据分析工程师。给我一段 pandas 代码，不要解释。

环境里已准备好：
    df          由采集数据构建的 DataFrame（已做基础预处理）
    pd / np     pandas / numpy
    plt         matplotlib.pyplot（无界面后端）
    math / statistics / datetime / timedelta / re / json

硬性要求：
1. 只做数据分析与绘图：不要读写文件、不要联网、不要写 import。
2. 需要输出数值时用 print()，或赋给变量 result（会被读取）。
3. 需要画图就用 plt；一张图即可，标题用英文（避免中文字体缺失）。
4. 原始字段是字符串！数值列用 <列名>_num，时间列用 <列名>_dt，
   直接对原始列做算术或 corr 会得到 nan。
5. 先做必要清洗（类型转换、去空、排序），再计算。
6. 只输出代码，不要 markdown 代码块。"""

CODE_USER = """【数据概况】
{overview}

【分析目标】
{question}

请写出完成这个分析的 pandas 代码。"""

CODE_REPAIR_USER = """上一步的代码执行失败。

【代码】
{code}

【报错】
{error}

请修正后重新输出一段完整可运行的代码（只输出代码）。"""

SUMMARY_SYSTEM = """你是数据分析助手。基于给定的分析过程作答。

要求：
1. 先给结论（数字/排名/趋势），再给一句解释。
2. 只依据给到的数据，不要编造。
3. 数据为空就直说"没有可用数据"，并给出可能原因。
4. 用中文，简洁。"""

SUMMARY_USER = """【问题】
{question}

【分析过程与结果】
{context}

请作答。"""


def _extract_question(state: AnalystState) -> str:
    question = state.get("question")
    if question:
        return str(question)
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def detect_question_url(text: str) -> str:
    """从问题里抽 URL；抽不到返回空串。

    单独写一份而不是复用 supervisor：analyst 只关心"要不要先抓"，
    不需要认识 crawler 的关键词表。
    """
    import re

    match = re.search(r"https?://[^\s，。；、）)\"']+", text or "")
    return match.group(0).rstrip(".,;") if match else ""


async def dispatch_node(state: AnalystState) -> dict[str, Any]:
    """问题里带 URL 就先抓再分析，否则直接分析已有数据。"""
    question = _extract_question(state)
    url = detect_question_url(question)
    logger.info("analyst.dispatch", question=question[:40], has_url=bool(url))
    return {"question": question, "entry_url": url, "answer_mode": "chain" if url else "explore"}


def route_after_dispatch(state: AnalystState) -> str:
    return "chain" if state.get("entry_url") else "explore"


async def _latest_task(tenant_id: str) -> int:
    """没有指定任务时，取最近一个已完成任务。"""
    client = await get_mcp_client("ops")
    tools = {t.name: t for t in await client.get_tools()}
    raw = await tools["list_tasks"].ainvoke({"limit": 10})
    tasks = parse_mcp_payload(raw)
    for task in tasks:
        if task.get("status") == "done":
            return int(task.get("id") or 0)
    return int(tasks[0].get("id") or 0) if tasks else 0


def field_distribution(records: list[dict], top: int = 8) -> list[dict]:
    """统计每个字段的取值分布（不依赖 LLM 的兜底手段）。"""
    counters: dict[str, dict[str, int]] = {}
    for record in records:
        data = record.get("data") if isinstance(record, dict) else None
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            text = str(value).strip()
            if not text:
                continue
            bucket = counters.setdefault(key, {})
            bucket[text] = bucket.get(text, 0) + 1

    stats: list[dict] = []
    for key, bucket in counters.items():
        items = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)[:top]
        stats.append(
            {
                "field": key,
                "distinct": len(bucket),
                "top": [{"value": v, "count": c} for v, c in items],
            }
        )
    stats.sort(key=lambda item: item["distinct"])
    return stats


async def explore_node(state: AnalystState) -> dict[str, Any]:
    """用 MCP 工具取采集记录。"""
    tenant = str(state.get("tenant_id") or "tenant_default")
    task_id = int(state.get("task_id") or 0)
    try:
        if not task_id:
            task_id = await _latest_task(tenant)
        if not task_id:
            return {
                "answer": "还没有任何采集任务，先让爬虫抓一轮数据再分析。",
                "answer_mode": "no_data",
                "structured_output": {"answer_mode": "no_data"},
            }

        client = await get_mcp_client("ops")
        tools = {t.name: t for t in await client.get_tools()}
        raw = await tools["task_records"].ainvoke(
            {"task_id": task_id, "limit": 500, "tenant_id": tenant}
        )
        payload = parse_mcp_dict(raw)
        if not payload.get("ok"):
            return {
                "answer": f"读取任务 #{task_id} 的记录失败：{payload.get('error', '未知原因')}",
                "answer_mode": "error",
                "structured_output": {"answer_mode": "error"},
            }
        items = [{"data": item} for item in payload.get("items") or []]
    except Exception as exc:
        logger.warning("analyst.explore_failed", error=str(exc))
        return {
            "answer": f"读取采集数据失败：{exc}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error"},
        }

    logger.info("analyst.explored", task=task_id, records=len(items))
    return {"task_id": task_id, "records": items}


async def code_node(state: AnalystState) -> dict[str, Any]:
    """让模型写 pandas 代码，在沙箱里跑，拿结果和图片。"""
    question = _extract_question(state)
    records = state.get("records") or []
    code = str(state.get("code") or "")
    error = str(state.get("code_error") or "")

    if not records:
        return {"code_finished": True}

    frame, report = build_frame(records)
    overview = frame_overview(frame)

    llm = LLMFactory.get_llm("analyst", temperature=0)
    prompt = (
        CODE_REPAIR_USER.format(code=code, error=error)
        if error and code
        else CODE_USER.format(overview=overview, question=question)
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=CODE_SYSTEM), HumanMessage(content=prompt)]
        )
        draft = str(response.content).strip()
    except Exception as exc:
        return {"code_error": f"生成代码失败：{exc}", "code_finished": True}

    draft = draft.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    # 直接把预处理好的 frame 交给沙箱：
    # 转换过的 _num / _dt 列在沙箱里仍然可用，模型不用再从字符串转一遍
    result = await run_code(code=draft, frame=frame)

    if not result.ok and not state.get("code_retried"):
        logger.info("analyst.code_retry", error=result.error[:80])
        return {
            "code": draft,
            "code_error": result.error,
            "code_retried": True,
            "code_finished": False,
            "prep": report.as_dict(),
        }

    return {
        "code": draft,
        "code_error": "" if result.ok else result.error,
        "code_stdout": result.stdout,
        "code_result": result.result,
        "figures": result.figures,
        "code_ms": result.elapsed_ms,
        "prep": report.as_dict(),
        "code_retried": True,
        "code_finished": True,
    }


async def narrative_node(state: AnalystState) -> dict[str, Any]:
    """汇总代码执行结果，讲成人话。"""
    question = _extract_question(state)
    task_id = int(state.get("task_id") or 0)
    records = state.get("records") or []
    stdout = str(state.get("code_stdout") or "")
    code_result = str(state.get("code_result") or "")
    code_error = str(state.get("code_error") or "")
    figures = state.get("figures") or []
    code = str(state.get("code") or "")
    prep = state.get("prep") or {}

    base_output = {
        "answer_mode": "pandas",
        "task_id": task_id,
        "row_count": len(records),
        "prep": prep,
        "code": code,
        "stdout": stdout[:2000],
        "figures": figures,
        "chart_count": len(figures),
    }

    if not records:
        answer = f"任务 #{task_id} 没有可分析的记录。"
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "structured_output": {**base_output, "row_count": 0},
        }

    if code_error and not stdout and not figures:
        answer = (
            f"分析代码没能跑通：{code_error}\n\n"
            "可以换个问法，或者先去爬虫分区确认这个任务抽到的字段里有没有你要的数据。"
        )
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "structured_output": {**base_output, "error": code_error},
        }

    detail = [f"数据：任务 #{task_id}，{len(records)} 条记录"]
    if prep.get("numeric"):
        detail.append(f"转成数值的列：{'、'.join(prep['numeric'])}")
    if prep.get("datetime_columns"):
        detail.append(f"转成时间的列：{'、'.join(prep['datetime_columns'])}")
    if stdout.strip():
        detail.append(f"程序输出：\n{stdout.strip()[:1200]}")
    if code_result:
        detail.append(f"计算结果：{code_result}")
    if figures:
        detail.append(f"已生成 {len(figures)} 张图（见下方）")

    context = "\n".join(detail)
    llm = LLMFactory.get_llm("analyst", temperature=0.2, streaming=True)
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(content=SUMMARY_USER.format(question=question, context=context)),
            ]
        )
        answer = str(response.content).strip()
    except Exception as exc:
        answer = f"（模型总结失败：{exc}）\n\n{context}"

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "structured_output": base_output,
    }


async def chain_node(state: AnalystState) -> dict[str, Any]:
    """先让爬虫抓一轮，再回到 explore 分析。"""
    question = _extract_question(state)
    tenant = str(state.get("tenant_id") or "tenant_default")
    entry_url = str(state.get("entry_url") or detect_question_url(question))

    if not entry_url:
        return {
            "answer": "要「先抓再分析」，我需要一个入口网址。",
            "answer_mode": "need_url",
            "structured_output": {"answer_mode": "need_url"},
        }

    try:
        client = await get_mcp_client("ops")
        tools = {t.name: t for t in await client.get_tools()}
        raw = await tools["crawl"].ainvoke(
            {"entry_url": entry_url, "goal": question, "max_pages": 3, "tenant_id": tenant}
        )
        result = parse_mcp_dict(raw)
    except Exception as exc:
        return {
            "answer": f"抓取失败：{exc}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error"},
        }

    task_id = int(result.get("task_id") or 0)
    if not result.get("ok") or not task_id:
        return {
            "answer": (
                f"抓取没有产出数据：{result.get('message') or result.get('error') or '未知原因'}"
            ),
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error"},
        }

    logger.info("analyst.chained", url=entry_url, task=task_id)
    return {
        "task_id": task_id,
        "crawl_summary": str(result.get("message") or ""),
    }
