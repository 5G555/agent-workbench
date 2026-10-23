"""数据分析 Agent 的节点：只看"抓回来的数据"，用 pandas 算、用 matplotlib 画。

不再包含"查系统指标"的 SQL 路径——那部分已归运维（agents/ops/metrics）。

流程：
    dispatch →（问题里有 URL）→ chain（先抓）→ explore → code → narrative
             →（没有 URL）──────────────→ explore → code → narrative
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.analyst.charts import chart_hint, detect_chart_type
from agents.analyst.code_runner import run_code
from agents.analyst.frame import build_frame, frame_overview
from agents.analyst.result_cache import cache_key, data_fingerprint, get_cached, put_cached
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
   样式（配色/字号/网格/尺寸）已经统一设好，不要自己 set figsize 或颜色，
   除非某张图确实需要特殊处理（比如双轴图的右轴标签）。
4. 原始字段是字符串！数值列用 <列名>_num，时间列用 <列名>_dt，
   直接对原始列做算术或 corr 会得到 nan。
5. 先做必要清洗（类型转换、去空、排序），再计算。
6. 如果数据里有一列 task_id，说明合并了多个采集任务，
   对比时按它分组（groupby），图例用「任务 <编号>」。
7. 只输出代码，不要 markdown 代码块。"""

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
    chart = detect_chart_type(question)
    logger.info(
        "analyst.dispatch",
        question=question[:40],
        has_url=bool(url),
        chart=chart or "-",
    )
    return {
        "question": question,
        "entry_url": url,
        "chart_type": chart,
        "answer_mode": "chain" if url else "explore",
    }


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


async def cache_check_node(state: AnalystState) -> dict[str, Any]:
    """命中缓存就直接返回，省掉 4~8 秒的重新分析。"""
    question = _extract_question(state)
    tenant = str(state.get("tenant_id") or "tenant_default")
    chart = str(state.get("chart_type") or "")

    # 带 URL 的是"先抓再分析"，数据还没抓，缓存没意义
    if state.get("entry_url"):
        return {"cached": False}

    try:
        task_ids = await _resolve_task_ids(state, tenant)
    except Exception as exc:
        logger.warning("analyst.cache_resolve_failed", error=str(exc)[:120])
        return {"cached": False}

    fingerprint = await data_fingerprint(task_ids)
    if not fingerprint:
        return {"cached": False, "task_ids": task_ids}

    key = cache_key(tenant, task_ids, question, fingerprint, chart)
    payload = await get_cached(key)
    if payload:
        return {
            "cached": True,
            "cache_key": key,
            "task_id": int(payload.get("task_id") or (task_ids[0] if task_ids else 0)),
            "tasks": [int(t) for t in (payload.get("tasks") or task_ids)],
            "answer": str(payload.get("answer") or ""),
            "answer_mode": "pandas",
            "structured_output": {
                **(payload.get("structured_output") or {}),
                "cached": True,
                "chart_type": chart,
            },
        }
    return {"cached": False, "cache_key": key, "task_ids": task_ids}


def route_after_cache(state: AnalystState) -> str:
    """命中缓存直接结束；否则按有没有 URL 决定先抓还是直接分析。"""
    if state.get("cached"):
        return "done"
    return "chain" if state.get("entry_url") else "explore"


async def _resolve_task_ids(state: AnalystState, tenant: str) -> list[int]:
    """决定分析哪些任务：显式多任务 > 单个任务 > 最近一个已完成任务。"""
    explicit = [int(t) for t in (state.get("task_ids") or []) if int(t) > 0]
    if explicit:
        return explicit[:5]  # 最多 5 个，防止一次拉爆
    single = int(state.get("task_id") or 0)
    if single:
        return [single]
    latest = await _latest_task(tenant)
    return [latest] if latest else []


async def explore_node(state: AnalystState) -> dict[str, Any]:
    """用 MCP 工具取采集记录；多个任务时合并成一张表（带 task_id 列）。"""
    tenant = str(state.get("tenant_id") or "tenant_default")
    tool = None
    try:
        client = await get_mcp_client("ops")
        tool = {t.name: t for t in await client.get_tools()}["task_records"]
        task_ids = await _resolve_task_ids(state, tenant)
        if not task_ids:
            return {
                "answer": "还没有任何采集任务，先让爬虫抓一轮数据再分析。",
                "answer_mode": "no_data",
                "structured_output": {"answer_mode": "no_data"},
            }

        items: list[dict] = []
        used: list[int] = []
        for task_id in task_ids:
            raw = await tool.ainvoke({"task_id": task_id, "limit": 500, "tenant_id": tenant})
            payload = parse_mcp_dict(raw)
            if not payload.get("ok"):
                logger.warning(
                    "analyst.task_read_failed",
                    task=task_id,
                    error=str(payload.get("error"))[:120],
                )
                continue
            for item in payload.get("items") or []:
                if not isinstance(item, dict):
                    continue
                # 多任务时把 task_id 塞进记录本身，
                # 这样预处理后它就是一列，模型可以 groupby("task_id")
                items.append({"data": {**item, "task_id": task_id}})
            used.append(task_id)

        if not items:
            return {
                "answer": f"任务 {task_ids} 都没有可分析的记录。",
                "answer_mode": "no_data",
                "structured_output": {"answer_mode": "no_data", "tasks": task_ids},
            }
    except Exception as exc:
        logger.warning("analyst.explore_failed", error=str(exc))
        return {
            "answer": f"读取采集数据失败：{exc}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error"},
        }

    logger.info("analyst.explored", tasks=used, records=len(items))
    return {
        "task_id": used[0],
        "tasks": used,
        "records": items,
    }


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

    chart = str(state.get("chart_type") or "")
    hint = chart_hint(chart)
    target = f"{question}\n\n{hint}" if hint else question

    llm = LLMFactory.get_llm("analyst", temperature=0)
    prompt = (
        CODE_REPAIR_USER.format(code=code, error=error)
        if error and code
        else CODE_USER.format(overview=overview, question=target)
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
            "frame_overview": overview,
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
        "frame_overview": overview,
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

    tasks = [int(t) for t in (state.get("tasks") or [])]
    base_output = {
        "answer_mode": "pandas",
        "task_id": task_id,
        "tasks": tasks,
        "row_count": len(records),
        "prep": prep,
        "overview": str(state.get("frame_overview") or ""),
        "code": code,
        "stdout": stdout[:2000],
        "figures": figures,
        "chart_count": len(figures),
        "chart_type": str(state.get("chart_type") or ""),
        "cached": False,
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

    # 写缓存：key 用实际分析到的任务重新算（chain 路径抓完才知道任务号）
    if not state.get("entry_url"):
        tenant = str(state.get("tenant_id") or "tenant_default")
        tasks = [int(t) for t in (state.get("tasks") or [])]
        fingerprint = await data_fingerprint(tasks)
        if fingerprint:
            key = cache_key(
                tenant,
                tasks,
                question,
                fingerprint,
                str(state.get("chart_type") or ""),
            )
            await put_cached(
                key,
                {
                    "answer": answer,
                    "task_id": task_id,
                    "tasks": tasks,
                    "structured_output": base_output,
                },
            )

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
