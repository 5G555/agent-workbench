"""数据分析 Agent 的节点：路由 → 取数/分析 → 汇总。

三条路径（对应架构图）：
    A  Text2SQL        精确取数：会话数、满意度、命中率这类"要一个确定数字"的问题
    B  MCP 分析        探索统计：已经采集到的记录里有什么规律（用 MCP 工具取数）
    C  先抓再分析      没有现成数据时，先让爬虫抓一轮，再走 B
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.analyst.schema import catalog_prompt
from agents.analyst.sql_guard import describe_rules, validate
from agents.analyst.sql_runner import run_sql
from agents.analyst.state import PATH_CHAIN, PATH_EXPLORE, PATH_SQL, AnalystState
from agents.research.mcp_client import get_mcp_client, parse_mcp_dict, parse_mcp_payload
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

ROUTE_SYSTEM = "你是数据分析的调度器，只输出 JSON。"
ROUTE_USER = """把用户的分析请求分配到一条路径。

A（Text2SQL，精确取数）：问的是本系统里已有的定量事实，例如
   "这个月有多少会话"、"FAQ 命中率多少"、"哪个部门点踩最多"、"采集了多少条记录"
B（MCP 分析，探索统计）：问的是"已经采集到的网页数据里"的规律，例如
   "抓到的数据里作者分布如何"、"这些记录里最常见的标签是什么"
C（先抓再分析）：库里还没有相关数据，需要先抓取再分析，例如
   "去看看竞品官网的价格再帮我总结"、"抓一下这个站然后统计标题关键词"

用户请求：{question}

只返回 JSON：{{"path": "A" 或 "B" 或 "C", "reason": "一句话理由"}}"""

SQL_SYSTEM = (
    "你是 PostgreSQL 分析师，只写 SQL，不要解释。"
    "只能查询下列视图，必须明确列出需要的列，不要 SELECT *。"
)
SQL_USER = """{catalog}

【问题】
{question}

【要求】
1. 只输出一条 SELECT 或 WITH 查询，不要分号，不要注释。
2. 只能查询上面的 ana_* 视图（它们已经按当前租户过滤，不要再加 tenant_id 条件）。
3. 中文别名用双引号，例如 count(*) AS "会话数"。
4. 需要聚合就聚合，别把明细全拉出来；结果控制在 200 行内。

只输出 SQL。"""

REPAIR_USER = """刚才这条 SQL 执行失败：

【SQL】
{sql}

【报错】
{error}

请修正后重新输出一条 SQL（只输出 SQL 本身）。"""

SUMMARY_SYSTEM = """你是数据分析助手。基于给定的查询结果作答。

要求：
1. 先给结论（数字/排名），再给一句解释。
2. 只依据给到的数据，不要编造。
3. 数据为空就直说"没有查到数据"，并给出可能原因。
4. 用中文，简洁。"""

SUMMARY_USER = """【问题】
{question}

【执行的 SQL】
{sql}

【结果（{count} 行）】
{table}

请作答。"""


class RoutePlan(BaseModel):
    """路由结果。"""

    path: str = Field(default=PATH_SQL)
    reason: str = ""


def _extract_question(state: AnalystState) -> str:
    question = state.get("question")
    if question:
        return str(question)
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def route_node(state: AnalystState) -> dict[str, Any]:
    """① 决定走哪条路径。"""
    question = _extract_question(state)
    if not question:
        return {"path": PATH_SQL, "path_reason": "空问题，默认精确取数"}

    try:
        llm = LLMFactory.get_structured_llm("analyst", RoutePlan)
        plan = await llm.ainvoke(
            [
                SystemMessage(content=ROUTE_SYSTEM),
                HumanMessage(content=ROUTE_USER.format(question=question)),
            ]
        )
        path = plan.path if plan.path in {PATH_SQL, PATH_EXPLORE, PATH_CHAIN} else PATH_SQL
        reason = plan.reason
    except Exception as exc:
        logger.warning("analyst.route_failed", error=str(exc))
        path, reason = PATH_SQL, "路由失败，默认精确取数"

    logger.info("analyst.routed", question=question[:40], path=path, reason=reason[:60])
    return {"question": question, "path": path, "path_reason": reason}


async def sql_node(state: AnalystState) -> dict[str, Any]:
    """②A 生成 SQL → 守卫校验 → 执行（失败自愈一次）。"""
    question = _extract_question(state)
    sql = str(state.get("sql") or "")
    error = str(state.get("sql_error") or "")
    retried = bool(state.get("sql_retried"))

    # 第一次生成 / 或者拿着报错重写
    llm = LLMFactory.get_llm("analyst", temperature=0)
    prompt = (
        REPAIR_USER.format(sql=sql, error=error)
        if error
        else SQL_USER.format(catalog=catalog_prompt(), question=question)
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=SQL_SYSTEM), HumanMessage(content=prompt)]
        )
        raw_sql = str(response.content).strip()
    except Exception as exc:
        return {
            "answer": f"生成 SQL 失败：{exc}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error"},
        }

    guard = validate(raw_sql)
    if not guard.ok:
        logger.info("analyst.sql_rejected", reason=guard.reason)
        return {
            "sql": raw_sql,
            "sql_error": f"守卫拒绝：{guard.reason}",
            "sql_retried": True,
            "answer": (
                f"这条 SQL 没通过安全校验：{guard.reason}\n\n允许的规则：\n{describe_rules()}"
            ),
            "answer_mode": "blocked",
            "structured_output": {"answer_mode": "blocked", "reason": guard.reason},
        }

    result = await run_sql(guard.sql, str(state.get("tenant_id") or "tenant_default"))
    if not result.ok and not retried:
        # 自愈一次：把报错丢回给模型改
        return {"sql": guard.sql, "sql_error": result.error, "sql_retried": True}

    if not result.ok:
        return {
            "sql": guard.sql,
            "answer": f"SQL 执行失败：{result.error}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error", "sql": guard.sql},
        }

    return {
        "sql": guard.sql,
        "sql_columns": result.columns,
        "sql_rows": result.rows,
        "sql_error": "",
        "answer_mode": "sql",
        "structured_output": {
            "answer_mode": "sql",
            "sql": guard.sql,
            "row_count": result.row_count,
            "elapsed_ms": result.elapsed_ms,
        },
    }


async def describe_node(state: AnalystState) -> dict[str, Any]:
    """③A 把查询结果讲成人话。"""
    question = _extract_question(state)
    sql = str(state.get("sql") or "")
    rows = state.get("sql_rows") or []
    columns = state.get("sql_columns") or []

    if not rows:
        answer = "没有查到数据。可能是这个时间范围内没有记录，或者条件太窄。"
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "structured_output": {"answer_mode": "sql", "sql": sql, "row_count": 0},
        }

    table = "| " + " | ".join(map(str, columns)) + " |\n"
    table += "| " + " | ".join("---" for _ in columns) + " |\n"
    for row in rows[:20]:
        table += "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n"

    llm = LLMFactory.get_llm("analyst", temperature=0.2, streaming=True)
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(
                    content=SUMMARY_USER.format(
                        question=question, sql=sql, count=len(rows), table=table
                    )
                ),
            ]
        )
        answer = str(response.content).strip()
    except Exception as exc:
        answer = f"（模型总结失败：{exc}）\n\n原始结果：\n{table}"

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "structured_output": {
            "answer_mode": "sql",
            "sql": sql,
            "row_count": len(rows),
            "columns": columns,
        },
    }


# ── 路径 B：MCP 分析 ───────────────────────────────────────────
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
    """统计每条记录的每个字段有多少种取值、最常见的几个是什么。

    纯函数，方便单测：这是"探索统计"的核心，不依赖 LLM。
    """
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
    """②B 用 MCP 工具取采集记录，做字段分布统计。"""
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
            {"task_id": task_id, "limit": 200, "tenant_id": tenant}
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

    stats = field_distribution(items)
    logger.info("analyst.explored", task=task_id, records=len(items), fields=len(stats))
    return {
        "task_id": task_id,
        "records": items,
        "field_stats": stats,
        "answer_mode": "explore",
        "structured_output": {
            "answer_mode": "explore",
            "task_id": task_id,
            "records": len(items),
            "fields": [s["field"] for s in stats],
        },
    }


async def narrative_node(state: AnalystState) -> dict[str, Any]:
    """③B 把统计结果讲成人话。"""
    question = _extract_question(state)
    task_id = int(state.get("task_id") or 0)
    stats = state.get("field_stats") or []
    records = state.get("records") or []

    if not records:
        answer = f"任务 #{task_id} 没有可分析的记录。"
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "structured_output": {"answer_mode": "explore", "row_count": 0},
        }

    lines = [f"任务 #{task_id}，共 {len(records)} 条记录，字段分布如下："]
    for item in stats[:6]:
        top = "、".join(f"{t['value']}({t['count']})" for t in item["top"][:5])
        lines.append(f"- {item['field']}：{item['distinct']} 种取值，最多的是 {top}")
    detail = "\n".join(lines)

    llm = LLMFactory.get_llm("analyst", temperature=0.2, streaming=True)
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(content=f"【问题】{question}\n\n【统计】\n{detail}"),
            ]
        )
        answer = str(response.content).strip()
    except Exception:
        answer = detail

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "structured_output": {
            "answer_mode": "explore",
            "task_id": task_id,
            "row_count": len(records),
            "fields": [s["field"] for s in stats],
        },
    }


# ── 路径 C：先抓再分析 ────────────────────────────────────────
async def chain_node(state: AnalystState) -> dict[str, Any]:
    """②C 让爬虫先抓一轮，再回到 B 分析。"""
    question = _extract_question(state)
    tenant = str(state.get("tenant_id") or "tenant_default")

    from agents.supervisor.router import detect_url

    entry_url = detect_url(question)
    if not entry_url:
        return {
            "answer": (
                "要「先抓再分析」，我需要一个入口网址。"
                "例如：抓取 https://example.com/list 然后统计标题关键词"
            ),
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
        "answer_mode": "chain",
    }
