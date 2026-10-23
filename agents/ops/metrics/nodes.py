"""系统指标问答（原 analyst 的 A 路径，现归运维）。

职责：回答"系统自己跑得怎么样"——会话量、命中率、点踩、采集量。
    模型写 SQL → 守卫校验 → 只读执行 → 解读。
    视图 ana_* 已经按租户过滤，模型不需要（也不允许）自己加 tenant 条件。

和 analyst 的分界：
    metrics  看系统内部记录（本模块）
    analyst  看外部抓回来的数据（pandas + 绘图）
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.ops.metrics.sql_guard import describe_rules, validate
from agents.ops.metrics.sql_runner import run_sql
from agents.ops.metrics.state import MetricState
from agents.ops.metrics.views import catalog_prompt
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

# 固定用 SQL 路径（这里只做指标查询，不需要三选一）
SQL_SYSTEM = (
    "你是 PostgreSQL 分析师，只写 SQL，不要解释。"
    "只能查询给定视图，必须明确列出需要的列，不要 SELECT *。"
)
SQL_USER = """{catalog}

【问题】
{question}

【要求】
1. 只输出一条 SELECT 或 WITH 查询，不要分号，不要注释。
2. 只能查询上面的 ana_* 视图（已按当前租户过滤，不要再加 tenant_id 条件）。
3. 中文别名用双引号，例如 count(*) AS "会话数"。
4. 需要聚合就聚合，别把明细全拉出来；结果控制在 200 行内。

只输出 SQL。"""

REPAIR_USER = """刚才这条 SQL 执行失败：

【SQL】
{sql}

【报错】
{error}

请修正后重新输出一条 SQL（只输出 SQL 本身）。"""

SUMMARY_SYSTEM = """你是业务分析助手。基于给定的查询结果作答。

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


class SqlPlan(BaseModel):
    """占位：当前不需要路由，保留给将来扩展（比如多步指标）。"""

    reason: str = Field(default="")


def _table(columns: list[str], rows: list[dict], limit: int = 20) -> str:
    head = "| " + " | ".join(map(str, columns)) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows[:limit]]
    return "\n".join([head, sep, *body])


def _question_from_messages(state: MetricState) -> str:
    """state.question 没传时，从最后一条用户消息里取。"""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


async def metric_node(state: MetricState) -> dict[str, Any]:
    """生成 SQL → 守卫 → 只读执行（失败自愈一次）→ 解读。"""
    question = str(state.get("question") or "").strip() or _question_from_messages(state)
    tenant_id = str(state.get("tenant_id") or "tenant_default")
    sql = str(state.get("sql") or "")
    error = str(state.get("sql_error") or "")
    retried = bool(state.get("sql_retried"))

    if not question:
        return {
            "answer": "请说明想看的指标，例如：一共有多少会话。",
            "answer_mode": "empty",
            "structured_output": {"answer_mode": "empty"},
        }

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
        draft = str(response.content).strip()
    except Exception as exc:
        return {
            "answer": f"生成 SQL 失败：{exc}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error"},
        }

    draft = draft.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    guard = validate(draft)
    if not guard.ok:
        logger.info("metrics.sql_rejected", reason=guard.reason)
        return {
            "sql": draft,
            "sql_error": f"守卫拒绝：{guard.reason}",
            "sql_retried": True,
            "answer": (
                f"这条 SQL 没通过安全校验：{guard.reason}\n\n允许的规则：\n{describe_rules()}"
            ),
            "answer_mode": "blocked",
            "structured_output": {"answer_mode": "blocked", "reason": guard.reason},
        }

    result = await run_sql(guard.sql, tenant_id)
    if not result.ok and not retried:
        return {"sql": guard.sql, "sql_error": result.error, "sql_retried": True}

    if not result.ok:
        return {
            "sql": guard.sql,
            "answer": f"SQL 执行失败：{result.error}",
            "answer_mode": "error",
            "structured_output": {"answer_mode": "error", "sql": guard.sql},
        }

    if not result.rows:
        answer = "没有查到数据。可能是这个时间范围内没有记录，或者条件太窄。"
        return {
            "sql": guard.sql,
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "answer_mode": "sql",
            "structured_output": {
                "answer_mode": "sql",
                "sql": guard.sql,
                "row_count": 0,
                "columns": result.columns,
                "rows": [],
                "elapsed_ms": result.elapsed_ms,
            },
        }

    table = _table(result.columns, result.rows)
    summarize = LLMFactory.get_llm("analyst", temperature=0.2, streaming=True)
    try:
        response = await summarize.ainvoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(
                    content=SUMMARY_USER.format(
                        question=question, sql=guard.sql, count=len(result.rows), table=table
                    )
                ),
            ]
        )
        answer = str(response.content).strip()
    except Exception as exc:
        answer = f"（模型总结失败：{exc}）\n\n原始结果：\n{table}"

    logger.info("metrics.answered", question=question[:40], rows=len(result.rows))
    return {
        "sql": guard.sql,
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "answer_mode": "sql",
        "structured_output": {
            "answer_mode": "sql",
            "sql": guard.sql,
            "row_count": len(result.rows),
            "columns": result.columns,
            "rows": result.rows[:50],
            "elapsed_ms": result.elapsed_ms,
        },
    }
