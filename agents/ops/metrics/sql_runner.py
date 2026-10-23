"""执行分析 SQL：只读事务 + 租户上下文 + 超时 + 行数上限。

三件必须做的事：
1. `set_config('app.tenant_id', ..., true)` —— 让 ana_* 视图知道"现在是谁在查"，
   且 is_local=true 只在当前事务有效，不会污染连接池
2. `SET TRANSACTION READ ONLY` —— 就算守卫漏了，数据库也不允许写
3. 超时 —— 模型可能写出笛卡尔积，别把库拖死
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

STATEMENT_TIMEOUT_MS = 8000


@dataclass
class QueryResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str = ""

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def as_table(self, limit: int = 20) -> str:
        """渲染成 markdown 表格，方便直接塞进提示词。"""
        if not self.columns:
            return "（无数据）"
        head = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join(str(row.get(c, "")) for c in self.columns) + " |"
            for row in self.rows[:limit]
        ]
        return "\n".join([head, sep, *body])


async def run_sql(sql: str, tenant_id: str) -> QueryResult:
    """在只读事务里执行，返回结果或错误。"""
    started = time.perf_counter()
    conn = await lease_connection()
    try:
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            await conn.execute(
                "SELECT set_config('statement_timeout', $1, true)",
                str(STATEMENT_TIMEOUT_MS),
            )
            records = await conn.fetch(sql)
    except Exception as exc:  # 语法错误 / 超时 / 权限
        logger.warning("analyst.sql_failed", error=str(exc)[:200])
        return QueryResult(
            ok=False,
            error=str(exc),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    finally:
        await conn.close()

    columns = list(records[0].keys()) if records else []
    rows = [dict(r) for r in records]
    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info("analyst.sql_ok", rows=len(rows), ms=elapsed)
    return QueryResult(ok=True, columns=columns, rows=rows, elapsed_ms=elapsed)
