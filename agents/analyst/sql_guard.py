"""SQL 守卫：只放行"对分析视图的只读查询"。

模型写的 SQL 属于不可信输入，这里做四件事（全部是确定性规则，不看模型怎么说）：

1. 语句类型：只允许单条 SELECT / WITH；出现 ; 多语句、DDL、DML 一律拒绝
2. 表白名单：只能查 ana_* 视图。写 base 表（users / crawl_pages ...）一律拒绝，
   因为视图已经强制了租户隔离，绕过视图就等于绕过隔离
3. 危险函数：pg_read_file、pg_sleep、COPY、lo_import 等禁用
4. 兜底限流：没有 LIMIT 就补一个；禁止 SELECT *

租户隔离不靠这里——靠视图里的 current_setting('app.tenant_id')，
所以即使模型漏写条件也查不到别家数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_VIEW_PREFIX = "ana_"
MAX_ROWS = 200

# 出现在 SQL 里就直接拒绝的关键字（按词边界匹配，避免误伤列名）
FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "vacuum",
    "analyze",
    "call",
    "do",
)
FORBIDDEN_FUNCTIONS = (
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_sleep",
    "lo_import",
    "lo_export",
    "dblink",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "set_config",
)

CTE_PATTERN = re.compile(r"(?:\bwith\b|,)\s*([a-zA-Z_][\w]*)\s+as\s*\(", re.I)

IDENT_PATTERN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)
KEYWORD_PATTERN = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.I)
FUNCTION_PATTERN = re.compile(r"\b(" + "|".join(FORBIDDEN_FUNCTIONS) + r")\s*\(", re.I)


@dataclass
class GuardResult:
    ok: bool
    sql: str = ""
    reason: str = ""


def _strip_comments(sql: str) -> str:
    """去掉注释——避免用 /* */ 或 -- 藏关键字。"""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def cte_names(sql: str) -> set[str]:
    """WITH 子句里定义的临时表名——它们不是真实表，不该被白名单拦下。"""
    return {m.group(1).lower() for m in CTE_PATTERN.finditer(sql)}


def referenced_views(sql: str) -> list[str]:
    """抽出 FROM / JOIN 后面的表名。"""
    return [match.group(1).lower().strip('"') for match in IDENT_PATTERN.finditer(sql)]


def validate(sql: str) -> GuardResult:
    """校验并规范化一条分析 SQL。"""
    raw = (sql or "").strip()
    if not raw:
        return GuardResult(ok=False, reason="SQL 为空")

    cleaned = _strip_comments(raw).strip()
    # 去掉结尾分号（单语句允许带一个）
    cleaned = cleaned.rstrip(";").strip()

    if ";" in cleaned:
        return GuardResult(ok=False, reason="只允许单条语句（检测到多条）")

    if not re.match(r"^(select|with)\b", cleaned, re.I):
        return GuardResult(ok=False, reason="只允许 SELECT / WITH 查询")

    keyword = KEYWORD_PATTERN.search(cleaned)
    if keyword:
        return GuardResult(ok=False, reason=f"包含禁止的关键字：{keyword.group(1)}")

    function = FUNCTION_PATTERN.search(cleaned)
    if function:
        return GuardResult(ok=False, reason=f"包含禁止的函数：{function.group(1)}")

    views = referenced_views(cleaned)
    if not views:
        return GuardResult(ok=False, reason="没有识别到 FROM 子句")
    ctes = cte_names(cleaned)
    for name in views:
        if name in ctes:
            continue  # CTE 别名，允许
        if not name.startswith(ALLOWED_VIEW_PREFIX):
            return GuardResult(
                ok=False,
                reason=f"只能查询分析视图（ana_*），不允许直接查 {name}",
            )

    if re.search(r"select\s+\*", cleaned, re.I):
        return GuardResult(ok=False, reason="禁止 SELECT *，请明确列出需要的列")

    # 兜底加 LIMIT，防止一次拉全表
    if not re.search(r"\blimit\s+\d+", cleaned, re.I):
        # SQL 已过白名单与关键字校验，这里只包一层限流
        cleaned = f"SELECT * FROM ({cleaned}) AS _guard_wrap LIMIT {MAX_ROWS}"  # noqa: S608
    return GuardResult(ok=True, sql=cleaned)


def describe_rules() -> str:
    """给用户的说明（失败时回显）。"""
    return (
        "• 只能查询 ana_* 分析视图\n"
        "• 只允许单条 SELECT / WITH\n"
        "• 禁止 SELECT *、DDL/DML、系统函数\n"
        f"• 默认最多返回 {MAX_ROWS} 行"
    )
