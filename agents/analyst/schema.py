"""数据层：面向分析的租户隔离视图。

为什么要建视图，而不是让模型自己写 WHERE tenant_id：
    模型忘写一次就是跨租户泄露，这种错误不能靠提示词防。
    这里把租户过滤下沉到视图定义里，用 current_setting('app.tenant_id') 取值：
        设置了租户 → 只看本租户
        没设置     → current_setting 返回 NULL → 条件恒假 → 零行（Fail-Closed）

    查询时由 runner 在同一个事务里 `SET LOCAL app.tenant_id = ...`，
    事务结束自动失效，不会污染连接池里的连接。

表与租户的关系（原始表并不都带 tenant_id）：
    users / faq_entries / crawl_tasks            直接有 tenant_id
    qa_sessions / qa_messages / qa_feedback       没有，要 join users
    crawl_pages / crawl_records                   没有，要 join crawl_tasks
"""

from __future__ import annotations

from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

VIEWS_SQL = """
-- 采集任务
CREATE OR REPLACE VIEW ana_crawl_task AS
SELECT t.id AS task_id, t.tenant_id, t.user_id, t.goal, t.entry_url, t.status,
       t.pages_fetched, t.records_extracted, t.strategy, t.stop_reason,
       t.started_at, t.finished_at
FROM crawl_tasks t
WHERE t.tenant_id = current_setting('app.tenant_id', true);

-- 采集到的页面
CREATE OR REPLACE VIEW ana_crawl_page AS
SELECT p.id AS page_id, p.task_id, t.tenant_id, p.url, p.title, p.status,
       p.mode, p.char_count, p.fetched_at
FROM crawl_pages p
JOIN crawl_tasks t ON t.id = p.task_id
WHERE t.tenant_id = current_setting('app.tenant_id', true);

-- 结构化记录（data 是 JSON 文本，用 ->> 取字段）
CREATE OR REPLACE VIEW ana_crawl_record AS
SELECT r.id AS record_id, r.task_id, t.tenant_id, r.page_url, r.strategy,
       r.data, r.created_at
FROM crawl_records r
JOIN crawl_tasks t ON t.id = r.task_id
WHERE t.tenant_id = current_setting('app.tenant_id', true);

-- 问答轮次（消息 + 所属会话）
CREATE OR REPLACE VIEW ana_qa_turn AS
SELECT m.id AS message_id, m.session_id, u.tenant_id, u.username, u.department,
       m.role, m.answer_mode, m.created_at
FROM qa_messages m
JOIN users u ON u.id::text = m.user_id
WHERE u.tenant_id = current_setting('app.tenant_id', true);

-- 会话
CREATE OR REPLACE VIEW ana_qa_session AS
SELECT s.session_id, u.tenant_id, u.username, u.department, s.title,
       s.created_at, s.updated_at
FROM qa_sessions s
JOIN users u ON u.id::text = s.user_id
WHERE u.tenant_id = current_setting('app.tenant_id', true);

-- 用户反馈
CREATE OR REPLACE VIEW ana_qa_feedback AS
SELECT f.id AS feedback_id, f.session_id, u.tenant_id, u.username, u.department,
       f.rating, f.comment, f.created_at
FROM qa_feedback f
JOIN users u ON u.id::text = f.user_id
WHERE u.tenant_id = current_setting('app.tenant_id', true);

-- FAQ 条目
CREATE OR REPLACE VIEW ana_faq_entry AS
SELECT f.id AS faq_id, f.tenant_id, f.department, f.subject, f.question,
       f.created_at
FROM faq_entries f
WHERE f.tenant_id = current_setting('app.tenant_id', true);

-- 用户
CREATE OR REPLACE VIEW ana_user AS
SELECT u.id AS user_id, u.tenant_id, u.username, u.role, u.department,
       u.is_active, u.created_at
FROM users u
WHERE u.tenant_id = current_setting('app.tenant_id', true);
"""

# 视图名 → 中文说明 + 主要字段（给模型看的"数据字典"）
VIEW_CATALOG: dict[str, dict[str, object]] = {
    "ana_crawl_task": {
        "desc": "网页采集任务（一个任务=一次抓取）",
        "columns": [
            "task_id",
            "goal",
            "entry_url",
            "status",
            "pages_fetched",
            "records_extracted",
            "strategy",
            "stop_reason",
            "started_at",
        ],
    },
    "ana_crawl_page": {
        "desc": "采集到的页面（正文元信息）",
        "columns": [
            "page_id",
            "task_id",
            "url",
            "title",
            "status",
            "mode",
            "char_count",
            "fetched_at",
        ],
    },
    "ana_crawl_record": {
        "desc": "采集抽取出的结构化记录；data 是 JSON 文本，用 data->>'字段名' 取值",
        "columns": ["record_id", "task_id", "page_url", "strategy", "data", "created_at"],
    },
    "ana_qa_turn": {
        "desc": "问答消息（role=user/assistant，answer_mode=faq/rag/llm_direct/web）",
        "columns": [
            "message_id",
            "session_id",
            "username",
            "department",
            "role",
            "answer_mode",
            "created_at",
        ],
    },
    "ana_qa_session": {
        "desc": "问答会话",
        "columns": ["session_id", "username", "department", "title", "created_at", "updated_at"],
    },
    "ana_qa_feedback": {
        "desc": "用户反馈（rating=1 点赞 / -1 点踩）",
        "columns": [
            "feedback_id",
            "session_id",
            "username",
            "department",
            "rating",
            "comment",
            "created_at",
        ],
    },
    "ana_faq_entry": {
        "desc": "FAQ 条目（department='*' 表示全公司可见）",
        "columns": ["faq_id", "department", "subject", "question", "created_at"],
    },
    "ana_user": {
        "desc": "账号（role=user/admin）",
        "columns": ["user_id", "username", "role", "department", "is_active", "created_at"],
    },
}

# 指标字典：常见问题的标准口径。模型可以照着写 SQL，避免每次口径不一样。
METRIC_CATALOG: dict[str, dict[str, str]] = {
    "会话数": {
        "sql": "SELECT count(*) AS 会话数 FROM ana_qa_session",
        "note": "按会话去重，不是消息数",
    },
    "消息数": {
        "sql": "SELECT count(*) AS 消息数 FROM ana_qa_turn WHERE role = 'user'",
        "note": "只数用户提问，不含回答",
    },
    "FAQ 命中率": {
        "sql": (
            "SELECT round(100.0 * count(*) FILTER (WHERE answer_mode = 'faq') "
            '/ greatest(count(*), 1), 1) AS "FAQ命中率" '
            "FROM ana_qa_turn WHERE role = 'assistant'"
        ),
        "note": "分母是助手回答数；零除用 greatest 兜住",
    },
    "点踩数": {
        "sql": "SELECT count(*) AS 点踩数 FROM ana_qa_feedback WHERE rating = -1",
        "note": "rating=-1",
    },
    "采集任务数": {
        "sql": "SELECT count(*) AS 采集任务数 FROM ana_crawl_task",
        "note": "",
    },
    "抓取记录数": {
        "sql": "SELECT count(*) AS 抓取记录数 FROM ana_crawl_record",
        "note": "",
    },
}


def catalog_prompt() -> str:
    """把视图与指标拼成给模型的 schema 说明。"""
    lines: list[str] = ["可用的分析视图（已经按当前租户过滤，不要自己加 tenant_id 条件）："]
    for name, meta in VIEW_CATALOG.items():
        cols = ", ".join(str(c) for c in meta["columns"])  # type: ignore[union-attr]
        lines.append(f"- {name}：{meta['desc']}")
        lines.append(f"    字段：{cols}")
    lines.append("")
    lines.append("常用指标口径（优先复用，保证口径一致）：")
    for name, meta in METRIC_CATALOG.items():
        note = f"（{meta['note']}）" if meta["note"] else ""
        lines.append(f"- {name}{note}：{meta['sql']}")
    return "\n".join(lines)


async def ensure_views() -> int:
    """建/更新所有分析视图，返回视图数量。"""
    conn = await lease_connection()
    try:
        await conn.execute(VIEWS_SQL)
    finally:
        await conn.close()
    logger.info("analyst.views_ready", count=len(VIEW_CATALOG))
    return len(VIEW_CATALOG)
