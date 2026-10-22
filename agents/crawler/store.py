"""爬取存储层：任务日志 / 原始页面 / 结构化记录，三张表分开。

为什么拆三张而不是一张大表：
    crawl_tasks    一次抓取任务的元信息与结果统计（编排 Agent 写、接口读）
    crawl_pages    原始页面快照（排查"为什么抽错了"时要能回看原文）
    crawl_records  结构化记录（处理 Agent 的唯一输入）

    三者的读写方不同、生命周期也不同：页面可以定期清理，记录要长期留作分析，
    任务日志用于审计。合在一起反而互相牵制。

记录用 TEXT 存 JSON 而不是 JSONB：
    asyncpg 对 jsonb 需要注册编解码器，多一处初始化就容易忘；
    现在的量级（单任务几十到几百条）用文本完全够，Python 侧解析即可。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config import Settings, get_settings
from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

CREATE_TABLES_SQL = """
-- 任务日志
CREATE TABLE IF NOT EXISTS crawl_tasks (
    id                 BIGSERIAL PRIMARY KEY,
    tenant_id          VARCHAR(64)  NOT NULL DEFAULT 'tenant_default',
    user_id            VARCHAR(64)  NOT NULL DEFAULT '',
    goal               VARCHAR(1024) NOT NULL DEFAULT '',
    entry_url          VARCHAR(1024) NOT NULL DEFAULT '',
    status             VARCHAR(16)  NOT NULL DEFAULT 'running',
    pages_fetched      INTEGER      NOT NULL DEFAULT 0,
    records_extracted  INTEGER      NOT NULL DEFAULT 0,
    strategy           VARCHAR(32)  NOT NULL DEFAULT '',
    stop_reason        VARCHAR(64)  NOT NULL DEFAULT '',
    error              TEXT         NOT NULL DEFAULT '',
    started_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finished_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_crawl_tasks_tenant ON crawl_tasks (tenant_id, id DESC);

-- 原始页面
CREATE TABLE IF NOT EXISTS crawl_pages (
    id          BIGSERIAL PRIMARY KEY,
    task_id     BIGINT        NOT NULL REFERENCES crawl_tasks(id) ON DELETE CASCADE,
    url         VARCHAR(1024) NOT NULL,
    final_url   VARCHAR(1024) NOT NULL DEFAULT '',
    title       VARCHAR(512)  NOT NULL DEFAULT '',
    status      INTEGER       NOT NULL DEFAULT 0,
    mode        VARCHAR(16)   NOT NULL DEFAULT 'http',
    char_count  INTEGER       NOT NULL DEFAULT 0,
    content     TEXT          NOT NULL DEFAULT '',
    html        TEXT          NOT NULL DEFAULT '',
    error       TEXT          NOT NULL DEFAULT '',
    fetched_at  TIMESTAMPTZ   NOT NULL DEFAULT now(),
    UNIQUE (task_id, url)
);
-- 迁移：老表没有 html 列时补上（存原始 HTML，用于复现抽取问题）
ALTER TABLE crawl_pages ADD COLUMN IF NOT EXISTS html TEXT NOT NULL DEFAULT '';

-- 结构化记录
CREATE TABLE IF NOT EXISTS crawl_records (
    id          BIGSERIAL PRIMARY KEY,
    task_id     BIGINT        NOT NULL REFERENCES crawl_tasks(id) ON DELETE CASCADE,
    page_url    VARCHAR(1024) NOT NULL DEFAULT '',
    strategy    VARCHAR(32)   NOT NULL DEFAULT '',
    data        TEXT          NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_crawl_records_task ON crawl_records (task_id, id);
"""


@dataclass
class TaskInfo:
    """一次抓取任务的概要。"""

    id: int
    status: str
    goal: str = ""
    entry_url: str = ""
    pages_fetched: int = 0
    records_extracted: int = 0
    strategy: str = ""
    stop_reason: str = ""
    error: str = ""


class CrawlStore:
    """三张表的读写。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ready = False

    async def _connect(self):
        return await lease_connection()

    async def ensure_tables(self) -> None:
        if self._ready:
            return
        conn = await self._connect()
        try:
            await conn.execute(CREATE_TABLES_SQL)
            self._ready = True
            logger.info("crawler.tables_ready")
        finally:
            await conn.close()

    # ── 任务 ────────────────────────────────────────────────
    async def create_task(
        self, tenant_id: str, goal: str, entry_url: str, user_id: str = ""
    ) -> int:
        await self.ensure_tables()
        conn = await self._connect()
        try:
            task_id = await conn.fetchval(
                """
                INSERT INTO crawl_tasks (tenant_id, user_id, goal, entry_url)
                VALUES ($1, $2, $3, $4) RETURNING id
                """,
                tenant_id,
                user_id,
                goal[:1000],
                entry_url[:1000],
            )
        finally:
            await conn.close()
        logger.info("crawler.task_created", task=task_id, goal=goal[:40])
        return int(task_id or 0)

    async def finish_task(
        self,
        task_id: int,
        status: str,
        pages: int = 0,
        records: int = 0,
        strategy: str = "",
        stop_reason: str = "",
        error: str = "",
    ) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                """
                UPDATE crawl_tasks
                SET status = $2, pages_fetched = $3, records_extracted = $4,
                    strategy = $5, stop_reason = $6, error = $7, finished_at = now()
                WHERE id = $1
                """,
                task_id,
                status,
                pages,
                records,
                strategy[:30],
                stop_reason[:60],
                error[:2000],
            )
        finally:
            await conn.close()

    async def get_task(self, task_id: int, tenant_id: str) -> TaskInfo | None:
        await self.ensure_tables()
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, status, goal, entry_url, pages_fetched,
                       records_extracted, strategy, stop_reason, error
                FROM crawl_tasks WHERE id = $1 AND tenant_id = $2
                """,
                task_id,
                tenant_id,
            )
        finally:
            await conn.close()
        if row is None:
            return None
        return TaskInfo(
            id=int(row["id"]),
            status=str(row["status"]),
            goal=str(row["goal"]),
            entry_url=str(row["entry_url"]),
            pages_fetched=int(row["pages_fetched"]),
            records_extracted=int(row["records_extracted"]),
            strategy=str(row["strategy"]),
            stop_reason=str(row["stop_reason"]),
            error=str(row["error"]),
        )

    async def recent_tasks(self, tenant_id: str, limit: int = 10) -> list[dict[str, Any]]:
        await self.ensure_tables()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT id, goal, entry_url, status, pages_fetched,
                       records_extracted, started_at
                FROM crawl_tasks WHERE tenant_id = $1
                ORDER BY id DESC LIMIT $2
                """,
                tenant_id,
                limit,
            )
        finally:
            await conn.close()
        return [
            {
                "id": int(r["id"]),
                "goal": str(r["goal"]),
                "entry_url": str(r["entry_url"]),
                "status": str(r["status"]),
                "pages_fetched": int(r["pages_fetched"]),
                "records_extracted": int(r["records_extracted"]),
                "started_at": r["started_at"].isoformat(),
            }
            for r in rows
        ]

    # ── 页面 ────────────────────────────────────────────────
    async def save_page(
        self,
        task_id: int,
        url: str,
        final_url: str = "",
        title: str = "",
        status: int = 0,
        mode: str = "http",
        content: str = "",
        html: str = "",
        error: str = "",
    ) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO crawl_pages
                    (task_id, url, final_url, title, status, mode,
                     char_count, content, html, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (task_id, url) DO UPDATE
                SET final_url = EXCLUDED.final_url, title = EXCLUDED.title,
                    status = EXCLUDED.status, mode = EXCLUDED.mode,
                    char_count = EXCLUDED.char_count, content = EXCLUDED.content,
                    html = EXCLUDED.html, error = EXCLUDED.error, fetched_at = now()
                """,
                task_id,
                url[:1000],
                final_url[:1000],
                title[:500],
                status,
                mode[:10],
                len(content),
                content,
                truncate_html(html),
                error[:2000],
            )
        finally:
            await conn.close()

    async def list_pages(self, task_id: int) -> list[dict[str, Any]]:
        """某任务抓过的页面列表（不带正文，避免返回体过大）。"""
        await self.ensure_tables()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT id, url, title, status, mode, char_count,
                       length(html) AS html_len, fetched_at
                FROM crawl_pages WHERE task_id = $1 ORDER BY id
                """,
                task_id,
            )
        finally:
            await conn.close()
        return [
            {
                "id": int(r["id"]),
                "url": str(r["url"]),
                "title": str(r["title"]),
                "status": int(r["status"]),
                "mode": str(r["mode"]),
                "char_count": int(r["char_count"]),
                "html_len": int(r["html_len"] or 0),
                "fetched_at": r["fetched_at"].isoformat(),
            }
            for r in rows
        ]

    async def get_page(self, page_id: int, tenant_id: str) -> dict[str, Any] | None:
        """取单页完整内容（含原始 HTML），用于排查抽取问题。"""
        await self.ensure_tables()
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT p.id, p.task_id, p.url, p.title, p.status, p.mode,
                       p.char_count, p.content, p.html
                FROM crawl_pages p
                JOIN crawl_tasks t ON t.id = p.task_id
                WHERE p.id = $1 AND t.tenant_id = $2
                """,
                page_id,
                tenant_id,
            )
        finally:
            await conn.close()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "task_id": int(row["task_id"]),
            "url": str(row["url"]),
            "title": str(row["title"]),
            "status": int(row["status"]),
            "mode": str(row["mode"]),
            "char_count": int(row["char_count"]),
            "content": str(row["content"]),
            "html": str(row["html"]),
        }

    async def page_urls(self, task_id: int) -> set[str]:
        """本任务已经抓过的 URL，用于去重。"""
        await self.ensure_tables()
        conn = await self._connect()
        try:
            rows = await conn.fetch("SELECT url FROM crawl_pages WHERE task_id = $1", task_id)
        finally:
            await conn.close()
        return {str(r["url"]) for r in rows}

    # ── 记录 ────────────────────────────────────────────────
    async def save_records(
        self, task_id: int, page_url: str, strategy: str, records: list[dict[str, str]]
    ) -> int:
        if not records:
            return 0
        conn = await self._connect()
        try:
            await conn.executemany(
                """
                INSERT INTO crawl_records (task_id, page_url, strategy, data)
                VALUES ($1, $2, $3, $4)
                """,
                [
                    (task_id, page_url[:1000], strategy[:30], json.dumps(r, ensure_ascii=False))
                    for r in records
                ],
            )
        finally:
            await conn.close()
        return len(records)

    async def list_records(self, task_id: int, limit: int = 500) -> list[dict[str, Any]]:
        await self.ensure_tables()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT id, page_url, strategy, data FROM crawl_records
                WHERE task_id = $1 ORDER BY id LIMIT $2
                """,
                task_id,
                limit,
            )
        finally:
            await conn.close()
        result = []
        for r in rows:
            try:
                data = json.loads(r["data"])
            except json.JSONDecodeError:
                data = {}
            result.append(
                {
                    "id": int(r["id"]),
                    "page_url": str(r["page_url"]),
                    "strategy": str(r["strategy"]),
                    "data": data,
                }
            )
        return result


_store: CrawlStore | None = None


def get_crawl_store() -> CrawlStore:
    global _store
    if _store is None:
        _store = CrawlStore()
    return _store


def truncate_html(html: str, limit: int | None = None) -> str:
    """原始 HTML 落库前截断。

    为什么必须截断：
        有的页面单文件就几 MB（内联的 base64 图片、超长脚本），
        直接入库会让 crawl_pages 迅速膨胀，查询也变慢。
        截断到上限足够复现抽取问题——抽取依赖的是结构，不是尾部内容。
    """
    if not html:
        return ""
    cap = limit if limit is not None else get_settings().crawl_max_html_chars
    if cap <= 0:
        return ""
    if len(html) <= cap:
        return html
    logger.info("crawler.html_truncated", original=len(html), kept=cap)
    return html[:cap] + "\n<!-- truncated -->"
