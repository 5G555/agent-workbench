"""编排 Agent 的节点。

流程（带循环）：
    START → init（建任务、入口 URL 入队）
          → decide ─┬─ fetch → extract → decide …（直到该停）
                    └─ finalize → END

停止条件全在代码里，可预测：
    1. 待抓队列空
    2. 抓够 max_pages 页
    3. 抽够目标记录数（防无限翻页）
    4. 连续失败 3 次
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agents.crawler.planner import choose_links
from agents.crawler.state import CrawlState
from agents.crawler.store import get_crawl_store
from agents.crawler.tools import extract_records, fetch_page, find_links, paginate
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

TARGET_RECORDS = 50
MAX_CONSECUTIVE_FAILURES = 3
MAX_FRONTIER = 20


async def init_node(state: CrawlState) -> dict[str, Any]:
    """建任务记录，把入口 URL 放进待抓队列。"""
    settings = get_settings()
    entry = str(state.get("entry_url") or "").strip()
    goal = str(state.get("goal") or state.get("query") or "").strip()

    if not entry.startswith(("http://", "https://")):
        return {
            "stop_reason": "bad_entry_url",
            "error": "入口 URL 必须是 http/https 地址",
            "frontier": [],
        }

    store = get_crawl_store()
    task_id = await store.create_task(
        tenant_id=str(state.get("tenant_id") or settings.default_tenant_id),
        goal=goal,
        entry_url=entry,
        user_id=str(state.get("user_id") or ""),
    )
    return {
        "task_id": task_id,
        "goal": goal,
        "entry_url": entry,
        "frontier": [entry],
        "visited": [],
        "pages": [],
        "record_count": 0,
        "strategies": [],
        "consecutive_failures": 0,
    }


async def decide_node(state: CrawlState) -> dict[str, Any]:
    """决定继续抓还是收工。只判断，不做事。"""
    settings = get_settings()
    frontier = list(state.get("frontier") or [])
    visited = list(state.get("visited") or [])
    pages = state.get("pages") or []
    records = int(state.get("record_count") or 0)
    max_pages = int(state.get("max_pages") or settings.crawler_max_pages)
    failures = int(state.get("consecutive_failures") or 0)

    if state.get("stop_reason"):
        return {"next_url": ""}
    if failures >= MAX_CONSECUTIVE_FAILURES:
        return {"stop_reason": "too_many_failures", "next_url": ""}
    if not frontier:
        return {"stop_reason": "no_more_urls", "next_url": ""}
    if len(visited) >= max_pages:
        return {"stop_reason": "max_pages", "next_url": ""}
    if records >= TARGET_RECORDS:
        return {"stop_reason": "enough_records", "next_url": ""}

    logger.info("crawler.decide", next=frontier[0], visited=len(visited), pages=len(pages))
    return {"next_url": frontier[0], "frontier": frontier[1:]}


async def fetch_node(state: CrawlState) -> dict[str, Any]:
    """抓一页，原始内容存进 crawl_pages。"""
    url = str(state.get("next_url") or "")
    task_id = int(state.get("task_id") or 0)
    store = get_crawl_store()
    visited = [*list(state.get("visited") or []), url]
    failures = int(state.get("consecutive_failures") or 0)

    result = await fetch_page(url)
    await store.save_page(
        task_id=task_id,
        url=url,
        final_url=result.final_url,
        title=result.title,
        status=result.status,
        mode=result.mode,
        content=result.text,
        error=result.error,
    )

    if not result.ok:
        logger.info("crawler.page_failed", url=url, status=result.status, error=result.error)
        return {
            "visited": visited,
            "consecutive_failures": failures + 1,
            "current_html": "",
            "current_text": "",
        }

    pages = list(state.get("pages") or [])
    pages.append(
        {"url": url, "title": result.title, "chars": result.char_count, "mode": result.mode}
    )
    logger.info("crawler.page_fetched", url=url, chars=result.char_count, mode=result.mode)
    return {
        "visited": visited,
        "pages": pages,
        "consecutive_failures": 0,
        "current_url": url,
        "current_html": result.html,
        "current_text": result.text,
    }


async def extract_node(state: CrawlState) -> dict[str, Any]:
    """抽记录并落库，同时把值得继续抓的链接塞进队列。"""
    html = str(state.get("current_html") or "")
    text = str(state.get("current_text") or "")
    current_url = str(state.get("current_url") or "")
    task_id = int(state.get("task_id") or 0)
    goal = str(state.get("goal") or "")

    if not html and not text:
        return {}  # 上一页抓失败了，跳过抽取

    store = get_crawl_store()
    records, strategy = extract_records(html, text)
    payload = [r.data for r in records if r.data]
    saved = await store.save_records(task_id, current_url, strategy, payload)

    strategies = list(state.get("strategies") or [])
    if strategy not in strategies:
        strategies.append(strategy)

    visited = set(state.get("visited") or [])
    frontier = list(state.get("frontier") or [])

    # 先尝试翻页（URL 层能翻就翻，最省事）
    next_page = paginate(html, current_url)
    if next_page and next_page not in visited:
        frontier.append(next_page)

    # 再让模型从同站链接里挑相关的
    candidates = [
        link
        for link in find_links(html, current_url)
        if link not in visited and link != current_url
    ]
    for link in await choose_links(goal, current_url, candidates, limit=2):
        if link not in visited and link not in frontier:
            frontier.append(link)

    logger.info(
        "crawler.extracted",
        url=current_url,
        strategy=strategy,
        records=saved,
        frontier=len(frontier),
    )
    return {
        "record_count": int(state.get("record_count") or 0) + saved,
        "strategies": strategies,
        "frontier": frontier[:MAX_FRONTIER],
    }


async def finalize_node(state: CrawlState) -> dict[str, Any]:
    """收尾：写任务统计，产出摘要。"""
    task_id = int(state.get("task_id") or 0)
    pages = state.get("pages") or []
    records = int(state.get("record_count") or 0)
    strategies = state.get("strategies") or []
    stop_reason = str(state.get("stop_reason") or "done")
    error = str(state.get("error") or "")
    strategy = strategies[0] if len(strategies) == 1 else ("mixed" if strategies else "")

    store = get_crawl_store()
    if task_id:
        await store.finish_task(
            task_id,
            status="failed" if error else "done",
            pages=len(pages),
            records=records,
            strategy=strategy,
            stop_reason=stop_reason,
            error=error,
        )

    if error:
        summary = f"抓取失败：{error}"
        mode = "error"
    elif not pages:
        summary = f"没有抓到任何页面（停止原因：{stop_reason}），请检查入口 URL 是否可访问。"
        mode = "no_result"
    else:
        chars = sum(int(p.get("chars") or 0) for p in pages)
        summary = (
            f"任务 #{task_id}：抓取 {len(pages)} 个页面（共 {chars} 字），"
            f"抽取 {records} 条结构化记录（策略 {strategy}）。停止原因：{stop_reason}"
        )
        mode = "crawl"

    logger.info(
        "crawler.task_done", task=task_id, pages=len(pages), records=records, stop=stop_reason
    )
    return {
        "summary": summary,
        "answer_mode": mode,
        "answer": summary,
        "messages": [AIMessage(content=summary)],
        "structured_output": {
            "answer_mode": mode,
            "task_id": task_id,
            "pages": len(pages),
            "records": records,
            "strategy": strategy,
            "stop_reason": stop_reason,
            "urls": [str(p.get("url", "")) for p in pages],
        },
    }
