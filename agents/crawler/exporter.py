"""把一次爬取任务导出成文件。

为什么要导出：
    数据本来只存在 Postgres 里，用 SQL 看没问题，但有两类需求它满足不了：
        1. 想用编辑器/Diff 工具直接看原文和抽取结果
        2. 想把抓到的语料交给别的工具（比如灌进 Milvus、喂给标注平台）

目录结构（每个任务一个目录，天然隔离）：
    data/crawl/{task_id}/
        manifest.json          任务元信息 + 文件清单
        records.jsonl          结构化记录，一行一条
        pages/0001-标题.html    原始 HTML
        pages/0001-标题.txt     清洗后的正文

文件名里的标题会做安全处理：只留字母数字和中文，
其余字符（含路径分隔符）一律换成 '-'，避免出现 ../ 这类路径穿越。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from agents.crawler.store import get_crawl_store
from core.logger import get_logger

logger = get_logger(__name__)

# 默认导出到项目下的 data/crawl
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "crawl"
# 文件名里保留的字符：字母、数字、中文、下划线、连字符
SAFE_PATTERN = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff_-]+")


@dataclass
class ExportResult:
    """导出结果。"""

    ok: bool = False
    task_id: int = 0
    directory: str = ""
    pages: int = 0
    records: int = 0
    bytes_written: int = 0
    files: list[str] = field(default_factory=list)
    error: str = ""


def safe_name(text: str, fallback: str = "page", limit: int = 40) -> str:
    """把标题变成安全的文件名片段。

    空标题、纯符号标题都会退化成 fallback，避免出现 "0001-.html" 这种命名。
    """
    cleaned = SAFE_PATTERN.sub("-", (text or "").strip()).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)[:limit].strip("-")
    return cleaned or fallback


def page_filename(page_id: int, title: str, url: str, suffix: str) -> str:
    """`0001-标题.html`；没标题就用域名兜底，别一堆同名文件。"""
    label = safe_name(title) if title.strip() else safe_name(url.split("//")[-1], "page")
    return f"{page_id:04d}-{label}{suffix}"


async def export_task(
    task_id: int,
    tenant_id: str,
    out_dir: str | Path | None = None,
) -> ExportResult:
    """把某任务的页面与记录写到文件，返回清单。"""
    store = get_crawl_store()
    task = await store.get_task(task_id, tenant_id)
    if task is None:
        return ExportResult(ok=False, task_id=task_id, error="任务不存在或不属于当前租户")

    pages = await store.all_pages(task_id, tenant_id)
    if not pages:
        return ExportResult(ok=False, task_id=task_id, error="该任务没有抓到任何页面")

    root = Path(out_dir) if out_dir else DEFAULT_ROOT
    target = root / str(task_id)

    # 防路径穿越：解析后的绝对路径必须落在 root 下
    root_abs = root.resolve()
    target_abs = target.resolve()
    if root_abs not in target_abs.parents and target_abs != root_abs:
        return ExportResult(ok=False, task_id=task_id, error="导出目录不合法")

    pages_dir = target_abs / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    total_bytes = 0

    for page in pages:
        html_name = page_filename(page["id"], page["title"], page["url"], ".html")
        text_name = page_filename(page["id"], page["title"], page["url"], ".txt")
        html_path = pages_dir / html_name
        text_path = pages_dir / text_name

        if page["html"]:
            html_path.write_text(page["html"], encoding="utf-8")
            written.append(f"pages/{html_name}")
            total_bytes += len(page["html"].encode("utf-8"))

        body = page["content"] or ""
        # 正文文件带上来源行，方便脱离数据库也能对上号
        header = (
            f"# {page['title']}\n"
            f"# {page['url']}\n"
            f"# 状态 {page['status']} / {page['char_count']} 字\n\n"
        )
        text_path.write_text(header + body, encoding="utf-8")
        written.append(f"pages/{text_name}")
        total_bytes += len((header + body).encode("utf-8"))

    records = await store.list_records(task_id, limit=10000)
    records_path = target_abs / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "page_url": record["page_url"],
                        "strategy": record["strategy"],
                        "data": record["data"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    written.append("records.jsonl")
    total_bytes += records_path.stat().st_size

    manifest = {
        "task_id": task.id,
        "goal": task.goal,
        "entry_url": task.entry_url,
        "status": task.status,
        "pages": len(pages),
        "records": len(records),
        "strategy": task.strategy,
        "stop_reason": task.stop_reason,
        "files": written,
    }
    manifest_path = target_abs / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append("manifest.json")
    total_bytes += manifest_path.stat().st_size

    logger.info(
        "crawler.exported",
        task=task_id,
        pages=len(pages),
        records=len(records),
        directory=str(target_abs),
    )
    return ExportResult(
        ok=True,
        task_id=task_id,
        directory=str(target_abs),
        pages=len(pages),
        records=len(records),
        bytes_written=total_bytes,
        files=written,
    )
