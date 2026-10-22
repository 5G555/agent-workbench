"""把 data/test_kb 下的文档灌进 Milvus。

用法：
    python -m scripts.ingest                 # 灌 data/test_kb
    python -m scripts.ingest <目录>          # 灌指定目录
    python -m scripts.ingest --query "问题"  # 只做一次检索验证
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

import pymupdf
from docx import Document

from agents.qa.cache import bump_cache_version
from agents.qa.retrieval.vector_store import get_vector_store
from agents.qa.splitter import split_document
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_kb"
SUFFIXES = {".md", ".txt", ".docx", ".pdf", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".bmp"}

_ocr_engine = None


def _get_ocr():
    """懒加载 OCR 引擎（首次加载模型约 1 秒，别在 import 时做）。"""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_image(path: Path) -> str:
    """对图片做 OCR，返回识别出的文字。"""
    result, _ = _get_ocr()(str(path))
    if not result:
        return ""
    return "\n".join(line[1] for line in result if len(line) > 1)


def _read_pptx(path: Path) -> str:
    """PPT/PPTX：按页提取文本框 + 表格文字。"""
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: list[str] = []
    for index, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    texts.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        texts.append(" | ".join(cells))
        if texts:
            parts.append(f"第 {index} 页\n" + "\n".join(texts))
    return "\n\n".join(parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:  # 表格里的文字也要，否则会漏内容
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _read_pdf(path: Path) -> str:
    """PDF：优先抽文字层；某页没有文字层说明是扫描件，转图片走 OCR。"""
    parts: list[str] = []
    with pymupdf.open(str(path)) as pdf:
        for page in pdf:
            text = page.get_text().strip()
            if text:
                # PDF 里的换行多半是排版换行而非段落，中文直接接起来；
                # 空行（\n\n）才当作段落分隔。
                parts.append(re.sub(r"(?<!\n)\n(?!\n)", "", text))
                continue

            pix = page.get_pixmap(dpi=200)
            result, _ = _get_ocr()(pix.tobytes("png"))
            if result:
                parts.append("\n".join(line[1] for line in result if len(line) > 1))
                logger.info("ingest.pdf_page_ocred", file=path.name, page=page.number + 1)
    return "\n\n".join(parts)


def _read_image(path: Path) -> str:
    return _ocr_image(path)


_READERS = {
    ".md": _read_text,
    ".txt": _read_text,
    ".docx": _read_docx,
    ".pdf": _read_pdf,
    ".pptx": _read_pptx,
    ".ppt": _read_pptx,
    ".png": _read_image,
    ".jpg": _read_image,
    ".jpeg": _read_image,
    ".bmp": _read_image,
}


def load_documents(directory: Path) -> list:
    docs = []
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUFFIXES:
            continue
        reader = _READERS[suffix]
        text = reader(path)
        if not text.strip():
            logger.warning("ingest.empty_document", file=path.name, suffix=suffix)
            continue
        source = path.stem
        chunks = split_document(text, source=source)
        docs.extend(chunks)
        logger.info(
            "ingest.file_loaded", file=path.name, suffix=suffix, chars=len(text), chunks=len(chunks)
        )
    return docs


def _pop_option(args: list[str], name: str) -> str | None:
    """从参数里取出 --name value 并移除，返回 value。"""
    if name in args:
        i = args.index(name)
        value = args[i + 1] if i + 1 < len(args) else None
        del args[i : i + (2 if value else 1)]
        return value
    return None


def main() -> None:
    args = list(sys.argv[1:])

    # 租户参数先摘出来，剩下的位置参数才是目录
    tenant = _pop_option(args, "--tenant") or get_settings().default_tenant_id

    if args and args[0] == "--query":
        query = args[1] if len(args) > 1 else "你们怎么收费"
        store = get_vector_store()
        results = store.hybrid_search_with_rerank(query, tenant_id=tenant)
        print(f"\n查询: {query}（租户 {tenant}）\n命中 {len(results)} 个父块\n")
        for i, doc in enumerate(results, 1):
            print(f"--- [{i}] source={doc.metadata.get('source')} ---")
            print(doc.page_content[:300].replace("\n", " "))
            print()
        return

    directory = Path(args[0]) if args and not args[0].startswith("-") else DATA_DIR
    if not directory.exists():
        raise SystemExit(f"目录不存在: {directory}")

    documents = load_documents(directory)
    if not documents:
        raise SystemExit(f"{directory} 下没有可导入的 {'/'.join(SUFFIXES)} 文件")

    print(f"切出 {len(documents)} 个子块，开始向量化并写入 Milvus（租户 {tenant}）…")
    store = get_vector_store()
    if "--reset" in args:
        print("重建集合（切分参数或 schema 变了必须重建）…")
        store.reset()
    count = store.add_documents(documents, tenant_id=tenant)
    time.sleep(1.5)  # Milvus 写入后有短暂可见性延迟，立刻统计会得到 0
    print(f"写入 {count} 条（租户 {tenant}），集合当前总条数: {store.count()}")
    print(f"该租户可见条数: {store.count(tenant_id=tenant)}")

    # 知识库变了，旧答案缓存必须失效
    version = asyncio.run(bump_cache_version())
    print(f"缓存版本号已提升到 {version}，旧缓存失效")


if __name__ == "__main__":
    main()
