"""父块 / 子块切分。

为什么要两层：
  子块小（300 字）→ 算向量时语义聚焦，检索命中率高
  父块大（1200 字）→ 送给 LLM 时上下文完整，不会把一句话切断
所以向量只在子块上算，命中后回溯到父块给模型。
"""

from __future__ import annotations

import re
from datetime import datetime

from langchain_core.documents import Document

from config import Settings, get_settings


def _paragraphs(text: str) -> list[str]:
    """按空行切段，去掉纯空白段。"""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk(text: str, size: int, overlap: int = 0) -> list[str]:
    """把长文本按字符数切成带重叠的块（按段落边界优先，避免切断句子）。"""
    if size <= 0:
        raise ValueError("chunk size 必须大于 0")

    chunks: list[str] = []
    buf = ""
    for para in _paragraphs(text):
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
            continue
        if buf:
            chunks.append(buf)
        buf = para
        # 单段本身就超过 size：硬切
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap :] if overlap else buf[size:]
    if buf:
        chunks.append(buf)
    return chunks


def split_document(
    text: str,
    *,
    source: str,
    settings: Settings | None = None,
    timestamp: str | None = None,
) -> list[Document]:
    """把一篇文档切成「父块 → 子块」，返回带完整元数据的子块列表。"""
    s = settings or get_settings()
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    docs: list[Document] = []
    parents = _chunk(text, s.parent_chunk_size)
    for p_idx, parent in enumerate(parents):
        parent_id = f"{source}_p{p_idx}"
        for child in _chunk(parent, s.child_chunk_size, s.chunk_overlap):
            docs.append(
                Document(
                    page_content=child,
                    metadata={
                        "parent_id": parent_id,
                        "parent_content": parent,
                        "source": source,
                        "timestamp": ts,
                    },
                )
            )
    return docs
