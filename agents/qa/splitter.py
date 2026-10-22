"""父块 / 子块切分。

为什么要两层：
  子块小（200 字）→ 算向量时语义聚焦，检索命中率高
  父块大（600 字）→ 送 LLM 时上下文完整，不会把一句话切断
所以向量只在子块上算，命中后回溯到父块给模型。

切分策略:ChineseRecursiveTextSplitter：
  按「段落 → 换行 → 句号 → 问号 → 感叹号 → 分号 → 逗号 → 空格」的顺序递归尝试，
  每层切不动了才落到下一层，保证尽量在语义边界断开，而不是数着字符硬切。
"""

from __future__ import annotations

import re
from datetime import datetime

from langchain_core.documents import Document

from config import Settings, get_settings

# 分隔符从"粗"到"细"，中文标点放在英文之前
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _split_with_separator(text: str, separator: str, size: int, overlap: int) -> list[str] | None:
    """用单个分隔符切；切出来的块仍超长则返回 None，交给下一层。"""
    if separator == "":
        step = max(1, size - overlap)
        return [text[i : i + size] for i in range(0, len(text), step)]

    parts = text.split(separator)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        piece = part + separator
        if len(piece) > size:
            return None  # 单段就超长，这个分隔符不合适
        if len(buf) + len(piece) <= size:
            buf += piece
        else:
            if buf:
                chunks.append(buf)
            buf = piece
    if buf:
        chunks.append(buf)
    return chunks


def _recursive_split(text: str, size: int, overlap: int, level: int = 0) -> list[str]:
    """递归切分：当前分隔符切不动就换更细的。"""
    if len(text) <= size:
        return [text] if text.strip() else []
    if level >= len(SEPARATORS):
        step = max(1, size - overlap)
        return [text[i : i + size] for i in range(0, len(text), step)]

    result = _split_with_separator(text, SEPARATORS[level], size, overlap)
    if result is None:
        return _recursive_split(text, size, overlap, level + 1)

    out: list[str] = []
    for chunk in result:
        if len(chunk) <= size:
            out.append(chunk)
        else:
            out.extend(_recursive_split(chunk, size, overlap, level + 1))

    # 加重叠：把上一块尾部拼到下一块前面，避免边界信息丢失
    if overlap > 0 and len(out) > 1:
        out = [out[0]] + [out[i - 1][-overlap:] + out[i] for i in range(1, len(out))]
    return [c.strip() for c in out if c.strip()]


def chunk_text(text: str, size: int, overlap: int = 0) -> list[str]:
    """对外暴露的切分函数。先按空行分段，再逐段递归切。"""
    if size <= 0:
        raise ValueError("chunk size 必须大于 0")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        chunks.extend(_recursive_split(para, size, overlap))
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
    parents = chunk_text(text, s.parent_chunk_size)
    for p_idx, parent in enumerate(parents):
        parent_id = f"{source}_p{p_idx}"
        for child in chunk_text(parent, s.child_chunk_size, s.chunk_overlap):
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
