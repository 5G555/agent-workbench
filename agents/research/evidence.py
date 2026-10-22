"""统一证据结构：内部知识库与公网检索共用一套表示。

按这张表的核心判断——联网搜索是 RAG 的增强组件，不是并行的另一套系统——
两种数据源应该有同一个形状，只靠 `source_type` 区分，这样：
    1. 可以放进同一个上下文一起生成，而不是各答一遍再拼接
    2. 引用能标明"这条来自内部文档 / 这条来自公网"
    3. 权限策略只需要针对 web 这一种类型做判断

字段说明：
    source_type  kb（内部知识库）/ web（公网）
    trust        可信度提示，内部文档是权威来源，公网只能当参考
    fetched_at   公网结果的时间戳（静态知识库不需要）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

KB = "kb"
WEB = "web"

TRUST_LABELS = {KB: "内部文档", WEB: "公网资料"}


@dataclass
class Evidence:
    """一条可被引用的依据。"""

    content: str
    source: str  # 文档名 / 网页标题
    source_type: str = KB
    url: str = ""  # 公网结果才有
    score: float = 0.0  # 精排分或搜索相关度
    fetched_at: str = ""

    @classmethod
    def from_kb(cls, content: str, source: str, score: float = 0.0) -> Evidence:
        return cls(content=content, source=source, source_type=KB, score=score)

    @classmethod
    def from_web(cls, title: str, url: str, content: str, score: float = 0.0) -> Evidence:
        return cls(
            content=content,
            source=title or url,
            source_type=WEB,
            url=url,
            score=score,
            fetched_at=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

    def as_line(self, index: int) -> str:
        """拼进提示词的一段。公网条目额外标出 URL 和抓取时间。"""
        label = TRUST_LABELS.get(self.source_type, self.source_type)
        head = f"[{index}]（{label}）{self.source}"
        if self.source_type == WEB and self.url:
            when = f"，抓取于 {self.fetched_at}" if self.fetched_at else ""
            head += f"\n链接：{self.url}{when}"
        else:
            head += "\n链接：无（内部文档）"
        body = self.content.strip().replace("\n", " ")[:800]
        return f"{head}\n正文：{body}"


@dataclass
class EvidenceSet:
    """一次问答的全部依据。"""

    items: list[Evidence] = field(default_factory=list)

    def add(self, evidence: Evidence) -> None:
        self.items.append(evidence)

    def extend(self, items: list[Evidence]) -> None:
        self.items.extend(items)

    @property
    def empty(self) -> bool:
        return not self.items

    def of_type(self, source_type: str) -> list[Evidence]:
        return [e for e in self.items if e.source_type == source_type]

    @property
    def has_kb(self) -> bool:
        return bool(self.of_type(KB))

    @property
    def has_web(self) -> bool:
        return bool(self.of_type(WEB))

    def as_context(self, limit: int = 6) -> str:
        """编号后拼成提示词上下文。编号即引用序号。"""
        chosen = self.items[:limit]
        return "\n\n".join(item.as_line(i) for i, item in enumerate(chosen, 1))

    def cited_by(self, answer: str) -> list[Evidence]:
        """答案里出现了 [n] 的条目，就是真正被引用的条目。"""
        return [e for i, e in enumerate(self.items[:6], 1) if f"[{i}]" in answer]

    def sources_labeled(self, limit: int = 6) -> list[str]:
        """给前端/API 的来源列表，带类型前缀，一眼能分清内外。"""
        out: list[str] = []
        for item in self.items[:limit]:
            prefix = "公网" if item.source_type == WEB else "内部"
            out.append(f"[{prefix}] {item.source}"[:120])
        return out
