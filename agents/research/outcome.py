"""联网检索的结果类型：把"失败"和"没搜到"分开。

为什么需要这个：
    原来所有异常都被吞成空列表，最后统一返回"没有找到足够的公开资料"。
    但"Tavily 挂了 / 被限流 / 没配 key"和"网上确实没有这个信息"是两件事，
    前者应该提示稍后重试，后者才该说没找到——混在一起会让用户误判成知识库缺失。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agents.research.evidence import Evidence


class SearchStatus(StrEnum):
    """一次联网检索的结果状态。"""

    OK = "ok"  # 拿到资料
    NO_RESULT = "no_result"  # 服务正常，但确实没搜到
    RATE_LIMITED = "rate_limited"  # 搜索服务方限流（429）
    QUOTA_EXCEEDED = "quota_exceeded"  # 本租户当日额度用完
    NO_KEY = "no_key"  # 没配置搜索密钥
    FAILED = "failed"  # 其他错误（网络、服务异常等）


# 每种状态对应的用户提示语
MESSAGES: dict[SearchStatus, str] = {
    SearchStatus.OK: "",
    SearchStatus.NO_RESULT: "公开资料中没有找到与这个问题相关的内容。",
    SearchStatus.RATE_LIMITED: (
        "联网检索服务当前被限流，暂时无法查询公开资料。请稍后再试，"
        "或把问题补充得更具体一些以便从知识库中查找。"
    ),
    SearchStatus.QUOTA_EXCEEDED: (
        "本租户今天的联网检索额度已用完，明天的额度会自动恢复。如需临时提高额度，请联系管理员。"
    ),
    SearchStatus.NO_KEY: "系统没有配置联网检索的密钥，无法查询公开资料，请联系管理员。",
    SearchStatus.FAILED: "联网检索服务暂时不可用，请稍后重试。",
}


@dataclass
class SearchOutcome:
    """检索结果：资料 + 状态 + 面向用户的说明。"""

    status: SearchStatus = SearchStatus.NO_RESULT
    items: list[Evidence] = field(default_factory=list)
    detail: str = ""  # 给日志/排查用的原始错误信息

    @property
    def ok(self) -> bool:
        return self.status is SearchStatus.OK

    @property
    def usable(self) -> bool:
        """有没有可以拿来作答的资料。"""
        return self.ok and bool(self.items)

    def user_message(self) -> str:
        """给用户看的一句话（不含内部堆栈）。"""
        return MESSAGES.get(self.status, "")

    def describe(self) -> str:
        """日志用：状态 + 细节。"""
        return f"{self.status.value}{'：' + self.detail if self.detail else ''}"


def classify_error(message: str) -> SearchStatus:
    """把底层错误文案归类成状态。越具体的判断放前面。"""
    text = (message or "").lower()
    if "429" in text or "限流" in text or "rate limit" in text:
        return SearchStatus.RATE_LIMITED
    if "额度" in text or "quota" in text:
        return SearchStatus.QUOTA_EXCEEDED
    if "tavily_api_key" in text or "未配置" in text or "no api key" in text:
        return SearchStatus.NO_KEY
    return SearchStatus.FAILED
