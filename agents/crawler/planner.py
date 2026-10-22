"""编排 Agent 的决策：从候选链接里挑该抓哪些。

为什么只让 LLM 干"挑链接"这一件事：
    停止条件（页数上限、记录够不够）用代码判断更可预测、可测试；
    但"这 20 个链接里哪 3 个跟任务目标有关"是语义判断，规则写不好。
    所以分工是：代码管流程，模型管相关性。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

SELECT_SYSTEM = "你是一个网页采集助手，只输出 JSON，不解释。"

SELECT_USER = """任务目标：{goal}

当前页面：{current_url}
候选链接（编号: 地址）：
{candidates}

从上表中选出最多 {limit} 个**最有助于完成目标**的链接。

判断标准：
1. 优先"列表页 / 详情页 / 文档页"，避免登录页、关于我们、隐私政策这类。
2. 与目标主题无关的不要选。
3. 同一个栏目下的链接不要选太多，避免重复内容。
4. 如果一个都不相关，返回空列表。

只返回 JSON：{{"picks": [编号, ...]}}"""


class LinkPicks(BaseModel):
    """模型选出的链接编号。"""

    picks: list[int] = Field(default_factory=list)


async def choose_links(
    goal: str,
    current_url: str,
    candidates: list[str],
    limit: int = 3,
) -> list[str]:
    """让模型从候选里挑链接；失败或模型返回空则退回"按顺序取前几个"。"""
    if not candidates:
        return []

    shown = candidates[:30]
    listing = "\n".join(f"{i}. {url}" for i, url in enumerate(shown, 1))
    try:
        llm = LLMFactory.get_structured_llm("crawler", LinkPicks)
        result = await llm.ainvoke(
            [
                SystemMessage(content=SELECT_SYSTEM),
                HumanMessage(
                    content=SELECT_USER.format(
                        goal=goal or "（未指定，尽量抓内容页）",
                        current_url=current_url,
                        candidates=listing,
                        limit=limit,
                    )
                ),
            ]
        )
        picks = result.picks if isinstance(result, LinkPicks) else []
    except Exception as exc:
        logger.warning("crawler.choose_failed", error=str(exc))
        picks = []

    chosen: list[str] = []
    for index in picks:
        if 1 <= index <= len(shown):
            url = shown[index - 1]
            if url not in chosen:
                chosen.append(url)
        if len(chosen) >= limit:
            break

    if not chosen:
        # 模型没给有用的结果时不要空转：按顺序取前几个
        chosen = shown[:limit]
        logger.info("crawler.choose_fallback", count=len(chosen))

    logger.info("crawler.links_chosen", goal=goal[:30], picked=len(chosen))
    return chosen
