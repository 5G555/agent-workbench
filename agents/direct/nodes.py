"""direct：不查任何数据，直接让模型回答。

存在的理由：
    "K 线怎么画""帮我写一封道歉邮件""什么是向量检索"这类问题，
    要的不是检索，是模型的表达与常识。以前它们会被关键词或三分类模型
    塞进 research，花了 8 秒搜回来一堆无关资料，然后答"没找到"。

answer-then-flag：
    模型自己也不清楚"哪些事实已经过时"，所以不逼它判断"要不要联网"，
    而是让它答完顺手标出拿不准的事实点。标出来了就显式告诉用户，
    而不是默默编一个数字。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.direct.state import DirectState
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一个务实的中文助手，直接回答问题。

规则：
1. 能用常识和方法论回答的，直接给出可执行的答案，别绕。
2. 涉及**可能过时的具体事实**（价格、版本号、现任负责人、最新政策、实时数据），
   如你不确定，把这些点放进 uncertain 列表，正文里可以给出你已知的大致情况但标注不确定。
3. 不要编造精确数字。宁可说"大约"或放进 uncertain。
4. 用中文，简洁；需要步骤就给编号步骤。"""

USER_PROMPT = """【问题】
{question}

【上下文】
{context}"""


class DirectAnswer(BaseModel):
    """结构化输出：答案 + 拿不准的事实点。"""

    answer: str = Field(description="直接给用户的回答")
    uncertain: list[str] = Field(
        default_factory=list,
        description="拿不准、可能需要联网核实的事实点（如 'AAPL 当前价格'）",
    )


async def answer_node(state: DirectState) -> dict[str, Any]:
    """直接作答；标记不确定的事实点。"""
    question = str(state.get("question") or "").strip()
    if not question:
        for message in reversed(state.get("messages") or []):
            if isinstance(message, HumanMessage):
                question = str(message.content).strip()
                break

    # 最近两轮上下文够用了，太长反而干扰
    history = []
    for message in (state.get("messages") or [])[-4:]:
        if isinstance(message, HumanMessage):
            history.append(f"用户：{message.content}")
        elif isinstance(message, AIMessage):
            history.append(f"助手：{message.content}")
    context = "\n".join(history[:-1]) or "（无）"

    try:
        llm = LLMFactory.get_structured_llm("qa", DirectAnswer)
        result = await llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=USER_PROMPT.format(question=question, context=context)),
            ]
        )
        answer = result.answer.strip()
        uncertain = [str(item) for item in result.uncertain if str(item).strip()]
    except Exception as exc:
        # 结构化输出失败时退化成普通调用，别让用户看到报错
        logger.warning("direct.structured_failed", error=str(exc)[:120])
        try:
            llm = LLMFactory.get_llm("qa", temperature=0.3, streaming=True)
            response = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=USER_PROMPT.format(question=question, context=context)),
                ]
            )
            answer = str(response.content).strip()
            uncertain = []
        except Exception as inner:
            logger.error("direct.failed", error=str(inner)[:120])
            return {
                "answer": f"模型暂时不可用（{inner}），请稍后再试。",
                "answer_mode": "error",
                "structured_output": {"answer_mode": "error"},
            }

    if uncertain:
        # 把不确定性显式交给用户，而不是替他猜
        listed = "、".join(uncertain[:3])
        answer += (
            f"\n\n以上内容基于我已有的知识；其中「{listed}」可能已经变化，需要我去查最新数据吗？"
        )

    logger.info("direct.answered", question=question[:40], uncertain=len(uncertain))
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "answer_mode": "direct",
        "structured_output": {
            "answer_mode": "direct",
            "uncertain": uncertain,
            "needs_check": bool(uncertain),
        },
    }
