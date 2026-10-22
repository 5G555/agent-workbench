"""统一作答：把内部证据和公网证据放在一份上下文里生成答案。

这是"联网搜索是 RAG 的增强组件"的落地：
    - 不区分调用方是 qa 还是 research，只有一份提示词、一次生成
    - 每条证据都带类型（内部文档 / 公网资料），答案里用 [n] 引用
    - 内部无资料时必须先说明，再说公网结论，避免让用户以为公网内容也是官方口径
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.research.evidence import EvidenceSet
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """你是一个严谨的企业问答助手，只能依据给定资料作答。

资料分两类，必须区别对待：
- （内部文档）：企业自己的知识库，是权威口径，直接采信。
- （公网资料）：只作参考。若公网资料与内部文档冲突，以内部文档为准，并指出冲突。

硬性要求：
1. 每条事实性陈述后用 [序号] 标注来源。
2. 内部文档没有相关资料时，开头必须明确说明"知识库中没有相关资料"，
   再说"以下内容来自公开资料，仅供参考"。
3. 资料不足就直说没找到，绝不编造。
4. 用中文，先结论后依据，不要输出"根据搜索结果"这类元话术。"""


async def compose_answer(
    query: str,
    evidence: EvidenceSet,
    history: str = "无历史对话",
) -> tuple[str, list[str]]:
    """生成答案，返回 (答案文本, 被引用的 URL 列表)。"""
    if evidence.empty:
        return "知识库和公开资料中都没有找到相关内容。", []

    llm = LLMFactory.get_llm("qa", temperature=0.2)
    messages: list[Any] = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"【用户问题】\n{query}\n\n"
                f"【可用资料】\n{evidence.as_context()}\n\n"
                f"【历史对话】\n{history}"
            )
        ),
    ]

    try:
        response = await llm.ainvoke(messages)
        answer = str(response.content).strip()
    except Exception as exc:
        logger.error("compose.failed", error=str(exc))
        return "生成答案时出错了，请稍后重试。", []

    cited = [e.url for e in evidence.cited_by(answer) if e.url]
    logger.info("compose.done", chars=len(answer), cited=len(cited), total=len(evidence.items))
    return answer, cited
