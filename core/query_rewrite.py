"""追问改写：把"那包年呢"补全成"你们包年怎么收费"。

为什么需要：
    FAQ 匹配和向量检索只看当前这句话。追问往往只有几个字
    （"那包年呢""换成季度"），里面没有任何可检索的实体，
    于是检索落空、模型拿到"知识库没有"的证据，只能说"没找到"。
    记忆停在了生成层，没进检索层——这是 RAG 系统的经典缺陷。

为什么不是每轮都改写：
    改写要一次模型调用（0.5~1s）。独立的完整问题不需要改写，
    所以先用规则判断"这句话是否依赖上一轮"，再决定要不要花这次调用。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

# 依赖上下文的信号：指示代词、省略、追问词
COREFERENCE_WORDS = (
    "那",
    "这个",
    "那个",
    "它",
    "他们",
    "上面",
    "刚才",
    "刚刚",
    "前面",
    "再说",
    "换成",
    "改为",
    "呢",
    "还有",
    "另外",
    "同样",
)

SYSTEM_PROMPT = """你的任务是改写追问，让它脱离上下文也能看懂。

规则：
1. 只输出改写后的问题，不要解释、不要引号。
2. 如果问题本身已经完整（不依赖上一轮），原样输出。
3. 保留用户原意，不要添加用户没提的条件。
4. 不要把上一轮回答里的具体价格、条款写进问题。
4. 不要回答问题，只改写问题。"""

USER_PROMPT = """【最近的对话】
{history}

【用户这次的问题】
{question}

输出改写后的问题："""


def should_rewrite(question: str, history: list[dict[str, str]]) -> bool:
    """这句话是否依赖上一轮。

    判据（任一命中就改写）：
        · 很短（≤12 字）—— 通常是追问
        · 含指示代词/省略/追问词（那、这个、它、刚才、换成…）
    没有历史时永远不改写。
    """
    if not history:
        return False
    text = (question or "").strip()
    if not text:
        return False
    # 含指示代词/省略词：明确依赖上一轮
    if any(word in text for word in COREFERENCE_WORDS):
        return True
    # 很短、且没有任何独立要素（怎么/什么/多少/哪/谁/你们…）：当追问处理
    independent = ("怎么", "什么", "多少", "为什么", "哪", "谁", "你们", "我们", "公司", "请问")
    return len(text) <= 6 and not any(word in text for word in independent)


def format_history(history: list[dict[str, str]], max_turns: int = 4) -> str:
    """只取最近几轮，太长反而干扰改写。"""
    # 只取用户的提问：把助手的答案拼进去，模型会把答案里的事实写进问题
    # （实测"那包年呢"被改写成"一号部门内部结算价包年怎么收费"）
    questions = [
        str(item.get("content") or "").strip()
        for item in history
        if str(item.get("role")) == "user"
    ]
    questions = [q for q in questions if q][-max_turns:]
    return "\n".join(f"用户：{q[:120]}" for q in questions) or "（无）"


async def rewrite_question(
    question: str,
    history: list[dict[str, str]],
    *,
    max_length: int = 120,
) -> str:
    """改写追问；任何异常都回落成原问题（改写失败不能影响正常问答）。"""
    original = (question or "").strip()
    if not should_rewrite(original, history):
        return original

    try:
        llm = LLMFactory.get_llm("qa", temperature=0)
        response = await llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=USER_PROMPT.format(history=format_history(history), question=original)
                ),
            ]
        )
        rewritten = str(response.content).strip().strip('"').strip("'")
    except Exception as exc:
        logger.warning("rewrite.failed", error=str(exc)[:120])
        return original

    # 兜底：改写结果为空/过长/明显跑题时，宁可不用
    if not rewritten or len(rewritten) > max_length:
        logger.info("rewrite.rejected", reason="空或过长", length=len(rewritten or ""))
        return original

    logger.info(
        "rewrite.done",
        original=original[:30],
        rewritten=rewritten[:60],
    )
    return rewritten


def rewrite_meta(original: str, rewritten: str) -> dict[str, Any]:
    """给响应带上改写痕迹，方便前端/日志排查。"""
    return {
        "rewritten": rewritten != original,
        "original_question": original if rewritten != original else "",
    }
