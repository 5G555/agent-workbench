"""API 层共享的会话历史读写。

qa 和 research 两个 Agent 共用同一张 qa_messages 表：
    会话键统一是 "{user_id}:{session_id}"
    这样前端的会话列表能同时看到两个 Agent 的对话，切换 Agent 不用换会话。

放在独立模块而不是 import api.v1.qa 里的私有函数，是为了避免两个路由模块互相耦合。
"""

from __future__ import annotations

from agents.qa.history import get_history_store
from core.logger import get_logger

logger = get_logger(__name__)


def session_key(user_id: str, session_id: str) -> str:
    """会话存库时统一加用户前缀，防止猜到别人的 session_id 就能读历史。"""
    return f"{user_id}:{session_id}"


async def load_history(user_id: str, session_id: str, limit: int = 6) -> list[dict[str, str]]:
    """取最近几条用于多轮上下文；读失败不影响主流程。"""
    try:
        return await get_history_store().recent(session_key(user_id, session_id), limit)
    except Exception as exc:
        logger.warning("history.load_failed", session_id=session_id, error=str(exc))
        return []


async def save_turn(user_id: str, session_id: str, question: str, answer: str, mode: str) -> None:
    """落库一问一答；顺带用首问生成会话标题。"""
    try:
        store = get_history_store()
        key = session_key(user_id, session_id)
        await store.touch_session(key, user_id, question)
        await store.append(key, user_id, "user", question)
        await store.append(key, user_id, "assistant", answer, answer_mode=mode)
    except Exception as exc:
        logger.warning("history.save_failed", session_id=session_id, error=str(exc))
