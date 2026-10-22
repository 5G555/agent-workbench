"""三层兜底：自动重试 → Agent 级降级 → 系统级兜底。

三层各管什么：
    重试      短暂故障（网络抖动、超时），等一下再来一次通常就好了
    降级      换个更省/更稳的实现，尽量把功能保住（deep→fast、浏览器→HTTP）
    系统兜底  实在不行也要给用户一句话，绝不把异常抛到 HTTP 层变成 500

降级不是"随便返回一句话"：能真跑的就真跑一遍。research 的 deep 模式失败时，
降级会**重新用 fast 模式跑一次**（不读网页正文，只靠搜索摘要），
这比直接告诉用户"服务不可用"有用得多。
"""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

from core.exceptions import (
    AuthenticationError,
    InvalidInputError,
    LLMAPIError,
    MilvusConnectionError,
)
from core.logger import get_logger

logger = get_logger(__name__)

# 可重试：多半是短暂故障（网络抖动、超时），重试一下可能就好
RETRYABLE_ERRORS = (
    LLMAPIError,
    MilvusConnectionError,
    TimeoutError,
    ConnectionError,
)
# 不可重试：重试也没用（输入非法、认证失败），应立即抛出
NON_RETRYABLE_ERRORS = (
    InvalidInputError,
    AuthenticationError,
)

MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 3.0]
# 单次调用超时。必须大于模型层超时（llm_factory 里是 120s），
# 否则外层先熔断，模型还没超时就报 TimeoutError，白白浪费一次重试机会。
TIMEOUT_PER_ATTEMPT = 150.0

# 降级原因（写进响应 metadata，前端/排查都能看到这次"退到哪一层"）
DEGRADE_RETRY_EXHAUSTED = "retry_exhausted"  # 重试全部失败后进入降级
DEGRADE_DEEP_TO_FAST = "deep_to_fast"  # 联网深度从 deep 退到 fast
DEGRADE_AGENT_UNAVAILABLE = "agent_unavailable"  # 该 Agent 无可用降级，只能提示
DEGRADE_SYSTEM = "system_fallback"  # 走到最后一层


def with_retry(agent_type: str = ""):
    """三层兜底装饰器工厂。

    用法：
        @with_retry(agent_type="qa")
        async def _invoke(state, config):
            return await graph.ainvoke(state, config=config)

    注意：被装饰的函数要能被"重放"——降级层会拿到同一批 args/kwargs 再跑一次。
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # ── 第一层：自动重试 ──────────────────────────
            last_error: Exception | None = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=TIMEOUT_PER_ATTEMPT,
                    )
                    if attempt > 0:
                        logger.info("retry.succeeded", agent_type=agent_type, attempt=attempt + 1)
                    return result

                except NON_RETRYABLE_ERRORS as e:
                    logger.warning("retry.non_retryable_error", agent_type=agent_type, error=str(e))
                    raise

                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "retry.attempt_failed",
                            agent_type=agent_type,
                            attempt=attempt + 1,
                            max_retries=MAX_RETRIES,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retry.all_attempts_failed", agent_type=agent_type, error=str(e)
                        )

            if last_error is None:
                last_error = RuntimeError("未知错误：重试循环未捕获到异常")

            # ── 第二层：Agent 级降级 ──────────────────────
            try:
                fallback_result = await AgentFallbackHandler.handle(
                    agent_type=agent_type,
                    original_error=last_error,
                    retry_call=func,
                    args=args,
                    kwargs=kwargs,
                )
                logger.info("retry.fallback_succeeded", agent_type=agent_type)
                return fallback_result
            except Exception as fallback_error:
                logger.error(
                    "retry.fallback_failed",
                    agent_type=agent_type,
                    error=str(fallback_error),
                )

            # ── 第三层：系统级兜底 ────────────────────────
            logger.error(
                "retry.system_fallback",
                agent_type=agent_type,
                original_error=str(last_error),
            )
            return _system_fallback_response(agent_type)

        return wrapper

    return decorator


class AgentFallbackHandler:
    """第二层降级：各 Agent 的专项策略。

    每个 Agent 都要有：
        qa        知识库/模型挂了 → 明确告知，不编答案
        research  deep 失败 → 用 fast 再跑一次；fast 也失败 → 告知联网不可用
        crawler   抓取服务挂了 → 告知入口和原因（采集失败不能假装成功）
    """

    @classmethod
    async def handle(
        cls,
        agent_type: str,
        original_error: Exception,
        retry_call: Callable | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        fallback_map = {
            "qa": cls._qa_fallback,
            "research": cls._research_fallback,
            "crawler": cls._crawler_fallback,
        }
        handler = fallback_map.get(agent_type)
        if handler is None:
            raise original_error  # 没有降级策略，交给系统兜底

        return await handler(original_error, retry_call, args, kwargs or {})

    # ── qa：只读本地，失败时不能编答案 ────────────────────
    @classmethod
    async def _qa_fallback(cls, error, retry_call, args, kwargs) -> dict:
        logger.info("fallback.qa_service_unavailable", error=str(error))
        return {
            "fallback_used": True,
            "degrade_reason": DEGRADE_AGENT_UNAVAILABLE,
            "content": "⚠️ 知识库检索暂时不可用，请稍后重试或直接联系有关工作人员。",
            "structured_output": {"answer_mode": "error", "degraded": True},
        }

    # ── research：能降级成 fast 就真再跑一次 ───────────────
    @classmethod
    async def _research_fallback(cls, error, retry_call, args, kwargs) -> dict:
        state = args[0] if args and isinstance(args[0], dict) else {}
        depth = str(state.get("depth") or "fast")

        if depth == "deep" and retry_call is not None:
            logger.info("fallback.research_deep_to_fast", error=str(error))
            degraded_state = {**state, "depth": "fast"}
            try:
                return await asyncio.wait_for(
                    retry_call(degraded_state, **kwargs),
                    timeout=TIMEOUT_PER_ATTEMPT,
                )
            except Exception as exc:
                logger.error("fallback.research_fast_failed", error=str(exc))
                error = exc

        logger.info("fallback.research_unavailable", error=str(error))
        return {
            "fallback_used": True,
            "degrade_reason": DEGRADE_AGENT_UNAVAILABLE,
            "content": (
                "⚠️ 联网检索暂时不可用（可能是网络或搜索服务故障）。"
                "你可以稍后重试，或把问题补充得更具体，我先从知识库里找找。"
            ),
            "structured_output": {"answer_mode": "error", "degraded": True},
        }

    # ── crawler：采集失败就明确说失败，不能返回空结果装作成功 ──
    @classmethod
    async def _crawler_fallback(cls, error, retry_call, args, kwargs) -> dict:
        state = args[0] if args and isinstance(args[0], dict) else {}
        entry = str(state.get("entry_url") or "")
        logger.info("fallback.crawler_unavailable", entry=entry, error=str(error))
        return {
            "fallback_used": True,
            "degrade_reason": DEGRADE_AGENT_UNAVAILABLE,
            "content": (
                f"⚠️ 网页采集失败（{entry or '未知入口'}）。"
                "可能是目标站点不可达或被反爬拦截，稍后可以重试；"
                "如果地址需要登录才能看，这边抓不到内容。"
            ),
            "structured_output": {
                "answer_mode": "error",
                "degraded": True,
                "entry_url": entry,
            },
        }


# Agent 各自的"服务不可用"提示语，找不到就用最后那句通用兜底
_SYSTEM_MESSAGES = {
    "qa": "非常抱歉，智能问答服务暂时不可用，请稍后再试，或直接联系有关人员。",
    "research": "非常抱歉，联网检索服务暂时不可用，请稍后再试。",
    "crawler": "非常抱歉，网页采集服务暂时不可用，请稍后再试。",
}


def _system_fallback_response(agent_type: str) -> dict:
    """第三层：系统级兜底。保证用户始终能收到响应，不会看到 500。"""
    content = _SYSTEM_MESSAGES.get(agent_type, "服务暂时不可用，请稍后再试。")
    return {
        "messages": [],
        "content": content,
        "fallback_used": True,
        "system_fallback": True,
        "degrade_reason": DEGRADE_SYSTEM,
        "structured_output": {"answer_mode": "error", "degraded": True},
    }
