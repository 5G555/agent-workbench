# 三层兜底机制：自动重试 → Agent 级降级 → 系统级兜底

import asyncio  # 异步：用于超时控制和等待
from collections.abc import Callable  # 类型注解：可调用对象
from functools import wraps  # @wraps：装饰器里保留原函数的名字/文档
from typing import Any  # 任意类型

from core.exceptions import (  # 引入 3.3 定义的异常（已去掉 Judge0 的 Sandbox 异常）
    AuthenticationError,
    InvalidInputError,
    LLMAPIError,
    MilvusConnectionError,
)
from core.logger import get_logger

logger = get_logger(__name__)

# ── 异常分类 ───────────────────────────────────────────────
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

MAX_RETRIES = 2  # 最多重试 2 次（加上首次 = 共 3 次尝试）
RETRY_DELAYS = [1.0, 3.0]  # 第 1 次重试前等 1 秒，第 2 次前等 3 秒
# 单次调用超时。必须大于模型层超时（llm_factory 里是 120s），
# 否则外层先熔断，模型还没超时就报 TimeoutError，白白浪费一次重试机会。
TIMEOUT_PER_ATTEMPT = 150.0


def with_retry(agent_type: str = ""):
    """三层兜底装饰器工厂。给异步函数套上「重试 → 降级 → 系统兜底」三层保护。

    用法：
        @with_retry(agent_type="qa")
        async def _invoke():
            return await graph.ainvoke(state, config=config)
    """

    def decorator(func: Callable) -> Callable:  # 中间层：接收被装饰的函数
        @wraps(func)  # 保留原函数的元信息（名字、docstring）
        async def wrapper(*args, **kwargs) -> Any:  # 最内层：真正执行逻辑

            # ── 第一层：自动重试 ──────────────────────────
            last_error: Exception | None = None  # 记录最后一次的错误，留给后面降级用
            for attempt in range(MAX_RETRIES + 1):  # 循环 3 次：attempt = 0, 1, 2
                try:
                    # 给单次调用套一个超时；超过 30 秒就抛 TimeoutError
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=TIMEOUT_PER_ATTEMPT,
                    )
                    if attempt > 0:  # 如果是重试后成功的，记一条日志
                        logger.info("retry.succeeded", agent_type=agent_type, attempt=attempt + 1)
                    return result  # 成功，直接返回，结束

                except NON_RETRYABLE_ERRORS as e:  # 不可重试异常：立即抛出，不再重试
                    logger.warning("retry.non_retryable_error", agent_type=agent_type, error=str(e))
                    raise  # 原样抛出，交给上层处理

                except Exception as e:  # 其它（可重试）异常
                    last_error = e  # 记下来
                    if attempt < MAX_RETRIES:  # 还没到上限：等待后重试
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "retry.attempt_failed",
                            agent_type=agent_type,
                            attempt=attempt + 1,
                            max_retries=MAX_RETRIES,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)  # 等 1s 或 3s 再重试
                    else:  # 到上限了：记录失败，跳出循环去降级
                        logger.error(
                            "retry.all_attempts_failed", agent_type=agent_type, error=str(e)
                        )

            # ── 第二层：Agent 级降级 ──────────────────────
            # 走到这里说明至少失败过一次；last_error 理论上有值，兜一下让类型收敛
            if last_error is None:
                last_error = RuntimeError("未知错误：重试循环未捕获到异常")
            try:
                fallback_result = await AgentFallbackHandler.handle(  # 按 agent_type 找降级策略
                    agent_type=agent_type,
                    original_error=last_error,
                )
                logger.info("retry.fallback_succeeded", agent_type=agent_type)
                return fallback_result  # 降级成功，返回降级结果
            except Exception as fallback_error:  # 连降级都失败
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
            return _system_fallback_response(agent_type)  # 最后的保底，永远不会再失败

        return wrapper

    return decorator


class AgentFallbackHandler:
    """第二层降级：各 Agent 的专项降级策略（尽量保留核心功能，退化为更简单的实现）。"""

    @classmethod
    async def handle(cls, agent_type: str, original_error: Exception) -> Any:
        """根据 agent_type 选择对应的降级策略。"""
        fallback_map = {  # 类型 → 降级方法 的映射表
            "qa": cls._qa_fallback,
        }
        handler = fallback_map.get(agent_type)  # 查表
        if handler:
            return await handler()
        raise original_error  # 没有对应降级策略，原样抛出（交给系统兜底）

    @classmethod
    async def _qa_fallback(cls) -> dict:
        """问答降级：知识库或模型不可用，返回提示语。"""
        logger.info("fallback.qa_service_unavailable")
        return {
            "fallback_used": True,
            "content": "⚠️ 知识库检索暂时不可用，请稍后重试或直接联系有关工作人员。",
            "structured_output": None,
        }


def _system_fallback_response(agent_type: str) -> dict:
    """第三层：系统级兜底。所有降级都失败后返回它，保证用户始终能收到响应。"""
    messages = {  # 按 agent_type 给不同的友好提示
        "qa": "非常抱歉，智能问答服务暂时不可用，请稍后再试，或直接联系有关人员。",
    }
    content = messages.get(agent_type, "服务暂时不可用，请稍后再试。")  # 找不到就用通用提示
    return {
        "messages": [],
        "content": content,
        "fallback_used": True,
        "system_fallback": True,  # 标记：走到了最后一层系统兜底
        "structured_output": None,
    }
