"""统一的流式生成封装。

项目里有三种调用场景，之前各写各的，导致有的地方能逐字输出、有的地方"憋完再吐"：

    1. 图节点内（qa 的 generate_node、research 的 synthesize_node）
       —— LangGraph 提供 get_stream_writer()，往里 write 就变成 custom stream 事件
    2. 图之外（orchestrator 拼证据作答的 compose）
       —— 拿不到 writer，需要调用方传一个回调进来
    3. 脚本 / 测试
       —— 两者都没有，直接一次性返回

这里把三种情况收成一个函数，调用方只关心"给我最终文本"。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

TokenSink = Callable[[str], Awaitable[None]]


def _graph_writer() -> Callable[[str], None] | None:
    """当前是否运行在 LangGraph 节点里；是则返回写入器。"""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


async def stream_or_invoke(
    llm: Any,
    messages: list[Any],
    on_token: TokenSink | None = None,
) -> str:
    """调用模型；能流式就边生成边推送，返回完整文本。

    优先级：
        on_token 回调  >  LangGraph writer  >  一次性 invoke

    回调优先是有意的：orchestrator 在 escalation 时需要自己接管事件序列，
    这时图 writer 不可用，只能靠回调。
    """
    writer = None if on_token else _graph_writer()

    if on_token is None and writer is None:
        response = await llm.ainvoke(messages)
        return str(response.content).strip()

    parts: list[str] = []
    async for chunk in llm.astream(messages):
        text = str(chunk.content or "")
        if not text:
            continue
        parts.append(text)
        if on_token is not None:
            await on_token(text)
        elif writer is not None:
            writer(text)
    return "".join(parts).strip()
