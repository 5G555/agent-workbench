"""请求级 trace_id 中间件。

目标：同一次请求产生的所有日志都能串起来。
做法：为每个请求生成（或沿用客户端传来的）trace_id，绑定到 structlog 的 contextvars，
     logger 的处理链里有 merge_contextvars，所以每条日志会自动带上它；
     同时把 trace_id 回写到响应头 X-Trace-Id，出问题时前端能直接报这个 ID。

实现成纯 ASGI 中间件而不是 BaseHTTPMiddleware：
后者会包一层 response body 迭代器，对 SSE 流式接口有缓冲风险。
"""

from __future__ import annotations

from uuid import uuid4

import structlog


class TraceMiddleware:
    """给每个 HTTP 请求注入 trace_id。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        trace_id = incoming.strip() or uuid4().hex[:16]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)

        async def send_with_trace(message):
            if message["type"] == "http.response.start":
                raw = list(message.get("headers") or [])
                raw.append((b"x-trace-id", trace_id.encode("latin-1")))
                message["headers"] = raw
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            structlog.contextvars.clear_contextvars()
