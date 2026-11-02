import logging
from uuid import uuid4

from app.core.observability.log_helper import trace_id_var

logger = logging.getLogger("http")


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        trace_id = uuid4().hex
        token = trace_id_var.set(trace_id)

        async def capture_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-trace-id", trace_id.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, capture_send)
        except Exception:
            logger.exception("HTTP 请求异常 | method=%s | path=%s", scope["method"], scope["path"])
            raise
        finally:
            trace_id_var.reset(token)
