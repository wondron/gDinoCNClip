import logging
from time import perf_counter
from uuid import uuid4

from app.core.observability.log_helper import show_text, trace_id_var

logger = logging.getLogger("http")


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        trace_id = uuid4().hex
        token = trace_id_var.set(trace_id)
        status = 500
        started = perf_counter()

        async def capture_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-trace-id", trace_id.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, capture_send)
        except Exception:
            logger.exception("HTTP 请求异常 | method=%s | path=%s", scope["method"], scope["path"])
            raise
        finally:
            elapsed = round((perf_counter() - started) * 1000, 2)
            level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
            logger.log(level, "HTTP 请求完成 | method=%s | path=%s | status=%d | cost_ms=%.2f",
                       scope["method"], scope["path"], status, elapsed)
            show_text("HTTP 请求完成", logger_level=level, method=scope["method"],
                      path=scope["path"], status=status, spendTime=elapsed)
            trace_id_var.reset(token)
