import logging
from contextvars import ContextVar
from uuid import uuid4

trace_id_var = ContextVar("grounding_trace_id", default="")


def get_k8s_logger(trace_id="", **kwargs):
    return logging.LoggerAdapter(logging.getLogger("platform"), {"traceId": trace_id or trace_id_var.get() or uuid4().hex, **kwargs,})


def show_text(message, trace_id="", logger_level=logging.INFO, exc_info=None, **kwargs):
    get_k8s_logger(trace_id, **kwargs).log(logger_level, message, exc_info=exc_info)
