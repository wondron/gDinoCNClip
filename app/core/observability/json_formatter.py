import json
import logging
from datetime import datetime, timedelta, timezone

from app.core.config.settings import settings

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class CustomJsonFormatter(logging.Formatter):
    def __init__(self, app_settings=settings):
        super().__init__()
        self.settings = app_settings
        self.tz = timezone(timedelta(hours=app_settings.logging.timezone_offset_hours))

    def format(self, record):
        result = {
            "index": self.settings.logging.index,
            "time": datetime.fromtimestamp(record.created, self.tz).isoformat(timespec="milliseconds"),
            "env": self.settings.application.env,
            "logger": record.name,
            "level": record.levelname,
            "thread": record.threadName,
            "class": f"{record.name}.{record.funcName}:{record.lineno}",
            "message": record.getMessage(),
            "traceId": getattr(record, "traceId", ""),
            "stack_trace": self.formatException(record.exc_info) if record.exc_info else "",
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in result and value is not None:
                result[key] = value
        return json.dumps(result, ensure_ascii=False, default=str)
