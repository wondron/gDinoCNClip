import copy
import logging
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler

from app.core.config.settings import local_path, settings
from app.core.observability.json_formatter import CustomJsonFormatter
from app.core.observability.log_helper import trace_id_var


class TraceFilter(logging.Filter):
    def filter(self, record):
        if not getattr(record, "traceId", ""):
            record.traceId = trace_id_var.get()
        return True


class CustomFormatter(logging.Formatter):
    def __init__(self, app_settings=settings):
        super().__init__("{asctime} | {levelname:<5} | {name:^15} | {message}", style="{")
        self.tz = timezone(timedelta(hours=app_settings.logging.timezone_offset_hours))

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, self.tz).strftime(datefmt or "%H:%M:%S")

    def format(self, record):
        # Formatter fields must not leak into the JSON handler's LogRecord.
        return super().format(copy.copy(record))


class DailyFileHandler(TimedRotatingFileHandler):
    """Rotate on the configured display timezone's midnight, including Windows."""
    def __init__(self, path, app_settings):
        self.tz = timezone(timedelta(hours=app_settings.logging.timezone_offset_hours))
        super().__init__(path, when="midnight", backupCount=app_settings.logging.backup_count,
                         encoding="utf-8", delay=True)
        self.suffix = "%Y-%m-%d"

    def computeRollover(self, current_time):
        current = datetime.fromtimestamp(current_time, self.tz)
        midnight = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight.timestamp())

    def rotation_filename(self, default_name):
        day = datetime.fromtimestamp(self.rolloverAt - 86400, self.tz).strftime(self.suffix)
        return self.baseFilename + "." + day


def setup_logger(save_to_file=None, save_k8s_file=None, *, app_settings=settings):
    cfg = app_settings.logging
    save_to_file = cfg.save_to_file if save_to_file is None else save_to_file
    save_k8s_file = cfg.save_k8s_file if save_k8s_file is None else save_k8s_file
    root = logging.getLogger()
    try:
        root.setLevel(cfg.level.upper())
    except ValueError as exc:
        raise ValueError(f"Invalid LOG_LEVEL: {cfg.level}") from exc
    platform = logging.getLogger("platform")
    platform.setLevel(root.level)
    platform.propagate = False
    for target in (root, platform):
        for handler in target.handlers[:]:
            if getattr(handler, "_grounding_handler", False):
                target.removeHandler(handler)
                handler.close()

    def attach(target, handler, formatter):
        handler._grounding_handler = True
        handler.addFilter(TraceFilter())
        handler.setFormatter(formatter)
        target.addHandler(handler)

    attach(root, logging.StreamHandler(sys.stderr), CustomFormatter(app_settings))
    if save_to_file:
        log_dir = local_path(app_settings.paths.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        attach(root, DailyFileHandler(log_dir / "app.log", app_settings), CustomFormatter(app_settings))
        attach(platform, DailyFileHandler(log_dir / "platform.log", app_settings), CustomJsonFormatter(app_settings))
    if save_k8s_file:
        attach(platform, logging.StreamHandler(sys.stdout), CustomJsonFormatter(app_settings))
    if not platform.handlers:
        attach(platform, logging.NullHandler(), CustomJsonFormatter(app_settings))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "starlette"):
        target = logging.getLogger(name)
        for handler in target.handlers[:]:
            target.removeHandler(handler)
            handler.close()
        target.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    root.info("日志初始化完成 | output=%s | file=%s | platform_stdout=%s",
              local_path(app_settings.paths.output_dir), save_to_file, save_k8s_file)
    return root
