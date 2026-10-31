"""Typed startup settings, following Image-Embedding/backend conventions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import constant as c


def local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else c.BASE_DIR / path


def _nacos_profile() -> str:
    """NACOS_ENV 是测试环境标识：非空选择 test，否则选择 prod。"""
    return "test" if os.getenv("NACOS_ENV", "").strip() else c.NACOS_DEFAULT_PROFILE


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    service_name: str = c.SERVICE_NAME
    version: str = c.VERSION
    env: str = c.APP_ENV
    host: str = c.DEFAULT_HOST
    port: int = c.DEFAULT_PORT
    reload: bool = c.APP_RELOAD


@dataclass(frozen=True, slots=True)
class ConfigSettings:
    yaml_path: str = field(default_factory=lambda: os.getenv("GROUNDING_ONNX_CONFIG", c.DEFAULT_YAML_PATH))


@dataclass(frozen=True, slots=True)
class PathSettings:
    output_dir: str = c.OUTPUT_DIR
    save_images_dir: str = c.SAVE_IMAGES_DIR
    save_images_enabled: bool = c.SAVE_IMAGES_ENABLED
    jpeg_quality: int = c.JPEG_QUALITY


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    index: str = c.LOGGER_NAME
    level: str = c.LOG_LEVEL
    save_to_file: bool = c.LOG_SAVE_TO_FILE
    save_k8s_file: bool = c.LOG_SAVE_K8S_FILE
    backup_count: int = c.LOG_BACKUP_COUNT
    timezone_offset_hours: float = c.LOG_TIMEZONE_OFFSET_HOURS


@dataclass(frozen=True, slots=True)
class WarmupSettings:
    enabled: bool = c.WARMUP_ENABLED
    runs: int = c.WARMUP_RUNS
    image_path: str = c.WARMUP_IMAGE_PATH
    caption: str = c.WARMUP_CAPTION


@dataclass(frozen=True, slots=True)
class NacosSettings:
    profile: str = field(default_factory=_nacos_profile)
    enabled: bool = field(default_factory=lambda: c.NACOS_ENABLED)
    poll_enabled: bool = c.NACOS_POLL_ENABLED
    connect_timeout_seconds: float = c.NACOS_CONNECT_TIMEOUT_SECONDS
    request_timeout_seconds: float = c.NACOS_REQUEST_TIMEOUT_SECONDS
    poll_interval_seconds: float = c.NACOS_POLL_INTERVAL_SECONDS
    retry_interval_seconds: float = c.NACOS_RETRY_INTERVAL_SECONDS
    profiles: tuple[c.NacosProfile, ...] = field(default_factory=lambda: c.NACOS_CONFIG_PROFILES)

    def __post_init__(self):
        if not self.enabled:
            return
        if self.profile not in {profile["profile"] for profile in self.profiles}:
            raise ValueError("Selected Nacos profile must be configured in NACOS_CONFIG_PROFILES")
        if min(self.connect_timeout_seconds, self.request_timeout_seconds,
               self.poll_interval_seconds, self.retry_interval_seconds) <= 0:
            raise ValueError("Nacos timeout and polling intervals must be positive")


@dataclass(frozen=True, slots=True)
class Settings:
    application: ApplicationSettings = field(default_factory=ApplicationSettings)
    config: ConfigSettings = field(default_factory=ConfigSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    warmup: WarmupSettings = field(default_factory=WarmupSettings)
    nacos: NacosSettings = field(default_factory=NacosSettings)


settings = Settings()
