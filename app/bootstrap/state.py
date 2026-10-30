"""One application-owned configuration and model state, shared with HTTP handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from app.core.config.settings import Settings, local_path

if TYPE_CHECKING:
    from app.core.config.nacos import NacosConfigProvider
    from app.domain.groundingdino.predictor import OnnxPredictor


@dataclass
class ApplicationState:
    settings: Settings
    config: dict
    config_info: dict
    config_provider: NacosConfigProvider
    predictor: OnnxPredictor | None = None
    stopping: bool = False
    reload_error: str | None = None

    lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @property
    def upload_dir(self) -> Path:
        return local_path(self.settings.paths.save_images_dir)
