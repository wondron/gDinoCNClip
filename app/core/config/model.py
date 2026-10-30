"""Validated algorithm parameters shared by the HTTP service and CLI."""

from dataclasses import dataclass
import math

from app.core.config.settings import local_path


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def channels(value, name, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must have three numeric RGB values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or (positive and v <= 0) for v in value):
        raise ValueError(f"{name} must contain finite {'positive ' if positive else ''}numbers")
    return tuple(float(v) for v in value)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    model_path: str
    tokenizer_path: str
    use_gpu: bool = True
    device_id: int = 0
    threads: int = 4
    short_side: int = 800
    max_size: int = 1333
    max_text_len: int = 256
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    @property
    def provider(self):
        return "cuda" if self.use_gpu else "cpu"

    @classmethod
    def from_config(cls, config):
        model = config["GROUNDINGDINO_CONFIG"]
        total = config.get("TOTAL_CONFIG", {})
        if not isinstance(total, dict):
            raise ValueError("TOTAL_CONFIG must be a mapping")
        use_gpu = model.get("use_gpu", True)
        if not isinstance(use_gpu, bool):
            raise ValueError("GROUNDINGDINO_CONFIG.use_gpu must be true or false")
        max_text_len = integer(model.get("max_text_len", 256), "max_text_len")
        if max_text_len > 256:
            raise ValueError("max_text_len must not exceed this model's 256 token limit")
        return cls(
            model_path=str(local_path(model["model_path"]).resolve()),
            tokenizer_path=str(local_path(model["tokenizer_path"]).resolve()),
            use_gpu=use_gpu,
            device_id=integer(total.get("device_id", 0), "device_id", 0),
            threads=integer(model.get("threads", 4), "threads", 0),
            short_side=integer(model.get("short_side", 800), "short_side"),
            max_size=integer(model.get("max_size", 1333), "max_size"),
            max_text_len=max_text_len,
            mean=channels(model.get("mean", [0.485, 0.456, 0.406]), "mean"),
            std=channels(model.get("std", [0.229, 0.224, 0.225]), "std", True),
        )
