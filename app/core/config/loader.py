"""Read local/Nacos YAML using project-root-relative model paths."""

import math
from copy import deepcopy
from pathlib import Path

import yaml

from app.core.config.settings import local_path, settings


def parse_config_text(text: str, source: str = "configuration") -> dict:
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a YAML mapping at the document root")
    return config


def load_yaml(path: str | Path) -> dict:
    path = local_path(path)
    return parse_config_text(path.read_text(encoding="utf-8"), str(path))


def _check_integer(value, name: str, minimum: int) -> None:
    # bool 是 int 的子类：YAML 的 true/false 与 4.0 这类浮点数都不能当作整数参数。
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _check_number(value, name: str, minimum: float, maximum: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be a finite number between {minimum:g} and {maximum:g}")


def _check_channels(value, name: str, positive: bool) -> None:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or (positive and v <= 0) for v in value)):
        raise ValueError(f"{name} must be three finite {'positive ' if positive else ''}numbers")


def validate_config(config: dict) -> None:
    """Reject GroundingDINO values before any model is loaded or hot-swapped.

    Only keys that are present are checked; missing keys keep GdinoDetect's defaults.
    CLIP initialization failures are reported through ClipDetect.ini_flag instead.
    """
    model = config.get("GROUNDINGDINO_CONFIG")
    if not isinstance(model, dict):
        raise ValueError("GROUNDINGDINO_CONFIG must be a YAML mapping")
    for key in ("model_path", "tokenizer_path"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f"GROUNDINGDINO_CONFIG.{key} is required")
    total = config.get("TOTAL_CONFIG", {})
    if not isinstance(total, dict):
        raise ValueError("TOTAL_CONFIG must be a YAML mapping")
    if "device_id" in total:
        _check_integer(total["device_id"], "TOTAL_CONFIG.device_id", 0)
    for key, minimum in (("threads", 0), ("short_side", 1), ("max_size", 1), ("max_text_len", 1)):
        if key in model:
            _check_integer(model[key], f"GROUNDINGDINO_CONFIG.{key}", minimum)
    if model.get("max_text_len", 0) > 256:
        raise ValueError("GROUNDINGDINO_CONFIG.max_text_len must not exceed the model's 256 token limit")
    for key in ("use_gpu", "nms_agnostic"):
        if not isinstance(model.get(key, False), bool):
            raise ValueError(f"GROUNDINGDINO_CONFIG.{key} must be true or false")
    for key, positive in (("mean", False), ("std", True)):
        if key in model:
            _check_channels(model[key], f"GROUNDINGDINO_CONFIG.{key}", positive)
    if "caption" in model and (not isinstance(model["caption"], str) or not model["caption"].strip()):
        raise ValueError("GROUNDINGDINO_CONFIG.caption must be a nonempty string")
    for key in ("box_threshold", "text_threshold"):
        if key in model:
            _check_number(model[key], f"GROUNDINGDINO_CONFIG.{key}", 0, 1)
    if model.get("nms_threshold") is not None:  # null 表示关闭 NMS
        _check_number(model["nms_threshold"], "GROUNDINGDINO_CONFIG.nms_threshold", 0, 1)


def prepare_config(config: dict) -> dict:
    config = deepcopy(config)
    # Accept old deployment files while using the reference project's model_path
    # and use_gpu naming for all new configurations.
    if "GROUNDINGDINO_CONFIG" not in config and "ONNX_MODEL" in config:
        legacy = config["ONNX_MODEL"]
        if not isinstance(legacy, dict):
            raise ValueError("ONNX_MODEL must be a mapping")
        provider = legacy.get("provider", "cuda")
        if provider not in ("cpu", "cuda"):
            raise ValueError("ONNX_MODEL.provider must be cpu or cuda")
        config["GROUNDINGDINO_CONFIG"] = {
            **legacy,
            "model_path": legacy.get("path", "config/model/groundingdino/groundingdino_swint_ogc.onnx"),
            "tokenizer_path": legacy.get("tokenizer", "config/model/groundingdino/bert-base-uncased"),
            "use_gpu": provider == "cuda",
        }
    validate_config(config)
    model = config["GROUNDINGDINO_CONFIG"]
    for key in ("model_path", "tokenizer_path"):
        model[key] = str(local_path(model[key]).resolve())
    return config


def load_config(yaml_path: str | Path | None = None) -> dict:
    return prepare_config(load_yaml(yaml_path or settings.config.yaml_path))
