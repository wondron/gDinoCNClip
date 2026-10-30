"""Read local/Nacos YAML using project-root-relative model paths."""

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
    model = config.get("GROUNDINGDINO_CONFIG")
    if not isinstance(model, dict):
        raise ValueError("GROUNDINGDINO_CONFIG must be a YAML mapping")
    for key in ("model_path", "tokenizer_path"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f"GROUNDINGDINO_CONFIG.{key} is required")
        model[key] = str(local_path(model[key]).resolve())
    return config


def load_config(yaml_path: str | Path | None = None) -> dict:
    return prepare_config(load_yaml(yaml_path or settings.config.yaml_path))
