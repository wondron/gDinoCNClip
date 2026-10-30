"""GroundingDINO ONNX detector shared by the pipeline, HTTP service and CLI."""

import logging
import math
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import numpy as np

from app.core.observability.log_helper import show_text
from app.domain.common.onnx_utils import create_session
from app.domain.groundingdino.postprocess import postprocess
from app.domain.groundingdino.preprocess import INPUT_NAMES, prepare_inputs
from app.domain.groundingdino.tokenizer import BertTokenizer

logger = logging.getLogger("groundingdino")



def _project_path(value):
    """Resolve configured paths relative to the project, independent of cwd."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path.resolve()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _channels(value, name, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must have three numeric RGB values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or (positive and v <= 0) for v in value):
        raise ValueError(f"{name} must contain finite {'positive ' if positive else ''}numbers")
    return tuple(float(v) for v in value)


def _probability(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite number between zero and one")
    return float(value)


class GdinoDetect:
    def __init__(self, config_data: Dict[str, Any]):
        self.ini_flag = False
        self.lock = threading.Lock()
        self._load_config(config_data)
        self.init_model()

    def _load_config(self, config: Dict[str, Any]) -> None:
        gdino_cfg = config.get("GROUNDINGDINO_CONFIG", {})
        total_cfg = config.get("TOTAL_CONFIG", {})
        if not isinstance(gdino_cfg, dict):
            raise ValueError("GROUNDINGDINO_CONFIG must be a mapping")
        if not isinstance(total_cfg, dict):
            raise ValueError("TOTAL_CONFIG must be a mapping")
        for key in ("model_path", "tokenizer_path"):
            if not isinstance(gdino_cfg.get(key), str) or not gdino_cfg[key].strip():
                raise ValueError(f"GROUNDINGDINO_CONFIG.{key} is required")

        self.model_path = str(_project_path(gdino_cfg["model_path"]))
        self.tokenizer_path = str(_project_path(gdino_cfg["tokenizer_path"]))
        self.use_gpu = gdino_cfg.get("use_gpu", True)
        if not isinstance(self.use_gpu, bool):
            raise ValueError("GROUNDINGDINO_CONFIG.use_gpu must be true or false")
        self.provider = "cuda" if self.use_gpu else "cpu"
        self.device_id = _integer(total_cfg.get("device_id", 0), "device_id", 0)
        self.threads = _integer(gdino_cfg.get("threads", 4), "threads", 0)
        self.short_side = _integer(gdino_cfg.get("short_side", 800), "short_side")
        self.max_size = _integer(gdino_cfg.get("max_size", 1333), "max_size")
        self.max_text_len = _integer(gdino_cfg.get("max_text_len", 256), "max_text_len")
        if self.max_text_len > 256:
            raise ValueError("max_text_len must not exceed this model's 256 token limit")
        self.mean = _channels(gdino_cfg.get("mean", [0.485, 0.456, 0.406]), "mean")
        self.std = _channels(gdino_cfg.get("std", [0.229, 0.224, 0.225]), "std", True)
        self.caption = gdino_cfg.get("caption", "object .")
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise ValueError("GROUNDINGDINO_CONFIG.caption must be a nonempty string")
        self.box_threshold = _probability(gdino_cfg.get("box_threshold", 0.35), "box_threshold")
        self.text_threshold = _probability(gdino_cfg.get("text_threshold", 0.25), "text_threshold")

    def get_providers(self):
        """Return the requested provider; session creation checks availability."""
        return ["CUDAExecutionProvider" if self.use_gpu else "CPUExecutionProvider"]

    def init_model(self) -> None:
        """Load the tokenizer and ONNX graph; initialization errors propagate."""
        self.ini_flag = False
        model_path = Path(self.model_path)
        tokenizer_path = Path(self.tokenizer_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        started = perf_counter()
        logger.info("加载模型 | path=%s | tokenizer=%s | provider=%s | device_id=%s",
                    model_path, tokenizer_path, self.provider, self.device_id)
        self.tokenizer = BertTokenizer(tokenizer_path)
        self.session = create_session(model_path, self.provider, self.threads, device_id=self.device_id)
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        if set(self.input_shapes) != set(INPUT_NAMES):
            raise ValueError(f"Unexpected ONNX model inputs: {self.input_shapes}")
        outputs = {item.name for item in self.session.get_outputs()}
        if not {"pred_logits", "pred_boxes"} <= outputs:
            raise ValueError(f"Unexpected ONNX model outputs: {outputs}")
        self.ini_flag = True
        elapsed = (perf_counter() - started) * 1000
        logger.info("模型加载完成 | providers=%s | inputs=%s | cost_ms=%.2f",
                    self.session.get_providers(), self.input_shapes, elapsed)
        show_text("模型加载完成", model=str(model_path), providers=self.session.get_providers(),
                  spendTime=round(elapsed, 2))

    def warmup(self, runs: int = 3) -> None:
        """Warm the same image/text inference path used by detect()."""
        shape = self.input_shapes["images"]
        height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else self.short_side
        width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else self.short_side
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        started = perf_counter()
        for _ in range(max(1, runs)):
            self.detect(dummy)
        logger.info("预热完成 | runs=%d | cost_ms=%.2f", max(1, runs), (perf_counter() - started) * 1000)

    def _prepare_caption(self, text_range) -> str:
        if text_range is None:
            return self.caption
        if isinstance(text_range, str):
            caption = text_range.strip()
        elif isinstance(text_range, (list, tuple)):
            if not text_range or any(not isinstance(text, str) or not text.strip(" .") for text in text_range):
                raise ValueError("text_range must contain nonempty text prompts")
            caption = " . ".join(text.strip().rstrip(".").strip() for text in text_range) + " ."
        else:
            raise TypeError("text_range must be a caption string or a list of text prompts")
        if not caption:
            raise ValueError("text_range must not be empty")
        return caption

    def preprocess(self, image, caption):
        """Build graph feeds from a BGR array, RGB PIL image or image path."""
        return prepare_inputs(
            image, caption, self.tokenizer, self.input_shapes,
            short_side=self.short_side, max_size=self.max_size, max_text_len=self.max_text_len,
            mean=self.mean, std=self.std,
        )

    def detect(self, imgbgr, text_range=None, box_threshold=None, text_threshold=None):
        """Return boxes, logits and phrases; boxes are normalized [cx, cy, w, h].

        NumPy input uses OpenCV BGR uint8 layout. text_range accepts a caption
        or a list such as ["glass", "straw"]. Omitted prompts/thresholds use
        GROUNDINGDINO_CONFIG defaults. Results retain query order, without NMS.
        """
        result, _ = self.predict_with_metadata(
            imgbgr, self._prepare_caption(text_range),
            self.box_threshold if box_threshold is None else box_threshold,
            self.text_threshold if text_threshold is None else text_threshold,
        )
        return result

    def predict(self, image_path, caption, box_threshold, text_threshold):
        """HTTP-compatible entry point preserving the original threshold policy."""
        result, _ = self.predict_with_metadata(
            image_path, caption, box_threshold, text_threshold, validate_thresholds=False,
        )
        return result

    def predict_with_metadata(self, image_path, caption, box_threshold, text_threshold,
                              *, validate_thresholds=True):
        """Return detections and preprocessing metadata for HTTP/CLI callers."""
        if not self.ini_flag:
            raise RuntimeError("GroundingDINO model is not initialized")
        started = perf_counter()
        with self.lock:
            acquired = perf_counter()
            feeds, metadata = self.preprocess(image_path, caption)
            prepared = perf_counter()
            logits, boxes = self.session.run(["pred_logits", "pred_boxes"], feeds)
            inferred = perf_counter()
            result = postprocess(logits, boxes, self.tokenizer, metadata,
                                 box_threshold, text_threshold, validate_thresholds=validate_thresholds)
        ended = perf_counter()
        timings = {
            "queue_ms": round((acquired - started) * 1000, 2),
            "preprocess_ms": round((prepared - acquired) * 1000, 2),
            "inference_ms": round((inferred - prepared) * 1000, 2),
            "postprocess_ms": round((ended - inferred) * 1000, 2),
        }
        logger.info("推理完成 | image=%s | text_length=%d | detections=%d | timings=%s",
                    feeds["images"].shape, metadata["text_length"], len(result["boxes"]), timings)
        show_text("推理完成", imageShape=list(feeds["images"].shape),
                  textLength=metadata["text_length"], detections=len(result["boxes"]), **timings)
        return result, metadata


if __name__ == "__main__":
    from app.core.config.loader import load_config
    from app.core.observability.logging import setup_logger

    setup_logger()
    config = load_config()
    detector = GdinoDetect(config)
    image_path = config["GROUNDINGDINO_CONFIG"].get("test_image_path", "config/images/test.jpg")
    detections = detector.detect(_project_path(image_path))
    logger.info("检测结果：%s", detections)
