"""Shared GroundingDINO model lifecycle and inference for HTTP and CLI."""

import logging
import threading
from pathlib import Path
from time import perf_counter

from app.core.config.model import ModelSettings
from app.core.observability.log_helper import show_text
from app.adapter.onnx.session import create_session
from app.domain.groundingdino.postprocess import postprocess
from app.domain.groundingdino.preprocess import INPUT_NAMES, prepare_inputs
from app.domain.groundingdino.tokenizer import BertTokenizer

logger = logging.getLogger("groundingdino")


class OnnxPredictor:
    def __init__(self, config):
        self.model_settings = ModelSettings.from_config(config)
        cfg = self.model_settings
        model_path = Path(cfg.model_path)
        tokenizer_path = Path(cfg.tokenizer_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        started = perf_counter()
        logger.info("加载模型 | path=%s | tokenizer=%s | provider=%s | device_id=%s",
                    model_path, tokenizer_path, cfg.provider, cfg.device_id)
        self.tokenizer = BertTokenizer(tokenizer_path)
        self.session = create_session(model_path, cfg.provider, cfg.threads, device_id=cfg.device_id)
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        if set(self.input_shapes) != set(INPUT_NAMES):
            raise ValueError(f"Unexpected ONNX model inputs: {self.input_shapes}")
        outputs = {item.name for item in self.session.get_outputs()}
        if not {"pred_logits", "pred_boxes"} <= outputs:
            raise ValueError(f"Unexpected ONNX model outputs: {outputs}")
        self.lock = threading.Lock()
        elapsed = (perf_counter() - started) * 1000
        logger.info("模型加载完成 | providers=%s | inputs=%s | cost_ms=%.2f",
                    self.session.get_providers(), self.input_shapes, elapsed)
        show_text("模型加载完成", model=str(model_path), providers=self.session.get_providers(),
                  spendTime=round(elapsed, 2))

    def predict(self, image_path, caption, box_threshold, text_threshold):
        result, _ = self.predict_with_metadata(
            image_path, caption, box_threshold, text_threshold, validate_thresholds=False,
        )
        return result

    def predict_with_metadata(self, image_path, caption, box_threshold, text_threshold,
                              *, validate_thresholds=True):
        """Return detections and input metadata; CLI validates threshold bounds."""
        cfg = self.model_settings
        started = perf_counter()
        with self.lock:
            acquired = perf_counter()
            feeds, metadata = prepare_inputs(
                image_path, caption, self.tokenizer, self.input_shapes,
                short_side=cfg.short_side, max_size=cfg.max_size, max_text_len=cfg.max_text_len,
                mean=cfg.mean, std=cfg.std,
            )
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
