"""GroundingDINO ONNX detector shared by the pipeline, HTTP service and CLI."""

import logging
import os
import threading
from pathlib import Path
from time import perf_counter

import numpy as np

from app.domain.groundingdino.postprocess import postprocess
from app.domain.groundingdino.preprocess import prepare_inputs
from app.domain.groundingdino.tokenizer import BertTokenizer

logger = logging.getLogger("groundingdino")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_path(value):
    """Resolve a configured path relative to the project root, independent of cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


class GdinoDetect:
    def __init__(self, config):
        if isinstance(config, (str, Path)):
            from app.core.config.loader import load_config
            config = load_config(config)

        cfg = config["GROUNDINGDINO_CONFIG"]
        self.model_path = str(_project_path(cfg["model_path"]))
        self.tokenizer_path = str(_project_path(cfg["tokenizer_path"]))
        self.provider = "cuda" if cfg.get("use_gpu", True) else "cpu"
        self.device_id = config.get("TOTAL_CONFIG", {}).get("device_id", 0)
        self.threads = cfg.get("threads", 4)
        self.short_side = cfg.get("short_side", 800)
        self.max_size = cfg.get("max_size", 1333)
        self.max_text_len = cfg.get("max_text_len", 256)
        self.mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
        self.caption = cfg.get("caption", "object .")
        self.box_threshold = cfg.get("box_threshold", 0.35)
        self.text_threshold = cfg.get("text_threshold", 0.25)
        self.lock = threading.Lock()
        self.ini_flag = False
        self.init_model()

    def init_model(self):
        started = perf_counter()
        self.tokenizer = BertTokenizer(self.tokenizer_path)
        self.session = create_session(self.model_path, self.provider, self.threads, device_id=self.device_id)
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        self.ini_flag = True
        logger.info("模型加载完成 path=%s | providers=%s | cost_ms=%.2f", self.model_path, self.session.get_providers(), (perf_counter() - started) * 1000)

    def warmup(self, runs=3):
        """Run dummy images through the same path as detect()."""
        _, _, h, w = self.input_shapes["images"]
        h = h if isinstance(h, int) and h > 0 else self.short_side
        w = w if isinstance(w, int) and w > 0 else self.short_side
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect(dummy)

    def _prepare_caption(self, text_range):
        if text_range is None:
            return self.caption
        if isinstance(text_range, str):
            return text_range.strip()
        return " . ".join(t.strip().rstrip(".").strip() for t in text_range) + " ."

    def preprocess(self, image, caption):
        """Build graph feeds from a BGR array, RGB PIL image or image path."""
        return prepare_inputs(
            image, caption, self.tokenizer, self.input_shapes,
            short_side=self.short_side, max_size=self.max_size, max_text_len=self.max_text_len,
            mean=self.mean, std=self.std,
        )

    def detect(self, imgbgr, text_range=None, box_threshold=None, text_threshold=None):
        result, _ = self.predict_with_metadata(
            imgbgr, self._prepare_caption(text_range),
            self.box_threshold if box_threshold is None else box_threshold,
            self.text_threshold if text_threshold is None else text_threshold,
        )
        boxes = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
        return boxes, np.asarray(result["logits"], dtype=np.float32), result["phrases"]

    def predict(self, image_path, caption, box_threshold, text_threshold):
        result, _ = self.predict_with_metadata(image_path, caption, box_threshold, text_threshold, validate_thresholds=False)
        return result

    def predict_with_metadata(self, image, caption, box_threshold, text_threshold, *, validate_thresholds=True):
        t0 = perf_counter()
        with self.lock:
            t1 = perf_counter()
            feeds, metadata = self.preprocess(image, caption)
            t2 = perf_counter()
            logits, boxes = self.session.run(["pred_logits", "pred_boxes"], feeds)
            t3 = perf_counter()
            result = postprocess(logits, boxes, self.tokenizer, metadata, box_threshold,
                                 text_threshold, validate_thresholds=validate_thresholds)
        t4 = perf_counter()
        logger.info("推理完成 | image=%s | detections=%d | queue=%.1fms pre=%.1fms infer=%.1fms post=%.1fms",
                    feeds["images"].shape, len(result["boxes"]),
                    (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000)
        return result, metadata


def create_session(model, provider="cpu", intra_op_num_threads=0, *, device_id=0,
                   log_severity_level=None):
    """Create an ORT session; if CUDA is requested it must actually be used."""
    import onnxruntime as ort

    if log_severity_level is None:
        from app.core.config.constant import ORT_LOG_SEVERITY_LEVEL
        log_severity_level = ORT_LOG_SEVERITY_LEVEL
    ort.set_default_logger_severity(log_severity_level)

    options = ort.SessionOptions()
    options.log_severity_level = log_severity_level
    if intra_op_num_threads:
        options.intra_op_num_threads = intra_op_num_threads

    if provider == "cuda":
        # Windows + pip-installed nvidia-* wheels: let ORT (>= 1.21) load the CUDA/cuDNN DLLs.
        if os.name == "nt" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        # use_tf32=0 keeps FP32 scores stable near the thresholds.
        providers = [("CUDAExecutionProvider", {"device_id": str(device_id), "use_tf32": "0"})]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    session.disable_fallback()
    return session


if __name__ == "__main__":
    from app.core.config.loader import load_config
    from app.core.observability.logging import setup_logger

    setup_logger()
    config = load_config()
    detector = GdinoDetect(config)
    image = config["GROUNDINGDINO_CONFIG"].get("test_image_path", "config/images/test.jpg")
    logger.info("检测结果：%s", detector.detect(_project_path(image)))
