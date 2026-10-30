"""Chinese CLIP inference using the image and text ONNX models."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import app.domain.common.tokenizer as tokenizer
from app.domain.common.image_utils import box2xyxy
from app.domain.pipeline.a_gdinoDtct import create_session

logger = logging.getLogger("clip")


class ClipDetect:
    def __init__(self, config_data: Dict[str, Any] | str | os.PathLike):
        if isinstance(config_data, (str, os.PathLike)):
            from app.core.config.loader import load_config

            config_data = load_config(config_data)
        self._load_config(config_data)
        self.init_model()

    def _load_config(self, config):
        clip_cfg = config.get("CLIP_CONFIG", {})
        total_cfg = config.get("TOTAL_CONFIG", {})
        if not isinstance(clip_cfg, dict) or not isinstance(total_cfg, dict):
            raise ValueError("CLIP_CONFIG and TOTAL_CONFIG must be mappings")
        project_root = Path(__file__).resolve().parents[3]

        def configured_path(key):
            value = clip_cfg.get(key)
            if value is None:
                return None
            path = Path(value).expanduser()
            return str((path if path.is_absolute() else project_root / path).resolve())

        self.model_path = configured_path("model_path")
        self.text_model_path = configured_path("text_model_path")
        self.use_gpu = clip_cfg.get("use_gpu", True)
        if not isinstance(self.use_gpu, bool):
            raise ValueError("CLIP_CONFIG.use_gpu must be true or false")
        self.provider = "cuda" if self.use_gpu else "cpu"
        self.device_id = total_cfg.get("device_id", 0)
        self.threads = clip_cfg.get("threads", 4)
        for name, value in (("device_id", self.device_id), ("threads", self.threads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.context_length = clip_cfg.get("context_length", 52)
        self.image_size = clip_cfg.get("image_size", 224)
        for name, value in (("context_length", self.context_length), ("image_size", self.image_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"CLIP_CONFIG.{name} must be a positive integer")
        self.threshold = float(clip_cfg.get("threshold", 0.5))
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("CLIP_CONFIG.threshold must be between zero and one")
        self.topk = clip_cfg.get("topk", 5)

    def get_providers(self):
        return ["CUDAExecutionProvider" if self.use_gpu else "CPUExecutionProvider"]

    def init_model(self):
        self.ini_flag = False
        self.ini_flag = self.load_image_model() and self.load_text_model()
        if self.ini_flag:
            logger.info("CLIP模型初始化完成！")
        else:
            logger.error("CLIP模型初始化失败！")

    def warmup(self, runs=3):
        height = self.shape[2] if isinstance(self.shape[2], int) and self.shape[2] > 0 else self.image_size
        width = self.shape[3] if isinstance(self.shape[3], int) and self.shape[3] > 0 else self.image_size
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.perf_counter()
        for _ in range(max(1, runs)):
            self.extract_image_feature(dummy)
            self.extract_text_feature_with_model(["warmup"])
        logger.info("预热完成 | runs=%d | cost_ms=%.2f", max(1, runs),
                    (time.perf_counter() - started) * 1000)

    def load_image_model(self):
        try:
            if not self.model_path or not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"CLIP model not found: {self.model_path}")
            self.sess = create_session(self.model_path, self.provider, self.threads, device_id=self.device_id)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
            self.shape = self.sess.get_inputs()[0].shape
            logger.info("clip imgs 输入名：%s", self.input_name)
            logger.info("clip imgs 输出名：%s", self.output_name)
            return True
        except Exception as exc:
            logger.error("加载CLIP图像模型失败：%s", exc)
            return False

    def load_text_model(self):
        try:
            if not self.text_model_path or not os.path.isfile(self.text_model_path):
                raise FileNotFoundError(f"CLIP text model not found: {self.text_model_path}")
            self.text_sess = create_session(self.text_model_path, self.provider, self.threads,
                                            device_id=self.device_id)
            self.text_input_name = self.text_sess.get_inputs()[0].name
            self.text_output_name = self.text_sess.get_outputs()[0].name
            self.text_shape = self.text_sess.get_inputs()[0].shape
            logger.info("clip text 输入名：%s", self.text_input_name)
            logger.info("clip text 输出名：%s", self.text_output_name)
            return True
        except Exception as exc:
            logger.error("加载CLIP文本模型失败：%s", exc)
            return False

    def preprocess_image(self, img_bgr, *, reference=False):
        try:
            if (not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3
                    or img_bgr.shape[2] != 3 or img_bgr.size == 0 or img_bgr.dtype != np.uint8):
                raise ValueError("CLIP images must be nonempty uint8 BGR arrays with shape [H,W,3]")
            height = self.shape[2] if isinstance(self.shape[2], int) and self.shape[2] > 0 else self.image_size
            width = self.shape[3] if isinstance(self.shape[3], int) and self.shape[3] > 0 else self.image_size
            if reference:
                image = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
                source_width, source_height = image.size
                side = max(width, height)
                if source_width <= source_height:
                    resized_width, resized_height = side, int(side * source_height / source_width)
                else:
                    resized_width, resized_height = int(side * source_width / source_height), side
                image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
                left = int(round((resized_width - width) / 2.0))
                top = int(round((resized_height - height) / 2.0))
                img = np.asarray(image.crop((left, top, left + width, top + height)))
            else:
                img = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            img = (img - mean) / std
            return np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        except Exception as exc:
            logger.error("图片处理失败：%s", exc)
            return None

    def extract_image_feature(self, img, *, reference=False):
        try:
            if not self.ini_flag:
                raise RuntimeError("CLIP model is not initialized")
            pixels = self.preprocess_image(img, reference=reference)
            if pixels is None:
                return None
            features = self.sess.run([self.output_name], {self.input_name: pixels})[0]
            return features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-7)
        except Exception as exc:
            logger.error("CLIP图像特征提取失败：%s", exc)
            return None

    def extract_text_feature_with_model(self, text_list=None):
        """Encode the supplied text directly; no stored text features are used."""
        if isinstance(text_list, str):
            text_list = [text_list]
        if (not isinstance(text_list, (list, tuple)) or not text_list
                or any(not isinstance(text, str) or not text.strip() for text in text_list)):
            raise ValueError("Provide one or more nonempty CLIP text prompts")
        text_list = [text.strip() for text in text_list]
        seq_len = self.text_shape[1] if len(self.text_shape) > 1 else self.context_length
        if not isinstance(seq_len, int) or seq_len < 1:
            seq_len = self.context_length
        tokens = np.asarray(tokenizer.tokenize(text_list, context_length=seq_len), dtype=np.int64)
        expected_bs = self.text_shape[0]
        if not isinstance(expected_bs, int) or expected_bs in (0, -1) or expected_bs == len(tokens):
            features = self.text_sess.run([self.text_output_name], {self.text_input_name: tokens})[0]
        elif expected_bs == 1:
            features = np.concatenate([
                self.text_sess.run([self.text_output_name], {self.text_input_name: row[None]})[0]
                for row in tokens
            ], axis=0)
        else:
            raise ValueError(f"CLIP text model expects batch {expected_bs}, got {len(tokens)}")
        features = np.asarray(features, dtype=np.float32)
        if (features.ndim != 2 or features.shape[0] != len(text_list) or features.shape[1] == 0
                or not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0)):
            raise RuntimeError("CLIP text model returned invalid features")
        features /= np.linalg.norm(features, axis=-1, keepdims=True) + 1e-7
        return text_list, features

    def extract_text_feature(self, text_list=None):
        return self.extract_text_feature_with_model(text_list)

    def _similarity_scores(self, image_features, text_features):
        """Map normalized image/text cosine similarities from [-1,1] to [0,1]."""
        image_features = np.asarray(image_features, dtype=np.float64)
        text_features = np.asarray(text_features, dtype=np.float64)
        if image_features.ndim == 2 and image_features.shape[0] == 1:
            image_features = image_features[0]
        if (image_features.ndim != 1 or text_features.ndim != 2
                or image_features.size == 0 or text_features.shape[1] != image_features.size
                or not np.isfinite(image_features).all() or not np.isfinite(text_features).all()
                or np.linalg.norm(image_features) == 0
                or np.any(np.linalg.norm(text_features, axis=1) == 0)):
            raise RuntimeError("CLIP feature extraction returned invalid features")
        if text_features.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        cosine = text_features @ image_features
        if not np.isfinite(cosine).all():
            raise RuntimeError("CLIP similarity is not finite")
        return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

    def _prompt_features(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("CLIP text prompt must not be empty")
        _, features = self.extract_text_feature_with_model([text.strip()])
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[0] != 1:
            raise RuntimeError("CLIP text feature extraction failed")
        return features

    def batch_detect(self, image_data: Image.Image, boxes, single_text: str,
                     threshold: Optional[float] = None):
        """Filter pixel xyxy rectangles by mapped cosine similarity scores."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return [], []
        if not self.ini_flag:
            raise RuntimeError("CLIP model is not initialized")
        threshold = float(self.threshold if threshold is None else threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("CLIP threshold must be between zero and one")
        if not isinstance(image_data, Image.Image):
            raise TypeError("CLIP batch_detect requires a PIL image")
        text_features = self._prompt_features(single_text)
        image = image_data.convert("RGB")
        rects = box2xyxy(image, boxes)
        selected_rects, confidences = [], []
        for rect in rects:
            crop = image.crop(tuple(rect))
            bgr = np.ascontiguousarray(np.asarray(crop)[:, :, ::-1])
            image_features = self.extract_image_feature(bgr, reference=True)
            score = float(self._similarity_scores(image_features, text_features)[0])
            if score > threshold:
                selected_rects.append(rect)
                confidences.append(score)
        return selected_rects, confidences

    def similarity(self, img_bgr: np.ndarray, text: str) -> float:
        """Return (cosine + 1) / 2 for a directly encoded text prompt."""
        if not self.ini_flag:
            raise RuntimeError("CLIP model is not initialized")
        text_features = self._prompt_features(text)
        image_features = self.extract_image_feature(img_bgr, reference=True)
        return float(self._similarity_scores(image_features, text_features)[0])

    def detect_with_text_model(self, img: np.ndarray, text_range=None) -> List[Tuple[str, float]]:
        if not self.ini_flag:
            logger.error("CLIP模型未初始化！")
            return []
        try:
            texts, text_features = self.extract_text_feature_with_model(text_range)
            image_features = self.extract_image_feature(img)
            scores = self._similarity_scores(image_features, text_features)
            indices = np.argsort(-scores)[:min(self.topk, len(scores))]
            results = [(texts[index], float(scores[index])) for index in indices
                       if scores[index] >= self.threshold]
            logger.info("CLIP模型识别结果：%s", results)
            return results
        except Exception as exc:
            logger.error("CLIP模型识别失败：%s", exc)
            return []

    def detect(self, imgbgr: np.ndarray, text_range=None) -> List[Tuple[str, float]]:
        return self.detect_with_text_model(imgbgr, text_range)


if __name__ == "__main__":
    from app.core.config.loader import load_config
    from app.core.observability.logging import setup_logger

    setup_logger()
    config = load_config()
    image_path = Path(config["GROUNDINGDINO_CONFIG"].get("test_image_path", "config/images/test.jpg"))
    if not image_path.is_absolute():
        image_path = Path(__file__).resolve().parents[3] / image_path
    clip = ClipDetect(config)
    print(clip.detect(cv2.imread(str(image_path)), ["鱼", "虾", "鸡肉"]))
