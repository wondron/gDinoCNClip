"""Standalone ONNX food pipeline: GroundingDINO detection, then Chinese-CLIP on every detected box.

Copied from the service (app/domain/groundingdino, pipeline/a_gdinoDtct.py, pipeline/b_clipdtct.py,
food_service.py and scripts/infer_folder.py) so this folder runs without the project code or PyTorch.
Keep the numerics in sync when those files change. Every public call takes an optional `timings`
dict that accumulates milliseconds per stage (see STAGES).
"""

import ctypes
import io
import os
import site
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

import cn_tokenizer

INPUT_NAMES = (
    "images", "input_ids", "token_type_ids", "attention_mask", "position_ids",
    "text_self_attention_masks",
)
STAGES = (
    "total_ms", "decode_ms", "gdino_ms", "gdino_preprocess_ms", "gdino_infer_ms",
    "gdino_postprocess_ms", "clip_ms", "clip_image_infer_ms", "clip_text_infer_ms",
)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Hold NVIDIA DLL handles for the lifetime of this process.
_CUDA_DLL_DIRECTORIES = []
_CUDA_DLL_HANDLES = []


@contextmanager
def timed(timings, key):
    started = perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + (perf_counter() - started) * 1000


# ONNX Runtime ---------------------------------------------------------------

def _preload_windows_cuda():
    """Load NVIDIA pip-wheel CUDA 12 / cuDNN 9 DLLs without importing torch."""
    if os.name != "nt" or _CUDA_DLL_HANDLES:
        return
    roots = set(site.getsitepackages())
    roots.add(site.getusersitepackages())
    roots.update(path for path in sys.path if path)
    directories = {}
    for root in roots:
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            for directory in nvidia_root.glob("*/bin"):
                if directory.is_dir():
                    resolved = directory.resolve()
                    directories[str(resolved).lower()] = resolved
    if not directories:
        # Also permit a correctly installed system CUDA runtime on PATH.
        return
    for directory in sorted(directories.values()):
        _CUDA_DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
    files = {}
    for directory in directories.values():
        for path in directory.glob("*.dll"):
            files[path.name.lower()] = path.resolve()
    # Dependency order follows ORT's official preload_dlls implementation.
    names = [
        "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll",
        "cudnn_graph64_9.dll", "cudnn64_9.dll",
    ]
    missing = sorted({name.lower() for name in names} - set(files))
    if missing:
        raise RuntimeError("Incomplete NVIDIA CUDA 12 runtime packages; missing: " + ", ".join(missing))
    for name in names:
        _CUDA_DLL_HANDLES.append(ctypes.WinDLL(str(files[name.lower()])))


GPU_PROVIDERS = {"cuda": "CUDAExecutionProvider", "musa": "MUSAExecutionProvider"}
LOAD_HINTS = {
    "cuda": "on Linux start through run.sh so LD_LIBRARY_PATH contains the pip NVIDIA libraries",
    "musa": "use the ORT-MUSA image (onnxruntime 1.23.0+musa) and do not install requirements.txt in it: "
            "onnxruntime-gpu replaces the MUSA build",
}


def create_session(model, provider="cuda", threads=0, *, device_id=0, log_severity_level=3, profile_prefix=None):
    """Create an ORT session. Unlike the service, a requested GPU provider must load: no silent CPU fallback.

    With profile_prefix, ORT profiling is on until session.end_profiling() (see benchmark.node_placement).
    """
    import onnxruntime as ort

    if provider != "cpu" and provider not in GPU_PROVIDERS:
        raise ValueError(f"provider must be cpu or one of {sorted(GPU_PROVIDERS)}")
    ort.set_default_logger_severity(log_severity_level)
    options = ort.SessionOptions()
    options.log_severity_level = log_severity_level
    if threads:
        options.intra_op_num_threads = threads
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        _preload_windows_cuda()
        # use_tf32=0 keeps FP32 scores stable near the thresholds and comparable across GPUs.
        providers.insert(0, ("CUDAExecutionProvider", {"device_id": str(device_id), "use_tf32": "0"}))
    elif provider == "musa":
        providers.insert(0, ("MUSAExecutionProvider", {"device_id": str(device_id)}))
    if provider != "cpu" and GPU_PROVIDERS[provider] not in ort.get_available_providers():
        raise RuntimeError(f"onnxruntime {ort.__version__} has no {GPU_PROVIDERS[provider]}; {LOAD_HINTS[provider]}. "
                           f"Available providers: {ort.get_available_providers()}")
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if provider != "cpu" and session.get_providers()[0] != GPU_PROVIDERS[provider]:
        raise RuntimeError(f"{GPU_PROVIDERS[provider]} failed to load; {LOAD_HINTS[provider]}. "
                           f"Active providers: {session.get_providers()}")
    return session


# GroundingDINO --------------------------------------------------------------

class BertTokenizer:
    """Minimal local tokenizer interface used by GroundingDINO inference."""

    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer

        tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        # GroundingDINO constructs its sentence masks before truncating text.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.pad_token_id = self._tokenizer.token_to_id("[PAD]")
        self._unk_token_id = self._tokenizer.token_to_id("[UNK]")

    def __call__(self, texts, *, padding="longest", return_tensors="np"):
        if padding != "longest" or return_tensors != "np":
            raise ValueError("Only longest padding and NumPy tensors are supported")
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        length = max((len(encoding.ids) for encoding in encodings), default=0)
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        result = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for encoding in encodings:
            pad_length = length - len(encoding.ids)
            result["input_ids"].append(encoding.ids + [pad_id] * pad_length)
            result["token_type_ids"].append(encoding.type_ids + [0] * pad_length)
            result["attention_mask"].append(encoding.attention_mask + [0] * pad_length)
        return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}

    def convert_tokens_to_ids(self, tokens):
        ids = [self._tokenizer.token_to_id(token) for token in tokens]
        return [self._unk_token_id if token_id is None else token_id for token_id in ids]

    def decode(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=False)


def preprocess_caption(caption):
    caption = caption.lower().strip()
    return caption if caption.endswith(".") else caption + "."


def _fixed_dimension(value):
    return int(value) if isinstance(value, (int, np.integer)) and value > 0 else None


def _resize_shape(width, height, short_side=800, max_size=1333):
    """Match groundingdino.datasets.transforms.resize's integer rounding."""
    if min(width, height, short_side, max_size) <= 0:
        raise ValueError("Image dimensions and resize settings must be positive")
    size = int(short_side)
    if max(width, height) / min(width, height) * size > max_size:
        size = int(round(max_size * min(width, height) / max(width, height)))
    if (width <= height and width == size) or (height <= width and height == size):
        return height, width
    if width < height:
        return int(size * height / width), size
    return size, int(size * width / height)


def preprocess_image(image, image_shape=None, short_side=800, max_size=1333,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize a path or RGB PIL image to NCHW."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    else:
        image = image.convert("RGB")
    width, height = image.size
    target_height, target_width = _resize_shape(width, height, short_side, max_size)
    if image_shape is not None:
        if len(image_shape) != 4:
            raise ValueError(f"images input must be NCHW, got {image_shape}")
        batch, channels, fixed_height, fixed_width = map(_fixed_dimension, image_shape)
        if batch not in (None, 1) or channels not in (None, 3):
            raise ValueError(f"Only batch-one RGB graphs are supported: {image_shape}")
        if fixed_height is not None and fixed_width is not None:
            target_height, target_width = fixed_height, fixed_width
        elif fixed_height is not None:
            target_height = fixed_height
            target_width = max(1, int(fixed_height * width / height))
        elif fixed_width is not None:
            target_width = fixed_width
            target_height = max(1, int(fixed_width * height / width))
    resized = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    return tensor, {"original_size": [width, height], "resized_size": [target_width, target_height]}


def prepare_text_inputs(caption, tokenizer, max_text_len=256, fixed_text_len=None):
    """Build original GroundingDINO sentence masks before truncation/padding."""
    if max_text_len < 1:
        raise ValueError("max_text_len must be positive")
    if fixed_text_len is not None and not 1 <= fixed_text_len <= max_text_len:
        raise ValueError(f"Graph text length must be between 1 and {max_text_len}")
    caption = preprocess_caption(caption)
    tokenized = tokenizer([caption], padding="longest", return_tensors="np")
    input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
    token_types = np.asarray(tokenized.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int64)
    token_mask = np.asarray(tokenized["attention_mask"], dtype=np.bool_)
    length = input_ids.shape[1]
    special_ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    special_mask = np.isin(input_ids[0], special_ids)
    sentence_mask = np.eye(length, dtype=np.bool_)[None]
    positions = np.zeros((1, length), dtype=np.int64)
    previous_col = 0
    for col in np.flatnonzero(special_mask):
        if col == 0 or col == length - 1:
            sentence_mask[0, col, col] = True
            positions[0, col] = 0
        else:
            sentence_mask[0, previous_col + 1:col + 1, previous_col + 1:col + 1] = True
            positions[0, previous_col + 1:col + 1] = np.arange(col - previous_col)
        previous_col = col
    target_length = min(length, max_text_len) if fixed_text_len is None else fixed_text_len
    kept = min(length, target_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    ids_out = np.full((1, target_length), pad_id, dtype=np.int64)
    types_out = np.zeros((1, target_length), dtype=np.int64)
    mask_out = np.zeros((1, target_length), dtype=np.bool_)
    positions_out = np.zeros((1, target_length), dtype=np.int64)
    sentence_out = np.eye(target_length, dtype=np.bool_)[None]
    ids_out[:, :kept] = input_ids[:, :kept]
    types_out[:, :kept] = token_types[:, :kept]
    mask_out[:, :kept] = token_mask[:, :kept]
    positions_out[:, :kept] = positions[:, :kept]
    sentence_out[:, :kept, :kept] = sentence_mask[:, :kept, :kept]
    return {
        "input_ids": np.ascontiguousarray(ids_out),
        "token_type_ids": np.ascontiguousarray(types_out),
        "attention_mask": np.ascontiguousarray(mask_out),
        "position_ids": np.ascontiguousarray(positions_out),
        "text_self_attention_masks": np.ascontiguousarray(sentence_out),
    }, {"caption": caption, "token_ids": input_ids[0, :kept].tolist()}


def prepare_inputs(image, caption, tokenizer, input_shapes=None, short_side=800,
                   max_size=1333, max_text_len=256,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Return (feed dictionary, metadata) for a batch-one ONNX model."""
    shapes = input_shapes or {}
    images, metadata = preprocess_image(image, shapes.get("images"), short_side, max_size, mean, std)
    fixed_lengths = set()
    for name in INPUT_NAMES[1:]:
        for dimension in (shapes.get(name) or [])[1:]:
            fixed = _fixed_dimension(dimension)
            if fixed is not None:
                fixed_lengths.add(fixed)
    if len(fixed_lengths) > 1:
        raise ValueError(f"Conflicting fixed graph text lengths: {sorted(fixed_lengths)}")
    text, text_metadata = prepare_text_inputs(caption, tokenizer, max_text_len, next(iter(fixed_lengths), None))
    metadata.update(text_metadata)
    return {"images": images, **text}, metadata


def postprocess(pred_logits, pred_boxes, tokenizer, metadata, box_threshold=0.35, text_threshold=0.25):
    """Match original predict(): strict thresholds, normalized cxcywh, no NMS."""
    if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise ValueError("Thresholds must be between zero and one")
    pred_logits = np.asarray(pred_logits)
    pred_boxes = np.asarray(pred_boxes)
    if np.isnan(pred_logits).any() or not np.isfinite(pred_boxes).all():
        raise ValueError("ONNX outputs contain invalid numeric values")
    probabilities = np.exp(-np.logaddexp(np.float32(0), -pred_logits[0]))
    scores = probabilities.max(axis=1)
    selected = scores > box_threshold
    token_ids = metadata["token_ids"]
    phrases = []
    for row in probabilities[selected]:
        # get_phrases_from_posmap excludes token 0 and all indices >=255.
        indices = np.flatnonzero(row > text_threshold)
        indices = indices[(indices > 0) & (indices < min(255, len(token_ids)))]
        phrases.append(tokenizer.decode([token_ids[i] for i in indices]).replace(".", ""))
    return {"boxes": pred_boxes[0, selected].tolist(), "logits": scores[selected].tolist(), "phrases": phrases}


def nms(boxes, scores, iou_threshold, labels=None):
    boxes = np.asarray(boxes, dtype=np.float64)
    lt = boxes[:, :2] - boxes[:, 2:] / 2
    rb = boxes[:, :2] + boxes[:, 2:] / 2
    area = np.prod(np.clip(rb - lt, 0, None), axis=1)
    labels = None if labels is None else np.asarray(labels, dtype=object)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while order.size:
        i, order = order[0], order[1:]
        keep.append(i)
        wh = np.clip(np.minimum(rb[i], rb[order]) - np.maximum(lt[i], lt[order]), 0, None)
        inter = wh[:, 0] * wh[:, 1]
        suppress = inter / np.maximum(area[i] + area[order] - inter, 1e-12) > iou_threshold
        if labels is not None:
            suppress &= labels[order] == labels[i]
        order = order[~suppress]
    return np.sort(np.asarray(keep, dtype=np.int64))


class GroundingDino:
    """GdinoDetect equivalent; detect() matches GdinoDetect.detect() including per-phrase NMS."""

    def __init__(self, model_path, tokenizer_path, provider="cuda", device_id=0, threads=4,
                 short_side=800, max_size=1333, max_text_len=256, nms_threshold=0.5, nms_agnostic=False):
        self.short_side = short_side
        self.max_size = max_size
        self.max_text_len = max_text_len
        self.nms_threshold = nms_threshold
        self.nms_agnostic = nms_agnostic
        self.tokenizer = BertTokenizer(tokenizer_path)
        self.session = create_session(model_path, provider, threads, device_id=device_id)
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}

    def detect(self, image, caption, box_threshold, text_threshold, timings=None):
        with timed(timings, "gdino_preprocess_ms"):
            feeds, metadata = prepare_inputs(image, caption.strip(), self.tokenizer, self.input_shapes,
                                             short_side=self.short_side, max_size=self.max_size,
                                             max_text_len=self.max_text_len)
        with timed(timings, "gdino_infer_ms"):
            logits, boxes = self.session.run(["pred_logits", "pred_boxes"], feeds)
        with timed(timings, "gdino_postprocess_ms"):
            result = postprocess(logits, boxes, self.tokenizer, metadata, box_threshold, text_threshold)
            boxes = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(result["logits"], dtype=np.float32)
            phrases = result["phrases"]
            iou = self.nms_threshold
            if iou is not None and iou < 1 and len(boxes) > 1:
                keep = nms(boxes, scores, iou, labels=None if self.nms_agnostic else phrases)
                boxes, scores, phrases = boxes[keep], scores[keep], [phrases[i] for i in keep]
        return boxes, scores, phrases


# Chinese-CLIP ---------------------------------------------------------------

def box2xyxy(image, boxes):
    """Normalized cxcywh -> integer pixel xyxy, as app/domain/common/image_utils.box2xyxy."""
    w, h = image.size
    boxes = np.asarray(boxes, dtype=np.float32) * np.array([w, h, w, h], dtype=np.float32)
    cx, cy, bw, bh = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1)
    return xyxy.astype(int).tolist()


class ChineseClip:
    """ClipDetect equivalent: text features of the food, common and recipe ranges are computed at load time."""

    def __init__(self, image_model, text_model, food_range, common_range, recipe_range, provider="cuda",
                 device_id=0):
        if not food_range and not common_range:
            raise ValueError("food_range and common_range must not both be empty")
        self.image_session = create_session(image_model, provider, device_id=device_id)
        self.text_session = create_session(text_model, provider, device_id=device_id)
        self.input_name = self.image_session.get_inputs()[0].name
        self.output_name = self.image_session.get_outputs()[0].name
        self.shape = self.image_session.get_inputs()[0].shape
        self.text_input_name = self.text_session.get_inputs()[0].name
        self.text_output_name = self.text_session.get_outputs()[0].name
        self.text_shape = self.text_session.get_inputs()[0].shape
        self.recipe_range = list(recipe_range)
        self.food_feats, self.common_feats, self.recipe_feats = (
            dict(zip(names, self.extract_text_feature(list(names)))) if names else {}
            for names in (food_range, common_range, recipe_range))

    def extract_image_feature(self, img_bgr, timings=None):
        img = cv2.resize(img_bgr, (self.shape[2], self.shape[3]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = ((img - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)[None].astype(np.float32)
        with timed(timings, "clip_image_infer_ms"):
            feats = self.image_session.run([self.output_name], {self.input_name: img})[0]
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-7)

    def extract_text_feature(self, text_list, timings=None):
        seq_len = self.text_shape[1] if len(self.text_shape) > 1 else 52
        tokens = np.array(cn_tokenizer.tokenize(text_list, context_length=seq_len), dtype=np.int64)
        with timed(timings, "clip_text_infer_ms"):
            if isinstance(self.text_shape[0], int) and self.text_shape[0] == 1:
                output = np.concatenate([self.text_session.run([self.text_output_name],
                                                               {self.text_input_name: tokens[i:i + 1]})[0]
                                         for i in range(len(tokens))])
            else:
                output = self.text_session.run([self.text_output_name], {self.text_input_name: tokens})[0]
        return output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-7)

    def text_candidates(self, input_name, timings=None):
        """(names, feature matrix) input_name competes with: recipes only against common_range, others
        against food + common; a caption outside every range is encoded once per request."""
        if input_name in self.recipe_range:
            pool = self.common_feats
        else:
            pool = {**self.food_feats, **self.common_feats}
        names, feats = list(pool), list(pool.values())
        if input_name not in pool:
            feat = self.recipe_feats.get(input_name)
            if feat is None:
                feat = self.extract_text_feature([input_name], timings)[0]
            names.append(input_name)
            feats.append(feat)
        return names, np.stack(feats)

    def detect(self, img_bgr, input_name, candidates, timings=None):
        """Return the top cosine score if input_name is the best match, otherwise 0.0."""
        img_feat = self.extract_image_feature(img_bgr, timings)
        names, feats = candidates
        scores = feats @ img_feat[0]
        best = int(np.argmax(scores))
        return float(scores[best]) if names[best] == input_name else 0.0

    def batch_detect(self, image, boxes, clip_caption, timings=None):
        """Crop normalized cxcywh boxes; return pixel xyxy rects whose best CLIP match is clip_caption."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes = boxes[np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > 0).all(axis=1)]
        if not len(boxes):
            return [], []
        candidates = self.text_candidates(clip_caption, timings)
        width, height = image.size
        rects, confidences = [], []
        for rect in box2xyxy(image, boxes):
            x1, y1, x2, y2 = rect
            if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
                continue
            crop_bgr = cv2.cvtColor(np.asarray(image.crop(rect)), cv2.COLOR_RGB2BGR)
            score = self.detect(crop_bgr, clip_caption, candidates, timings)
            if np.isfinite(score) and score > 0:
                rects.append(rect)
                confidences.append(score)
        return rects, confidences


def food_detect(dino, clip, image_bytes, dino_cap, dino_box, dino_text, clip_cap, clip_thre, timings=None):
    """Same flow as Detector.detect() behind /food_detect/; errors propagate instead of returning ([], [])."""
    with timed(timings, "total_ms"):
        with timed(timings, "decode_ms"):
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        with timed(timings, "gdino_ms"):
            boxes, scores, phrases = dino.detect(image, dino_cap, dino_box, dino_text, timings)
        if clip_thre <= 0:
            keep = [i for i, phrase in enumerate(phrases) if dino_cap in phrase]
            rects, confidences = box2xyxy(image, boxes[keep]), [0.6] * len(keep)
        else:
            with timed(timings, "clip_ms"):
                rects, confidences = clip.batch_detect(image, boxes, clip_cap, timings)
    return {"original_size": list(image.size),
            "gdino": {"boxes": boxes.tolist(), "scores": scores.tolist(), "phrases": phrases},
            "rects": rects, "confidences": confidences}
