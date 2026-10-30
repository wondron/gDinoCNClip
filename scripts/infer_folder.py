"""Standalone GroundingDINO ONNX folder inference.

All model loading, tokenization, preprocessing, inference, decoding, timing and
JSON writing are implemented here. No other project Python modules are imported.

Edit the configuration block below, then run:
    python scripts/infer_folder.py
"""

import ctypes
import json
import logging
import os
import site
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# 在这里修改参数，运行时无需传入命令行参数。
INPUT_DIR = ROOT / "config/images"  # 例如 Path(r"D:\images")
MODEL_PATH = ROOT / "config/model/groundingdino/groundingdino_swint_ogc.onnx"
TOKENIZER_PATH = ROOT / "config/model/groundingdino/bert-base-uncased"
OUTPUT_JSON = ROOT / "output/batch_predictions.json"
CAPTION = "glass . straw . lemon ."
PROVIDER = "cuda"  # "cuda" 或 "cpu"
RECURSIVE = False  # True 时扫描子目录
DEVICE_ID = 0
THREADS = 4
SHORT_SIDE = 800
MAX_SIZE = 1333
MAX_TEXT_LEN = 256
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
ORT_LOG_SEVERITY = 3  # 0详细、1信息、2警告、3错误、4致命

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".jfif"}
logger = logging.getLogger("onnx.folder")

# Hold NVIDIA DLL handles for the lifetime of this process.
_CUDA_DLL_DIRECTORIES = []
_CUDA_DLL_HANDLES = []

INPUT_NAMES = (
    "images", "input_ids", "token_type_ids", "attention_mask", "position_ids",
    "text_self_attention_masks",
)


def _preload_windows_cuda():
    """Load NVIDIA pip-wheel CUDA 12 / cuDNN 9 DLLs without importing torch."""
    if os.name != "nt" or _CUDA_DLL_HANDLES:
        return
    loaded_torch = sys.modules.get("torch")
    torch_cuda = getattr(getattr(loaded_torch, "version", None), "cuda", None)
    if torch_cuda and not torch_cuda.startswith("12."):
        raise RuntimeError(
            f"PyTorch CUDA {torch_cuda} is already loaded in this process, but this "
            "ONNX Runtime CUDA provider requires CUDA 12 / cuDNN 9. Run ONNX GPU "
            "inference in a separate process to avoid conflicting CUDA DLLs."
        )
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
    # Dependency order follows ORT's official preload_dlls implementation;
    # CUDA runtime is loaded first, before the libraries that use it.
    names = [
        "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll",
        "cudnn_graph64_9.dll", "cudnn64_9.dll",
    ]
    required = {name.lower() for name in names}
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(
            "Incomplete NVIDIA CUDA 12 runtime packages; missing: " + ", ".join(missing)
            + ". Install nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, "
            "nvidia-cufft-cu12 and nvidia-cudnn-cu12."
        )
    try:
        for name in names:
            path = files.get(name.lower())
            if path is not None:
                _CUDA_DLL_HANDLES.append(ctypes.WinDLL(str(path)))
    except OSError as exc:
        _CUDA_DLL_HANDLES.clear()
        raise RuntimeError(f"Could not preload NVIDIA CUDA runtime DLL {path}: {exc}") from exc


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
    """Return normalized NCHW float32 array and JSON-compatible image metadata."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    else:
        image = image.convert("RGB")
    width, height = image.size
    target_height, target_width = _resize_shape(width, height, short_side, max_size)
    policy = "short_side_preserving_aspect_ratio"
    if image_shape is not None:
        if len(image_shape) != 4:
            raise ValueError(f"images input must be NCHW, got {image_shape}")
        batch, channels, fixed_height, fixed_width = map(_fixed_dimension, image_shape)
        if batch not in (None, 1) or channels not in (None, 3):
            raise ValueError(f"Only batch-one RGB graphs are supported: {image_shape}")
        if fixed_height is not None and fixed_width is not None:
            target_height, target_width = fixed_height, fixed_width
            policy = "fixed_model_dimensions"
        elif fixed_height is not None:
            target_height = fixed_height
            target_width = max(1, int(fixed_height * width / height))
            policy = "fixed_height_preserving_aspect_ratio"
        elif fixed_width is not None:
            target_width = fixed_width
            target_height = max(1, int(fixed_width * height / width))
            policy = "fixed_width_preserving_aspect_ratio"
    resized = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if (mean_array.shape != (3,) or std_array.shape != (3,)
            or not np.isfinite(mean_array).all() or not np.isfinite(std_array).all()
            or np.any(std_array <= 0)):
        raise ValueError("mean/std must have three finite RGB values; std must be positive")
    pixels = (pixels - mean_array) / std_array
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    return tensor, {
        "original_size": [width, height],
        "resized_size": [target_width, target_height],
        "resize_policy": policy,
    }


def prepare_text_inputs(caption, tokenizer, max_text_len=256, fixed_text_len=None):
    """Build original GroundingDINO sentence masks before truncation/padding.

    Returns (input_arrays, metadata). All arrays have batch size one. The BERT
    tokenizer remains outside the ONNX graph so captions can change at runtime.
    """
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
    }, {
        "caption": caption,
        "token_ids": input_ids[0, :kept].tolist(),
        "token_count_before_truncation": int(length),
        "text_length": target_length,
    }


def prepare_inputs(image, caption, tokenizer, input_shapes=None, short_side=800,
                   max_size=1333, max_text_len=256,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Return (feed dictionary, metadata) for a batch-one ONNX model.

    input_shapes is an optional mapping from ONNX input names to declared shapes.
    Symbolic dimensions (strings/None) remain dynamic. Static text inputs are
    padded with the tokenizer pad id; truncation is performed after mask creation.
    """
    shapes = input_shapes or {}
    images, metadata = preprocess_image(image, shapes.get("images"), short_side, max_size, mean, std)
    fixed_lengths = set()
    for name in INPUT_NAMES[1:]:
        shape = shapes.get(name)
        if shape is None:
            continue
        expected_rank = 3 if name == "text_self_attention_masks" else 2
        if len(shape) != expected_rank or _fixed_dimension(shape[0]) not in (None, 1):
            raise ValueError(f"Unsupported {name} input shape: {shape}")
        for dimension in shape[1:]:
            fixed = _fixed_dimension(dimension)
            if fixed is not None:
                fixed_lengths.add(fixed)
    if len(fixed_lengths) > 1:
        raise ValueError(f"Conflicting fixed graph text lengths: {sorted(fixed_lengths)}")
    fixed_length = next(iter(fixed_lengths), None)
    text, text_metadata = prepare_text_inputs(caption, tokenizer, max_text_len, fixed_length)
    metadata.update(text_metadata)
    feeds = {"images": images, **text}
    for name, shape in shapes.items():
        if name not in feeds:
            raise ValueError(f"Unknown graph input {name!r}; supported inputs: {INPUT_NAMES}")
        actual = feeds[name].shape
        if len(shape) != len(actual) or any(
            _fixed_dimension(expected) is not None and int(expected) != observed
            for expected, observed in zip(shape, actual)
        ):
            raise ValueError(f"Graph input {name} expects {shape}, prepared {actual}")
    return feeds, metadata


def postprocess(pred_logits, pred_boxes, tokenizer, metadata, box_threshold=0.35,
                text_threshold=0.25, *, validate_thresholds=True):
    """Match original predict(): strict thresholds, normalized cxcywh, no NMS."""
    if validate_thresholds and (not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1):
        raise ValueError("Thresholds must be between zero and one")
    pred_logits = np.asarray(pred_logits)
    pred_boxes = np.asarray(pred_boxes)
    if pred_logits.ndim != 3 or pred_logits.shape[0] != 1:
        raise ValueError(f"Expected batch-one logits [1,Q,T], got {pred_logits.shape}")
    if pred_boxes.shape != (1, pred_logits.shape[1], 4):
        raise ValueError(f"Expected boxes [1,Q,4], got {pred_boxes.shape}")
    if np.isnan(pred_logits).any() or not np.isfinite(pred_boxes).all():
        raise ValueError("ONNX outputs contain invalid numeric values")
    probabilities = np.exp(-np.logaddexp(np.float32(0), -pred_logits[0]))
    scores = probabilities.max(axis=1)
    selected = scores > box_threshold
    selected_logits = probabilities[selected]
    token_ids = metadata["token_ids"]
    phrases = []
    for row in selected_logits:
        # get_phrases_from_posmap excludes token 0 and all indices >=255.
        indices = np.flatnonzero(row > text_threshold)
        indices = indices[(indices > 0) & (indices < min(255, len(token_ids)))]
        phrases.append(tokenizer.decode([token_ids[i] for i in indices]).replace(".", ""))
    return {
        "boxes": pred_boxes[0, selected].tolist(),
        "logits": scores[selected].tolist(),
        "phrases": phrases,
    }


def create_session(model, provider="cpu", intra_op_num_threads=0, *, device_id=0,
                   log_severity_level=3):
    """Create ORT session; a requested CUDA provider must actually initialize."""
    import onnxruntime as ort

    if (isinstance(log_severity_level, bool) or not isinstance(log_severity_level, int)
            or not 0 <= log_severity_level <= 4):
        raise ValueError("ORT_LOG_SEVERITY_LEVEL must be an integer from 0 to 4")
    if provider not in ("cpu", "cuda"):
        raise ValueError("provider must be cpu or cuda")
    if intra_op_num_threads < 0 or device_id < 0:
        raise ValueError("threads and device_id must be nonnegative")
    # Graph partition warnings use the session logger; CUDA ScatterND kernel
    # warnings use the default logger. Configure both, preserving errors.
    ort.set_default_logger_severity(log_severity_level)
    requested = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    available = ort.get_available_providers()
    logger.info("ONNX Runtime providers | available=%s | requested=%s | device_id=%d | ort_log_severity=%d",
                available, requested, device_id, log_severity_level)
    if requested not in available:
        raise RuntimeError(f"{requested} unavailable; installed ORT providers: {available}")
    if provider == "cuda":
        _preload_windows_cuda()
    options = ort.SessionOptions()
    options.log_severity_level = log_severity_level
    if intra_op_num_threads:
        options.intra_op_num_threads = intra_op_num_threads
    # TF32 can change detection scores around thresholds; prefer FP32 agreement
    # with the reference model when CUDA is explicitly requested.
    providers = [(requested, {"use_tf32": "0", "device_id": str(device_id)})] if provider == "cuda" else [requested]
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    session.disable_fallback()
    if provider == "cuda" and session.get_providers()[0] != requested:
        raise RuntimeError(
            "CUDA provider initialization failed; refusing to run on CPU. "
            f"Active providers: {session.get_providers()}"
        )
    return session

class OnnxPredictor:
    """Load the model/tokenizer once and reuse them for every input image."""

    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH,
                 provider="cuda", device_id=0, threads=4, short_side=800,
                 max_size=1333, max_text_len=256, ort_log_severity=3):
        self.model_path = Path(model_path).expanduser().resolve()
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_path}")
        if min(short_side, max_size, max_text_len) < 1 or max_text_len > 256:
            raise ValueError("Resize sizes must be positive; max_text_len must be in [1, 256]")
        self.short_side = short_side
        self.max_size = max_size
        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer(self.tokenizer_path)
        self.session = create_session(
            self.model_path, provider, threads, device_id=device_id,
            log_severity_level=ort_log_severity,
        )
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        if set(self.input_shapes) != set(INPUT_NAMES):
            raise ValueError(f"Unexpected ONNX inputs: {self.input_shapes}")
        outputs = {item.name for item in self.session.get_outputs()}
        if not {"pred_logits", "pred_boxes"} <= outputs:
            raise ValueError(f"Unexpected ONNX outputs: {sorted(outputs)}")

    def predict_with_metadata(self, image, caption, box_threshold, text_threshold):
        started = perf_counter()
        feeds, metadata = prepare_inputs(
            image, caption, self.tokenizer, self.input_shapes,
            short_side=self.short_side, max_size=self.max_size, max_text_len=self.max_text_len,
        )
        prepared = perf_counter()
        # session.run returns host outputs; this timing includes execution and
        # the transfers needed by ORT, and excludes Python preprocessing/decoding.
        logits, boxes = self.session.run(["pred_logits", "pred_boxes"], feeds)
        inferred = perf_counter()
        result = postprocess(logits, boxes, self.tokenizer, metadata, box_threshold, text_threshold)
        ended = perf_counter()
        metadata["timings_ms"] = {
            "preprocess_ms": (prepared - started) * 1000,
            "inference_ms": (inferred - prepared) * 1000,
            "postprocess_ms": (ended - inferred) * 1000,
            "total_ms": (ended - started) * 1000,
        }
        return result, metadata


def discover_images(input_dir, recursive=False):
    """Return supported image paths in stable, case-insensitive relative order."""
    directory = Path(input_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Input path is not a directory: {directory}")
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    images = [path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda path: (path.relative_to(directory).as_posix().casefold(),
                                            path.relative_to(directory).as_posix()))


def save_json(report, output):
    """Atomically replace the JSON snapshot, keeping the last completed image."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_batch(input_dir, caption, output, predictor, *, box_threshold=0.35,
              text_threshold=0.25, recursive=False, model_load_ms=0.0):
    """Infer every image, recording errors while continuing with other images."""
    if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise ValueError("Thresholds must be between zero and one")
    input_dir = Path(input_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    images = discover_images(input_dir, recursive)
    if not images:
        raise ValueError(f"No supported images found in: {input_dir}")
    if output in {path.resolve() for path in images}:
        raise ValueError("JSON output must not overwrite an input image")
    report = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "model": str(predictor.model_path),
        "tokenizer": str(predictor.tokenizer_path),
        "providers": predictor.session.get_providers(),
        "caption": preprocess_caption(caption),
        "box_format": "normalized_cxcywh",
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "recursive": recursive,
        "timing_unit": "ms",
        "summary": {
            "total_images": len(images), "succeeded": 0, "failed": 0,
            "total_detections": 0, "model_load_ms": model_load_ms,
            "processing_time_ms": 0.0, "total_inference_ms": 0.0,
            "mean_inference_ms": None, "mean_image_time_ms": None,
        },
        "results": [],
    }
    summary = report["summary"]
    inference_sum = 0.0
    image_time_sum = 0.0
    started = perf_counter()

    def checkpoint():
        summary["processing_time_ms"] = (perf_counter() - started) * 1000
        summary["total_inference_ms"] = inference_sum
        succeeded = summary["succeeded"]
        summary["mean_inference_ms"] = inference_sum / succeeded if succeeded else None
        summary["mean_image_time_ms"] = image_time_sum / succeeded if succeeded else None
        save_json(report, output)

    checkpoint()
    try:
        for index, path in enumerate(images, 1):
            image_started = perf_counter()
            relative = path.relative_to(input_dir).as_posix()
            try:
                result, metadata = predictor.predict_with_metadata(
                    path, caption, box_threshold, text_threshold,
                )
                timings = metadata["timings_ms"]
                record = {
                    "image": relative, "status": "ok", **result,
                    "original_size": metadata["original_size"],
                    "model_input_size": metadata["resized_size"],
                    "timings_ms": timings,
                }
                inference_sum += timings["inference_ms"]
                image_time_sum += timings["total_ms"]
                summary["succeeded"] += 1
                summary["total_detections"] += len(result["boxes"])
                logger.info(
                    "[%d/%d] %s | detections=%d | inference=%.2f ms | total=%.2f ms",
                    index, len(images), relative, len(result["boxes"]),
                    timings["inference_ms"], timings["total_ms"],
                )
            except Exception as exc:
                summary["failed"] += 1
                record = {
                    "image": relative, "status": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "timings_ms": {"total_ms": (perf_counter() - image_started) * 1000},
                }
                logger.exception("[%d/%d] Failed: %s", index, len(images), relative)
            report["results"].append(record)
            checkpoint()
    except KeyboardInterrupt:
        report["interrupted"] = True
        raise
    finally:
        checkpoint()
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    try:
        images = discover_images(INPUT_DIR, RECURSIVE)
        if not images:
            raise ValueError(f"No supported images found in: {INPUT_DIR}")
        if not 0 <= BOX_THRESHOLD <= 1 or not 0 <= TEXT_THRESHOLD <= 1:
            raise ValueError("Thresholds must be between zero and one")
        if THREADS < 0 or DEVICE_ID < 0:
            raise ValueError("THREADS and DEVICE_ID must be nonnegative")
        if min(SHORT_SIDE, MAX_SIZE, MAX_TEXT_LEN) < 1 or MAX_TEXT_LEN > 256:
            raise ValueError("Resize sizes must be positive; MAX_TEXT_LEN must be in [1, 256]")
        if PROVIDER not in ("cpu", "cuda"):
            raise ValueError("PROVIDER must be cpu or cuda")
        if not isinstance(RECURSIVE, bool):
            raise ValueError("RECURSIVE must be True or False")
    except (OSError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    try:
        logger.info("Found %d images | model=%s | provider=%s", len(images), MODEL_PATH, PROVIDER)
        load_started = perf_counter()
        predictor = OnnxPredictor(
            MODEL_PATH, TOKENIZER_PATH, PROVIDER, DEVICE_ID, THREADS,
            SHORT_SIDE, MAX_SIZE, MAX_TEXT_LEN, ORT_LOG_SEVERITY,
        )
        model_load_ms = (perf_counter() - load_started) * 1000
        logger.info("Model loaded once | elapsed=%.2f ms", model_load_ms)
        report = run_batch(
            INPUT_DIR, CAPTION, OUTPUT_JSON, predictor,
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            recursive=RECURSIVE, model_load_ms=model_load_ms,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2, allow_nan=False))
        logger.info("JSON saved: %s", Path(OUTPUT_JSON).expanduser().resolve())
        return 1 if report["summary"]["failed"] else 0
    except KeyboardInterrupt:
        logger.warning("Interrupted; results for completed images were saved")
        return 130
    except Exception:
        logger.exception("Folder inference failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

