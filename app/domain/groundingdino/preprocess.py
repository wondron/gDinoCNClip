"""GroundingDINO image/text preprocessing with the original resize and token masks."""

import numpy as np
from PIL import Image

INPUT_NAMES = (
    "images", "input_ids", "token_type_ids", "attention_mask", "position_ids",
    "text_self_attention_masks",
)


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
