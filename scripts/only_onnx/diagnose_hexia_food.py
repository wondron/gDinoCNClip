#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 heXia_260311_0030.jpg / food 用例，采集并比较 H20 与 S5000 的
GroundingDINO 原始输出、阈值过滤、phrase、NMS、CLIP crop/feature/score。

依赖：项目原有 onnx_pipeline.py、cn_tokenizer.py、模型文件，以及 numpy/opencv/Pillow/onnxruntime。

一、在 H20 上采集：
python diagnose_hexia_food.py capture --tag H20 --provider cuda --clip-provider cuda

二、在 S5000 上采集：
python diagnose_hexia_food.py capture --tag S5000 --provider musa --clip-provider musa

三、把两台机器生成的目录拷贝到同一台机器后比较：
python diagnose_hexia_food.py compare \
    diagnostics/H20 diagnostics/S5000

每个 capture 目录包含：
  diagnostics.json       可读的候选框、phrase、NMS、CLIP 结果
  dino_raw.npz           GroundingDINO 原始 pred_logits / pred_boxes
  dino_inputs.npz        GroundingDINO 实际输入 Tensor
  clip_intermediates.npz 每个候选框的 CLIP 输入、原始输出、归一化特征
  crops/                 实际送入 CLIP 的 crop 图片

注意：provider 名称沿用项目 benchmark.py：cuda / musa / cpu。
本脚本通过项目 onnx_pipeline.py 创建模型 Session，保持和现有推理路径一致。
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

import onnx_pipeline as pipe


# 与 benchmark.py 保持一致
HERE = Path(__file__).resolve().parent
DEFAULT_GDINO_MODEL = HERE / "models/groundingdino/groundingdino_swint_ogc.onnx"
DEFAULT_TOKENIZER = HERE / "models/groundingdino/bert-base-uncased"
DEFAULT_CLIP_IMAGE_MODEL = HERE / "models/clip/clip_cn_vit-b-16.img.fp32.onnx"
DEFAULT_CLIP_TEXT_MODEL = HERE / "models/clip/clip_cn_vit-b-16.txt.fp32.onnx"
DEFAULT_IMAGE = HERE / "images/heXia_260311_0030.jpg"

DINO_CAPTION = "food"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.40
CLIP_CAPTION = "食物"
CLIP_THRESHOLD = 0.50
SHORT_SIDE = 800
MAX_SIZE = 1333
MAX_TEXT_LEN = 256
NMS_THRESHOLD = 0.50
THREADS = 4
DEVICE_ID = 0

CLIP_FOOD_RANGE = ["猪肉", "鱼", "螃蟹", "鱿鱼", "蘑菇", "皮皮虾"]
CLIP_COMMON_RANGE = ["水泥", "猫", "狗", "无食材", "人"]
CLIP_RECIPE_RANGE = ["一道菜", "食物"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_copy(value):
    return np.asarray(value).copy()


def to_json_safe(obj):
    """将 NumPy 类型转换为 JSON 可序列化的 Python 类型。"""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    return obj


def json_dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_obj = to_json_safe(obj)
    path.write_text(
        json.dumps(safe_obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )


def get_clip_model(image_model, text_model, provider, device_id):
    """
    兼容两种项目版本：
    1. ChineseClip(image_model, text_model, detect_range, provider, device_id)
    2. ChineseClip(image_model, text_model, food_range, common_range,
                   recipe_range, provider, device_id)
    """
    params = list(inspect.signature(pipe.ChineseClip.__init__).parameters.values())
    positional = [
        p for p in params
        if p.name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) >= 7:
        return pipe.ChineseClip(
            str(image_model), str(text_model),
            CLIP_FOOD_RANGE, CLIP_COMMON_RANGE, CLIP_RECIPE_RANGE,
            provider, device_id,
        )
    detect_range = CLIP_FOOD_RANGE + CLIP_COMMON_RANGE + CLIP_RECIPE_RANGE
    return pipe.ChineseClip(
        str(image_model), str(text_model), detect_range, provider, device_id
    )


def token_string(tokenizer, token_id):
    try:
        raw = tokenizer._tokenizer.id_to_token(int(token_id))
        return raw if raw is not None else ""
    except Exception:
        try:
            return tokenizer.decode([int(token_id)])
        except Exception:
            return ""


def build_dino_diagnostics(logits, boxes, tokenizer, metadata, box_threshold, text_threshold):
    """
    与 onnx_pipeline.postprocess() 使用相同的 sigmoid、box threshold 和 phrase token 规则。
    额外保留原始 query index，便于跨 GPU 对齐同一个 query。
    """
    logits = np.asarray(logits)
    boxes = np.asarray(boxes)
    probabilities = np.exp(-np.logaddexp(np.float32(0), -logits[0]))
    scores_all = probabilities.max(axis=1)
    selected_ids = np.flatnonzero(scores_all > box_threshold)
    token_ids = metadata["token_ids"]

    post = pipe.postprocess(
        logits, boxes, tokenizer, metadata,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )

    candidates = []
    for local_i, query_id in enumerate(selected_ids):
        row = probabilities[query_id]
        top_indices = np.argsort(row)[::-1][:10]
        above = np.flatnonzero(row > text_threshold)
        phrase_indices = above[
            (above > 0) & (above < min(255, len(token_ids)))
        ]
        phrase = tokenizer.decode(
            [token_ids[i] for i in phrase_indices]
        ).replace(".", "")

        top_tokens = []
        for idx in top_indices:
            token_id = token_ids[idx] if idx < len(token_ids) else None
            top_tokens.append({
                "token_index": int(idx),
                "token_id": int(token_id) if token_id is not None else None,
                "token": token_string(tokenizer, token_id) if token_id is not None else "",
                "probability": float(row[idx]),
                "is_valid_phrase_token": bool(
                    idx > 0 and idx < min(255, len(token_ids))
                ),
            })

        candidates.append({
            "query_id": int(query_id),
            "selected_by_box_threshold": True,
            "score": float(scores_all[query_id]),
            "box_cxcywh": boxes[0, query_id].astype(float).tolist(),
            "phrase": phrase,
            "phrase_token_indices": phrase_indices.astype(int).tolist(),
            "all_token_indices_above_text_threshold": above.astype(int).tolist(),
            "top_tokens": top_tokens,
            "local_postprocess_index": int(local_i),
        })

    # 额外记录分数最高的 query，以及距离 box threshold 最近的 query。
    top_ids = np.argsort(scores_all)[::-1][:30]
    near_ids = np.argsort(np.abs(scores_all - box_threshold))[:30]

    def compact_query(qid):
        row = probabilities[qid]
        idx = int(np.argmax(row))
        token_id = token_ids[idx] if idx < len(token_ids) else None
        return {
            "query_id": int(qid),
            "score": float(scores_all[qid]),
            "selected": bool(scores_all[qid] > box_threshold),
            "box_cxcywh": boxes[0, qid].astype(float).tolist(),
            "top_token_index": idx,
            "top_token_id": int(token_id) if token_id is not None else None,
            "top_token": token_string(tokenizer, token_id) if token_id is not None else "",
            "top_token_probability": float(row[idx]),
        }

    return {
        "candidates_pre_nms": candidates,
        "top_30_queries": [compact_query(i) for i in top_ids],
        "nearest_30_to_box_threshold": [compact_query(i) for i in near_ids],
        "selected_query_ids": selected_ids.astype(int).tolist(),
        "postprocess": post,
    }


def calculate_nms(dino, dino_info):
    candidates = dino_info["candidates_pre_nms"]
    if not candidates:
        return [], []

    boxes = np.asarray([x["box_cxcywh"] for x in candidates], dtype=np.float32)
    scores = np.asarray([x["score"] for x in candidates], dtype=np.float32)
    phrases = [x["phrase"] for x in candidates]

    threshold = getattr(dino, "nms_threshold", NMS_THRESHOLD)
    agnostic = getattr(dino, "nms_agnostic", False)
    if threshold is not None and threshold < 1 and len(boxes) > 1:
        keep_local = pipe.nms(
            boxes, scores, threshold,
            labels=None if agnostic else phrases,
        ).tolist()
    else:
        keep_local = list(range(len(boxes)))

    kept = []
    for local_i in keep_local:
        item = dict(candidates[local_i])
        item["nms_kept"] = True
        kept.append(item)
    return kept, [int(candidates[i]["query_id"]) for i in keep_local]


class SessionCapture:
    """代理 ORT Session：不改变实际推理，只记录 run() 的输入和输出。"""
    def __init__(self, session, name, state):
        self._session = session
        self._name = name
        self._state = state

    def __getattr__(self, name):
        return getattr(self._session, name)

    def run(self, output_names, input_feed, *args, **kwargs):
        outputs = self._session.run(output_names, input_feed, *args, **kwargs)
        if self._name == "gdino":
            names = list(output_names) if output_names is not None else []
            if "pred_logits" in names and "pred_boxes" in names:
                self._state["dino_inputs"] = {
                    k: array_copy(v) for k, v in input_feed.items()
                }
                self._state["dino_outputs"] = {
                    name: array_copy(value) for name, value in zip(names, outputs)
                }
        elif self._name == "clip_image":
            active = self._state.get("active_clip")
            if active is not None:
                active["image_inputs"] = {
                    k: array_copy(v) for k, v in input_feed.items()
                }
                active["image_outputs"] = [
                    array_copy(value) for value in outputs
                ]
        elif self._name == "clip_text":
            active = self._state.get("active_clip")
            if active is not None:
                active.setdefault("text_runs", []).append({
                    "inputs": {k: array_copy(v) for k, v in input_feed.items()},
                    "outputs": [array_copy(value) for value in outputs],
                })
        return outputs


def npz_save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    np.savez_compressed(path, **clean)


def save_clip_text_features(clip, arrays, meta):
    """
    保存常见 ChineseClip.texts 字典中的文本特征。
    兼容其他实现：同时记录实例属性名称，便于判断文本特征是否按不同范围拆分。
    """
    meta["clip_attributes"] = sorted(list(vars(clip).keys()))
    texts = getattr(clip, "texts", None)
    if isinstance(texts, dict) and texts:
        names, feats = [], []
        for name, feat in texts.items():
            arr = np.asarray(feat)
            if arr.size:
                names.append(str(name))
                feats.append(arr.reshape(-1))
        if feats and len({x.shape for x in feats}) == 1:
            arrays["clip_text_features"] = np.stack(feats)
            meta["clip_text_names"] = names


def capture(args):
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    for p in (args.gdino_model, args.tokenizer, args.clip_image_model, args.clip_text_model):
        if not p.exists():
            raise FileNotFoundError(f"模型或 tokenizer 路径不存在：{p}")

    tag = args.tag
    out_dir = args.out_dir.resolve() / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as im:
        image = im.convert("RGB")

    print(f"[{tag}] 初始化 GroundingDINO: {args.provider}", flush=True)
    dino = pipe.GroundingDino(
        str(args.gdino_model), str(args.tokenizer),
        args.provider, args.device_id, args.threads,
        SHORT_SIDE, MAX_SIZE, MAX_TEXT_LEN, NMS_THRESHOLD,
    )

    print(f"[{tag}] 初始化 CLIP: {args.clip_provider}", flush=True)
    clip = get_clip_model(
        args.clip_image_model, args.clip_text_model,
        args.clip_provider, args.device_id,
    )

    state = {}
    dino.session = SessionCapture(dino.session, "gdino", state)
    clip.image_session = SessionCapture(clip.image_session, "clip_image", state)
    clip.text_session = SessionCapture(clip.text_session, "clip_text", state)

    # 为每次 clip.detect() 记录实际 crop，并让 SessionCapture 将该次 ORT 输入/输出挂到对应记录。
    clip_records = []
    pending_query_ids = []

    original_extract_image_feature = clip.extract_image_feature

    def extract_image_feature_with_capture(img_bgr, timings=None):
        feature = original_extract_image_feature(img_bgr, timings)
        active = state.get("active_clip")
        if active is not None:
            active["normalized_image_feature"] = array_copy(feature)
        return feature

    clip.extract_image_feature = extract_image_feature_with_capture
    

    original_detect = clip.detect
    original_text_candidates = getattr(clip, "text_candidates", None)

    detect_signature = inspect.signature(original_detect)
    detect_accepts_candidates = (
        "candidates" in detect_signature.parameters
    )


    def detect_with_capture(img_bgr, input_name, *args, **kwargs):
        """
        同时兼容新旧版本：
        新版：detect(img_bgr, input_name, candidates, timings=None)
        旧版：detect(img_bgr, input_name, timings=None)
        """
        query_id = pending_query_ids.pop(0) if pending_query_ids else None

        rec = {
            "query_id": query_id,
            "input_name": str(input_name),
            "crop_shape_bgr": list(img_bgr.shape),
            "crop_sha256": hashlib.sha256(
                np.ascontiguousarray(img_bgr).tobytes()
            ).hexdigest(),
        }

        # 新版：candidates 由 batch_detect 外部传入
        if detect_accepts_candidates:
            candidates = kwargs.get("candidates")

            if candidates is None and args:
                candidates = args[0]

            if candidates is not None:
                names, features = candidates
                rec["candidate_names"] = [str(x) for x in names]
                rec["candidate_features"] = np.asarray(features).copy()

        state["active_clip"] = rec

        try:
            score = original_detect(
                img_bgr, input_name, *args, **kwargs
            )
        finally:
            state["active_clip"] = None

        rec["returned_score"] = float(score)
        clip_records.append(rec)

        return score


    # 旧版：detect 内部调用 text_candidates()
    # 在这里捕获候选文本特征
    if callable(original_text_candidates):

        def text_candidates_with_capture(input_name, *args, **kwargs):

            candidates = original_text_candidates(
                input_name, *args, **kwargs
            )

            active = state.get("active_clip")

            if active is not None and candidates is not None:
                names, features = candidates

                active["candidate_names"] = [
                    str(x) for x in names
                ]

                active["candidate_features"] = np.asarray(
                    features
                ).copy()

            return candidates

        clip.text_candidates = text_candidates_with_capture


    clip.detect = detect_with_capture



    # 用相同参数生成 DINO 输入，保存 tokenizer 和预处理元信息。
    feeds, metadata = pipe.prepare_inputs(
        image, DINO_CAPTION, dino.tokenizer, dino.input_shapes,
        short_side=SHORT_SIDE, max_size=MAX_SIZE, max_text_len=MAX_TEXT_LEN,
    )

    # 直接调用与 GroundingDino.detect 相同的 ONNX 输出节点，保留未后处理的原始 Tensor。
    raw_logits, raw_boxes = dino.session.run(
        ["pred_logits", "pred_boxes"], feeds
    )
    if "dino_outputs" not in state:
        raise RuntimeError("未捕获到 GroundingDINO pred_logits / pred_boxes。")

    dino_info = build_dino_diagnostics(
        raw_logits, raw_boxes, dino.tokenizer, metadata,
        BOX_THRESHOLD, TEXT_THRESHOLD,
    )
    nms_candidates, nms_query_ids = calculate_nms(dino, dino_info)

    # 生成 batch_detect 会实际遍历的 query id 顺序，保证 CLIP crop 与 DINO query 对齐。
    nms_boxes = np.asarray(
        [x["box_cxcywh"] for x in nms_candidates], dtype=np.float32
    ).reshape(-1, 4)
    nms_ids = [x["query_id"] for x in nms_candidates]
    valid_mask = (
        np.isfinite(nms_boxes).all(axis=1)
        & (nms_boxes[:, 2:] > 0).all(axis=1)
    ) if len(nms_boxes) else np.zeros((0,), dtype=bool)
    valid_boxes = nms_boxes[valid_mask]
    valid_ids = [qid for qid, ok in zip(nms_ids, valid_mask) if ok]
    rects_for_clip = pipe.box2xyxy(image, valid_boxes) if len(valid_boxes) else []
    width, height = image.size
    for qid, rect in zip(valid_ids, rects_for_clip):
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            continue
        pending_query_ids.append(qid)

    # 与 benchmark.py 一样调用 batch_detect；该方法内部会逐框调用 clip.detect。
    clip_rects, clip_confidences = clip.batch_detect(
        image, nms_boxes, CLIP_CAPTION
    )

    # 将实际最终保留框映射回 DINO query id。
    final_rect_to_qid = {}
    for rec in clip_records:
        qid = rec.get("query_id")
        if qid is not None:
            # rect 在上面按同样的 box2xyxy 规则计算。
            cand = next((x for x in nms_candidates if x["query_id"] == qid), None)
            if cand is not None:
                rect = tuple(pipe.box2xyxy(
                    image, np.asarray([cand["box_cxcywh"]], dtype=np.float32)
                )[0])
                final_rect_to_qid[rect] = qid
    final_kept_query_ids = []
    for rect in clip_rects:
        qid = final_rect_to_qid.get(tuple(rect))
        if qid is not None:
            final_kept_query_ids.append(qid)

    # 保存可视化：蓝色为 NMS 后 DINO 框，红色为 CLIP 最终保留框。
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for item in nms_candidates:
        cx, cy, bw, bh = item["box_cxcywh"]
        x0, y0 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x1, y1 = (cx + bw / 2) * width, (cy + bh / 2) * height
        draw.rectangle((x0, y0, x1, y1), outline=(40, 120, 255), width=3)
        draw.text((x0 + 3, y0 + 3),
                  f"q{item['query_id']} {item['score']:.4f} {item['phrase']}",
                  fill=(40, 120, 255))
    for rect, conf in zip(clip_rects, clip_confidences):
        draw.rectangle(tuple(rect), outline=(255, 0, 0), width=4)
        draw.text((rect[0] + 3, max(0, rect[1] - 14)),
                  f"CLIP {conf:.4f}", fill=(255, 0, 0))
    overlay.save(out_dir / "overlay.jpg", quality=92)

    # 保存 crop 文件及 NPZ 中间 Tensor。
    clip_arrays = {}
    for i, rec in enumerate(clip_records):
        qid = rec.get("query_id")
        suffix = f"q{qid}" if qid is not None else f"idx{i}"
        # 由 NMS query id 反查 crop，用于落盘可视化。
        cand = next((x for x in nms_candidates if x["query_id"] == qid), None)
        if cand is not None:
            rect = pipe.box2xyxy(
                image, np.asarray([cand["box_cxcywh"]], dtype=np.float32)
            )[0]
            crop = image.crop(tuple(rect))
            crop.save(crop_dir / f"{suffix}.jpg", quality=95)

        for key, value in rec.get("image_inputs", {}).items():
            clip_arrays[f"{suffix}__input__{key}"] = value
        for j, value in enumerate(rec.get("image_outputs", [])):
            clip_arrays[f"{suffix}__output_{j}"] = value
        if rec.get("normalized_image_feature") is not None:
            clip_arrays[f"{suffix}__normalized_feature"] = rec["normalized_image_feature"]
        if rec.get("candidate_features") is not None:
            clip_arrays[f"{suffix}__candidate_features"] = rec["candidate_features"]
        for j, run in enumerate(rec.get("text_runs", [])):
            for key, value in run["inputs"].items():
                clip_arrays[f"{suffix}__text_input_{j}__{key}"] = value
            for k, value in enumerate(run["outputs"]):
                clip_arrays[f"{suffix}__text_output_{j}_{k}"] = value

    clip_records_meta = [
        {
            key: value for key, value in rec.items()
            if key not in (
                "image_inputs", "image_outputs", "text_runs",
                "normalized_image_feature", "candidate_features",
            )
        }
        for rec in clip_records
    ]

    text_meta = {
        "tag": tag,
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_size": list(image.size),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "gdino_active_providers": dino.session.get_providers(),
        "clip_image_active_providers": clip.image_session.get_providers(),
        "clip_text_active_providers": clip.text_session.get_providers(),
        "provider_args": {
            "gdino": args.provider,
            "clip": args.clip_provider,
            "device_id": args.device_id,
        },
        "models": {
            "gdino": {"path": str(args.gdino_model), "sha256": sha256_file(args.gdino_model)},
            "clip_image": {"path": str(args.clip_image_model), "sha256": sha256_file(args.clip_image_model)},
            "clip_text": {"path": str(args.clip_text_model), "sha256": sha256_file(args.clip_text_model)},
        },
        "settings": {
            "dino_caption": DINO_CAPTION,
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "clip_caption": CLIP_CAPTION,
            "clip_threshold_reported": CLIP_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
            "short_side": SHORT_SIDE,
            "max_size": MAX_SIZE,
            "max_text_len": MAX_TEXT_LEN,
        },
        "preprocess_metadata": metadata,
        "dino_candidates_pre_nms": dino_info["candidates_pre_nms"],
        "dino_top_30_queries": dino_info["top_30_queries"],
        "dino_nearest_30_to_box_threshold": dino_info["nearest_30_to_box_threshold"],
        "dino_nms_candidates": nms_candidates,
        "dino_nms_query_ids": nms_query_ids,
        "clip_candidates": clip_records_meta,
        "clip_final_rects": [list(map(int, r)) for r in clip_rects],
        "clip_final_confidences": [float(x) for x in clip_confidences],
        "clip_final_kept_query_ids": final_kept_query_ids,
        "clip_candidate_count": len(clip_records),
        "clip_final_count": len(clip_rects),
    }

    # 保存输入、原始输出和 CLIP 中间 Tensor。
    npz_save(out_dir / "dino_inputs.npz", state["dino_inputs"])
    npz_save(out_dir / "dino_raw.npz", {
        "pred_logits": state["dino_outputs"]["pred_logits"],
        "pred_boxes": state["dino_outputs"]["pred_boxes"],
    })
    save_clip_text_features(clip, clip_arrays, text_meta)
    npz_save(out_dir / "clip_intermediates.npz", clip_arrays)
    json_dump(out_dir / "diagnostics.json", text_meta)

    print("\n采集完成")
    print(f"输出目录：{out_dir}")
    print(f"DINO 原始 logits：{np.asarray(raw_logits).shape}")
    print(f"DINO 原始 boxes：{np.asarray(raw_boxes).shape}")
    print(f"box threshold 后：{len(dino_info['candidates_pre_nms'])}")
    print(f"NMS 后：{len(nms_candidates)}")
    print(f"实际 CLIP 处理 crop：{len(clip_records)}")
    print(f"CLIP 最终保留：{len(clip_rects)}")
    print(f"可视化：{out_dir / 'overlay.jpg'}")
    print("请分别在 H20、S5000 执行 capture，再将两个输出目录拷贝到同一台机器执行 compare。")


def load_capture(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    meta_path = path / "diagnostics.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    return path, json.loads(meta_path.read_text(encoding="utf-8"))



def metrics(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "same_shape": False
        }

    # 只对两边均为有限数值的位置计算误差
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    finite_both = finite_a & finite_b

    # 判断非有限值是否一致，例如两边都是 -Inf
    same_nan = np.isnan(a) & np.isnan(b)
    same_pos_inf = np.isposinf(a) & np.isposinf(b)
    same_neg_inf = np.isneginf(a) & np.isneginf(b)

    same_nonfinite = same_nan | same_pos_inf | same_neg_inf

    # 一边有限、一边非有限，或非有限类型不同
    nonfinite_mismatch = ~(finite_both | same_nonfinite)

    d = np.abs(a[finite_both] - b[finite_both])

    if d.size:
        finite_indices = np.flatnonzero(finite_both)
        flat_index = int(finite_indices[int(np.argmax(d))])
        max_index = [
            int(x) for x in np.unravel_index(flat_index, a.shape)
        ]

        max_a = float(a.flat[flat_index])
        max_b = float(b.flat[flat_index])

        max_abs = float(d.max())
        mean_abs = float(d.mean())
        rmse = float(np.sqrt(np.mean(d ** 2)))
    else:
        max_index = []
        max_a = None
        max_b = None
        max_abs = None
        mean_abs = None
        rmse = None

    return {
        "same_shape": True,
        "shape": list(a.shape),

        "max_abs": max_abs,
        "max_abs_index": max_index,
        "a_at_max": max_a,
        "b_at_max": max_b,

        "mean_abs": mean_abs,
        "rmse": rmse,

        "finite_count": int(finite_both.sum()),
        "nonfinite_a": int((~finite_a).sum()),
        "nonfinite_b": int((~finite_b).sum()),
        "nonfinite_mismatch": int(nonfinite_mismatch.sum()),

        "exact_equal": bool(
            np.all((a == b) | same_nan)
        )
    }


def box_iou_cxcywh(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aa = np.array([a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2])
    bb = np.array([b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2])
    lt = np.maximum(aa[:2], bb[:2])
    rb = np.minimum(aa[2:], bb[2:])
    inter = np.prod(np.maximum(rb - lt, 0))
    area_a = np.prod(np.maximum(aa[2:] - aa[:2], 0))
    area_b = np.prod(np.maximum(bb[2:] - bb[:2], 0))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def compare_npz(path_a, path_b):
    if not path_a.exists() or not path_b.exists():
        return {}, ["某一侧 NPZ 文件缺失"]
    with np.load(path_a, allow_pickle=False) as aa, np.load(path_b, allow_pickle=False) as bb:
        keys_a, keys_b = set(aa.files), set(bb.files)
        result = {}
        for key in sorted(keys_a | keys_b):
            if key not in keys_a:
                result[key] = {"missing": "A"}
            elif key not in keys_b:
                result[key] = {"missing": "B"}
            else:
                result[key] = metrics(aa[key], bb[key])
        return result, []


def fmt_metric(m):
    if not m:
        return "无数据"

    if not m.get("same_shape", False):
        return f"形状不同：{m.get('shape_a')} vs {m.get('shape_b')}"

    if m.get("max_abs") is None:
        error_info = "无可比较的有限数值"
    else:
        error_info = (
            f"max_abs={m['max_abs']:.6g} "
            f"at={m['max_abs_index']} "
            f"(A={m['a_at_max']}, B={m['b_at_max']}), "
            f"mean_abs={m['mean_abs']:.6g}, "
            f"RMSE={m['rmse']:.6g}"
        )

    return (
        f"shape={m['shape']} {error_info}, "
        f"finite_count={m.get('finite_count', 0)}, "
        f"nonfinite_A={m.get('nonfinite_a', 0)}, "
        f"nonfinite_B={m.get('nonfinite_b', 0)}, "
        f"nonfinite_mismatch={m.get('nonfinite_mismatch', 0)}, "
        f"exact={m['exact_equal']}"
    )

def compare(args):
    dir_a, a = load_capture(args.dir_a)
    dir_b, b = load_capture(args.dir_b)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# heXia_260311_0030.jpg / food 诊断对比",
        "",
        f"- A：{a['tag']} ({dir_a})",
        f"- B：{b['tag']} ({dir_b})",
        f"- 图片 SHA256 一致：{a['image_sha256'] == b['image_sha256']}",
        f"- GroundingDINO 模型 SHA256 一致：{a['models']['gdino']['sha256'] == b['models']['gdino']['sha256']}",
        f"- CLIP image 模型 SHA256 一致：{a['models']['clip_image']['sha256'] == b['models']['clip_image']['sha256']}",
        f"- CLIP text 模型 SHA256 一致：{a['models']['clip_text']['sha256'] == b['models']['clip_text']['sha256']}",
        "",
        "## 1. DINO 输入 Tensor",
    ]

    input_cmp, _ = compare_npz(dir_a / "dino_inputs.npz", dir_b / "dino_inputs.npz")
    for key, m in input_cmp.items():
        lines.append(f"- `{key}`：{fmt_metric(m)}")

    lines += ["", "## 2. GroundingDINO 原始输出 Tensor"]
    raw_cmp, _ = compare_npz(dir_a / "dino_raw.npz", dir_b / "dino_raw.npz")
    for key, m in raw_cmp.items():
        lines.append(f"- `{key}`：{fmt_metric(m)}")

    # 按原始 query index 对齐，不使用 IoU 贪心配对。
    cand_a = {x["query_id"]: x for x in a["dino_candidates_pre_nms"]}
    cand_b = {x["query_id"]: x for x in b["dino_candidates_pre_nms"]}
    all_qids = sorted(set(cand_a) | set(cand_b))
    dino_rows = []
    for qid in all_qids:
        x, y = cand_a.get(qid), cand_b.get(qid)
        if x is None or y is None:
            present = y if x is None else x
            dino_rows.append({
                "query_id": qid,
                "status": "only_B" if x is None else "only_A",
                "score_a": x["score"] if x else None,
                "score_b": y["score"] if y else None,
                "score_delta": None,
                "iou": None,
                "phrase_a": x["phrase"] if x else None,
                "phrase_b": y["phrase"] if y else None,
                "box_a": x["box_cxcywh"] if x else None,
                "box_b": y["box_cxcywh"] if y else None,
                "top_token_a": x["top_tokens"][0] if x else None,
                "top_token_b": y["top_tokens"][0] if y else None,
            })
        else:
            dino_rows.append({
                "query_id": qid,
                "status": "both_selected",
                "score_a": x["score"],
                "score_b": y["score"],
                "score_delta": abs(x["score"] - y["score"]),
                "iou": box_iou_cxcywh(x["box_cxcywh"], y["box_cxcywh"]),
                "phrase_a": x["phrase"],
                "phrase_b": y["phrase"],
                "box_a": x["box_cxcywh"],
                "box_b": y["box_cxcywh"],
                "top_token_a": x["top_tokens"][0] if x["top_tokens"] else None,
                "top_token_b": y["top_tokens"][0] if y["top_tokens"] else None,
            })

    lines += [
        "",
        "## 3. DINO 候选框（按 query index 对齐）",
        "",
        f"- A：box threshold 后 {len(cand_a)} 个；NMS 后 {len(a['dino_nms_query_ids'])} 个",
        f"- B：box threshold 后 {len(cand_b)} 个；NMS 后 {len(b['dino_nms_query_ids'])} 个",
        f"- 仅 A 通过 box threshold：{len(set(cand_a) - set(cand_b))}",
        f"- 仅 B 通过 box threshold：{len(set(cand_b) - set(cand_a))}",
        f"- NMS query index 仅 A：{sorted(set(a['dino_nms_query_ids']) - set(b['dino_nms_query_ids']))}",
        f"- NMS query index 仅 B：{sorted(set(b['dino_nms_query_ids']) - set(a['dino_nms_query_ids']))}",
        "",
        "| query | 状态 | A score | B score | 差值 | IoU | A phrase | B phrase |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for r in dino_rows:
        lines.append(
            f"| {r['query_id']} | {r['status']} | "
            f"{r['score_a'] if r['score_a'] is not None else '-'} | "
            f"{r['score_b'] if r['score_b'] is not None else '-'} | "
            f"{r['score_delta'] if r['score_delta'] is not None else '-'} | "
            f"{r['iou'] if r['iou'] is not None else '-'} | "
            f"{r['phrase_a'] or '-'} | {r['phrase_b'] or '-'} |"
        )

    # 空 phrase 的 top token 诊断。
    empty_a = [x for x in a["dino_candidates_pre_nms"] if not x["phrase"]]
    empty_b = [x for x in b["dino_candidates_pre_nms"] if not x["phrase"]]
    lines += ["", "### 空 phrase 候选的 top token"]
    for label, items in ((a["tag"], empty_a), (b["tag"], empty_b)):
        if not items:
            lines.append(f"- {label}：无空 phrase 候选")
        for x in items:
            top = x["top_tokens"][0] if x["top_tokens"] else {}
            lines.append(
                f"- {label} query={x['query_id']} score={x['score']:.6f} "
                f"top_token={top.get('token')!r} index={top.get('token_index')} "
                f"prob={top.get('probability')}"
            )

    lines += ["", "## 4. CLIP crop / feature / score 对比"]
    clip_a = {x["query_id"]: x for x in a["clip_candidates"] if x.get("query_id") is not None}
    clip_b = {x["query_id"]: x for x in b["clip_candidates"] if x.get("query_id") is not None}
    lines.append(f"- A 实际送入 CLIP 的 crop 数：{len(clip_a)}")
    lines.append(f"- B 实际送入 CLIP 的 crop 数：{len(clip_b)}")
    lines.append(f"- 仅 A 有 CLIP crop query：{sorted(set(clip_a) - set(clip_b))}")
    lines.append(f"- 仅 B 有 CLIP crop query：{sorted(set(clip_b) - set(clip_a))}")
    lines.append("")
    lines.append("| query | A score | B score | 分数差 | crop SHA 一致 |")
    lines.append("|---:|---:|---:|---:|---|")
    clip_rows = []
    for qid in sorted(set(clip_a) | set(clip_b)):
        x, y = clip_a.get(qid), clip_b.get(qid)
        if x and y:
            delta = abs(x["returned_score"] - y["returned_score"])
            same_crop = x["crop_sha256"] == y["crop_sha256"]
            lines.append(
                f"| {qid} | {x['returned_score']:.6f} | {y['returned_score']:.6f} | "
                f"{delta:.6f} | {same_crop} |"
            )
            clip_rows.append({
                "query_id": qid,
                "score_a": x["returned_score"],
                "score_b": y["returned_score"],
                "score_delta": delta,
                "crop_hash_equal": same_crop,
            })
        else:
            lines.append(
                f"| {qid} | {x['returned_score'] if x else '-'} | "
                f"{y['returned_score'] if y else '-'} | - | - |"
            )

    clip_tensor_cmp, _ = compare_npz(
        dir_a / "clip_intermediates.npz",
        dir_b / "clip_intermediates.npz",
    )
    lines += ["", "### CLIP 中间 Tensor"]
    if not clip_tensor_cmp:
        lines.append("- 未发现可比较的 CLIP Tensor，检查采集目录。")
    else:
        for key, m in clip_tensor_cmp.items():
            lines.append(f"- `{key}`：{fmt_metric(m)}")

    lines += [
        "",
        "## 5. 结论定位提示",
        "",
        "- 若 DINO 输入 Tensor 不一致：先查图片解码、resize、tokenizer、caption 和预处理参数，暂时不要归因于 GPU 算子。",
        "- 若输入一致但 pred_logits 差异明显：差异发生在 GroundingDINO 模型计算阶段；继续定位最大误差的 query/token。",
        "- 若 pred_logits 接近但 box threshold 后候选数量不同：检查 box threshold 附近的 query 分数。",
        "- 若原始输出接近，但 phrase 不同或为空：重点检查 token_ids、text_threshold、token index 过滤和 tokenizer decode。",
        "- 若 pre-NMS 候选相同但 NMS query index 不同：检查 NMS 输入顺序、score、phrase label 和 NMS 实现。",
        "- 若 DINO 框 / crop 不同：CLIP 差异首先是上游输入变化，不能直接归因于 CLIP。",
        "- 若 crop 完全一致但 CLIP image feature 或 score 不同：再定位 CLIP 模型计算及文本特征。",
        "",
        "注意：本脚本比较的是同一张图片、同一模型、同一 query index 的中间结果。模型 SHA256、输入 Tensor 或推理参数不一致时，应先解决环境/配置不一致，再解释 GPU 数值差异。",
    ]

    report_path = out_dir / "hexia_food_comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_dump(out_dir / "hexia_food_comparison.json", {
        "a": str(dir_a),
        "b": str(dir_b),
        "dino_input": input_cmp,
        "dino_raw": raw_cmp,
        "dino_query_comparison": dino_rows,
        "clip_query_comparison": clip_rows,
        "clip_tensor_comparison": clip_tensor_cmp,
    })
    print("\n".join(lines))
    print(f"\n对比报告已保存：{report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="诊断 H20 / S5000 的 heXia_260311_0030.jpg · food 不一致问题"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture", help="在单台服务器采集中间 Tensor")
    p_cap.add_argument("--tag", required=True, help="服务器名称，例如 H20 / S5000")
    p_cap.add_argument("--provider", choices=("cuda", "musa", "cpu"), required=True)
    p_cap.add_argument("--clip-provider", choices=("cuda", "musa", "cpu"))
    p_cap.add_argument("--device-id", type=int, default=DEVICE_ID)
    p_cap.add_argument("--threads", type=int, default=THREADS)
    p_cap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p_cap.add_argument("--gdino-model", type=Path, default=DEFAULT_GDINO_MODEL)
    p_cap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p_cap.add_argument("--clip-image-model", type=Path, default=DEFAULT_CLIP_IMAGE_MODEL)
    p_cap.add_argument("--clip-text-model", type=Path, default=DEFAULT_CLIP_TEXT_MODEL)
    p_cap.add_argument("--out-dir", type=Path, default=Path("diagnostics"))

    p_cmp = sub.add_parser("compare", help="比较两台服务器的采集目录")
    p_cmp.add_argument("dir_a", type=Path, help="基准目录，例如 diagnostics/H20_xxx")
    p_cmp.add_argument("dir_b", type=Path, help="对比目录，例如 diagnostics/S5000_xxx")
    p_cmp.add_argument("--out-dir", type=Path, default=Path("diagnostics_compare"))

    args = parser.parse_args()
    if args.command == "capture":
        args.clip_provider = args.clip_provider or args.provider
        capture(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
