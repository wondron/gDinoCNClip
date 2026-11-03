"""End-to-end consistency and speed check: original PyTorch GroundingDINO vs this ONNX pipeline.

PyTorch runs the Grounding project the way its fast_api.py does: load_image() +
predict(), no NMS. ONNX runs GdinoDetect.predict_with_metadata(), i.e. this
service's preprocessing, tokenizer and postprocessing before NMS. Detections are
paired by IoU and checked against score/box tolerances; a detection reported by
only one side counts as "borderline" when its score is within MAX_SCORE_DIFF of
the box threshold (the other backend scored it just below). Each case is timed
after warm-up: "total" covers image decoding through phrases, "model" only the
forward pass (PyTorch forward with CUDA sync, ORT session.run).

Edit the settings below, then run from the project root with the onnxgpu conda
environment:
    python tools/compare_pt_onnx.py

Writes OUTPUT_DIR/<timestamp>/report.json and one overlay per case under vis/
(PyTorch green, ONNX red).
"""

import json
import multiprocessing
import os
import platform
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from validate_onnx import _xyxy, sha256_file

# Settings -------------------------------------------------------------------
ONNX_MODEL = ROOT / "config/model/groundingdino/groundingdino_swint_ogc.onnx"
ONNX_TOKENIZER = ROOT / "config/model/groundingdino/bert-base-uncased"
PT_ROOT = Path(r"D:\01-code\Grounding")  # provides the groundingdino package and bert-base-uncased
PT_CONFIG = PT_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
PT_CHECKPOINT = PT_ROOT / "weights/groundingdino_swint_ogc.pth"
PT_DEVICE = "cuda"  # "cuda" or "cpu"; ONNX uses CUDA when available, like the service
OUTPUT_DIR = ROOT / "output/consistency"  # each run writes OUTPUT_DIR/<timestamp>/
IMAGES = [ROOT / "config/images"]  # image files and/or folders (non-recursive)
CAPTIONS = ["food", "tofu . pickled vegetable . plate .", "mantis shrimp . shrimp . crab ."]
THRESHOLDS = [(0.35, 0.25), (0.2, 0.2)]  # (box_threshold, text_threshold)
WARMUP_RUNS = 1  # untimed runs per case: CUDA initialization and per-shape kernel selection
TIMED_RUNS = 3  # timed runs per case; the median is reported
MATCH_IOU = 0.5  # minimum IoU to pair a PyTorch and an ONNX detection
MIN_IOU = 0.98
MAX_SCORE_DIFF = 0.005
MAX_BOX_DIFF = 0.005  # normalized cxcywh units
# ----------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pairwise_iou(left, right):
    lo = np.maximum(left[:, None, :2], right[None, :, :2])
    hi = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(hi - lo, 0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0, None).prod(axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compare_detections(pt, onnx, box_threshold, image_size, *, match_iou=0.5, min_iou=0.98,
                       max_score_diff=0.005, max_box_diff=0.005):
    """Pair detections greedily by IoU and judge them against the tolerances.

    Status is "pass", "borderline" (every unpaired detection scores within
    max_score_diff of box_threshold) or "fail". Boxes are normalized cxcywh;
    image_size is the original [width, height] used for pixel differences.
    """
    pt_boxes = np.asarray(pt["boxes"], dtype=np.float64).reshape(-1, 4)
    onnx_boxes = np.asarray(onnx["boxes"], dtype=np.float64).reshape(-1, 4)
    iou = pairwise_iou(_xyxy(pt_boxes), _xyxy(onnx_boxes))
    scale = np.tile(np.asarray(image_size, dtype=np.float64), 2)
    pairs, used_pt, used_onnx = [], set(), set()
    for i, j in sorted(np.argwhere(iou >= match_iou).tolist(), key=lambda p: -iou[p[0], p[1]]):
        if i in used_pt or j in used_onnx:
            continue
        used_pt.add(i)
        used_onnx.add(j)
        diff = np.abs(pt_boxes[i] - onnx_boxes[j])
        pairs.append({"pt_index": i, "onnx_index": j, "iou": float(iou[i, j]),
                      "pt_phrase": pt["phrases"][i], "onnx_phrase": onnx["phrases"][j],
                      "pt_score": float(pt["logits"][i]), "onnx_score": float(onnx["logits"][j]),
                      "score_diff": abs(float(pt["logits"][i]) - float(onnx["logits"][j])),
                      "box_diff": float(diff.max()), "box_diff_px": float((diff * scale).max())})

    def unpaired(result, used):
        return [{"index": k, "phrase": result["phrases"][k], "score": float(result["logits"][k]),
                 "borderline": abs(float(result["logits"][k]) - box_threshold) <= max_score_diff}
                for k in range(len(result["phrases"])) if k not in used]

    only_pt, only_onnx = unpaired(pt, used_pt), unpaired(onnx, used_onnx)
    failures = []
    for pair in pairs:
        label = f"{pair['pt_phrase']!r}@{pair['pt_score']:.4f}"
        if pair["pt_phrase"] != pair["onnx_phrase"]:
            failures.append(f"phrase {pair['pt_phrase']!r} != {pair['onnx_phrase']!r}")
        if pair["iou"] < min_iou:
            failures.append(f"{label}: IoU {pair['iou']:.4f} < {min_iou}")
        if pair["score_diff"] > max_score_diff:
            failures.append(f"{label}: score diff {pair['score_diff']:.4f} > {max_score_diff}")
        if pair["box_diff"] > max_box_diff:
            failures.append(f"{label}: box diff {pair['box_diff']:.4f} > {max_box_diff}")
    for side, items in (("PyTorch", only_pt), ("ONNX", only_onnx)):
        failures += [f"only {side}: {item['phrase']!r}@{item['score']:.4f}"
                     for item in items if not item["borderline"]]
    status = "fail" if failures else "borderline" if only_pt or only_onnx else "pass"
    return {"status": status, "failures": failures, "pt_count": len(pt_boxes),
            "onnx_count": len(onnx_boxes), "pairs": pairs, "only_pt": only_pt, "only_onnx": only_onnx}


def timed(function, model_times, sync=None):
    """Wrap a forward call so measure() can separate model time from pre/postprocessing."""
    def wrapper(*args, **kwargs):
        if sync:
            sync()
        started = time.perf_counter()
        outputs = function(*args, **kwargs)
        if sync:
            sync()
        model_times.append(time.perf_counter() - started)
        return outputs
    return wrapper


def measure(run, model_times, warmup, repeat):
    """Return run()'s last result and median total/model milliseconds after warm-up calls."""
    for _ in range(warmup):
        run()
    model_times.clear()
    totals = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = run()
        totals.append(time.perf_counter() - started)
    return result, {"total_ms": round(statistics.median(totals) * 1000, 2),
                    "model_ms": round(statistics.median(model_times) * 1000, 2)}


def run_onnx(cases, model, tokenizer, warmup, repeat):
    """Runs in a spawned process: ORT's CUDA 12 DLLs cannot share a process with PyTorch's CUDA 11."""
    import onnxruntime as ort
    from app.core.config.loader import load_config
    from app.domain.pipeline.a_gdinoDtct import GdinoDetect

    config = load_config()
    config["GROUNDINGDINO_CONFIG"].update(model_path=str(model), tokenizer_path=str(tokenizer))
    detector = GdinoDetect(config)
    model_times = []
    detector.session.run = timed(detector.session.run, model_times)
    results = []
    for case in cases:
        run = partial(detector.predict_with_metadata, case["image"], case["caption"],
                      case["box_threshold"], case["text_threshold"])
        (result, metadata), timing = measure(run, model_times, warmup, repeat)
        results.append({**result, "token_ids": metadata["token_ids"],
                        "original_size": metadata["original_size"],
                        "resized_size": metadata["resized_size"], "timing": timing})
    return {"results": results, "providers": detector.session.get_providers(),
            "onnxruntime": ort.__version__}


def run_pytorch(cases, root, config, checkpoint, device, warmup, repeat):
    import torch
    import torchvision
    import transformers

    sys.path.insert(0, str(root))
    from groundingdino.util.inference import load_image, load_model, predict, preprocess_caption
    from app.domain.groundingdino.preprocess import preprocess_image

    try:
        from groundingdino import _C  # noqa: F401
        custom_ops = True
    except ImportError:
        custom_ops = False
    # ORT runs with use_tf32=0; keep PyTorch in full FP32 so differences come from the conversion.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cwd = os.getcwd()
    # Like fast_api.py: the config's "bert-base-uncased" resolves to Grounding's local copy.
    os.chdir(root)
    try:
        model = load_model(str(config), str(checkpoint), device=device)
    finally:
        os.chdir(cwd)
    model_times = []
    model.forward = timed(model.forward, model_times, torch.cuda.synchronize if device == "cuda" else None)

    def infer(case):
        _, image = load_image(case["image"])
        return image, predict(model, image, case["caption"], case["box_threshold"],
                              case["text_threshold"], device=device)

    results, preprocessing = [], {}
    for case in cases:
        (image, (boxes, logits, phrases)), timing = measure(partial(infer, case), model_times, warmup, repeat)
        results.append({"boxes": boxes.tolist(), "logits": logits.tolist(), "phrases": phrases,
                        "token_ids": model.tokenizer(preprocess_caption(case["caption"]))["input_ids"],
                        "resized_size": [image.shape[2], image.shape[1]], "timing": timing})
        if case["image"] not in preprocessing:
            onnx_image = preprocess_image(case["image"])[0][0]
            same_shape = onnx_image.shape == tuple(image.shape)
            preprocessing[case["image"]] = {
                "pt_shape": list(image.shape), "onnx_shape": list(onnx_image.shape),
                "max_abs_diff": float(np.abs(image.numpy() - onnx_image).max()) if same_shape else None}
    return {"results": results, "preprocessing": preprocessing, "custom_ops": custom_ops,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                         "transformers": transformers.__version__}}


def draw_overlay(path, pt, onnx, output):
    with Image.open(path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    size = max(14, width // 60)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    text = {"font": font, "stroke_width": 2, "stroke_fill": "black"}
    draw.text((8, 8), "PyTorch", fill=(0, 220, 0), **text)
    draw.text((8, 12 + size), "ONNX", fill=(255, 40, 40), **text)
    for name, result, color, line in (("PT", pt, (0, 220, 0), 6), ("ONNX", onnx, (255, 40, 40), 2)):
        for (cx, cy, bw, bh), score, phrase in zip(result["boxes"], result["logits"], result["phrases"]):
            x0, y0 = (cx - bw / 2) * width, (cy - bh / 2) * height
            x1, y1 = (cx + bw / 2) * width, (cy + bh / 2) * height
            draw.rectangle((x0, y0, x1, y1), outline=color, width=line)
            y = y0 + 4 if name == "PT" else y1 - size - 6
            draw.text((x0 + 4, y), f"{name} {phrase} {score:.3f}", fill=color, **text)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=90)


def check_settings():
    paths = (ONNX_MODEL, ONNX_TOKENIZER, PT_ROOT / "groundingdino", PT_CONFIG, PT_CHECKPOINT, *IMAGES)
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Settings point to missing paths: " + ", ".join(missing))
    if PT_DEVICE not in ("cuda", "cpu"):
        raise ValueError(f"PT_DEVICE must be 'cuda' or 'cpu', got {PT_DEVICE!r}")
    if WARMUP_RUNS < 0 or TIMED_RUNS < 1:
        raise ValueError("WARMUP_RUNS must be >= 0 and TIMED_RUNS >= 1")
    if not all(0 <= value <= 1 for pair in THRESHOLDS for value in pair):
        raise ValueError("THRESHOLDS must be (box, text) pairs within [0, 1]")


def build_cases():
    images = []
    for entry in map(Path, IMAGES):
        images += (sorted(p for p in entry.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
                   if entry.is_dir() else [entry])
    if not images:
        raise FileNotFoundError(f"No images found in {[str(p) for p in IMAGES]}")
    return [{"image": str(path.resolve()), "caption": caption, "box_threshold": box, "text_threshold": token}
            for path in images for caption in CAPTIONS for box, token in THRESHOLDS]


def main():
    check_settings()
    cases = build_cases()
    run_dir = Path(OUTPUT_DIR).resolve() / time.strftime("%Y%m%d_%H%M%S")
    tolerance = {"match_iou": MATCH_IOU, "min_iou": MIN_IOU,
                 "max_score_diff": MAX_SCORE_DIFF, "max_box_diff": MAX_BOX_DIFF}
    print(f"{len(cases)} cases; hashing model files...", flush=True)
    models = {"onnx": str(ONNX_MODEL), "onnx_sha256": sha256_file(ONNX_MODEL),
              "onnx_tokenizer": str(ONNX_TOKENIZER), "pytorch_config": str(PT_CONFIG),
              "pytorch_checkpoint": str(PT_CHECKPOINT), "pytorch_checkpoint_sha256": sha256_file(PT_CHECKPOINT)}

    print(f"Running ONNX pipeline in a separate process "
          f"({WARMUP_RUNS} warm-up + {TIMED_RUNS} timed runs per case)...", flush=True)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        onnx = pool.submit(run_onnx, cases, ONNX_MODEL, ONNX_TOKENIZER, WARMUP_RUNS, TIMED_RUNS).result()
    print(f"ONNX providers: {onnx['providers']}", flush=True)
    print(f"Running PyTorch pipeline on {PT_DEVICE}...", flush=True)
    pt = run_pytorch(cases, PT_ROOT, PT_CONFIG, PT_CHECKPOINT, PT_DEVICE, WARMUP_RUNS, TIMED_RUNS)

    report_cases, pairs = [], []
    for index, (case, pt_result, onnx_result) in enumerate(zip(cases, pt["results"], onnx["results"])):
        comparison = compare_detections(pt_result, onnx_result, case["box_threshold"],
                                        onnx_result["original_size"], **tolerance)
        checks = ((pt_result["token_ids"] == onnx_result["token_ids"], "token ids differ"),
                  (pt_result["resized_size"] == onnx_result["resized_size"],
                   f"resized size {pt_result['resized_size']} != {onnx_result['resized_size']}"))
        failures = [message for ok, message in checks if not ok]
        if failures:
            comparison["failures"] = failures + comparison["failures"]
            comparison["status"] = "fail"
        timing = {"pytorch": pt_result.pop("timing"), "onnx": onnx_result.pop("timing")}
        slug = re.sub(r"[^a-z0-9]+", "-", case["caption"].lower()).strip("-")[:40]
        vis = run_dir / "vis" / (f"{index:03d}_{Path(case['image']).stem}_{slug}"
                                 f"_b{case['box_threshold']}_t{case['text_threshold']}.jpg")
        draw_overlay(case["image"], pt_result, onnx_result, vis)
        pairs += comparison["pairs"]
        report_cases.append({**case, **comparison, "timing": timing, "visualization": str(vis),
                             "pytorch": pt_result, "onnx": onnx_result})
        worst = max((p["score_diff"] for p in comparison["pairs"]), default=0.0)
        worst_px = max((p["box_diff_px"] for p in comparison["pairs"]), default=0.0)
        print(f"[{comparison['status'].upper():>10}] {Path(case['image']).name} | {case['caption']!r} | "
              f"box={case['box_threshold']} text={case['text_threshold']} | "
              f"PT={comparison['pt_count']} ONNX={comparison['onnx_count']} | "
              f"score diff<={worst:.2e} box diff<={worst_px:.2f}px", flush=True)
        print(f"             time PT {timing['pytorch']['total_ms']:.1f}ms "
              f"(model {timing['pytorch']['model_ms']:.1f}) | ONNX {timing['onnx']['total_ms']:.1f}ms "
              f"(model {timing['onnx']['model_ms']:.1f})", flush=True)
        for message in comparison["failures"]:
            print(f"             {message}", flush=True)

    statuses = [case["status"] for case in report_cases]
    timing = {backend: {key: round(statistics.fmean(c["timing"][backend][key] for c in report_cases), 2)
                        for key in ("total_ms", "model_ms")} for backend in ("pytorch", "onnx")}
    timing["onnx_speedup"] = {key: round(timing["pytorch"][key] / timing["onnx"][key], 2)
                              for key in ("total_ms", "model_ms")}
    summary = {"cases": len(statuses), **{name: statuses.count(name) for name in ("pass", "borderline", "fail")},
               "pt_detections": sum(c["pt_count"] for c in report_cases),
               "onnx_detections": sum(c["onnx_count"] for c in report_cases),
               "paired_detections": len(pairs),
               "max_score_diff": max((p["score_diff"] for p in pairs), default=None),
               "max_box_diff": max((p["box_diff"] for p in pairs), default=None),
               "max_box_diff_px": max((p["box_diff_px"] for p in pairs), default=None),
               "min_iou": min((p["iou"] for p in pairs), default=None),
               "max_preprocess_abs_diff": max((d["max_abs_diff"] for d in pt["preprocessing"].values()
                                               if d["max_abs_diff"] is not None), default=None),
               "timing": timing}
    ms_deform_attn = "custom CUDA op" if pt["custom_ops"] else "PyTorch fallback"
    report = {"passed": summary["fail"] == 0, "summary": summary,
              "settings": {**tolerance, **models, "pt_device": PT_DEVICE, "tf32": False,
                           "warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS,
                           "timing": "median per case; total = image decoding through phrases, "
                                     "model = forward pass only (PyTorch with CUDA sync, ORT session.run)",
                           "pt_pipeline": "Grounding load_image() + predict(), no NMS",
                           "onnx_pipeline": "GdinoDetect.predict_with_metadata(), before NMS"},
              "environment": {"python": platform.python_version(), **pt["versions"],
                              "onnxruntime": onnx["onnxruntime"], "onnx_providers": onnx["providers"],
                              "cuda_device": pt["cuda_device"], "pt_ms_deform_attn": ms_deform_attn},
              "preprocessing": pt["preprocessing"], "cases": report_cases}
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{summary['pass']} pass, {summary['borderline']} borderline, {summary['fail']} fail "
          f"of {summary['cases']} cases; detections PT={summary['pt_detections']} "
          f"ONNX={summary['onnx_detections']} paired={summary['paired_detections']}")
    print(f"max score diff={summary['max_score_diff']}, max box diff={summary['max_box_diff_px']}px, "
          f"min IoU={summary['min_iou']}, max preprocess diff={summary['max_preprocess_abs_diff']}")
    print(f"Inference time, mean of per-case medians ({WARMUP_RUNS} warm-up + {TIMED_RUNS} timed runs):")
    print(f"  PyTorch ({PT_DEVICE}, {ms_deform_attn} ms_deform_attn): "
          f"total {timing['pytorch']['total_ms']} ms, model {timing['pytorch']['model_ms']} ms")
    print(f"  ONNX ({onnx['providers'][0]}): "
          f"total {timing['onnx']['total_ms']} ms, model {timing['onnx']['model_ms']} ms")
    print(f"  ONNX speed-up: total {timing['onnx_speedup']['total_ms']}x, "
          f"model {timing['onnx_speedup']['model_ms']}x")
    print(f"Consistency {'PASSED' if report['passed'] else 'FAILED'}; report: {target}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
