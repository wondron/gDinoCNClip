"""Benchmark the ONNX food pipeline on this server and save a report for compare.py.

Every image in IMAGES is run with every entry in CASES through the same flow as the service's
/food_detect/ (GroundingDINO + NMS, then Chinese-CLIP on each box). Each case gets WARMUP_RUNS
untimed runs and TIMED_RUNS timed runs; every stage is reported as median/mean/p90/min/max ms.

Edit the settings below if needed, then run from any directory:
    bash run.sh [--tag NAME]          # Linux: puts the pip CUDA libraries on LD_LIBRARY_PATH first
    python benchmark.py [--tag NAME]  # Windows, or when LD_LIBRARY_PATH is already set

Writes output/<tag>_<timestamp>/report.json and, with SAVE_VIS, one overlay per case under vis/
(GroundingDINO boxes blue, boxes kept by CLIP red).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from onnx_pipeline import STAGES, ChineseClip, GroundingDino, food_detect

HERE = Path(__file__).resolve().parent

# Settings (the command-line flags override the ones they name) ---------------
GDINO_MODEL = HERE / "models/groundingdino/groundingdino_swint_ogc.onnx"
GDINO_TOKENIZER = HERE / "models/groundingdino/bert-base-uncased"
CLIP_IMAGE_MODEL = HERE / "models/clip/clip_cn_vit-b-16.img.fp32.onnx"
CLIP_TEXT_MODEL = HERE / "models/clip/clip_cn_vit-b-16.txt.fp32.onnx"
IMAGES = [HERE / "images"]  # image files and/or folders (non-recursive)
OUTPUT_DIR = HERE / "output"
PROVIDER = "cuda"  # GroundingDINO: "cuda" or "cpu"
# The service lists CPU before CUDA for CLIP (b_clipdtct.get_providers), so CLIP runs on CPU there.
# "cpu" reproduces that; "cuda" measures CLIP on the GPU.
CLIP_PROVIDER = "cpu"
DEVICE_ID = 0
THREADS = 4  # GroundingDINO intra-op threads, as GROUNDINGDINO_CONFIG.threads
# The values below match config/param/config.yaml.
SHORT_SIDE = 800
MAX_SIZE = 1333
MAX_TEXT_LEN = 256
NMS_THRESHOLD = 0.5
CLIP_DETECT_RANGE = ["水泥", "猫", "狗", "无食材", "猪肉", "鱼", "螃蟹", "鱿鱼", "蘑菇", "人", "皮皮虾"]
# /food_detect/ form fields: (dino_cap, dino_box, dino_text, clip_cap, clip_thre). A clip_cap outside
# CLIP_DETECT_RANGE makes the service run the CLIP text model again for every box.
CASES = [
    ("food", 0.3, 0.4, "食物", 0.5),
    ("tofu . pickled vegetable . plate .", 0.35, 0.25, "豆腐", 0.5),
    ("mantis shrimp . shrimp . crab .", 0.35, 0.25, "皮皮虾", 0.5),
]
WARMUP_RUNS = 3  # untimed runs per case: CUDA initialization and per-shape kernel selection
TIMED_RUNS = 20
SAVE_VIS = True
# ----------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CASE_FIELDS = ("dino_cap", "dino_box", "dino_text", "clip_cap", "clip_thre")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or None


def memory_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    return None


def gpu_info(device_id):
    try:
        output = subprocess.run(
            ["nvidia-smi", "-i", str(device_id), "--format=csv,noheader",
             "--query-gpu=name,driver_version,memory.total"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    name, driver, memory = (part.strip() for part in output.splitlines()[0].split(","))
    return {"name": name, "driver": driver, "memory": memory}


def environment(device_id):
    import onnxruntime as ort

    return {"hostname": platform.node(), "os": platform.platform(), "python": platform.python_version(),
            "cpu": cpu_name(), "cpu_count": os.cpu_count(), "memory_gb": memory_gb(),
            "gpu": gpu_info(device_id), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "opencv": package_version("opencv-python-headless") or package_version("opencv-python"),
            "pillow": package_version("Pillow"), "tokenizers": package_version("tokenizers")}


def summarize(values):
    ordered = sorted(values)
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
    return {"median": round(statistics.median(ordered), 2), "mean": round(statistics.fmean(ordered), 2),
            "p90": round(p90, 2), "min": round(ordered[0], 2), "max": round(ordered[-1], 2)}


def list_images():
    images = []
    for entry in map(Path, IMAGES):
        images += (sorted(p for p in entry.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
                   if entry.is_dir() else [entry])
    if not images:
        raise FileNotFoundError(f"No images found in {[str(p) for p in IMAGES]}")
    return images


def draw_overlay(image_bytes, result, output):
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    gdino = result["gdino"]
    for (cx, cy, bw, bh), score, phrase in zip(gdino["boxes"], gdino["scores"], gdino["phrases"]):
        x0, y0 = (cx - bw / 2) * width, (cy - bh / 2) * height
        draw.rectangle((x0, y0, (cx + bw / 2) * width, (cy + bh / 2) * height), outline=(40, 120, 255), width=2)
        draw.text((x0 + 4, y0 + 4), f"{phrase} {score:.3f}", fill=(40, 120, 255))
    for (x1, y1, x2, y2), confidence in zip(result["rects"], result["confidences"]):
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 4, max(0, y1 - 14)), f"CLIP {confidence:.3f}", fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=90)


def run_case(dino, clip, image_bytes, case, warmup, repeat):
    args = [case[field] for field in CASE_FIELDS]
    warmup_ms = []
    for _ in range(warmup):
        started = time.perf_counter()
        food_detect(dino, clip, image_bytes, *args)
        warmup_ms.append(round((time.perf_counter() - started) * 1000, 2))
    runs = []
    for _ in range(repeat):
        timings = dict.fromkeys(STAGES, 0.0)
        result = food_detect(dino, clip, image_bytes, *args, timings=timings)
        runs.append(timings)
    timing = {stage: summarize([run[stage] for run in runs]) for stage in STAGES}
    return result, warmup_ms, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=platform.node(), help="server label for the report (default: hostname)")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default=PROVIDER, help="GroundingDINO provider")
    parser.add_argument("--clip-provider", choices=("cuda", "cpu"), default=CLIP_PROVIDER)
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=TIMED_RUNS)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs >= 1")

    images = list_images()
    cases = [(path, {"image": path.name, **dict(zip(CASE_FIELDS, case))}) for path in images for case in CASES]
    run_dir = OUTPUT_DIR / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.tag)}_{time.strftime('%Y%m%d_%H%M%S')}"
    env = environment(args.device_id)
    gpu = env["gpu"]["name"] if env["gpu"] else "未检测到 nvidia-smi"
    print(f"[{args.tag}] {env['cpu']} x{env['cpu_count']} | GPU {gpu} | onnxruntime {env['onnxruntime']}", flush=True)
    print(f"GroundingDINO={args.provider} CLIP={args.clip_provider} | {len(cases)} 个用例，"
          f"每个预热 {args.warmup} 次 + 计时 {args.runs} 次；正在计算模型 SHA256...", flush=True)
    models = {name: {"file": path.relative_to(HERE).as_posix(), "sha256": sha256_file(path)}
              for name, path in (("gdino", GDINO_MODEL), ("clip_image", CLIP_IMAGE_MODEL),
                                 ("clip_text", CLIP_TEXT_MODEL))}

    started = time.perf_counter()
    dino = GroundingDino(GDINO_MODEL, GDINO_TOKENIZER, args.provider, args.device_id, THREADS,
                         SHORT_SIDE, MAX_SIZE, MAX_TEXT_LEN, NMS_THRESHOLD)
    load_ms = {"gdino": round((time.perf_counter() - started) * 1000, 2)}
    started = time.perf_counter()
    clip = ChineseClip(CLIP_IMAGE_MODEL, CLIP_TEXT_MODEL, CLIP_DETECT_RANGE, args.clip_provider, args.device_id)
    load_ms["clip"] = round((time.perf_counter() - started) * 1000, 2)
    providers = {"gdino": dino.session.get_providers()[0], "clip": clip.image_session.get_providers()[0]}
    print(f"模型加载 GroundingDINO {load_ms['gdino']:.0f} ms, CLIP {load_ms['clip']:.0f} ms | {providers}", flush=True)

    report_cases = []
    for index, (path, case) in enumerate(cases):
        image_bytes = path.read_bytes()
        label = f"{case['image']} | {case['dino_cap']!r} {case['dino_box']}/{case['dino_text']} | {case['clip_cap']}"
        try:
            result, warmup_ms, timing = run_case(dino, clip, image_bytes, case, args.warmup, args.runs)
        except Exception as exc:
            report_cases.append({**case, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index + 1}/{len(cases)}] {label} | 失败: {type(exc).__name__}: {exc}", flush=True)
            continue
        entry = {**case, "status": "ok", "warmup_ms": warmup_ms, "timing": timing, **result}
        if SAVE_VIS:
            slug = re.sub(r"[^a-z0-9]+", "-", case["dino_cap"].lower()).strip("-")[:40]
            vis = run_dir / "vis" / f"{index:03d}_{Path(case['image']).stem}_{slug}.jpg"
            draw_overlay(image_bytes, result, vis)
            entry["vis"] = vis.relative_to(run_dir).as_posix()
        report_cases.append(entry)
        print(f"[{index + 1}/{len(cases)}] {label} | 框 {len(result['gdino']['boxes'])} -> CLIP 保留 "
              f"{len(result['rects'])} | 总 {timing['total_ms']['median']:.1f} ms (p90 {timing['total_ms']['p90']:.1f}) "
              f"= DINO {timing['gdino_ms']['median']:.1f} (推理 {timing['gdino_infer_ms']['median']:.1f}) "
              f"+ CLIP {timing['clip_ms']['median']:.1f}", flush=True)

    succeeded = [case for case in report_cases if case["status"] == "ok"]
    summary = {"cases": len(report_cases), "failed": len(report_cases) - len(succeeded),
               "timing": {stage: round(statistics.fmean(c["timing"][stage]["median"] for c in succeeded), 2)
                          for stage in STAGES} if succeeded else {}}
    report = {
        "schema": 1, "tag": args.tag, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env, "providers": providers, "models": models, "load_ms": load_ms,
        "settings": {"provider": args.provider, "clip_provider": args.clip_provider, "device_id": args.device_id,
                     "threads": THREADS, "warmup_runs": args.warmup, "timed_runs": args.runs,
                     "short_side": SHORT_SIDE, "max_size": MAX_SIZE, "max_text_len": MAX_TEXT_LEN,
                     "nms_threshold": NMS_THRESHOLD, "clip_detect_range": CLIP_DETECT_RANGE,
                     "timing": "per case: median/mean/p90/min/max over timed runs; summary: mean of case medians"},
        "summary": summary, "cases": report_cases}
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{len(succeeded)}/{len(report_cases)} 个用例成功；各阶段耗时（各用例中位数的平均，ms）：")
    for stage, value in summary["timing"].items():
        print(f"  {stage:<22}{value:>10.2f}")
    print(f"报告: {target}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
