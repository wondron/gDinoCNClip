"""Line up benchmark.py reports from several servers: environment, stage timings and output consistency.

    python compare.py output/serverA_20261102_231000 output/serverB_20261102_232500 [...]

Each argument is a report.json or the folder holding it; the first one is the baseline. Timing cells
read "ms (speed-up)" with speed-up = baseline ms / this server's ms, so >1 means faster than the
baseline. Detections of every other report are paired with the baseline's by IoU, using the
tolerances of tools/compare_pt_onnx.py; a GroundingDINO box found on one side only is "borderline"
when its score is within MAX_SCORE_DIFF of the box threshold.

Prints Markdown and saves it as output/compare_<timestamp>.md (--output to choose the file).
Exits with 1 when any case is inconsistent.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCH_IOU = 0.5  # minimum IoU to pair two detections
MIN_IOU = 0.98
MAX_SCORE_DIFF = 0.005
CASE_KEY = ("image", "dino_cap", "dino_box", "dino_text", "clip_cap", "clip_thre")
SAME_SETTINGS = ("provider", "clip_provider", "threads", "warmup_runs", "timed_runs", "short_side",
                 "max_size", "max_text_len", "nms_threshold", "clip_detect_range")
PLACEMENT_MODELS = {"gdino": ("GroundingDINO", "provider"), "clip_image": ("CLIP 图像", "clip_provider"),
                    "clip_text": ("CLIP 文本", "clip_provider")}
STAGE_NAMES = {
    "total_ms": "**端到端总耗时**", "decode_ms": "图片解码", "gdino_ms": "GroundingDINO 合计",
    "gdino_preprocess_ms": "└ 预处理", "gdino_infer_ms": "└ 推理 session.run",
    "gdino_postprocess_ms": "└ 后处理 + NMS", "clip_ms": "CLIP 合计（全部框）",
    "clip_image_infer_ms": "└ 图像推理", "clip_text_infer_ms": "└ 文本推理",
}


def load_report(path):
    path = Path(path)
    if path.is_dir():
        path = path / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_path"] = path
    return report


def cxcywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def pairwise_iou(left, right):
    lo = np.maximum(left[:, None, :2], right[None, :, :2])
    hi = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(hi - lo, 0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0, None).prod(axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def pair(left, right):
    """Greedy one-to-one pairing of xyxy boxes by IoU; returns [(i, j, iou)], unpaired i, unpaired j."""
    left = np.asarray(left, dtype=np.float64).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)
    iou = pairwise_iou(left, right)
    pairs, used_left, used_right = [], set(), set()
    for i, j in sorted(np.argwhere(iou >= MATCH_IOU).tolist(), key=lambda p: -iou[p[0], p[1]]):
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j, float(iou[i, j])))
    return (pairs, [i for i in range(len(left)) if i not in used_left],
            [j for j in range(len(right)) if j not in used_right])


def compare_case(base, other):
    """Return (status, messages, max score diff) for a case that succeeded in both reports."""
    failures, borderline, max_diff = [], [], 0.0
    b, o = base["gdino"], other["gdino"]
    pairs, only_base, only_other = pair(cxcywh_to_xyxy(b["boxes"]), cxcywh_to_xyxy(o["boxes"]))
    for i, j, iou in pairs:
        diff = abs(b["scores"][i] - o["scores"][j])
        max_diff = max(max_diff, diff)
        label = f"DINO {b['phrases'][i]!r}@{b['scores'][i]:.4f}"
        if b["phrases"][i] != o["phrases"][j]:
            failures.append(f"{label}: 短语不同 {o['phrases'][j]!r}")
        if iou < MIN_IOU:
            failures.append(f"{label}: IoU {iou:.4f} < {MIN_IOU}")
        if diff > MAX_SCORE_DIFF:
            failures.append(f"{label}: 分数差 {diff:.4f} > {MAX_SCORE_DIFF}")
    for side, result, indices in (("基准", b, only_base), ("对比", o, only_other)):
        for k in indices:
            near = abs(result["scores"][k] - base["dino_box"]) <= MAX_SCORE_DIFF
            (borderline if near else failures).append(f"DINO 仅{side}有 {result['phrases'][k]!r}@{result['scores'][k]:.4f}")
    # A box that DINO kept on one side only also reaches CLIP on that side only.
    clip_unpaired = borderline if borderline else failures
    pairs, only_base, only_other = pair(base["rects"], other["rects"])
    for i, j, _ in pairs:
        diff = abs(base["confidences"][i] - other["confidences"][j])
        max_diff = max(max_diff, diff)
        if diff > MAX_SCORE_DIFF:
            failures.append(f"CLIP 框 {base['rects'][i]}: 置信度差 {diff:.4f} > {MAX_SCORE_DIFF}")
    for side, result, indices in (("基准", base, only_base), ("对比", other, only_other)):
        clip_unpaired += [f"CLIP 仅{side}保留 {result['rects'][k]}@{result['confidences'][k]:.4f}" for k in indices]
    return ("fail" if failures else "borderline" if borderline else "pass"), failures + borderline, max_diff


def cell(value, base, higher_is_better=False):
    if value is None:
        return "—"
    if base is None or not base or not value:
        return f"{value:.1f}"
    ratio = value / base if higher_is_better else base / value
    return f"{value:.1f} ({ratio:.2f}x)"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def setting(report, key):
    value = report["settings"].get(key)
    # CUDA vs MUSA is what a cross-vendor run compares; only GPU vs CPU makes the settings differ.
    if key in ("provider", "clip_provider"):
        return "cpu" if value == "cpu" else "gpu"
    return value


def environment_section(reports):
    def gpu(r, key):
        return (r["environment"].get("gpu") or {}).get(key, "—")

    rows = [
        ("报告", lambda r: r["_path"].parent.name), ("时间", lambda r: r["created"]),
        ("主机名", lambda r: r["environment"]["hostname"]), ("OS", lambda r: r["environment"]["os"]),
        ("CPU", lambda r: r["environment"]["cpu"]), ("CPU 逻辑核数", lambda r: r["environment"]["cpu_count"]),
        ("内存 GB", lambda r: r["environment"]["memory_gb"] or "—"),
        ("GPU", lambda r: gpu(r, "name")), ("驱动", lambda r: gpu(r, "driver")), ("显存", lambda r: gpu(r, "memory")),
        ("Python", lambda r: r["environment"]["python"]),
        ("onnxruntime", lambda r: r["environment"]["onnxruntime"]),
        ("numpy / opencv / Pillow", lambda r: " / ".join(str(r["environment"][k]) for k in ("numpy", "opencv", "pillow"))),
        ("DINO / CLIP provider", lambda r: f"{r['providers']['gdino']} / {r['providers']['clip']}"),
        ("预热 / 计时次数", lambda r: f"{r['settings']['warmup_runs']} / {r['settings']['timed_runs']}"),
    ]
    lines = ["## 环境", ""] + table(["项目"] + [r["tag"] for r in reports],
                                   [[name] + [get(r) for r in reports] for name, get in rows])
    base = reports[0]
    warnings = []
    for r in reports[1:]:
        changed = [name for name in base["models"] if base["models"][name]["sha256"] != r["models"].get(name, {}).get("sha256")]
        if changed:
            warnings.append(f"{r['tag']} 的模型文件与基准不同（SHA256）：{', '.join(changed)}，请确认复制完整")
        changed = [key for key in SAME_SETTINGS if setting(base, key) != setting(r, key)]
        if changed:
            warnings.append(f"{r['tag']} 的评测设置与基准不同：{', '.join(changed)}")
    return lines + [""] + [f"> **注意**：{w}" for w in warnings]


def timing_section(reports):
    base = reports[0]
    rows = [["模型加载 GroundingDINO"] + [cell(r["load_ms"]["gdino"], base["load_ms"]["gdino"]) for r in reports],
            ["模型加载 CLIP"] + [cell(r["load_ms"]["clip"], base["load_ms"]["clip"]) for r in reports]]

    def cold(r):
        first = next((c for c in r["cases"] if c["status"] == "ok" and c["warmup_ms"]), None)
        return first["warmup_ms"][0] if first else None

    rows.append(["首次请求（冷启动）"] + [cell(cold(r), cold(base)) for r in reports])
    for stage, name in STAGE_NAMES.items():
        rows.append([name] + [cell(r["summary"]["timing"].get(stage), base["summary"]["timing"].get(stage))
                              for r in reports])

    def throughput(r):
        total = r["summary"]["timing"].get("total_ms")
        return 1000 / total if total else None

    rows.append(["吞吐（张/秒，串行）"] + [cell(throughput(r), throughput(base), higher_is_better=True) for r in reports])
    return ["## 耗时汇总", "", "各用例中位数的平均，单位 ms；括号内为相对基准的加速比（>1 更快）。", ""] + \
        table(["阶段"] + [r["tag"] for r in reports], rows)


def placement_section(reports):
    """Nodes each GPU provider left on CPU, from benchmark.check_placement; empty for older reports."""
    if not any(r.get("placement") for r in reports):
        return []

    def info(r, key):
        if r["settings"].get(PLACEMENT_MODELS[key][1]) == "cpu":
            return "CPU 推理"
        p = (r.get("placement") or {}).get(key)
        return "—" if p is None else "检查失败" if "error" in p else p

    def nodes(p):
        if isinstance(p, str):
            return p
        return f"{p['by_provider'].get('CPUExecutionProvider', 0)} / {p['nodes']}，Memcpy {p['memcpy']}，{p['cpu_ms']:.1f} ms"

    def ops(p, base):
        if isinstance(p, str):
            return p
        if base is None:
            return ", ".join(f"{op}×{count}" for op, count in list(p["cpu_ops"].items())[:8]) or "无"
        # Shape ops sit on CPU under every provider; list only what this server adds on top of the baseline.
        base_ops = base["cpu_ops"] if isinstance(base, dict) else {}
        extra = sorted(((op, count - base_ops.get(op, 0)) for op, count in p["cpu_ops"].items()
                        if count > base_ops.get(op, 0)), key=lambda item: -item[1])
        return "比基准多：" + ", ".join(f"{op}+{count}" for op, count in extra) if extra else "与基准相同"

    rows = []
    for key, (name, _) in PLACEMENT_MODELS.items():
        base = info(reports[0], key)
        rows.append([f"{name}：CPU 节点 / 总数"] + [nodes(info(r, key)) for r in reports])
        rows.append([f"{name}：CPU 上的算子"] + [ops(base, None)] + [ops(info(r, key), base) for r in reports[1:]])
    return ["## 算子分配", "",
            "GPU provider 不支持的算子会留在 CPU 上执行，并通过 Memcpy 节点在显存和内存之间拷贝数据。"
            "单元格为 CPU 节点数 / 总节点数、Memcpy 节点数、一次推理中 CPU 节点的耗时。"
            "CUDA 也会主动把形状计算类的小算子放在 CPU 上，所以基准列列出最多的 8 种 CPU 算子，"
            "其他列只列出比基准多的部分；完整列表见各 report.json 的 placement。", ""] + \
        table(["模型"] + [r["tag"] for r in reports], rows) + [""]


def case_tables(reports):
    keys = list(dict.fromkeys(tuple(c[k] for k in CASE_KEY) for r in reports for c in r["cases"]))
    indexed = [{tuple(c[k] for k in CASE_KEY): c for c in r["cases"]} for r in reports]
    rows = []
    for key in keys:
        cells = []
        for cases in indexed:
            case = cases.get(key)
            if case is None:
                cells.append("缺失")
            elif case["status"] != "ok":
                cells.append("失败")
            else:
                total = case["timing"]["total_ms"]
                cells.append(f"{total['median']:.1f} / {total['p90']:.1f} ({len(case['gdino']['boxes'])}→{len(case['rects'])})")
        rows.append([f"{key[0]} · {key[1]} · {key[2]}/{key[3]} · {key[4]}"] + cells)
    lines = ["## 各用例端到端耗时", "", "中位数 / p90，单位 ms；括号内为 DINO 框数→CLIP 保留数。", ""]
    lines += table(["用例（图片 · DINO 提示词 · 阈值 · CLIP 提示词）"] + [r["tag"] for r in reports], rows)

    lines += ["", "## 结果一致性（对比基准）", "",
              f"按 IoU≥{MATCH_IOU} 配对；要求短语相同、IoU≥{MIN_IOU}、分数差≤{MAX_SCORE_DIFF}。", ""]
    summary_rows, details, inconsistent = [], [], False
    for r, cases in zip(reports[1:], indexed[1:]):
        counts = {"pass": 0, "borderline": 0, "fail": 0, "missing": 0}
        worst = 0.0
        for key in keys:
            base, other = indexed[0].get(key), cases.get(key)
            if not base or not other or base["status"] != "ok" or other["status"] != "ok":
                counts["missing"] += 1
                continue
            status, messages, diff = compare_case(base, other)
            counts[status] += 1
            worst = max(worst, diff)
            details += [f"- {r['tag']} | {key[0]} · {key[1]} · {key[4]}：{m}" for m in messages]
        inconsistent |= counts["fail"] > 0
        summary_rows.append([r["tag"], counts["pass"], counts["borderline"], counts["fail"], counts["missing"], f"{worst:.2e}"])
    if summary_rows:
        lines += table(["服务器", "一致", "临界", "不一致", "缺失/失败", "最大分数差"], summary_rows)
        lines += [""] + (details or ["所有用例结果一致。"])
    else:
        lines.append("只有一份报告，无可对比的结果。")
    return lines, inconsistent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="report.json files or their folders; the first is the baseline")
    parser.add_argument("--output", help="Markdown file (default: output/compare_<timestamp>.md)")
    args = parser.parse_args()
    reports = [load_report(path) for path in args.reports]

    case_lines, inconsistent = case_tables(reports)
    lines = ["# ONNX 推理服务器对比", "",
             f"生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}，基准：**{reports[0]['tag']}**。", ""]
    lines += environment_section(reports) + [""] + timing_section(reports) + [""] + placement_section(reports) + case_lines
    text = "\n".join(lines) + "\n"
    output = Path(args.output) if args.output else HERE / "output" / f"compare_{time.strftime('%Y%m%d_%H%M%S')}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"已保存: {output}")
    return 1 if inconsistent else 0


if __name__ == "__main__":
    sys.exit(main())
