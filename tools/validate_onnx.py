"""Validate an exported GroundingDINO graph against its PyTorch model.

Example: python validate_onnx.py --model groundingdino.onnx --report validation.json
Multiple --image and --caption options are supported. The report distinguishes
raw, query-indexed tensor differences from IoU-matched detection differences.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("GROUNDING_SOURCE_ROOT", str(BUNDLE_ROOT.parent / "Grounding")))
sys.path.extend([str(BUNDLE_ROOT), str(SOURCE_ROOT)])

import numpy as np
from PIL import Image, ImageOps


DEFAULT_CAPTIONS = ("glass . straw . lemon .", "glass .", "straw . lemon .")
INPUT_NAMES = (
    "images", "input_ids", "token_type_ids", "attention_mask", "position_ids",
    "text_self_attention_masks",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metrics(left, right):
    """Summarize finite values and infinities without emitting invalid JSON."""
    left, right = np.asarray(left), np.asarray(right)
    result = {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    if left.shape != right.shape:
        return {**result, "shape_equal": False, "nonfinite_mask_equal": False}
    finite_left, finite_right = np.isfinite(left), np.isfinite(right)
    same_finite = np.array_equal(finite_left, finite_right)
    same_nonfinite = (
        same_finite
        and np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    valid = finite_left & finite_right
    diffs = np.abs(left[valid].astype(np.float64) - right[valid].astype(np.float64))
    result.update({
        "shape_equal": True,
        "nonfinite_mask_equal": bool(same_nonfinite),
        "has_nan": bool(np.isnan(left).any() or np.isnan(right).any()),
        "finite_count": int(valid.sum()),
        "max_abs_error": float(diffs.max()) if diffs.size else None,
        "mean_abs_error": float(diffs.mean()) if diffs.size else None,
    })
    return result


def _xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def detection_metrics(left, right, min_iou, max_score_diff, max_box_diff):
    """Compare thresholded detections by one-to-one maximum-IoU assignment."""
    from scipy.optimize import linear_sum_assignment

    lb = np.asarray(left["boxes"], dtype=np.float64).reshape(-1, 4)
    rb = np.asarray(right["boxes"], dtype=np.float64).reshape(-1, 4)
    result = {"pytorch_count": len(lb), "onnx_count": len(rb),
              "count_equal": len(lb) == len(rb), "matched_count": 0,
              "min_matched_iou": None, "max_matched_score_diff": None,
              "max_matched_box_diff": None, "phrase_mismatches": []}
    if not len(lb) or not len(rb):
        result["passed"] = len(lb) == len(rb)
        return result
    lxy, rxy = _xyxy(lb), _xyxy(rb)
    lo = np.maximum(lxy[:, None, :2], rxy[None, :, :2])
    hi = np.minimum(lxy[:, None, 2:], rxy[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=2)
    la = np.maximum(lxy[:, 2:] - lxy[:, :2], 0).prod(axis=1)
    ra = np.maximum(rxy[:, 2:] - rxy[:, :2], 0).prod(axis=1)
    union = la[:, None] + ra[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    li, ri = linear_sum_assignment(-iou)
    score_diff = np.abs(np.asarray(left["logits"])[li] - np.asarray(right["logits"])[ri])
    box_diff = np.abs(lb[li] - rb[ri])
    mismatches = [{"pytorch_index": int(i), "onnx_index": int(j),
                   "pytorch": left["phrases"][i], "onnx": right["phrases"][j]}
                  for i, j in zip(li, ri) if left["phrases"][i] != right["phrases"][j]]
    result.update({"matched_count": int(len(li)), "min_matched_iou": float(iou[li, ri].min()),
                   "max_matched_score_diff": float(score_diff.max()),
                   "max_matched_box_diff": float(box_diff.max()),
                   "phrase_mismatches": mismatches})
    result["passed"] = bool(result["count_equal"] and not mismatches
                            and result["min_matched_iou"] >= min_iou
                            and result["max_matched_score_diff"] <= max_score_diff
                            and result["max_matched_box_diff"] <= max_box_diff)
    return result


def _ort_worker(connection, model, provider, threads):
    """Create ORT in a fresh interpreter without importing torch."""
    try:
        from app.domain.common.onnx_utils import create_session

        if provider == "cuda" and "torch" in sys.modules:
            raise RuntimeError("CUDA ORT worker must not import PyTorch")
        session = create_session(model, provider, intra_op_num_threads=threads)
        if provider == "cuda" and "torch" in sys.modules:
            raise RuntimeError("PyTorch was unexpectedly loaded in the CUDA ORT worker")
        connection.send(("ready", {
            "inputs": [(item.name, item.shape) for item in session.get_inputs()],
            "outputs": [(item.name, item.shape) for item in session.get_outputs()],
            "providers": session.get_providers(),
            "torch_loaded": "torch" in sys.modules,
        }))
        while True:
            command = connection.recv()
            if command[0] == "close":
                return
            if command[0] != "run":
                raise ValueError(f"Unknown worker command: {command[0]}")
            connection.send(("result", session.run(command[1], command[2])))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class OrtProcessSession:
    """ORT-compatible interface backed by an isolated spawned process.

    CUDA 12 ORT and CUDA 11 PyTorch cannot safely load their DLLs in the same
    Windows process. A spawned worker gives each framework its own DLL space.
    """

    def __init__(self, model, provider, threads, timeout):
        self.timeout = timeout
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_ort_worker,
                                       args=(child, str(model), provider, threads))
        try:
            self.process.start()
            child.close()
            self.metadata = self._receive("ready")
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, expected):
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"ORT worker timed out waiting for {expected} after {self.timeout}s")
            if self.connection.poll(min(remaining, 0.5)):
                try:
                    kind, payload = self.connection.recv()
                except EOFError as exc:
                    raise RuntimeError("ORT worker closed its pipe unexpectedly") from exc
                if kind == "error":
                    raise RuntimeError(f"ORT worker failed:\n{payload}")
                if kind != expected:
                    raise RuntimeError(f"Unexpected ORT worker response: {kind!r}")
                return payload
            if not self.process.is_alive():
                raise RuntimeError(f"ORT worker exited unexpectedly (exit code {self.process.exitcode})")

    def get_inputs(self):
        return [SimpleNamespace(name=name, shape=shape) for name, shape in self.metadata["inputs"]]

    def get_outputs(self):
        return [SimpleNamespace(name=name, shape=shape) for name, shape in self.metadata["outputs"]]

    def get_providers(self):
        return self.metadata["providers"]

    def run(self, output_names, feeds):
        try:
            self.connection.send(("run", output_names, feeds))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("Cannot send data to ORT worker") from exc
        return self._receive("result")

    def close(self):
        try:
            if self.process.is_alive():
                try:
                    self.connection.send(("close",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(timeout=3)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(timeout=5)
        finally:
            self.connection.close()


def run(args, report):
    import onnx
    import onnxruntime as ort
    import torch
    from app.domain.common.onnx_utils import create_session

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    report["versions"] = {"torch": torch.__version__, "onnx": onnx.__version__,
                          "onnxruntime": ort.__version__, "transformers": version("transformers")}
    report["files"] = {"onnx": str(Path(args.model).resolve()),
                       "onnx_sha256": sha256_file(args.model),
                       "checkpoint": str(Path(args.checkpoint).resolve()),
                       "checkpoint_sha256": sha256_file(args.checkpoint)}
    print("Checking ONNX structure and model fingerprints...", flush=True)
    onnx.checker.check_model(str(args.model))
    report["onnx_checker"] = "passed"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch device requested but unavailable")
    print(f"Initializing ONNX Runtime ({args.provider})...", flush=True)
    if args.provider == "cuda":
        session = OrtProcessSession(args.model, args.provider, args.threads, args.ort_timeout)
    else:
        session = create_session(args.model, args.provider, intra_op_num_threads=args.threads)
    try:
        run_cases(args, report, session)
    finally:
        if isinstance(session, OrtProcessSession):
            session.close()


def run_cases(args, report, session):
    import torch
    from onnx_export import GroundingDINOOnnx, load_export_model
    from app.domain.groundingdino.postprocess import postprocess
    from app.domain.groundingdino.preprocess import prepare_inputs

    print(f"ONNX Runtime ready: {session.get_providers()}", flush=True)
    inputs = {item.name: item.shape for item in session.get_inputs()}
    outputs = {item.name: item.shape for item in session.get_outputs()}
    report["graph"] = {"inputs": inputs, "outputs": outputs,
                       "active_providers": session.get_providers()}
    if isinstance(session, OrtProcessSession):
        report["graph"]["ort_isolated_process"] = True
        report["graph"]["ort_process_torch_loaded"] = session.metadata["torch_loaded"]
    if set(inputs) != set(INPUT_NAMES):
        raise ValueError(f"Unexpected model inputs: {list(inputs)}")
    if set(outputs) != {"pred_logits", "pred_boxes"}:
        raise ValueError(f"Unexpected model outputs: {list(outputs)}")
    model = load_export_model(args.config, args.checkpoint, device=args.device)
    wrapper = GroundingDINOOnnx(model).to(args.device).eval()
    tokenizer = model.tokenizer
    captions = args.caption or list(DEFAULT_CAPTIONS)
    paths = args.image or [str(BUNDLE_ROOT / "config/images/test.jpg")]
    seen_images, seen_lengths = set(), set()
    total_positive = 0
    # Different content and token lengths, plus H/W changes and standard resize.
    for path in paths:
        with Image.open(path) as source:
            original = source.convert("RGB")
        cases = [(original, "original", 384, caption) for caption in captions]
        cases += [(ImageOps.mirror(original), "flipped", 385, captions[0]),
                  (original.transpose(Image.Transpose.TRANSPOSE), "transposed", 417, captions[-1]),
                  (original, "standard_preprocessing", 800, captions[0])]
        for image, variant, short_side, caption in cases:
            case = {"image": str(path), "variant": variant, "short_side": short_side,
                    "caption": caption, "passed": False}
            report["cases"].append(case)
            feeds, metadata = prepare_inputs(image, caption, tokenizer, input_shapes=inputs,
                                             short_side=short_side, max_text_len=model.max_text_len)
            case["metadata"] = metadata
            case["input_shapes"] = {k: list(v.shape) for k, v in feeds.items()}
            seen_images.add(tuple(feeds["images"].shape[-2:]))
            seen_lengths.add(int(feeds["input_ids"].shape[-1]))
            tensors = [torch.from_numpy(feeds[name]).to(args.device) for name in INPUT_NAMES]
            with torch.no_grad():
                native = model(tensors[0], captions=[metadata["caption"]])
                wrapped = wrapper(*tensors)
            native_arrays = [native["pred_logits"].detach().cpu().numpy(),
                             native["pred_boxes"].detach().cpu().numpy()]
            wrapped_arrays = [item.detach().cpu().numpy() for item in wrapped]
            del native, wrapped, tensors
            if args.device == "cuda":
                # The separate CUDA ORT process shares the same GPU; return
                # PyTorch's unused cached allocations before each ORT run.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            wrapper_metrics = {name: tensor_metrics(a, b) for name, a, b in
                               zip(("pred_logits", "pred_boxes"), native_arrays, wrapped_arrays)}
            case["native_vs_wrapper"] = wrapper_metrics
            wrapper_pass = all(m["shape_equal"] and m["nonfinite_mask_equal"]
                               and not m["has_nan"] and m["finite_count"] > 0
                               and m["max_abs_error"] <= 1e-5
                               for m in wrapper_metrics.values())
            ort_values = session.run(["pred_logits", "pred_boxes"], feeds)
            raw_metrics = {name: tensor_metrics(a, b) for name, a, b in
                           zip(("pred_logits", "pred_boxes"), native_arrays, ort_values)}
            raw_metrics["sigmoid_probabilities"] = tensor_metrics(
                np.exp(-np.logaddexp(np.float32(0), -native_arrays[0])),
                np.exp(-np.logaddexp(np.float32(0), -ort_values[0])))
            case["raw_query_indexed"] = raw_metrics
            raw_valid = all(m["shape_equal"] and m["nonfinite_mask_equal"]
                            and not m["has_nan"] and m["finite_count"] > 0
                            for m in raw_metrics.values())
            if not raw_valid:
                case["failure"] = "Invalid ONNX output shapes, NaNs, or mismatched finite masks"
                continue
            pt_det = postprocess(*native_arrays, tokenizer, metadata,
                                 box_threshold=args.box_threshold, text_threshold=args.text_threshold)
            ort_det = postprocess(*ort_values, tokenizer, metadata,
                                  box_threshold=args.box_threshold, text_threshold=args.text_threshold)
            detection = detection_metrics(pt_det, ort_det, args.min_iou,
                                          args.max_score_diff, args.max_box_diff)
            case["matched_detections"] = detection
            total_positive += detection["pytorch_count"]
            case["passed"] = bool(wrapper_pass and detection["passed"])
            print(f"{path} {variant} {short_side} {caption!r}: "
                  f"{'PASS' if case['passed'] else 'FAIL'} "
                  f"detections={detection['pytorch_count']}/{detection['onnx_count']}", flush=True)
    report["coverage"] = {"image_sizes": [list(s) for s in sorted(seen_images)],
                          "text_lengths": sorted(seen_lengths),
                          "positive_detections": total_positive}
    report["passed"] = bool(report["cases"] and all(c["passed"] for c in report["cases"])
                            and len(seen_images) >= 2 and len(seen_lengths) >= 2
                            and total_positive > 0)
    if len(seen_images) < 2:
        report["failures"].append("Fewer than two distinct image sizes were exercised")
    if len(seen_lengths) < 2:
        report["failures"].append("Fewer than two distinct text lengths were exercised")
    if total_positive == 0:
        report["failures"].append("No positive PyTorch detections were observed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--image", action="append", help="Image path; repeatable (default: config/images/test.jpg)")
    parser.add_argument("--caption", action="append", help="Caption; repeatable")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch device")
    parser.add_argument("--report", default="onnx_validation.json")
    parser.add_argument("--config", default=str(SOURCE_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--checkpoint", default=str(SOURCE_ROOT / "weights/groundingdino_swint_ogc.pth"))
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-iou", type=float, default=0.98)
    parser.add_argument("--max-score-diff", type=float, default=0.005)
    parser.add_argument("--max-box-diff", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ort-timeout", type=float, default=300,
                        help="CUDA ORT worker startup/inference timeout in seconds")
    args = parser.parse_args()
    report = {"passed": False, "cases": [], "failures": [],
              "comparison": {"wrapper_max_abs_error": 1e-5, "min_iou": args.min_iou,
                             "max_score_diff": args.max_score_diff,
                             "max_box_diff": args.max_box_diff,
                             "box_threshold": args.box_threshold,
                             "text_threshold": args.text_threshold,
                             "raw_metrics_scope": "query-indexed diagnostics; TopK can reorder queries",
                             "detection_metrics_scope": "thresholded detections matched one-to-one by maximum IoU"}}
    try:
        run(args, report)
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validation {'PASSED' if report['passed'] else 'FAILED'}; report: {target}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
