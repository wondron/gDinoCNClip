"""Run GroundingDINO ONNX inference using the project's YAML configuration."""

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

# Support both `python scripts/infer.py` and `python -m scripts.infer`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.groundingdino.predictor import OnnxPredictor
from app.domain.groundingdino.visualize import annotate

logger = logging.getLogger("onnx.cli")


def main():
    from app.core.config.constant import DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD
    from app.core.config.loader import load_config
    from app.core.config.model import ModelSettings
    from app.core.observability.logging import setup_logger

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML configuration, relative to the project root")
    parser.add_argument("--model", help="Override the configured ONNX model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--caption", required=True, help='For example: "person. car."')
    parser.add_argument("--tokenizer", help="Override the configured tokenizer directory")
    parser.add_argument("--provider", choices=("cpu", "cuda"), help="Override use_gpu in YAML")
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--short-side", type=int)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--max-text-len", type=int)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--output", help="JSON output file; stdout if omitted")
    parser.add_argument("--annotated", help="Optional annotated output image")
    parser.add_argument("--threads", type=int, help="ORT CPU intra-op thread count (0=ORT default)")
    args = parser.parse_args()
    setup_logger()
    config = load_config(args.config)
    model_config = config["GROUNDINGDINO_CONFIG"]
    for flag, key in (("model", "model_path"), ("tokenizer", "tokenizer_path"),
                      ("threads", "threads"), ("short_side", "short_side"),
                      ("max_size", "max_size"), ("max_text_len", "max_text_len")):
        value = getattr(args, flag)
        if value is not None:
            model_config[key] = value
    if args.provider is not None:
        model_config["use_gpu"] = args.provider == "cuda"
    if args.device_id is not None:
        config.setdefault("TOTAL_CONFIG", {})["device_id"] = args.device_id
    cfg = ModelSettings.from_config(config)
    logger.info("CLI 配置已读取 | model=%s | tokenizer=%s | provider=%s",
                cfg.model_path, cfg.tokenizer_path, cfg.provider)
    predictor = OnnxPredictor(config)
    started = perf_counter()
    result, metadata = predictor.predict_with_metadata(
        args.image, args.caption, args.box_threshold, args.text_threshold,
    )
    logger.info("输入信息 | providers=%s | image_size=%s | text_length=%s",
                predictor.session.get_providers(), metadata['resized_size'], metadata['text_length'])
    if metadata["resize_policy"] == "fixed_model_dimensions":
        print("This graph fixes image dimensions; resizing to its declared height/width. "
              "Use a dynamic-image graph to preserve the original resize policy.", file=sys.stderr)
    logger.info("CLI 推理完成 | detections=%d | cost_ms=%.2f",
                len(result['boxes']), (perf_counter() - started) * 1000)
    result.update({"caption": metadata["caption"], "box_format": "normalized_cxcywh",
                   "original_size": metadata["original_size"],
                   "model_input_size": metadata["resized_size"],
                   "providers": predictor.session.get_providers()})
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        logger.info("结果已保存 | detections=%d | path=%s", len(result['boxes']), args.output)
    else:
        print(serialized)
    if args.annotated:
        annotate(args.image, result, args.annotated)
        logger.info("标注图片已保存 | path=%s", args.annotated)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("ONNX 推理失败")
        sys.exit(1)

