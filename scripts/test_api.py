"""Call the original /process/ API contract without importing PyTorch."""

import argparse
import json
import mimetypes
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def convert_boxes_to_xyxy_pixel(boxes, image_size):
    """Convert normalized cxcywh boxes returned by the service to pixel xyxy."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    width, height = image_size
    centers = boxes[:, :2] * [width, height]
    half_sizes = boxes[:, 2:] * [width, height] / 2
    return np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)


def save_annotated(image_path, result, destination):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    rectangles = convert_boxes_to_xyxy_pixel(result["boxes"], image.size)
    draw = ImageDraw.Draw(image)
    for rectangle, phrase, score in zip(rectangles, result["phrases"], result["logits"]):
        coordinates = tuple(float(value) for value in rectangle)
        draw.rectangle(coordinates, outline="red", width=3)
        draw.text((coordinates[0], max(0, coordinates[1] - 14)),
                  f"{phrase} {float(score):.2f}", fill="red")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def call_service(url, image_path, discri, box_thre, text_thre, timeout=180):
    content_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    with open(image_path, "rb") as source:
        response = requests.post(
            url,
            files={"image": (Path(image_path).name, source, content_type)},
            data={"discri": discri, "box_thre": str(box_thre), "text_thre": str(text_thre)},
            timeout=timeout,
        )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    result = response.json()
    if set(result) != {"boxes", "phrases", "logits"}:
        raise RuntimeError(f"Unexpected response fields: {sorted(result)}")
    if not (len(result["boxes"]) == len(result["phrases"]) == len(result["logits"])):
        raise RuntimeError("Response boxes, phrases, and logits have different lengths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8678/process/")
    parser.add_argument("--image", default=str(ROOT / "config/images/test.jpg"))
    parser.add_argument("--discri", default="glass . straw . lemon .")
    parser.add_argument("--box-thre", type=float, default=0.35)
    parser.add_argument("--text-thre", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--annotated", help="Optional annotated image output path")
    args = parser.parse_args()
    try:
        result = call_service(args.url, args.image, args.discri,
                              args.box_thre, args.text_thre, args.timeout)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(serialized + "\n", encoding="utf-8")
            print(f"Saved {len(result['boxes'])} detections to {target}", file=sys.stderr)
        else:
            print(serialized)
        if args.annotated:
            save_annotated(args.image, result, args.annotated)
            print(f"Saved annotated image to {args.annotated}", file=sys.stderr)
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

