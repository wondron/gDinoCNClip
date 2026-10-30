"""Draw normalized detection results on images."""

from pathlib import Path
from PIL import Image, ImageDraw


def annotate(image, result, output):
    """Draw normalized cxcywh detections on the original RGB image."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    else:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box, score, phrase in zip(result["boxes"], result["logits"], result["phrases"]):
        cx, cy, bw, bh = box
        coords = ((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height)
        draw.rectangle(coords, outline="red", width=3)
        draw.text((coords[0], max(0, coords[1]-14)), f"{phrase} {score:.2f}", fill="red")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
