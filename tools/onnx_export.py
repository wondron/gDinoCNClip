"""Export the local GroundingDINO checkpoint to standard ONNX operators."""

import argparse
import json
import os
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GROUNDING_SOURCE_ROOT", str(BUNDLE_ROOT.parent / "Grounding")))
sys.path.extend([str(BUNDLE_ROOT), str(ROOT)])

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn
from torch.nn import functional as F

from groundingdino.models import build_model
from groundingdino.models.GroundingDINO import ms_deform_attn
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)
from groundingdino.util.misc import NestedTensor, clean_state_dict, inverse_sigmoid
from groundingdino.util.slconfig import SLConfig

INPUT_NAMES = [
    "images", "input_ids", "token_type_ids", "attention_mask",
    "position_ids", "text_self_attention_masks",
]


def exportable_attention(value, spatial_shapes, sampling_locations, attention_weights):
    """The existing grid_sample implementation, with dynamic per-level slices."""
    batch, _, heads, channels = value.shape
    queries = sampling_locations.shape[1]
    points = sampling_locations.shape[4]
    grids = sampling_locations * 2 - 1
    sampled = []
    start = 0
    # The number of feature levels is a model constant; their sizes are dynamic.
    for level in range(spatial_shapes.shape[0]):
        height, width = spatial_shapes[level, 0], spatial_shapes[level, 1]
        length = height * width
        feature = value[:, start:start + length].flatten(2).transpose(1, 2)
        feature = feature.reshape(batch * heads, channels, height, width)
        grid = grids[:, :, :, level].transpose(1, 2).reshape(batch * heads, queries, points, 2)
        sampled.append(F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        start = start + length
    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, -1)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return output.reshape(batch, heads * channels, queries).transpose(1, 2).contiguous()


def load_export_model(config=None, checkpoint=None, device="cpu"):
    config = Path(config or ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py").resolve()
    checkpoint = Path(checkpoint or ROOT / "weights/groundingdino_swint_ogc.pth").resolve()
    args = SLConfig.fromfile(str(config))
    args.device = device
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    if (ROOT / "bert-base-uncased").is_dir():
        args.text_encoder_type = str(ROOT / "bert-base-uncased")
    model = build_model(args)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    # The published checkpoint includes unused BERT buffers; missing trainable
    # weights, however, would make an equivalence claim meaningless.
    parameters = dict(model.named_parameters())
    missing_parameters = [key for key in incompatible.missing_keys if key in parameters]
    if missing_parameters:
        raise RuntimeError(f"Checkpoint is missing model weights: {missing_parameters}")
    for module in model.modules():
        for flag in ("use_checkpoint", "use_transformer_ckpt"):
            if hasattr(module, flag):
                setattr(module, flag, False)
    return model.to(device).eval()


class GroundingDINOOnnx(nn.Module):
    """Original model computation with tokenizer/Python mask generation outside."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, input_ids, token_type_ids, attention_mask,
                position_ids, text_self_attention_masks):
        model = self.model
        encoded = model.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=text_self_attention_masks,
            position_ids=position_ids,
        )["last_hidden_state"]
        text = {
            "encoded_text": model.feat_map(encoded),
            "text_token_mask": attention_mask.bool(),
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }
        mask = torch.zeros_like(images[:, 0], dtype=torch.bool)
        samples = NestedTensor(images, mask)
        features, positions = model.backbone(samples)
        sources, masks = [], []
        for level, feature in enumerate(features):
            source, feature_mask = feature.decompose()
            sources.append(model.input_proj[level](source))
            masks.append(feature_mask)
        for level in range(len(sources), model.num_feature_levels):
            source = model.input_proj[level](features[-1].tensors if level == len(features) else sources[-1])
            feature_mask = F.interpolate(mask[None].float(), size=source.shape[-2:]).bool()[0]
            position = model.backbone[1](NestedTensor(source, feature_mask)).to(source.dtype)
            sources.append(source)
            masks.append(feature_mask)
            positions.append(position)
        hidden, reference, _, _, _ = model.transformer(
            sources, masks, None, positions, None, None, text
        )
        boxes = (model.bbox_embed[-1](hidden[-1]) + inverse_sigmoid(reference[-2])).sigmoid()
        logits = model.class_embed[-1](hidden[-1], text)
        return logits, boxes


def example_inputs(model, height, width, caption, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    tokenized = model.tokenizer([caption], padding="longest", return_tensors="pt").to(device)
    masks, positions, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, model.specical_tokens, model.tokenizer
    )
    limit = model.max_text_len
    return (
        torch.zeros(1, 3, height, width, device=device),
        tokenized["input_ids"][:, :limit],
        tokenized["token_type_ids"][:, :limit],
        tokenized["attention_mask"][:, :limit].bool(),
        positions[:, :limit],
        masks[:, :limit, :limit],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--checkpoint", default=str(ROOT / "weights/groundingdino_swint_ogc.pth"))
    parser.add_argument("--output", default=str(BUNDLE_ROOT / "config/model/groundingdino/groundingdino_swint_ogc.onnx"))
    parser.add_argument("--height", type=int, default=256, help="Example input height, not a fixed deployment size")
    parser.add_argument("--width", type=int, default=384, help="Example input width, not a fixed deployment size")
    parser.add_argument("--caption", default="glass . straw . lemon .")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_export_model(args.config, args.checkpoint, args.device)
    wrapper = GroundingDINOOnnx(model).eval()
    inputs = example_inputs(model, args.height, args.width, args.caption, args.device)
    axes = {"images": {2: "height", 3: "width"}}
    axes.update({name: {1: "tokens"} for name in INPUT_NAMES[1:-1]})
    axes["text_self_attention_masks"] = {1: "tokens", 2: "tokens"}
    original_attention, original_extension = ms_deform_attn.multi_scale_deformable_attn_pytorch, ms_deform_attn._C
    try:
        ms_deform_attn._C = None
        ms_deform_attn.multi_scale_deformable_attn_pytorch = exportable_attention
        print(f"Exporting {output} (dynamic height, width, tokens; batch=1)", flush=True)
        with torch.no_grad():
            torch.onnx.export(
                wrapper, inputs, str(output), export_params=True,
                opset_version=17, do_constant_folding=True,
                input_names=INPUT_NAMES, output_names=["pred_logits", "pred_boxes"],
                dynamic_axes=axes, dynamo=False,
            )
    finally:
        ms_deform_attn.multi_scale_deformable_attn_pytorch = original_attention
        ms_deform_attn._C = original_extension
    import onnx
    onnx.checker.check_model(str(output))
    metadata = {
        "batch_size": 1, "dynamic_axes": ["height", "width", "tokens"],
        "opset": 17, "max_text_len": 256,
        "image_preprocessing": {"color": "RGB", "short_side": 800, "max_size": 1333,
                                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "outputs": {"pred_logits": "raw logits; padded token positions are -inf", "pred_boxes": "normalized cxcywh"},
        "torch_version": torch.__version__,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ONNX checker passed: {output} ({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
