"""Decode GroundingDINO scores and normalized boxes without NMS."""

import numpy as np


def postprocess(pred_logits, pred_boxes, tokenizer, metadata, box_threshold=0.35,
                text_threshold=0.25, *, validate_thresholds=True):
    """Match original predict(): strict thresholds, normalized cxcywh, no NMS."""
    if validate_thresholds and (not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1):
        raise ValueError("Thresholds must be between zero and one")
    pred_logits = np.asarray(pred_logits)
    pred_boxes = np.asarray(pred_boxes)
    if pred_logits.ndim != 3 or pred_logits.shape[0] != 1:
        raise ValueError(f"Expected batch-one logits [1,Q,T], got {pred_logits.shape}")
    if pred_boxes.shape != (1, pred_logits.shape[1], 4):
        raise ValueError(f"Expected boxes [1,Q,4], got {pred_boxes.shape}")
    if np.isnan(pred_logits).any() or not np.isfinite(pred_boxes).all():
        raise ValueError("ONNX outputs contain invalid numeric values")
    probabilities = np.exp(-np.logaddexp(np.float32(0), -pred_logits[0]))
    scores = probabilities.max(axis=1)
    selected = scores > box_threshold
    selected_logits = probabilities[selected]
    token_ids = metadata["token_ids"]
    phrases = []
    for row in selected_logits:
        # get_phrases_from_posmap excludes token 0 and all indices >=255.
        indices = np.flatnonzero(row > text_threshold)
        indices = indices[(indices > 0) & (indices < min(255, len(token_ids)))]
        phrases.append(tokenizer.decode([token_ids[i] for i in indices]).replace(".", ""))
    return {
        "boxes": pred_boxes[0, selected].tolist(),
        "logits": scores[selected].tolist(),
        "phrases": phrases,
    }
