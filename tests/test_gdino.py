"""Exercise the pipeline detector without model weights, CUDA or network access."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.domain.pipeline.a_gdinoDtct import GdinoDetect
from app.domain.groundingdino.preprocess import INPUT_NAMES


class ToyTokenizer:
    pad_token_id = 0
    vocabulary = {"[CLS]": 101, "[SEP]": 102, ".": 1012, "?": 1029,
                  "object": 200, "glass": 201, "straw": 202}

    def __call__(self, texts, *, padding="longest", return_tensors="np"):
        rows = [[101] + [self.vocabulary.get(word, 100)
                         for word in text.replace(".", " . ").split()] + [102]
                for text in texts]
        length = max(map(len, rows))
        return {
            "input_ids": np.array([row + [0] * (length - len(row)) for row in rows], dtype=np.int64),
            "token_type_ids": np.zeros((len(rows), length), dtype=np.int64),
            "attention_mask": np.array([[1] * len(row) + [0] * (length - len(row))
                                        for row in rows], dtype=np.int64),
        }

    def convert_tokens_to_ids(self, tokens):
        return [self.vocabulary.get(token, 100) for token in tokens]

    def decode(self, ids):
        words = {value: key for key, value in self.vocabulary.items()}
        return " ".join(words.get(token, "[UNK]") for token in ids)


def fake_session():
    session = Mock()
    shapes = {name: [1, "text_length"] for name in INPUT_NAMES[1:]}
    shapes["images"] = [1, 3, 2, 3]
    shapes["text_self_attention_masks"] = [1, "text_length", "text_length"]
    session.get_inputs.return_value = [SimpleNamespace(name=name, shape=shape)
                                       for name, shape in shapes.items()]
    session.get_outputs.return_value = [SimpleNamespace(name=name)
                                        for name in ("pred_logits", "pred_boxes")]
    session.get_providers.return_value = ["CPUExecutionProvider"]

    def infer(outputs, feeds):
        ids = feeds["input_ids"][0]
        words = np.flatnonzero(np.isin(ids, [200, 201, 202]))
        first, last = words[0], words[-1]
        probabilities = np.full((1, 3, 256), 0.01, dtype=np.float32)
        probabilities[0, 0, last] = 0.6
        probabilities[0, 0, first] = 0.9
        probabilities[0, 1, last] = 0.7
        probabilities[0, 2, first] = 0.2
        logits = np.log(probabilities / (1 - probabilities))
        boxes = np.array([[[0.2, 0.3, 0.1, 0.2], [0.6, 0.5, 0.2, 0.3],
                           [0.8, 0.8, 0.1, 0.1]]], dtype=np.float32)
        return logits, boxes

    session.run.side_effect = infer
    return session


class GdinoDetectorChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "model.onnx").touch()
        (self.root / "tokenizer").mkdir()
        self.config = {
            "TOTAL_CONFIG": {"device_id": 0},
            "GROUNDINGDINO_CONFIG": {
                "model_path": str(self.root / "model.onnx"),
                "tokenizer_path": str(self.root / "tokenizer"),
                "use_gpu": False,
                "short_side": 2,
                "max_size": 3,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
                "caption": "glass . straw .",
                "box_threshold": 0.8,
                "text_threshold": 0.65,
            },
        }
        self.session = fake_session()
        tokenizer_patch = patch("app.domain.pipeline.a_gdinoDtct.BertTokenizer", return_value=ToyTokenizer())
        session_patch = patch("app.domain.pipeline.a_gdinoDtct.create_session", return_value=self.session)
        tokenizer_patch.start()
        session_patch.start()
        self.addCleanup(tokenizer_patch.stop)
        self.addCleanup(session_patch.stop)
        self.bgr = np.array([[[5, 80, 240], [30, 20, 10], [90, 100, 110]],
                             [[150, 10, 50], [255, 0, 12], [23, 45, 67]]], dtype=np.uint8)

    def test_bgr_pil_and_file_inputs_have_identical_rgb_tensors_and_results(self):
        detector = GdinoDetect(self.config)
        original = self.bgr.copy()
        rgb = Image.fromarray(self.bgr[:, :, ::-1])
        path = self.root / "image.png"
        rgb.save(path)
        reference = None
        for image in (self.bgr, rgb, path):
            with self.subTest(input_type=type(image).__name__):
                result = detector.detect(image)
                feeds = self.session.run.call_args.args[1]
                expected = self.bgr[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / np.float32(255)
                np.testing.assert_array_equal(feeds["images"], expected)
                self.assertEqual(set(feeds), set(INPUT_NAMES))
                self.assertEqual(feeds["attention_mask"].dtype, np.bool_)
                self.assertEqual(feeds["input_ids"].dtype, np.int64)
                self.assertEqual(set(result), {"boxes", "phrases", "logits"})
                self.assertEqual(result["phrases"], ["glass"])
                np.testing.assert_allclose(result["boxes"], [[0.2, 0.3, 0.1, 0.2]])
                np.testing.assert_allclose(result["logits"], [0.9])
                if reference is None:
                    reference = result
                else:
                    self.assertEqual(result, reference)
        np.testing.assert_array_equal(self.bgr, original)
        self.assertTrue(detector.ini_flag)

    def test_runtime_prompt_list_matches_caption_and_threshold_overrides(self):
        detector = GdinoDetect(self.config)
        caption_result = detector.detect(self.bgr, "Glass . Straw .", box_threshold=0.65, text_threshold=0.65)
        list_result = detector.detect(self.bgr, ["glass", "straw"], box_threshold=0.65, text_threshold=0.65)
        self.assertEqual(caption_result, list_result)
        self.assertEqual(list_result["phrases"], ["glass", "straw"])
        np.testing.assert_allclose(list_result["logits"], [0.9, 0.7])
        detector.detect(self.bgr, ["straw"])
        self.assertIn(202, self.session.run.call_args.args[1]["input_ids"])
        self.assertNotIn(201, self.session.run.call_args.args[1]["input_ids"])
        self.assertEqual(detector.detect(self.bgr, box_threshold=1.0),
                         {"boxes": [], "phrases": [], "logits": []})

    def test_predict_retains_http_threshold_behavior_and_metadata_validation(self):
        detector = GdinoDetect(self.config)
        image = Image.fromarray(self.bgr[:, :, ::-1])
        result = detector.predict(image, "glass", -1, 2)
        self.assertEqual(len(result["boxes"]), 3)
        self.assertEqual(result["phrases"], ["", "", ""])
        with self.assertRaises(ValueError):
            detector.predict_with_metadata(image, "glass", -1, 0.25)
        _, metadata = detector.predict_with_metadata(image, "glass", 0.8, 0.65)
        self.assertEqual(metadata["original_size"], [3, 2])
        self.assertEqual(metadata["resized_size"], [3, 2])
        self.assertEqual(metadata["caption"], "glass.")

    def test_unexpected_graph_signature_fails_initialization(self):
        for change in ("missing_input", "missing_output"):
            with self.subTest(change=change):
                self.session = fake_session()
                if change == "missing_input":
                    self.session.get_inputs.return_value = self.session.get_inputs.return_value[:-1]
                else:
                    self.session.get_outputs.return_value = [SimpleNamespace(name="pred_logits")]
                with patch("app.domain.pipeline.a_gdinoDtct.create_session", return_value=self.session):
                    with self.assertRaises(ValueError):
                        GdinoDetect(self.config)

    def test_missing_model_fails_before_creating_runtime_session(self):
        (self.root / "model.onnx").unlink()
        with patch("app.domain.pipeline.a_gdinoDtct.create_session") as create:
            with self.assertRaises(FileNotFoundError):
                GdinoDetect(self.config)
            create.assert_not_called()

    def test_warmup_uses_model_dimensions_and_requested_run_count(self):
        detector = GdinoDetect(self.config)
        detector.warmup(runs=2)
        self.assertEqual(self.session.run.call_count, 2)
        feeds = self.session.run.call_args.args[1]
        np.testing.assert_array_equal(feeds["images"], np.zeros((1, 3, 2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
