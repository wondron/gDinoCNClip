"""Check direct text-model inference and bounded CLIP scores without weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.domain.pipeline.b_clipdtct import ClipDetect


class ClipFoodFilterChecks(unittest.TestCase):
    def detector(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.ini_flag = True
        detector.threshold = 0.5
        detector.topk = 2
        detector.extract_image_feature = Mock(return_value=np.array([[0.6, 0.8]], dtype=np.float32))
        detector.extract_text_feature_with_model = Mock(return_value=(
            ["a new caption"], np.array([[1.0, 0.0]], dtype=np.float32),
        ))
        return detector

    def test_constructor_accepts_yaml_without_feature_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                "GROUNDINGDINO_CONFIG:\n"
                "  model_path: config/model/groundingdino/model.onnx\n"
                "  tokenizer_path: config/model/groundingdino/tokenizer\n"
                "CLIP_CONFIG:\n"
                "  model_path: config/model/clip/image.onnx\n"
                "  text_model_path: config/model/clip/text.onnx\n"
                "  use_gpu: false\n"
                "  threads: 2\n"
                "  threshold: 0.5\n",
                encoding="utf-8",
            )
            for source in (config_path, str(config_path)):
                with self.subTest(source=type(source).__name__), patch.object(ClipDetect, "init_model"):
                    detector = ClipDetect(source)
                self.assertEqual(detector.provider, "cpu")
                self.assertEqual(detector.threads, 2)
                self.assertEqual(detector.threshold, 0.5)
                self.assertEqual(Path(detector.model_path),
                                 Path(__file__).resolve().parents[1] / "config/model/clip/image.onnx")

    def test_initialization_only_loads_image_and_text_models(self):
        with patch.object(ClipDetect, "load_image_model", return_value=True) as image, \
                patch.object(ClipDetect, "load_text_model", return_value=True) as text, \
                patch("builtins.open", side_effect=AssertionError("No feature files may be read")):
            detector = ClipDetect({"CLIP_CONFIG": {"use_gpu": False}})
        self.assertTrue(detector.ini_flag)
        image.assert_called_once()
        text.assert_called_once()

    def test_single_prompt_returns_mapped_cosine(self):
        detector = self.detector()
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertAlmostEqual(detector.similarity(image, "a new caption"), 0.8, places=6)
        detector.extract_text_feature_with_model.assert_called_once_with(["a new caption"])

    def test_mapping_handles_opposite_orthogonal_and_aligned_features(self):
        detector = self.detector()
        text = np.array([[1.0, 0.0]])
        with np.errstate(over="raise", invalid="raise"):
            for image, expected in (([[-1.0, 0.0]], 0.0), ([[0.0, 1.0]], 0.5),
                                    ([[1.0, 0.0]], 1.0), ([[1.0000001, 0.0]], 1.0),
                                    ([[-1.0000001, 0.0]], 0.0)):
                with self.subTest(image=image):
                    self.assertEqual(float(detector._similarity_scores(image, text)[0]), expected)

    def test_multiple_candidate_scores_are_independent_and_rank_labels(self):
        detector = self.detector()
        detector.threshold = 0.0
        detector.extract_text_feature_with_model.return_value = (
            ["fish", "shrimp"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        result = detector.detect(image, ["fish", "shrimp"])
        self.assertEqual([name for name, _ in result], ["shrimp", "fish"])
        np.testing.assert_allclose([score for _, score in result], [0.9, 0.8], rtol=1e-6)
        self.assertEqual(result, detector.detect_with_text_model(image, ["fish", "shrimp"]))
        detector.extract_text_feature_with_model.return_value = (
            ["fish"], np.array([[1.0, 0.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(result[1][1], detector.similarity(image, "fish"), places=6)

    def test_batch_encodes_prompt_once_and_filters_each_crop_independently(self):
        detector = self.detector()
        detector.extract_image_feature.side_effect = [
            np.array([[0.0, 1.0]]), np.array([[0.6, 0.8]]), np.array([[-0.6, 0.8]]),
        ]
        pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        image = Image.fromarray(pixels)
        boxes = np.array([[0.5, 0.5, 0.4, 0.5], [0.0, 0.5, 0.3, 0.5],
                          [0.8, 0.25, 0.2, 0.5]], dtype=np.float32)
        expected_rects = [[3, 2, 7, 6], [-1, 2, 1, 6], [7, 0, 9, 4]]
        rects, scores = detector.batch_detect(image, boxes, "fish", 0.5)
        self.assertEqual(rects, [expected_rects[1]])
        self.assertAlmostEqual(scores[0], 0.8, places=6)
        detector.extract_text_feature_with_model.assert_called_once_with(["fish"])
        for call, rect in zip(detector.extract_image_feature.call_args_list, expected_rects):
            crop = call.args[0]
            self.assertEqual(call.kwargs, {"reference": True})
            self.assertTrue(crop.flags.c_contiguous)
            np.testing.assert_array_equal(crop, np.asarray(image.crop(tuple(rect)))[:, :, ::-1])
        np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_empty_batch_returns_empty_lists_without_inference(self):
        detector = self.detector()
        for boxes in ([], np.empty((0, 4), dtype=np.float32)):
            self.assertEqual(detector.batch_detect(Image.new("RGB", (3, 2)), boxes, "", 0.5), ([], []))
        detector.extract_image_feature.assert_not_called()
        detector.extract_text_feature_with_model.assert_not_called()

    def test_batch_rejects_old_raw_logit_thresholds(self):
        detector = self.detector()
        with self.assertRaises(ValueError):
            detector.batch_detect(Image.new("RGB", (3, 2)), [[0.5, 0.5, 1, 1]], "fish", 21)
        detector.extract_image_feature.assert_not_called()

    def test_missing_or_invalid_features_raise(self):
        for features in (None, np.empty((0, 2)), np.zeros((1, 2)), np.array([[np.nan, 1.0]])):
            with self.subTest(features=features):
                detector = self.detector()
                detector.extract_image_feature.return_value = features
                with self.assertRaises(RuntimeError):
                    detector.similarity(np.zeros((2, 2, 3), dtype=np.uint8), "food")
        detector = self.detector()
        with self.assertRaises(ValueError):
            detector.similarity(np.zeros((2, 2, 3), dtype=np.uint8), " ")
        detector.ini_flag = False
        with self.assertRaises(RuntimeError):
            detector.similarity(np.zeros((2, 2, 3), dtype=np.uint8), "food")

    def test_text_model_requires_explicit_prompts(self):
        detector = ClipDetect.__new__(ClipDetect)
        for texts in (None, [], [""]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                detector.extract_text_feature_with_model(texts)

    def test_preprocessing_uses_declared_nchw_size_and_rgb(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.shape = [1, 3, 2, 3]
        image = np.full((4, 5, 3), [10, 40, 240], dtype=np.uint8)
        tensor = detector.preprocess_image(image)
        expected = ((np.array([240, 40, 10], dtype=np.float32) / 255.0
                     - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32))
                    / np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32))
        self.assertEqual(tensor.shape, (1, 3, 2, 3))
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected)

    def test_food_preprocessing_crops_rectangular_proposals_at_center(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.shape = [1, 3, 20, 20]
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :, 2] = np.linspace(0, 255, 80, dtype=np.uint8)
        reference = detector.preprocess_image(image, reference=True)
        legacy = detector.preprocess_image(image)
        reference_red = (reference[0, 0] * 0.26862954 + 0.48145466) * 255
        legacy_red = (legacy[0, 0] * 0.26862954 + 0.48145466) * 255
        self.assertGreater(reference_red.min(), 50)
        self.assertLess(reference_red.max(), 205)
        self.assertLess(legacy_red.min(), 15)
        self.assertGreater(legacy_red.max(), 240)

    def text_detector(self, shape):
        detector = ClipDetect.__new__(ClipDetect)
        detector.text_shape = shape
        detector.context_length = 6
        detector.text_input_name = "tokens"
        detector.text_output_name = "features"
        detector.text_sess = Mock()
        return detector

    def test_dynamic_text_graph_uses_configured_context_length(self):
        detector = self.text_detector(["batch", "tokens"])
        detector.text_sess.run.return_value = [np.array([[3.0, 4.0]], dtype=np.float32)]
        with patch("app.domain.pipeline.b_clipdtct.tokenizer.tokenize",
                   return_value=[[101, 102, 0, 0, 0, 0]]) as encode:
            texts, features = detector.extract_text_feature_with_model(["food"])
        encode.assert_called_once_with(["food"], context_length=6)
        self.assertEqual(texts, ["food"])
        np.testing.assert_allclose(features, [[0.6, 0.8]], rtol=1e-6)
        self.assertEqual(detector.text_sess.run.call_args.args[1]["tokens"].dtype, np.int64)

    def test_fixed_batch_one_graph_encodes_each_supplied_text(self):
        detector = self.text_detector([1, 6])
        detector.text_sess.run.side_effect = [
            [np.array([[3.0, 4.0]], dtype=np.float32)],
            [np.array([[0.0, 2.0]], dtype=np.float32)],
        ]
        with patch("app.domain.pipeline.b_clipdtct.tokenizer.tokenize",
                   return_value=[[101, 102, 0, 0, 0, 0], [101, 202, 102, 0, 0, 0]]):
            texts, features = detector.extract_text_feature_with_model(["fish", "shrimp"])
        self.assertEqual(texts, ["fish", "shrimp"])
        np.testing.assert_allclose(features, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)
        self.assertEqual(detector.text_sess.run.call_count, 2)

    def test_failed_initialization_stops_before_loading_text_model(self):
        with patch.object(ClipDetect, "load_image_model", return_value=False), \
                patch.object(ClipDetect, "load_text_model") as load_text:
            detector = ClipDetect({"CLIP_CONFIG": {"use_gpu": False}})
        self.assertFalse(detector.ini_flag)
        load_text.assert_not_called()

    def test_model_loading_honors_yaml_provider_device_and_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.touch()
            session = Mock()
            session.get_inputs.return_value = [SimpleNamespace(name="images", shape=[1, 3, 224, 224])]
            session.get_outputs.return_value = [SimpleNamespace(name="features")]
            detector = ClipDetect.__new__(ClipDetect)
            detector._load_config({"CLIP_CONFIG": {"model_path": str(model_path), "use_gpu": False,
                                                  "threads": 2}, "TOTAL_CONFIG": {"device_id": 3}})
            with patch("app.domain.pipeline.b_clipdtct.create_session", return_value=session) as create:
                self.assertTrue(detector.load_image_model())
            create.assert_called_once_with(str(model_path), "cpu", 2, device_id=3)


if __name__ == "__main__":
    unittest.main()
