"""Check CLIP food-filter logits without loading ONNX model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.domain.pipeline.b_clipdtct import ClipDetect


class ClipFoodFilterChecks(unittest.TestCase):
    def test_constructor_accepts_yaml_path(self):
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
                "  threads: 2\n",
                encoding="utf-8",
            )
            for value in (config_path, str(config_path)):
                with self.subTest(value_type=type(value).__name__), patch.object(ClipDetect, "init_model"):
                    detector = ClipDetect(value)
                self.assertEqual(detector.provider, "cpu")
                self.assertEqual(detector.threads, 2)
                self.assertEqual(Path(detector.model_path),
                                 Path(__file__).resolve().parents[1] / "config/model/clip/image.onnx")

    def test_batch_detect_uses_pil_crops_and_strict_raw_threshold(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.similarity = Mock(side_effect=[21.0, 21.25, 25.0])
        pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        image = Image.fromarray(pixels)
        boxes = np.array([[0.5, 0.5, 0.4, 0.5], [0.0, 0.5, 0.3, 0.5],
                          [0.8, 0.25, 0.2, 0.5]], dtype=np.float32)
        expected_rects = [[3, 2, 7, 6], [-1, 2, 1, 6], [7, 0, 9, 4]]
        rects, scores = detector.batch_detect(image, boxes, "鱼", 21)
        self.assertEqual(rects, expected_rects[1:])
        self.assertEqual(scores, [21.25, 25.0])
        for actual, rect in zip(detector.similarity.call_args_list, expected_rects):
            crop, caption = actual.args
            self.assertEqual(caption, "鱼")
            self.assertTrue(crop.flags.c_contiguous)
            np.testing.assert_array_equal(crop, np.asarray(image.crop(tuple(rect)))[:, :, ::-1])
        np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_empty_batch_returns_empty_lists_without_model_inference(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.similarity = Mock()
        for boxes in ([], np.empty((0, 4), dtype=np.float32)):
            with self.subTest(boxes=type(boxes).__name__):
                self.assertEqual(detector.batch_detect(Image.new("RGB", (3, 2)), boxes, "", 21), ([], []))
        detector.similarity.assert_not_called()

    def detector(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.ini_flag = True
        detector.logit_scale = 100.0
        detector.extract_image_feature = Mock(return_value=np.array([[0.6, 0.8]], dtype=np.float32))
        detector.extract_text_feature_with_model = Mock(return_value=(
            ["a new caption"], np.array([[1.0, 0.0]], dtype=np.float32),
        ))
        return detector

    def test_arbitrary_prompt_returns_scaled_cosine_instead_of_single_class_probability(self):
        detector = self.detector()
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertAlmostEqual(detector.similarity(image, "a new caption"), 60.0, places=5)
        detector.extract_text_feature_with_model.assert_called_once_with(["a new caption"])
        detector.logit_scale = 50.0
        self.assertAlmostEqual(detector.similarity(image, "a new caption"), 30.0, places=5)

    def test_raw_negative_similarity_remains_negative(self):
        detector = self.detector()
        detector.extract_image_feature.return_value = np.array([[-0.6, 0.8]], dtype=np.float32)
        self.assertAlmostEqual(detector.similarity(np.zeros((2, 2, 3), dtype=np.uint8), "food"),
                               -60.0, places=5)

    def test_legacy_empty_text_range_still_uses_all_texts_without_eager_cache(self):
        detector = self.detector()
        detector.feats_with_text_model = {}
        detector.topk = 2
        detector.threshold = 0
        detector.extract_text_feature_with_model.return_value = (
            ["food", "drink"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        result = detector.detect_with_text_model(np.zeros((2, 2, 3), dtype=np.uint8), [])
        self.assertEqual(len(result), 2)
        self.assertEqual({text for text, _ in result}, {"food", "drink"})
        detector.extract_text_feature_with_model.assert_called_once_with([])

    def test_missing_or_invalid_features_raise_instead_of_producing_empty_detections(self):
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

    def test_preprocessing_uses_declared_nchw_width_height_and_rgb_channels(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.shape = [1, 3, 2, 3]
        image = np.full((4, 5, 3), [10, 40, 240], dtype=np.uint8)
        tensor = detector.preprocess_image(image)
        expected_pixel = ((np.array([240, 40, 10], dtype=np.float32) / 255.0
                           - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32))
                          / np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32))
        self.assertEqual(tensor.shape, (1, 3, 2, 3))
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected_pixel)

    def test_food_similarity_crops_the_center_of_rectangular_proposals(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.shape = [1, 3, 20, 20]
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :, 2] = np.linspace(0, 255, 80, dtype=np.uint8)
        reference = detector.preprocess_image(image, reference=True)
        legacy = detector.preprocess_image(image)
        reference_red = (reference[0, 0] * 0.26862954 + 0.48145466) * 255
        legacy_red = (legacy[0, 0] * 0.26862954 + 0.48145466) * 255
        self.assertEqual(reference.shape, (1, 3, 20, 20))
        self.assertGreater(reference_red.min(), 50)
        self.assertLess(reference_red.max(), 205)
        self.assertLess(legacy_red.min(), 15)
        self.assertGreater(legacy_red.max(), 240)

    def test_dynamic_text_graph_uses_configured_context_length(self):
        detector = ClipDetect.__new__(ClipDetect)
        detector.text_shape = ["batch", "tokens"]
        detector.context_length = 6
        detector.text_input_name = "tokens"
        detector.text_output_name = "features"
        detector.text_sess = Mock()
        detector.text_sess.run.return_value = [np.array([[3.0, 4.0]], dtype=np.float32)]
        with patch("app.domain.pipeline.b_clipdtct.tokenizer.tokenize", return_value=[[101, 102, 0, 0, 0, 0]]) as encode:
            texts, features = detector.extract_text_feature_with_model(["food"])
        encode.assert_called_once_with(["food"], context_length=6)
        self.assertEqual(texts, ["food"])
        np.testing.assert_allclose(features, [[0.6, 0.8]], rtol=1e-6)
        self.assertEqual(detector.text_sess.run.call_args.args[1]["tokens"].dtype, np.int64)

    def test_failed_initialization_stops_before_loading_other_models(self):
        with patch.object(ClipDetect, "load_text_feature", return_value=False), \
                patch.object(ClipDetect, "load_image_model") as load_image, \
                patch.object(ClipDetect, "load_text_model") as load_text:
            detector = ClipDetect({"CLIP_CONFIG": {"use_gpu": False}})
        self.assertFalse(detector.ini_flag)
        load_image.assert_not_called()
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
