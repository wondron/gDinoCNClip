"""Check CLIP candidate selection, scoring and initialization without loading model weights."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.domain.pipeline.b_clipdtct import ClipDetect


class ClipScoreChecks(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((4, 5, 3), dtype=np.uint8)
        self.detector = ClipDetect.__new__(ClipDetect)
        self.detector.ini_flag = True
        self.detector.food_range, self.detector.common_range, self.detector.recipe_range = ["fish"], ["cat"], ["dish"]
        self.detector.food_feat_dict = {"fish": np.array([1.0, 0.0], dtype=np.float32)}
        self.detector.common_feat_dict = {"cat": np.array([0.0, 1.0], dtype=np.float32)}
        self.detector.recipe_feat_dict = {"dish": np.array([0.6, 0.8], dtype=np.float32)}
        # Cosine scores against this image: fish 0.6, cat 0.8, dish 1.0.
        self.detector.extract_image_feature = Mock(return_value=np.array([[0.6, 0.8]], dtype=np.float32))
        self.detector.extract_text_feature = Mock(return_value=(["new"], np.array([[0.6, 0.8]], dtype=np.float32)))
        for target in ("app.domain.pipeline.b_clipdtct.logger", "app.domain.pipeline.b_clipdtct.show_text"):
            patcher = patch(target)
            self.addCleanup(patcher.stop)
            patcher.start()

    def test_food_name_competes_with_food_and_common_ranges(self):
        self.assertAlmostEqual(self.detector.detect(self.image, "cat"), 0.8, places=6)
        self.assertEqual(self.detector.detect(self.image, "fish"), 0.0)
        self.assertEqual(self.detector.text_candidates("fish")[0], ["fish", "cat"])
        self.detector.extract_text_feature.assert_not_called()

    def test_recipe_name_competes_only_with_common_range(self):
        self.assertEqual(self.detector.text_candidates("dish")[0], ["cat", "dish"])
        # fish (1.0) would beat dish (0.6) here, but recipes are not compared with food names.
        self.detector.extract_image_feature.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(self.detector.detect(self.image, "dish"), 0.6, places=6)
        self.detector.extract_text_feature.assert_not_called()

    def test_unknown_name_is_encoded_once_per_batch_and_not_cached(self):
        image = Image.new("RGB", (20, 20))
        boxes = [[0.25, 0.25, 0.4, 0.4], [0.75, 0.75, 0.4, 0.4], [0.5, 0.5, 0.2, 0.2]]
        rects, confidences = self.detector.batch_detect(image, boxes, "new", 0.5)
        self.assertEqual(len(rects), 3)
        self.assertEqual(len(confidences), 3)
        self.detector.extract_text_feature.assert_called_once_with(["new"])
        self.assertEqual(self.detector.extract_image_feature.call_count, 3)
        for features in (self.detector.food_feat_dict, self.detector.common_feat_dict, self.detector.recipe_feat_dict):
            self.assertNotIn("new", features)

    def test_unknown_name_can_lose_to_a_known_candidate(self):
        self.detector.extract_text_feature.return_value = (["new"], np.array([[-1.0, 0.0]], dtype=np.float32))
        self.assertEqual(self.detector.detect(self.image, "new"), 0.0)

    def test_failed_text_extraction_returns_nothing(self):
        self.detector.extract_text_feature.return_value = ([], np.empty((0, 0), dtype=np.float32))
        self.assertEqual(self.detector.detect(self.image, "new"), 0.0)
        self.assertEqual(self.detector.batch_detect(Image.new("RGB", (20, 20)), [[0.5, 0.5, 0.2, 0.2]], "new", 0.5),
                         ([], []))
        self.detector.extract_image_feature.assert_not_called()

    def test_uninitialized_model_returns_zero_without_feature_extraction(self):
        self.detector.ini_flag = False
        self.assertEqual(self.detector.detect(self.image, "cat"), 0.0)
        self.assertEqual(self.detector.batch_detect(Image.new("RGB", (20, 20)), [[0.5, 0.5, 0.2, 0.2]], "cat", 0.5),
                         ([], []))
        self.detector.extract_image_feature.assert_not_called()
        self.detector.extract_text_feature.assert_not_called()

    def test_invalid_prompt_returns_zero(self):
        for text in ("", "   ", None, 123):
            with self.subTest(text=text):
                self.assertEqual(self.detector.detect(self.image, text), 0.0)
        self.detector.extract_image_feature.assert_not_called()

    def test_initialization_sets_the_flag_before_the_self_test_and_resets_it_on_failure(self):
        detector = self.detector
        features = {"fish": np.array([1.0, 0.0]), "皮皮虾": np.array([0.6, 0.8]),
                    "cat": np.array([0.0, 1.0]), "dish": np.array([1.0, 0.0])}
        detector.food_range = ["fish", "皮皮虾"]
        detector.test_image_path = "self-test.jpg"
        detector.load_image_model = Mock(return_value=True)
        detector.load_text_model = Mock(return_value=True)
        detector.load_text_feature = Mock(side_effect=lambda names: {name: features[name] for name in names})
        detector.sess = Mock()
        with patch("app.domain.pipeline.b_clipdtct.cv2.imread", return_value=self.image):
            detector.init_model()
            self.assertTrue(detector.ini_flag)
            # cat outscores 皮皮虾: the self-test fails.
            detector.extract_image_feature.return_value = np.array([[0.0, 1.0]], dtype=np.float32)
            detector.init_model()
            self.assertFalse(detector.ini_flag)
            detector.extract_image_feature.return_value = np.array([[0.6, 0.8]], dtype=np.float32)
            detector.load_text_feature = Mock(return_value={})
            detector.init_model()
            self.assertFalse(detector.ini_flag)
        detector.load_text_feature = Mock(side_effect=lambda names: {name: features[name] for name in names})
        with patch("app.domain.pipeline.b_clipdtct.cv2.imread", return_value=None):
            detector.init_model()
            self.assertFalse(detector.ini_flag)


if __name__ == "__main__":
    unittest.main()
