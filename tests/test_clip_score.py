"""Check the winning input text's softmax score without loading model weights."""

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.domain.pipeline.b_clipdtct import ClipDetect


class ClipScoreChecks(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((4, 5, 3), dtype=np.uint8)
        self.detector = ClipDetect.__new__(ClipDetect)
        self.detector.ini_flag = True
        self.detector.texts = {
            "fish": np.array([1.0, 0.0], dtype=np.float32),
            "vegetable": np.array([0.0, 1.0], dtype=np.float32),
        }
        self.detector.logit_scale = 2.0
        self.detector.threshold = 0.5
        self.detector.extract_image_feature = Mock(
            return_value=np.array([[0.6, 0.8]], dtype=np.float64))
        self.detector.extract_text_feature = Mock(
            return_value=(["new"], np.array([[0.6, 0.8]], dtype=np.float64)))
        logger = patch("app.domain.pipeline.b_clipdtct.logger")
        self.addCleanup(logger.stop)
        logger.start()

    def score(self, text):
        result = self.detector.detect(self.image, text)
        self.assertIs(type(result), float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
        return result

    def test_winning_key_returns_exact_probability_and_other_key_returns_zero(self):
        expected = 1.0 / (1.0 + math.exp(-0.4))
        self.assertAlmostEqual(self.score("vegetable"), expected, places=12)
        self.assertEqual(self.score("fish"), 0.0)
        self.detector.extract_text_feature.assert_not_called()

    def test_name_order_remains_aligned_with_feature_rows(self):
        self.detector.texts = {
            "last-alphabetically": np.array([0.0, 1.0]),
            "first-alphabetically": np.array([1.0, 0.0]),
        }
        self.assertAlmostEqual(self.score("last-alphabetically"),
                               1.0 / (1.0 + math.exp(-0.4)), places=12)
        self.assertEqual(self.score("first-alphabetically"), 0.0)

    def test_new_prompt_competes_without_changing_shared_text_features(self):
        original = self.detector.texts.copy()
        # Logits are [1.2, 1.6, 2.0]; the new candidate belongs in the denominator.
        expected = 1.0 / (math.exp(-0.8) + math.exp(-0.4) + 1.0)
        self.assertAlmostEqual(self.score("new"), expected, places=12)
        self.detector.extract_text_feature.assert_called_once_with(["new"])
        self.assertEqual(list(self.detector.texts), list(original))
        for key, value in original.items():
            self.assertIs(self.detector.texts[key], value)

    def test_new_prompt_can_lose_to_existing_candidate(self):
        self.detector.extract_text_feature.return_value = (
            ["new"], np.array([[-1.0, 0.0]], dtype=np.float32))
        self.assertEqual(self.score("new"), 0.0)
        self.assertNotIn("new", self.detector.texts)

    def test_one_dimensional_image_and_single_row_text_features_are_supported(self):
        self.detector.extract_image_feature.return_value = np.array([0.6, 0.8])
        self.detector.texts = {key: value[None, :] for key, value in self.detector.texts.items()}
        self.assertAlmostEqual(self.score("vegetable"),
                               1.0 / (1.0 + math.exp(-0.4)), places=12)

    def test_equal_scores_choose_first_candidate_in_dictionary_order(self):
        self.detector.texts = {"first": np.array([1.0, 0.0]),
                               "second": np.array([1.0, 0.0])}
        self.assertEqual(self.score("first"), 0.5)
        self.assertEqual(self.score("second"), 0.0)

    def test_threshold_and_topk_do_not_gate_the_highest_scoring_key(self):
        self.detector.threshold = 100
        self.assertFalse(hasattr(self.detector, "topk"))
        expected = 1.0 / (1.0 + math.exp(-0.4))
        self.assertAlmostEqual(self.score("vegetable"), expected, places=12)
        self.detector.topk = 0
        self.assertAlmostEqual(self.score("vegetable"), expected, places=12)

    def test_large_positive_scale_does_not_change_the_winner(self):
        self.detector.logit_scale = 1e308
        self.assertEqual(self.score("vegetable"), 1.0)
        self.assertEqual(self.score("fish"), 0.0)

    def test_optional_zero_and_negative_scale_follow_logit_ranking(self):
        cases = ((None, "vegetable", 1.0 / (1.0 + math.exp(-0.2))),
                 (0.0, "fish", 0.5),
                 (-2.0, "fish", 1.0 / (1.0 + math.exp(-0.4))))
        for scale, expected, probability in cases:
            with self.subTest(scale=scale):
                self.detector.logit_scale = scale
                self.assertAlmostEqual(self.score(expected), probability, places=12)
                other = "fish" if expected == "vegetable" else "vegetable"
                self.assertEqual(self.score(other), 0.0)

    def test_only_candidate_has_probability_one(self):
        self.detector.texts = {}
        self.assertEqual(self.score("new"), 1.0)
        self.assertEqual(self.detector.texts, {})

    def test_uninitialized_model_returns_zero_without_feature_extraction(self):
        self.detector.ini_flag = False
        self.assertEqual(self.score("vegetable"), 0.0)
        self.detector.extract_image_feature.assert_not_called()
        self.detector.extract_text_feature.assert_not_called()

    def test_invalid_prompt_returns_zero(self):
        for text in ("", "   ", None, 123):
            with self.subTest(text=text):
                self.assertEqual(self.score(text), 0.0)

    def test_invalid_image_features_return_zero(self):
        for feature in (None, np.array([]), np.zeros(2), np.array([np.nan, 1.0]),
                        np.array([np.inf, 1.0]), np.ones((2, 2)), np.ones(3)):
            with self.subTest(feature=feature):
                self.detector.extract_image_feature.return_value = feature
                self.assertEqual(self.score("vegetable"), 0.0)

    def test_invalid_cached_text_features_return_zero(self):
        for feature in (np.array([]), np.zeros(2), np.array([np.nan, 1.0]),
                        np.array([np.inf, 1.0]), np.ones(3), np.ones((2, 2))):
            with self.subTest(feature=feature):
                self.detector.texts["vegetable"] = feature
                self.assertEqual(self.score("vegetable"), 0.0)

    def test_failed_new_text_extraction_returns_zero(self):
        for features in (None, np.empty((0, 2)), np.array([[np.nan, 1.0]]),
                         np.zeros((1, 2)), np.ones((2, 2))):
            with self.subTest(features=features):
                self.detector.extract_text_feature.return_value = (["new"], features)
                self.assertEqual(self.score("new"), 0.0)
                self.assertNotIn("new", self.detector.texts)

    def test_image_and_text_inference_exceptions_return_zero(self):
        self.detector.extract_image_feature.side_effect = RuntimeError("image inference failed")
        self.assertEqual(self.score("vegetable"), 0.0)
        self.detector.extract_image_feature.side_effect = None
        self.detector.extract_text_feature.side_effect = RuntimeError("text inference failed")
        self.assertEqual(self.score("new"), 0.0)


if __name__ == "__main__":
    unittest.main()
