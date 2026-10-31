"""Check crop filtering and the CLIP batch calling contract without models."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.domain.pipeline.b_clipdtct import ClipDetect


class ClipBatchChecks(unittest.TestCase):
    def setUp(self):
        self.pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        self.image = Image.fromarray(self.pixels)
        self.detector = ClipDetect.__new__(ClipDetect)
        self.detector.ini_flag = True
        self.detector.detect = Mock(return_value=0.25)
        self.box = np.array([[0.5, 0.5, 0.4, 0.5]], dtype=np.float32)
        logger = patch("app.domain.pipeline.b_clipdtct.logger")
        self.addCleanup(logger.stop)
        logger.start()

    def batch(self, boxes, threshold=0.99, image=None):
        return self.detector.batch_detect(
            self.image if image is None else image, boxes, "fish", threshold)

    def test_crop_is_bgr_contiguous_and_caption_is_forwarded(self):
        rects, scores = self.batch(self.box)
        self.assertEqual(rects, [[3, 2, 7, 6]])
        self.assertEqual(scores, [0.25])
        self.detector.detect.assert_called_once()
        crop, caption = self.detector.detect.call_args.args
        np.testing.assert_array_equal(crop, self.pixels[2:6, 3:7, ::-1])
        self.assertTrue(crop.flags.c_contiguous)
        self.assertEqual(crop.dtype, np.uint8)
        self.assertEqual(caption, "fish")

    def test_positive_scores_below_threshold_remain_in_original_order(self):
        boxes = [[0.5, 0.5, 0.4, 0.5], [0.8, 0.25, 0.2, 0.5],
                 [0.2, 0.75, 0.2, 0.25]]
        self.detector.detect.side_effect = [0.125, 0.0, np.float32(0.25)]
        rects, scores = self.batch(boxes, threshold=1.0)
        self.assertEqual(rects, [[3, 2, 7, 6], [1, 5, 3, 7]])
        self.assertEqual(scores, [0.125, 0.25])
        self.assertEqual(self.detector.detect.call_count, 3)
        self.assertTrue(all(type(score) is float for score in scores))
        self.assertTrue(all(type(value) is int for rect in rects for value in rect))

    def test_clip_threshold_is_unused_even_when_not_numeric(self):
        for threshold in (0, 1, 100, None, object()):
            with self.subTest(threshold=threshold):
                self.assertEqual(self.batch(self.box, threshold=threshold),
                                 ([[3, 2, 7, 6]], [0.25]))

    def test_only_finite_positive_detect_results_are_retained(self):
        values = [0.0, -0.2, float("nan"), float("inf"), -float("inf"), 1e-10]
        self.detector.detect.side_effect = values
        rects, scores = self.batch(np.repeat(self.box, len(values), axis=0))
        self.assertEqual(rects, [[3, 2, 7, 6]])
        self.assertEqual(scores, [1e-10])

    def test_one_crop_failure_does_not_discard_later_successful_crops(self):
        self.detector.detect.side_effect = [RuntimeError("crop inference failed"), 0.25]
        rects, scores = self.batch([[0.5, 0.5, 0.4, 0.5], [0.8, 0.25, 0.2, 0.5]])
        self.assertEqual(rects, [[7, 0, 9, 4]])
        self.assertEqual(scores, [0.25])
        self.assertEqual(self.detector.detect.call_count, 2)

    def test_left_padding_preserves_original_unclipped_coordinates(self):
        self.assertEqual(self.batch([[0.0, 0.5, 0.4, 0.5]]),
                         ([[-2, 2, 2, 6]], [0.25]))
        crop = self.detector.detect.call_args.args[0]
        self.assertEqual(crop.shape, (4, 4, 3))
        np.testing.assert_array_equal(crop[:, :2], np.zeros((4, 2, 3), dtype=np.uint8))
        np.testing.assert_array_equal(crop[:, 2:], self.pixels[2:6, :2, ::-1])

    def test_right_and_bottom_padding_preserve_original_coordinates(self):
        self.assertEqual(self.batch([[1.0, 0.875, 0.4, 0.5]]),
                         ([[8, 5, 12, 9]], [0.25]))
        crop = self.detector.detect.call_args.args[0]
        self.assertEqual(crop.shape, (4, 4, 3))
        np.testing.assert_array_equal(crop[:3, :2], self.pixels[5:8, 8:10, ::-1])
        self.assertFalse(crop[:, 2:].any())
        self.assertFalse(crop[3].any())

    def test_coordinates_use_float32_and_truncate_toward_zero(self):
        rects, _ = self.batch([[-0.025, 0.5, 0.3, 0.5]])
        self.assertEqual(rects, [[-1, 2, 1, 6]])
        self.assertEqual(self.detector.detect.call_args.args[0].shape, (4, 2, 3))

    def test_single_box_vector_is_accepted(self):
        self.assertEqual(self.batch(self.box[0]), ([[3, 2, 7, 6]], [0.25]))

    def test_invalid_rows_are_skipped_without_losing_valid_rows(self):
        boxes = [[0.5, 0.5, 0.4, 0.5], [np.nan, 0.5, 0.4, 0.5],
                 [0.5, np.inf, 0.4, 0.5], [0.5, 0.5, 0.0, 0.5],
                 [0.5, 0.5, -0.4, 0.5], [0.5, 0.5, 0.4, -0.5],
                 [2.0, 0.5, 0.4, 0.5], [0.5, -2.0, 0.4, 0.5],
                 [0.55, 0.5, 0.01, 0.5], [0.8, 0.25, 0.2, 0.5]]
        rects, scores = self.batch(boxes)
        self.assertEqual(rects, [[3, 2, 7, 6], [7, 0, 9, 4]])
        self.assertEqual(scores, [0.25, 0.25])
        self.assertEqual(self.detector.detect.call_count, 2)

    def test_empty_and_all_invalid_boxes_return_empty_without_inference(self):
        for boxes in ([], np.empty((0, 4)), [[np.nan, 0.5, 0.4, 0.5]],
                      [[2.0, 0.5, 0.4, 0.5]]):
            with self.subTest(boxes=boxes):
                self.assertEqual(self.batch(boxes), ([], []))
        self.detector.detect.assert_not_called()

    def test_malformed_box_shapes_return_empty(self):
        for boxes in ([[1, 2, 3]], np.ones((2, 5)), np.ones((2, 2, 4)), "bad"):
            with self.subTest(boxes=boxes):
                self.assertEqual(self.batch(boxes), ([], []))
        self.detector.detect.assert_not_called()

    def test_non_pil_image_returns_empty(self):
        self.assertEqual(self.batch(self.box, image=self.pixels), ([], []))
        self.detector.detect.assert_not_called()

    def test_uninitialized_model_returns_empty_without_inference(self):
        self.detector.ini_flag = False
        self.assertEqual(self.batch(self.box), ([], []))
        self.detector.detect.assert_not_called()

    def test_original_image_and_box_array_remain_unchanged(self):
        boxes = np.array([[-0.025, 0.5, 0.3, 0.5], [0.8, 0.25, 0.2, 0.5]])
        before = boxes.copy()
        pixels_before = self.image.tobytes()
        self.batch(boxes)
        np.testing.assert_array_equal(boxes, before)
        self.assertEqual(self.image.tobytes(), pixels_before)


if __name__ == "__main__":
    unittest.main()
