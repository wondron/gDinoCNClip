"""Verify Detector's current GDINO and CLIP calling contracts."""

import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.domain.food_service import Detector


class FoodServiceChecks(unittest.TestCase):
    def setUp(self):
        self.pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        self.image = Image.fromarray(self.pixels)
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()
        self.boxes = np.array([[0.5, 0.5, 0.4, 0.5], [0.0, 0.5, 0.3, 0.5],
                               [0.8, 0.25, 0.2, 0.5]], dtype=np.float32)
        self.gdino = Mock()
        self.gdino.detect.return_value = (
            self.boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32),
            ["food bowl", "food", "Food"],
        )
        self.clip = Mock()
        self.rects = [[3, 2, 7, 6], [-1, 2, 1, 6], [7, 0, 9, 4]]

    def detector(self):
        with patch("app.domain.food_service.GdinoDetect", return_value=self.gdino), \
                patch("app.domain.food_service.ClipDetect", return_value=self.clip):
            return Detector("config/param/config.yaml")

    def test_gdino_only_uses_numpy_phrase_selection_and_fixed_confidence(self):
        for clip_threshold in (0, -1):
            with self.subTest(clip_threshold=clip_threshold):
                detector = self.detector()
                rects, confidence = detector.detect(
                    self.image_bytes, "food", 0.3, 0.4, "unused", clip_threshold,
                )
                self.assertEqual(rects, self.rects[:2])
                self.assertEqual(confidence, [0.6, 0.6])
                self.clip.batch_detect.assert_not_called()
                image, caption, box_threshold, text_threshold = self.gdino.detect.call_args.args
                self.assertEqual((caption, box_threshold, text_threshold), ("food", 0.3, 0.4))
                self.assertEqual(image.mode, "RGB")
                np.testing.assert_array_equal(np.asarray(image), self.pixels)

    def test_positive_clip_threshold_delegates_full_box_array_and_mapped_threshold(self):
        detector = self.detector()
        self.clip.batch_detect.return_value = ([self.rects[1]], [0.75])
        result = detector.detect(self.image_bytes, "food", 0.3, 0.4, "三文鱼", 0.5)
        self.assertEqual(result, ([self.rects[1]], [0.75]))
        self.clip.batch_detect.assert_called_once()
        image, boxes, caption, threshold = self.clip.batch_detect.call_args.args
        np.testing.assert_array_equal(np.asarray(image), self.pixels)
        np.testing.assert_array_equal(boxes, self.boxes)
        self.assertEqual((caption, threshold), ("三文鱼", 0.5))

    def test_empty_numpy_boxes_support_both_detection_branches(self):
        self.gdino.detect.return_value = (np.empty((0, 4), dtype=np.float32),
                                         np.empty((0,), dtype=np.float32), [])
        self.clip.batch_detect.return_value = ([], [])
        detector = self.detector()
        self.assertEqual(detector.detect(self.image_bytes, "food", 0.3, 0.4, "鱼", 0), ([], []))
        self.clip.batch_detect.assert_not_called()
        self.assertEqual(detector.detect(self.image_bytes, "food", 0.3, 0.4, "鱼", 0.5), ([], []))
        self.assertEqual(self.clip.batch_detect.call_args.args[1].shape, (0, 4))

    def test_configuration_path_initializes_models_once_for_multiple_requests(self):
        self.clip.batch_detect.return_value = (self.rects, [0.8] * 3)
        with patch("app.domain.food_service.GdinoDetect", return_value=self.gdino) as create_dino, \
                patch("app.domain.food_service.ClipDetect", return_value=self.clip) as create_clip:
            detector = Detector("config/param/config.yaml")
            first = detector.detect(self.image_bytes, "food", 0.3, 0.4, "鱼", 0.5)
            second = detector.detect(self.image_bytes, "food", 0.3, 0.4, "鱼", 0.7)
        self.assertEqual(first, (self.rects, [0.8] * 3))
        self.assertEqual(second, first)
        create_dino.assert_called_once_with("config/param/config.yaml")
        create_clip.assert_called_once_with("config/param/config.yaml")

    def test_detection_does_not_draw_or_save_images(self):
        detector = self.detector()
        self.clip.batch_detect.return_value = (self.rects, [0.8] * 3)
        with patch.object(detector, "draw_result") as draw_result, \
                patch.object(Image.Image, "save") as save:
            for clip_threshold in (0, 0.5):
                with self.subTest(clip_threshold=clip_threshold):
                    rects, confidence = detector.detect(
                        self.image_bytes, "food", 0.3, 0.4, "鱼", clip_threshold,
                    )
                    self.assertTrue(rects)
                    self.assertEqual(len(rects), len(confidence))
            draw_result.assert_not_called()
            save.assert_not_called()

    def test_draw_result_returns_a_new_image_without_modifying_or_saving_source(self):
        detector = self.detector()
        for source in (self.image, self.image_bytes):
            with self.subTest(source_type=type(source).__name__), \
                    patch.object(Image.Image, "save") as save:
                result = detector.draw_result(source, [[3, 2, 7, 6]], [0.8])
                self.assertIsInstance(result, Image.Image)
                self.assertIsNot(result, self.image)
                self.assertEqual((result.mode, result.size), ("RGB", self.image.size))
                self.assertFalse(np.array_equal(np.asarray(result), self.pixels))
                np.testing.assert_array_equal(np.asarray(self.image), self.pixels)
                save.assert_not_called()

    def test_draw_result_accepts_empty_boxes_without_saving(self):
        detector = self.detector()
        with patch.object(Image.Image, "save") as save:
            result = detector.draw_result(self.image_bytes, [], [])
            self.assertIsInstance(result, Image.Image)
            np.testing.assert_array_equal(np.asarray(result), self.pixels)
            save.assert_not_called()

    def test_bad_images_and_inference_failures_return_reference_empty_results(self):
        detector = self.detector()
        with patch("builtins.print"):
            self.assertEqual(detector.detect(b"not an image", "food", 0.3, 0.4, "鱼", 0.5),
                             ([], []))
        self.gdino.detect.assert_not_called()
        self.clip.batch_detect.assert_not_called()
        self.gdino.detect.side_effect = RuntimeError("inference failed")
        with patch("builtins.print"):
            self.assertEqual(detector.detect(self.image_bytes, "food", 0.3, 0.4, "鱼", 0),
                             ([], []))


if __name__ == "__main__":
    unittest.main()
