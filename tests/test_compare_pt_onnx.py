"""Pairing and tolerance rules of the PyTorch/ONNX consistency tool, without model weights."""

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compare_pt_onnx import compare_detections


def result(*detections):
    return {"boxes": [d[0] for d in detections], "logits": [d[1] for d in detections],
            "phrases": [d[2] for d in detections]}


FISH = ([0.5, 0.5, 0.2, 0.4], 0.62, "fish")
CRAB = ([0.2, 0.3, 0.1, 0.1], 0.48, "crab")


class CompareDetectionsTest(unittest.TestCase):
    def compare(self, pt, onnx, box_threshold=0.35):
        return compare_detections(pt, onnx, box_threshold, (1000, 500))

    def test_reordered_identical_detections_pass(self):
        comparison = self.compare(result(FISH, CRAB), result(CRAB, FISH))
        self.assertEqual(comparison["status"], "pass")
        self.assertEqual({(p["pt_index"], p["onnx_index"]) for p in comparison["pairs"]}, {(0, 1), (1, 0)})

    def test_small_differences_within_tolerance_pass_and_report_pixels(self):
        shifted = ([0.501, 0.5, 0.2, 0.4], 0.621, "fish")
        comparison = self.compare(result(FISH), result(shifted))
        self.assertEqual(comparison["status"], "pass")
        self.assertAlmostEqual(comparison["pairs"][0]["box_diff_px"], 1.0)
        self.assertAlmostEqual(comparison["pairs"][0]["score_diff"], 0.001)

    def test_unpaired_detection_at_threshold_is_borderline(self):
        edge = ([0.8, 0.8, 0.1, 0.1], 0.352, "plate")
        comparison = self.compare(result(FISH, edge), result(FISH))
        self.assertEqual(comparison["status"], "borderline")
        self.assertEqual(comparison["only_pt"][0]["phrase"], "plate")

    def test_unpaired_confident_detection_fails(self):
        comparison = self.compare(result(FISH), result(FISH, CRAB))
        self.assertEqual(comparison["status"], "fail")
        self.assertIn("only ONNX: 'crab'@0.4800", comparison["failures"])

    def test_phrase_or_score_mismatch_fails(self):
        renamed = (FISH[0], FISH[1], "fish crab")
        self.assertEqual(self.compare(result(FISH), result(renamed))["status"], "fail")
        rescored = (FISH[0], FISH[1] + 0.01, FISH[2])
        self.assertEqual(self.compare(result(FISH), result(rescored))["status"], "fail")

    def test_no_detections_on_either_side_pass(self):
        comparison = self.compare(result(), result())
        self.assertEqual((comparison["status"], comparison["pt_count"], comparison["onnx_count"]),
                         ("pass", 0, 0))


if __name__ == "__main__":
    unittest.main()
