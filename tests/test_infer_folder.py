"""Folder inference behavior checks without loading ONNX weights or CUDA."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer_folder import discover_images, run_batch


class FakePredictor:
    """Decode real image bytes while replacing the expensive model execution."""

    def __init__(self):
        self.calls = []
        self.model_path = "test-model.onnx"
        self.tokenizer_path = "test-tokenizer"
        self.provider = "cpu"
        self.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        self.model_settings = SimpleNamespace(
            model_path=self.model_path, tokenizer_path=self.tokenizer_path,
            provider=self.provider, short_side=800, max_size=1333, max_text_len=256,
        )

    def predict_with_metadata(self, image_path, caption, box_threshold, text_threshold):
        path = Path(image_path)
        self.calls.append((path.name, caption, box_threshold, text_threshold))
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        empty = "空检测" in path.name
        result = {
            "boxes": [] if empty else [[0.25, 0.5, 0.2, 0.3]],
            "phrases": [] if empty else ["glass"],
            "logits": [] if empty else [0.9],
        }
        inference_ms = 7.0 if empty else 5.0
        return result, {
            "caption": caption,
            "original_size": [width, height],
            "resized_size": [width, height],
            "timings_ms": {
                "queue_ms": 0.0, "preprocess_ms": 2.0,
                "inference_ms": inference_ms, "postprocess_ms": 0.5,
                "total_ms": inference_ms + 2.5,
            },
        }


class FolderInferenceChecks(unittest.TestCase):
    def test_discovery_filters_extensions_and_handles_recursive_case_variants(self):
        with tempfile.TemporaryDirectory(prefix="folder_discovery_") as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            for relative in ("A.JPG", "b.png", "03_大写.JPEG", "nested/C.WEBP", "ignore.txt"):
                (root / relative).write_bytes(b"discovery only")
            flat = [path.relative_to(root).as_posix() for path in discover_images(root)]
            recursive = [path.relative_to(root).as_posix() for path in discover_images(root, recursive=True)]
            self.assertEqual(flat, ["03_大写.JPEG", "A.JPG", "b.png"])
            self.assertEqual(recursive, ["03_大写.JPEG", "A.JPG", "b.png", "nested/C.WEBP"])

    def test_batch_continues_after_corruption_and_writes_detections_and_timings(self):
        with tempfile.TemporaryDirectory(prefix="folder_batch_") as directory:
            root = Path(directory)
            images = root / "图片目录"
            images.mkdir()
            Image.new("RGB", (16, 12), "white").save(images / "01_杯子.JPG")
            (images / "02_损坏.png").write_bytes(b"not an image")
            Image.new("RGB", (20, 10), "black").save(images / "03_空检测.png")
            predictor = FakePredictor()
            destination = root / "结果" / "batch.json"
            report = run_batch(images, "glass .", destination, predictor,
                               box_threshold=0.4, text_threshold=0.3, model_load_ms=123.0)
            summary = report["summary"]
            self.assertEqual(summary["total_images"], 3)
            self.assertEqual(summary["succeeded"], 2)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["total_detections"], 1)
            self.assertAlmostEqual(summary["mean_inference_ms"], 6.0)
            self.assertEqual(len(predictor.calls), 3)
            self.assertTrue(all(call[1:] == ("glass .", 0.4, 0.3) for call in predictor.calls))
            records = {record["image"]: record for record in report["results"]}
            positive = records["01_杯子.JPG"]
            self.assertEqual(positive["status"], "ok")
            self.assertEqual(positive["boxes"], [[0.25, 0.5, 0.2, 0.3]])
            self.assertEqual(positive["phrases"], ["glass"])
            self.assertEqual(positive["logits"], [0.9])
            self.assertEqual(positive["timings_ms"]["inference_ms"], 5.0)
            damaged = records["02_损坏.png"]
            self.assertEqual(damaged["status"], "error")
            self.assertEqual(damaged["error"]["type"], "UnidentifiedImageError")
            self.assertGreaterEqual(damaged["timings_ms"]["total_ms"], 0)
            empty = records["03_空检测.png"]
            self.assertEqual(empty["status"], "ok")
            self.assertEqual((empty["boxes"], empty["phrases"], empty["logits"]), ([], [], []))
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            self.assertIn("01_杯子.JPG", destination.read_text(encoding="utf-8"))

    def test_empty_missing_and_file_inputs_fail_before_inference(self):
        with tempfile.TemporaryDirectory(prefix="folder_invalid_") as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            file_path = root / "image.jpg"
            Image.new("RGB", (2, 2)).save(file_path)
            predictor = FakePredictor()
            for invalid in (empty, root / "missing", file_path):
                with self.subTest(path=invalid):
                    with self.assertRaises((ValueError, FileNotFoundError, NotADirectoryError)):
                        run_batch(invalid, "glass .", root / "result.json", predictor)
            self.assertFalse(predictor.calls)

    def test_production_script_has_no_imports_from_other_project_modules(self):
        tree = ast.parse((ROOT / "scripts/infer_folder.py").read_text(encoding="utf-8"))
        local_roots = {"app", "scripts", "tools", "tests", "main"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".", 1)[0], local_roots)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "Standalone script must not use relative imports")
                self.assertNotIn((node.module or "").split(".", 1)[0], local_roots)


if __name__ == "__main__":
    unittest.main(verbosity=2)
