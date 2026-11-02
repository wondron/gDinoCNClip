"""Check the food endpoint's multipart and reference response contracts."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.adapter.http.endpoints.predict import router
from app.domain.food_service import Detector


FORM = {"dino_cap": "food", "dino_box": "0.3", "dino_text": "0.4",
        "clip_cap": "鱼", "clip_thre": "0.5"}


class FoodApiChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.result_dir = self.directory / "output" / "save_res"
        self.service = Mock()
        self.service.detect.return_value = ([[1, 2, 7, 8]], [0.75])
        self.service.draw_result.side_effect = Detector.__new__(Detector).draw_result
        app = FastAPI()
        app.include_router(router)
        app.state.runtime = SimpleNamespace(
            food_service=self.service, result_dir=self.result_dir,
            settings=SimpleNamespace(paths=SimpleNamespace(jpeg_quality=95)),
        )
        self.app = app
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        buffer = io.BytesIO()
        Image.new("RGBA", (4, 3), (230, 70, 10, 128)).save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()

    def _post(self, *, data=None, image_bytes=None, mime="image/png", include_image=True):
        files = {"image": ("image.png", self.image_bytes if image_bytes is None else image_bytes, mime)}
        return self.client.post("/food_detect/", data=FORM if data is None else data,
                                files=files if include_image else {})

    def test_original_bytes_and_all_parameters_reach_service(self):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"rects": "[[1, 2, 7, 8]]", "conf": "[0.75]"})
        self.service.detect.assert_called_once_with(self.image_bytes, "food", 0.3, 0.4, "鱼", 0.5)

    def test_omitted_or_false_save_result_does_not_render_or_create_images(self):
        for save_result in (None, "false", "0"):
            with self.subTest(save_result=save_result):
                data = FORM if save_result is None else {**FORM, "save_result": save_result}
                self.assertEqual(self._post(data=data).status_code, 200)
                self.assertEqual(list(self.directory.rglob("*")), [])
        self.service.draw_result.assert_not_called()

    def test_save_result_writes_annotated_jpegs_without_overwriting_or_changing_response(self):
        self.service.detect.return_value = ([[8, 10, 28, 24]], [0.75])
        source = Image.new("RGB", (40, 30), (20, 120, 200))
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        for count in (1, 2):
            response = self._post(data={**FORM, "save_result": "true"}, image_bytes=buffer.getvalue())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"rects": "[[8, 10, 28, 24]]", "conf": "[0.75]"})
            saved = list(self.result_dir.glob("*.jpg"))
            self.assertEqual(len(saved), count)
            for path in saved:
                with Image.open(path) as result:
                    self.assertEqual(result.format, "JPEG")
                    self.assertEqual(result.size, (40, 30))
                    pixels = np.asarray(result).astype(int)
                    self.assertTrue(np.any((pixels[:, :, 0] > pixels[:, :, 1] + 60) &
                                           (pixels[:, :, 0] > pixels[:, :, 2] + 60)))
        self.assertEqual(set(self.directory.rglob("*.jpg")), set(saved))

    def test_invalid_save_result_is_rejected_before_detection(self):
        response = self._post(data={**FORM, "save_result": "not-a-bool"})
        self.assertEqual(response.status_code, 422)
        self.service.detect.assert_not_called()
        self.assertFalse(self.result_dir.exists())

    def test_saved_images_follow_dino_only_clip_and_empty_final_results(self):
        image = Image.new("RGB", (40, 30), (20, 120, 200))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        cases = (
            (0, ([[99, 99, 100, 100]], [0.9]), [[10, 7, 30, 22]], [0.6]),
            (0.5, ([[8, 10, 28, 24]], [0.75]), [[8, 10, 28, 24]], [0.75]),
            (0.5, ([], []), [], []),
        )
        for threshold, clip_result, expected_rects, expected_scores in cases:
            with self.subTest(threshold=threshold, expected_rects=expected_rects):
                gdino, clip = Mock(), Mock()
                gdino.detect.return_value = (np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
                                             np.array([0.9], dtype=np.float32), ["food"])
                clip.batch_detect.return_value = clip_result
                with patch("app.domain.food_service.GdinoDetect", return_value=gdino), \
                        patch("app.domain.food_service.ClipDetect", return_value=clip):
                    self.app.state.runtime.food_service = Detector({})
                before = set(self.result_dir.glob("*.jpg"))
                response = self._post(data={**FORM, "clip_thre": str(threshold), "save_result": "true"},
                                      image_bytes=buffer.getvalue())
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"rects": str(expected_rects), "conf": str(expected_scores)})
                saved = set(self.result_dir.glob("*.jpg")) - before
                self.assertEqual(len(saved), 1)
                with Image.open(saved.pop()) as rendered:
                    pixels = np.asarray(rendered).astype(int)
                    has_red = np.any((pixels[:, :, 0] > pixels[:, :, 1] + 60) &
                                     (pixels[:, :, 0] > pixels[:, :, 2] + 60))
                    self.assertEqual(bool(has_red), bool(expected_rects))
                if threshold <= 0:
                    clip.batch_detect.assert_not_called()

    def test_old_raw_score_threshold_is_rejected(self):
        self.assertEqual(self._post(data={**FORM, "clip_thre": "21"}).status_code, 422)
        self.service.detect.assert_not_called()

    def test_all_fields_are_required_and_numbers_must_parse(self):
        for field in ("image", *FORM):
            with self.subTest(missing=field):
                data = {key: value for key, value in FORM.items() if key != field}
                response = self._post(data=data, include_image=field != "image")
                self.assertEqual(response.status_code, 422)
                self.assertTrue(any(error["loc"][-1] == field for error in response.json()["detail"]))
        for field in ("dino_box", "dino_text", "clip_thre"):
            with self.subTest(invalid_number=field):
                self.assertEqual(self._post(data={**FORM, field: "invalid"}).status_code, 422)
        self.service.detect.assert_not_called()

    def test_non_image_mime_is_rejected_before_service(self):
        response = self._post(mime="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Uploaded file is not an image"})
        self.service.detect.assert_not_called()

    def test_real_image_decode_accepts_rgba_and_returns_reference_empty_on_corruption(self):
        gdino = Mock()
        gdino.detect.return_value = (np.empty((0, 4), dtype=np.float32),
                                     np.empty((0,), dtype=np.float32), [])
        with patch("app.domain.food_service.GdinoDetect", return_value=gdino), \
                patch("app.domain.food_service.ClipDetect"):
            self.app.state.runtime.food_service = Detector({})
        response = self._post(data={**FORM, "clip_thre": "0"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"rects": "[]", "conf": "[]"})
        image = gdino.detect.call_args.args[0]
        self.assertEqual(image.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(image), np.full((3, 4, 3), [230, 70, 10], dtype=np.uint8))
        gdino.reset_mock()
        with patch("builtins.print"):
            response = self._post(image_bytes=b"broken image")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"rects": "[]", "conf": "[]"})
        gdino.detect.assert_not_called()

    def test_inference_errors_return_500(self):
        self.service.detect.side_effect = RuntimeError("model failed")
        response = self._post()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["details"], "model failed")

    def test_json_decode_error_uses_reference_400_response(self):
        self.service.detect.side_effect = json.JSONDecodeError("invalid", "not-json", 0)
        response = self._post()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Invalid rect_list format, must be a valid JSON array."})

    def test_schema_exposes_required_food_fields_and_existing_process_route(self):
        document = self.client.get("/openapi.json").json()
        schema = document["paths"]["/food_detect/"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        if "$ref" in schema:
            schema = document["components"]["schemas"][schema["$ref"].split("/")[-1]]
        self.assertEqual(set(schema["required"]), {"image", *FORM})
        self.assertEqual(set(schema["properties"]), {"image", *FORM, "save_result"})
        self.assertEqual(schema["properties"]["save_result"]["type"], "boolean")
        self.assertIs(schema["properties"]["save_result"]["default"], False)
        self.assertIn("/process/", document["paths"])


if __name__ == "__main__":
    unittest.main()
