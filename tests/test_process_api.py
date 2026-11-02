"""Verify the original process contract and opt-in result images without real models."""

import io
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


FORM = {"discri": "food", "box_thre": "0.3", "text_thre": "0.4"}


class ProcessApiChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.result_dir = self.directory / "output" / "save_res"
        self.result = {"boxes": [[0.5, 0.5, 0.6, 0.6]], "phrases": ["food"], "logits": [0.8]}
        self.predictor = Mock()
        self.predictor.predict.side_effect = self._predict
        app = FastAPI()
        app.include_router(router)
        app.state.runtime = SimpleNamespace(
            predictor=self.predictor, result_dir=self.result_dir,
            upload_dir=self.directory / "uploads",
            settings=SimpleNamespace(paths=SimpleNamespace(jpeg_quality=95)),
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.pixels = np.arange(40 * 30 * 3, dtype=np.uint8).reshape(30, 40, 3)
        image = Image.fromarray(self.pixels).convert("RGBA")
        image.putalpha(128)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()

    def _predict(self, image, caption, box_threshold, text_threshold):
        self.prediction_pixels = np.asarray(image).copy()
        self.prediction_args = (image.mode, image.size, caption, box_threshold, text_threshold)
        return self.result

    def _post(self, *, data=None, image_bytes=None, mime="image/png", include_image=True):
        files = {"image": ("image.png", self.image_bytes if image_bytes is None else image_bytes, mime)}
        return self.client.post("/process/", data=FORM if data is None else data,
                                files=files if include_image else {})

    def test_old_call_preserves_response_and_jpeg_round_trip_without_disk_files(self):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.result)
        self.assertEqual(self.prediction_args, ("RGB", (40, 30), "food", 0.3, 0.4))
        expected_jpeg = io.BytesIO()
        with Image.open(io.BytesIO(self.image_bytes)) as source:
            source.convert("RGB").save(expected_jpeg, format="JPEG", quality=95)
        expected_jpeg.seek(0)
        with Image.open(expected_jpeg) as expected:
            np.testing.assert_array_equal(self.prediction_pixels, np.asarray(expected.convert("RGB")))
        self.assertEqual(list(self.directory.rglob("*")), [])

    def test_false_save_result_does_not_draw_or_create_images(self):
        with patch("app.adapter.http.endpoints.predict.annotate") as annotate:
            response = self._post(data={**FORM, "save_result": "false"})
        self.assertEqual(response.status_code, 200)
        annotate.assert_not_called()
        self.assertEqual(list(self.directory.rglob("*")), [])

    def test_save_result_creates_unique_readable_annotated_images(self):
        for count in (1, 2):
            response = self._post(data={**FORM, "save_result": "true"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), self.result)
            saved = list(self.result_dir.glob("*.jpg"))
            self.assertEqual(len(saved), count)
            for path in saved:
                with Image.open(path) as rendered:
                    self.assertEqual(rendered.format, "JPEG")
                    self.assertEqual(rendered.size, (40, 30))
                    pixels = np.asarray(rendered).astype(int)
                    self.assertTrue(np.any((pixels[:, :, 0] > pixels[:, :, 1] + 60) &
                                           (pixels[:, :, 0] > pixels[:, :, 2] + 60)))
        self.assertEqual(set(self.directory.rglob("*.jpg")), set(saved))
        self.assertFalse((self.directory / "uploads").exists())

    def test_invalid_flag_and_missing_original_fields_are_validation_errors(self):
        cases = [({**FORM, "save_result": "not-a-bool"}, True)]
        cases.extend(({key: value for key, value in FORM.items() if key != name}, True) for name in FORM)
        cases.append((FORM, False))
        for data, include_image in cases:
            with self.subTest(data=data, include_image=include_image):
                self.assertEqual(self._post(data=data, include_image=include_image).status_code, 422)
        self.predictor.predict.assert_not_called()
        self.assertFalse(self.result_dir.exists())

    def test_non_image_and_corrupt_uploads_preserve_error_response(self):
        response = self._post(mime="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Uploaded file is not an image"})
        response = self._post(image_bytes=b"invalid image", data={**FORM, "save_result": "true"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Invalid image format or corrupted image"})
        self.predictor.predict.assert_not_called()
        self.assertFalse(self.result_dir.exists())

    def test_prediction_failure_preserves_error_and_does_not_save(self):
        self.predictor.predict.side_effect = RuntimeError("model failed")
        response = self._post(data={**FORM, "save_result": "true"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "Model inference failed", "details": "model failed"})
        self.assertFalse(self.result_dir.exists())

    def test_schema_adds_only_an_optional_save_result_boolean(self):
        document = self.client.get("/openapi.json").json()
        schema = document["paths"]["/process/"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        if "$ref" in schema:
            schema = document["components"]["schemas"][schema["$ref"].split("/")[-1]]
        self.assertEqual(set(schema["required"]), {"image", *FORM})
        self.assertEqual(set(schema["properties"]), {"image", *FORM, "save_result"})
        self.assertEqual(schema["properties"]["save_result"]["type"], "boolean")
        self.assertIs(schema["properties"]["save_result"]["default"], False)


if __name__ == "__main__":
    unittest.main()
