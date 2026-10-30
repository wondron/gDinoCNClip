"""Check the food endpoint's multipart and reference response contracts."""

import io
import json
import unittest
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
        self.service = Mock()
        self.service.detect.return_value = ([[1, 2, 7, 8]], [0.75])
        app = FastAPI()
        app.include_router(router)
        app.state.runtime = SimpleNamespace(food_service=self.service)
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
        self.assertEqual(set(schema["properties"]), {"image", *FORM})
        self.assertIn("/process/", document["paths"])


if __name__ == "__main__":
    unittest.main()
