"""Check YAML inference defaults and explicit CLI overrides without model weights."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import yaml

from app.bootstrap.lifespan import warmup_models
from app.domain.pipeline.a_gdinoDtct import GdinoDetect
from scripts import infer


class InferenceConfigurationChecks(unittest.TestCase):
    def _run_cli(self, threshold_flags=()):
        detectors = []

        def build(config):
            with patch.object(GdinoDetect, "init_model"):
                detector = GdinoDetect(config)
            detector.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
            detector.predict_with_metadata = Mock(return_value=(
                {"boxes": [], "phrases": [], "logits": []},
                {"caption": "glass.", "original_size": [3, 2], "resized_size": [3, 2], "text_length": 4,
                 "resize_policy": "short_side_preserving_aspect_ratio"},
            ))
            detectors.append(detector)
            return detector

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump({"GROUNDINGDINO_CONFIG": {
                "model_path": "model.onnx", "tokenizer_path": "tokenizer", "use_gpu": False,
                "box_threshold": 0.87, "text_threshold": 0.62,
            }}), encoding="utf-8")
            argv = ["infer.py", "--config", str(config_path), "--image", "image.png",
                    "--caption", "glass", *threshold_flags]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO), \
                    patch("app.core.observability.logging.setup_logger"), \
                    patch.object(infer, "GdinoDetect", side_effect=build):
                infer.main()
        return detectors[0]

    def test_cli_uses_thresholds_from_selected_yaml(self):
        detector = self._run_cli()
        detector.predict_with_metadata.assert_called_once_with("image.png", "glass", 0.87, 0.62)

    def test_explicit_zero_cli_thresholds_override_yaml(self):
        detector = self._run_cli(("--box-threshold", "0", "--text-threshold", "0"))
        detector.predict_with_metadata.assert_called_once_with("image.png", "glass", 0.0, 0.0)

    def test_service_warmup_uses_detector_thresholds(self):
        predictor = SimpleNamespace(box_threshold=0.87, text_threshold=0.62, predict=Mock())
        image_path = Path(tempfile.gettempdir()) / "warmup.png"
        settings = SimpleNamespace(warmup=SimpleNamespace(
            enabled=True, runs=2, image_path=str(image_path), caption="glass .",
        ))
        warmup_models(predictor, settings)
        self.assertEqual(predictor.predict.call_args_list,
                         [call(image_path, "glass .", 0.87, 0.62)] * 2)


if __name__ == "__main__":
    unittest.main()
