"""Offline configuration, logging and Nacos checks; no GPU or remote service required."""

import copy
import json
import logging
import os
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.bootstrap.lifespan import apply_nacos_config_update
from app.bootstrap.state import ApplicationState
from app.core.config.loader import load_config, parse_config_text, prepare_config
from app.core.config.model import ModelSettings
from app.core.config.nacos import NacosConfigClient, NacosConfigProvider, NacosConfigSnapshot
from app.core.config.settings import NacosSettings, Settings
from app.core.observability.json_formatter import CustomJsonFormatter
from app.core.observability.log_helper import trace_id_var
from app.core.observability.logging import CustomFormatter, DailyFileHandler, TraceFilter, setup_logger
from app.adapter.onnx.session import create_session
from app.domain.groundingdino.preprocess import preprocess_image
from PIL import Image

PROFILE = {
    "profile": "test", "server_addresses": "http://127.0.0.1:8848", "namespace": "test",
    "username": "test-user", "password": "test-password", "group": "TEST_GROUP", "data_id_list": ["model", "runtime"],
}


def response(status=200, text="", payload=None):
    value = Mock(status_code=status, text=text)
    value.json.return_value = payload
    if status >= 400:
        value.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    return value


class ConfigurationChecks(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_paths_are_independent_of_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = load_config()
            finally:
                os.chdir(original)
        cfg = ModelSettings.from_config(config)
        self.assertTrue(Path(cfg.model_path).is_file())
        self.assertTrue(Path(cfg.tokenizer_path).is_dir())
        self.assertEqual(cfg.provider, "cuda")

    def test_invalid_and_missing_model_parameters_fail_early(self):
        for field, value in (("std", [1, 0, 1]), ("mean", [0, 1]), ("use_gpu", "false"),
                             ("short_side", 0), ("threads", -1), ("max_text_len", 257)):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config["GROUNDINGDINO_CONFIG"][field] = value
                with self.assertRaises(ValueError):
                    ModelSettings.from_config(config)
        with self.assertRaises(ValueError):
            prepare_config({})
        with self.assertRaises(ValueError):
            parse_config_text("- a\n- b")

    def test_old_onnx_model_keys_remain_readable(self):
        config = prepare_config({"ONNX_MODEL": {"path": "model.onnx", "tokenizer": "tokenizer", "provider": "cpu"}})
        cfg = ModelSettings.from_config(config)
        self.assertEqual(cfg.provider, "cpu")
        self.assertEqual(Path(cfg.model_path), ROOT / "model.onnx")

    def test_default_preprocessing_has_exactly_the_previous_values(self):
        image = Image.new("RGB", (32, 24), (128, 64, 240))
        actual, _ = preprocess_image(image, short_side=24)
        pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        expected = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected.transpose(2, 0, 1)[None])
        custom, _ = preprocess_image(image, short_side=24, mean=[0, 0, 0], std=[1, 1, 1])
        np.testing.assert_array_equal(custom, pixels.transpose(2, 0, 1)[None])

    def test_ort_logging_configures_session_and_kernel_logs(self):
        from app.core.config.constant import ORT_LOG_SEVERITY_LEVEL
        runtime = Mock()
        runtime.get_available_providers.return_value = ["CPUExecutionProvider"]
        session = runtime.InferenceSession.return_value
        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            self.assertIs(create_session("model.onnx"), session)
            runtime.set_default_logger_severity.assert_called_with(ORT_LOG_SEVERITY_LEVEL)
            self.assertEqual(runtime.SessionOptions.return_value.log_severity_level, ORT_LOG_SEVERITY_LEVEL)
            create_session("model.onnx", log_severity_level=2)
            runtime.set_default_logger_severity.assert_called_with(2)
            self.assertEqual(runtime.SessionOptions.return_value.log_severity_level, 2)
            for level in (-1, 5, True, "3"):
                with self.assertRaises(ValueError):
                    create_session("model.onnx", log_severity_level=level)

    def test_nacos_auth_retry_and_multiple_data_id_merge(self):
        session = Mock()
        session.post.side_effect = [response(payload={"accessToken": "expired"}), response(payload={"accessToken": "renewed"})]
        model_yaml = yaml.safe_dump({"GROUNDINGDINO_CONFIG": self.config["GROUNDINGDINO_CONFIG"]})
        session.request.side_effect = [response(403), response(text=model_yaml), response(text="TOTAL_CONFIG:\n  device_id: 1\n")]
        cfg = NacosSettings(profile="test", enabled=True, profiles=(PROFILE,))
        client = NacosConfigClient(PROFILE, cfg, session)
        snapshot = client.fetch_snapshot()
        self.assertEqual(snapshot.config["TOTAL_CONFIG"]["device_id"], 1)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.request.call_args_list[1].kwargs["params"]["accessToken"], "renewed")
        self.assertEqual(session.request.call_args_list[2].kwargs["params"]["dataId"], "runtime")
        client.close()
        session.close.assert_called_once()

    def test_nacos_failure_uses_local_yaml_without_leaking_tokens(self):
        session = Mock()
        session.post.return_value = response(payload={"accessToken": "private-token"})
        session.request.side_effect = requests.ConnectionError("https://host/config?accessToken=private-token")
        cfg = NacosSettings(profile="test", enabled=True, profiles=(PROFILE,))
        client = NacosConfigClient(PROFILE, cfg, session)
        provider = NacosConfigProvider(cfg, client_factory=lambda *_: client)
        with self.assertLogs("main.config.nacos", level="INFO") as logs:
            loaded, source = provider.load("config/param/config.yaml")
        provider.close()
        self.assertEqual(source["source"], "yaml")
        self.assertEqual(source["nacos_status"], "fallback")
        self.assertEqual(loaded, self.config)
        self.assertNotIn("private-token", "\n".join(logs.output))

    def test_nacos_poll_recovers_after_temporary_failure(self):
        cfg = NacosSettings(profile="test", enabled=True, poll_enabled=True, profiles=(PROFILE,),
                            poll_interval_seconds=0.01, retry_interval_seconds=0.01)
        provider = NacosConfigProvider(cfg)
        initial = NacosConfigSnapshot(self.config, PROFILE, {"model": "one", "runtime": "one"})
        changed = NacosConfigSnapshot(self.config, PROFILE, {"model": "two", "runtime": "one"})
        provider._active = initial
        fetched = Mock(side_effect=[None, changed, changed])
        complete = threading.Event()
        with patch.object(provider, "_fetch_selected_profile", fetched):
            try:
                provider.start_polling(lambda snapshot: complete.set() or True)
                self.assertTrue(complete.wait(3), "Polling failed to recover")
            finally:
                provider.close()
        self.assertFalse(provider._thread.is_alive())
        self.assertEqual(provider._active.fingerprint, changed.fingerprint)

    def _reload_fixture(self):
        original = SimpleNamespace(model_settings=ModelSettings.from_config(self.config))
        app_settings = Settings()
        app_settings = replace(app_settings, warmup=replace(app_settings.warmup, enabled=False))
        provider = Mock()
        provider.source_info.return_value = {"source": "nacos", "revision": "new"}
        runtime = ApplicationState(app_settings, self.config, {"source": "yaml"}, provider, original)
        app = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
        changed = copy.deepcopy(self.config)
        changed["GROUNDINGDINO_CONFIG"]["short_side"] = original.model_settings.short_side + 1
        snapshot = NacosConfigSnapshot(changed, PROFILE, {"model": "new", "runtime": "new"})
        replacement = SimpleNamespace(model_settings=ModelSettings.from_config(changed))
        return app, runtime, original, snapshot, replacement

    def test_rejected_model_reload_keeps_previous_predictor(self):
        app, runtime, original, snapshot, _ = self._reload_fixture()
        with patch("app.bootstrap.lifespan.OnnxPredictor", side_effect=RuntimeError("invalid replacement")):
            self.assertFalse(apply_nacos_config_update(app, snapshot))
        self.assertIs(runtime.predictor, original)
        self.assertIs(runtime.config, self.config)
        self.assertEqual(runtime.config_info, {"source": "yaml"})
        self.assertIn("invalid replacement", runtime.reload_error)

    def test_successful_reload_publishes_model_and_configuration_together(self):
        app, runtime, _, snapshot, replacement = self._reload_fixture()
        runtime.reload_error = "previous failure"
        with patch("app.bootstrap.lifespan.OnnxPredictor", return_value=replacement) as build:
            self.assertTrue(apply_nacos_config_update(app, snapshot))
        build.assert_called_once_with(snapshot.config)
        self.assertIs(runtime.predictor, replacement)
        self.assertIs(runtime.config, snapshot.config)
        self.assertEqual(runtime.config_info, {"source": "nacos", "revision": "new"})
        self.assertIsNone(runtime.reload_error)

    def test_invalid_reload_parameters_keep_previous_model_and_configuration(self):
        app, runtime, original, snapshot, _ = self._reload_fixture()
        snapshot.config["GROUNDINGDINO_CONFIG"]["std"] = [1, 0, 1]
        with patch("app.bootstrap.lifespan.OnnxPredictor") as build:
            self.assertFalse(apply_nacos_config_update(app, snapshot))
        build.assert_not_called()
        self.assertIs(runtime.predictor, original)
        self.assertIs(runtime.config, self.config)
        self.assertEqual(runtime.config_info, {"source": "yaml"})

    def test_stopping_prevents_publication_before_or_during_model_loading(self):
        for already_stopping in (True, False):
            with self.subTest(already_stopping=already_stopping):
                app, runtime, original, snapshot, replacement = self._reload_fixture()
                runtime.stopping = already_stopping

                def load_while_stopping(config):
                    runtime.stopping = True
                    return replacement

                with patch("app.bootstrap.lifespan.OnnxPredictor", side_effect=load_while_stopping) as build:
                    self.assertFalse(apply_nacos_config_update(app, snapshot))
                self.assertEqual(build.call_count, 0 if already_stopping else 1)
                self.assertIs(runtime.predictor, original)
                self.assertIs(runtime.config, self.config)
                self.assertEqual(runtime.config_info, {"source": "yaml"})

    def test_formatters_preserve_record_and_trace_context(self):
        record = logging.makeLogRecord({"name": "test", "msg": "中文日志", "levelno": logging.WARNING, "levelname": "WARNING"})
        token = trace_id_var.set("trace-test")
        try:
            TraceFilter().filter(record)
        finally:
            trace_id_var.reset(token)
        CustomFormatter().format(record)
        result = json.loads(CustomJsonFormatter().format(record))
        self.assertEqual(result["traceId"], "trace-test")
        self.assertEqual(result["level"], "WARNING")
        self.assertEqual(result["message"], "中文日志")
        self.assertNotIn("asctime", record.__dict__)
        self.assertNotIn("message", record.__dict__)

    def test_logger_initialization_is_idempotent_and_rotation_uses_config_timezone(self):
        app_settings = Settings()
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(app_settings, paths=replace(app_settings.paths, output_dir=directory))
            try:
                setup_logger(app_settings=cfg)
                first = len(logging.getLogger().handlers)
                setup_logger(app_settings=cfg)
                self.assertEqual(len(logging.getLogger().handlers), first)
                handler = next(h for h in logging.getLogger().handlers if isinstance(h, DailyFileHandler))
                # 2026-01-01 15:59:00 UTC, 23:59 in configured UTC+8.
                self.assertEqual(handler.computeRollover(1767283140), 1767283200)
                logging.getLogger("verify_config").info("配置日志检查")
                self.assertTrue((Path(directory) / "logs/app.log").exists())
            finally:
                for logger in (logging.getLogger(), logging.getLogger("platform")):
                    for handler in logger.handlers[:]:
                        if getattr(handler, "_grounding_handler", False):
                            logger.removeHandler(handler)
                            handler.close()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConfigurationChecks))
    destination = ROOT / "reports/config_checks.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({"passed": result.wasSuccessful(), "tests": result.testsRun,
                                       "failures": [str(item) for item in result.failures + result.errors]},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
