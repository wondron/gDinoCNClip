"""Offline configuration, logging and Nacos checks; no GPU or remote service required."""

import asyncio
import copy
import itertools
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

from app.adapter.http.middleware import RequestLoggingMiddleware
from app.bootstrap.lifespan import apply_nacos_config_update
from app.bootstrap.state import ApplicationState
from app.core.config.loader import load_config, parse_config_text, prepare_config
from app.core.config.nacos import NacosConfigClient, NacosConfigError, NacosConfigProvider, NacosConfigSnapshot
from app.core.config.settings import NacosSettings, Settings
from app.core.observability.json_formatter import CustomJsonFormatter
from app.core.observability.log_helper import show_text, trace_id_var
from app.core.observability.logging import CustomFormatter, DailyFileHandler, TraceFilter, setup_logger
from app.domain.pipeline.a_gdinoDtct import create_session
from app.domain.groundingdino.preprocess import preprocess_image
from app.domain.pipeline.a_gdinoDtct import GdinoDetect
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


class ScriptedStop:
    """Stands in for the poller's stop event: records every delay and stops after `rounds` polls."""

    def __init__(self, rounds):
        self.rounds, self.delays = rounds, []

    def wait(self, timeout):
        self.delays.append(timeout)
        return len(self.delays) > self.rounds

    def set(self):
        pass

    def clear(self):
        pass


class ConfigurationChecks(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def _detector_config(self, config):
        with patch.object(GdinoDetect, "init_model"):
            return GdinoDetect(config)

    def test_paths_are_independent_of_working_directory(self):
        original = Path.cwd()
        raw_config = parse_config_text((ROOT / "config/param/config.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for config in (load_config(), raw_config):
                    with self.subTest(raw_yaml=config is raw_config):
                        cfg = self._detector_config(config)
                        self.assertTrue(Path(cfg.model_path).is_file())
                        self.assertTrue(Path(cfg.tokenizer_path).is_dir())
                        self.assertEqual(cfg.provider, "cuda")
            finally:
                os.chdir(original)

    def test_invalid_and_missing_model_parameters_fail_early(self):
        # 参数在读取配置时校验（本地 YAML 与 Nacos 相同），早于任何模型加载。
        for field, value in (("std", [1, 0, 1]), ("mean", [0, 1]), ("use_gpu", "false"),
                             ("short_side", 0), ("threads", -1), ("max_text_len", 257),
                             ("box_threshold", 1.5), ("nms_threshold", -0.1), ("caption", " ")):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config["GROUNDINGDINO_CONFIG"][field] = value
                with self.assertRaises(ValueError):
                    prepare_config(config)
        disabled_nms = copy.deepcopy(self.config)
        disabled_nms["GROUNDINGDINO_CONFIG"]["nms_threshold"] = None
        self.assertIsNone(prepare_config(disabled_nms)["GROUNDINGDINO_CONFIG"]["nms_threshold"])
        with self.assertRaises(ValueError):
            prepare_config({})
        with self.assertRaises(ValueError):
            parse_config_text("- a\n- b")

    def test_old_onnx_model_keys_remain_readable(self):
        config = prepare_config({"ONNX_MODEL": {"path": "model.onnx", "tokenizer": "tokenizer", "provider": "cpu"}})
        cfg = self._detector_config(config)
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
        with self.assertLogs("nacos", level="INFO") as logs:
            loaded, source = provider.load("config/param/config.yaml")
            on_update = Mock()
            provider.start_polling(on_update)
        provider.close()
        self.assertEqual(source["source"], "yaml")
        self.assertEqual(source["nacos_status"], "fallback")
        self.assertEqual(loaded, self.config)
        # 回退本地后轮询线程继续重试 Nacos，关闭时随之退出。
        self.assertFalse(provider._thread.is_alive())
        on_update.assert_not_called()
        self.assertNotIn("private-token", "\n".join(logs.output))

    def test_nacos_disabled_loads_local_yaml_without_creating_remote_client(self):
        factory = Mock()
        cfg = NacosSettings(profile="test", enabled=False, profiles=(PROFILE,))
        provider = NacosConfigProvider(cfg, client_factory=factory)
        self.addCleanup(provider.close)
        loaded, source = provider.load("config/param/config.yaml")
        on_update = Mock()
        provider.start_polling(on_update)
        self.assertEqual(loaded, self.config)
        self.assertEqual(source["source"], "yaml")
        self.assertEqual(source["nacos_status"], "disabled")
        self.assertFalse(source["nacos_polling"])
        self.assertIsNone(provider._thread)
        factory.assert_not_called()
        on_update.assert_not_called()

    def test_invalid_nacos_yaml_falls_back_to_local_file(self):
        session = Mock()
        session.post.return_value = response(payload={"accessToken": "test-token"})
        session.request.return_value = response(text="- not a configuration mapping")
        cfg = NacosSettings(profile="test", enabled=True, profiles=(PROFILE,))
        client = NacosConfigClient(PROFILE, cfg, session)
        provider = NacosConfigProvider(cfg, client_factory=lambda *_: client)
        self.addCleanup(provider.close)
        loaded, source = provider.load("config/param/config.yaml")
        self.assertEqual(loaded, self.config)
        self.assertEqual(source["source"], "yaml")
        self.assertEqual(source["nacos_status"], "fallback")
        self.assertTrue(source["nacos_polling"])

    def test_disabling_polling_preserves_successfully_loaded_nacos_configuration(self):
        cfg = NacosSettings(profile="test", enabled=True, poll_enabled=False, profiles=(PROFILE,))
        remote = copy.deepcopy(self.config)
        remote["GROUNDINGDINO_CONFIG"]["caption"] = "remote caption ."
        snapshot = NacosConfigSnapshot(remote, PROFILE, {"model": "remote", "runtime": "remote"})
        client = Mock()
        client.fetch_snapshot.return_value = snapshot
        provider = NacosConfigProvider(cfg, client_factory=lambda *_: client)
        self.addCleanup(provider.close)
        with patch("app.core.config.nacos.load_config") as load_local:
            loaded, source = provider.load("config/param/config.yaml")
            provider.start_polling(Mock())
        self.assertIs(loaded, remote)
        self.assertEqual(source["source"], "nacos")
        self.assertFalse(source["nacos_polling"])
        self.assertIsNone(provider._thread)
        load_local.assert_not_called()

    def _raw(self, caption, **model):
        """Raw payloads for PROFILE's dataIds; the caption makes each version distinct."""
        gdino = {**self.config["GROUNDINGDINO_CONFIG"], "caption": f"{caption} .", **model}
        return {"model": yaml.safe_dump({"GROUNDINGDINO_CONFIG": gdino}, allow_unicode=True),
                "runtime": "TOTAL_CONFIG:\n  device_id: 0\n"}

    def _scripted_provider(self, fetches, *, startup=None):
        """Provider whose poll loop runs synchronously over scripted fetch results.

        startup is the payload loaded at startup; None simulates Nacos being unreachable then.
        """
        client = Mock()
        if startup is None:
            client.fetch_snapshot.side_effect = NacosConfigError("Nacos config network request failed")
        else:
            client.fetch_snapshot.return_value = NacosConfigSnapshot.from_raw(PROFILE, startup)
        client.fetch_raw_configs.side_effect = fetches
        cfg = NacosSettings(profile="test", enabled=True, profiles=(PROFILE,), poll_interval_seconds=10.0,
                            retry_interval_seconds=5.0, retry_max_interval_seconds=60.0)
        provider = NacosConfigProvider(cfg, client_factory=lambda *_: client)
        with patch("app.core.config.nacos.load_config", return_value=self.config), \
                self.assertLogs("nacos", level="INFO"):
            provider.load("config/param/config.yaml")
        provider._stop_event = ScriptedStop(len(fetches))
        return provider

    def test_nacos_poll_recovers_after_temporary_failure(self):
        cfg = NacosSettings(profile="test", enabled=True, poll_enabled=True, profiles=(PROFILE,),
                            poll_interval_seconds=0.01, retry_interval_seconds=0.01)
        changed = self._raw("two")
        client = Mock()
        client.fetch_snapshot.return_value = NacosConfigSnapshot.from_raw(PROFILE, self._raw("one"))
        client.fetch_raw_configs.side_effect = itertools.chain([NacosConfigError("offline")],
                                                               itertools.repeat(changed))
        provider = NacosConfigProvider(cfg, client_factory=lambda *_: client)
        complete = threading.Event()
        try:
            provider.load("config/param/config.yaml")
            provider.start_polling(lambda snapshot: complete.set() or True)
            self.assertTrue(complete.wait(3), "Polling failed to recover")
        finally:
            provider.close()
        self.assertFalse(provider._thread.is_alive())
        self.assertIs(provider._active.raw_configs, changed)
        client.close.assert_called_once_with()

    def test_nacos_poll_retries_after_failures_with_capped_backoff(self):
        offline = NacosConfigError("Nacos config network request failed")
        provider = self._scripted_provider([offline] * 6 + [self._raw("one")], startup=self._raw("one"))
        with self.assertLogs("nacos", level="DEBUG"):
            provider._poll_loop(Mock())
        # 失败后只等待重试间隔（不再叠加轮询间隔），随后指数退避到上限，恢复后回到轮询间隔。
        self.assertEqual(provider._stop_event.delays, [10.0, 5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 10.0])

    def test_nacos_outage_logs_one_warning_periodic_reminders_and_the_recovery(self):
        provider = self._scripted_provider([NacosConfigError("offline")] * 3 + [self._raw("one")],
                                           startup=self._raw("one"))
        with patch("app.core.config.nacos.time") as clock, self.assertLogs("nacos", level="DEBUG") as logs:
            clock.monotonic.side_effect = [0.0, 100.0, 400.0, 460.0]
            provider._poll_loop(Mock())
        failures = [record.levelno for record in logs.records if "poll failed" in record.getMessage()]
        self.assertEqual(failures, [logging.WARNING, logging.DEBUG, logging.WARNING])
        recovered = [record.getMessage() for record in logs.records if "poll recovered" in record.getMessage()]
        self.assertEqual(len(recovered), 1)
        self.assertIn("failures=3 | outage=460s", recovered[0])

    def test_rejected_nacos_content_is_retried_only_after_the_content_changes(self):
        one, two, three = self._raw("one"), self._raw("two"), self._raw("three")
        provider = self._scripted_provider([two, two, two, one, two, three], startup=one)
        on_update = Mock(side_effect=lambda snapshot: snapshot.raw_configs is three)
        with self.assertLogs("nacos", level="INFO"):
            provider._poll_loop(on_update)
        # 被拒绝的 two 不会每轮重载；内容回到 one 后再次发布 two 才重新尝试。
        self.assertEqual([call.args[0].raw_configs for call in on_update.call_args_list], [two, two, three])
        self.assertIs(provider._active.raw_configs, three)

    def test_invalid_nacos_content_during_polling_keeps_the_current_configuration(self):
        not_mapping = {"model": "- not a configuration mapping\n", "runtime": "TOTAL_CONFIG: {}\n"}
        float_threads = self._raw("two", threads=4.0)
        valid = self._raw("three")
        provider = self._scripted_provider([not_mapping, not_mapping, float_threads, float_threads, valid],
                                           startup=self._raw("one"))
        on_update = Mock(return_value=True)
        with self.assertLogs("nacos", level="INFO") as logs:
            provider._poll_loop(on_update)
        on_update.assert_called_once()
        self.assertIs(provider._active.raw_configs, valid)
        invalid = [line for line in logs.output if "configuration invalid" in line]
        self.assertEqual(len(invalid), 2)
        self.assertIn("threads must be an integer", invalid[1])

    def test_startup_fallback_switches_to_nacos_once_it_becomes_reachable(self):
        remote = self._raw("remote")
        provider = self._scripted_provider([NacosConfigError("still offline"), remote])
        on_update = Mock(return_value=True)
        with self.assertLogs("nacos", level="DEBUG"):
            provider._poll_loop(on_update)
        self.assertEqual(provider._stop_event.delays, [5.0, 10.0, 10.0])
        snapshot = on_update.call_args.args[0]
        self.assertEqual(snapshot.config["GROUNDINGDINO_CONFIG"]["caption"], "remote .")
        self.assertIs(provider._active, snapshot)
        self.assertEqual(provider.source_info(snapshot)["source"], "nacos")

    def test_every_nacos_poll_event_reaches_both_the_service_and_platform_logs(self):
        provider = self._scripted_provider([self._raw("one"), NacosConfigError("offline"), self._raw("two")],
                                           startup=self._raw("one"))
        with patch("app.core.config.nacos.show_text") as platform, self.assertLogs("nacos", level="DEBUG") as logs:
            provider._poll_loop(Mock(return_value=True))
        # 读取（含未变化）、失败、恢复、修改检测与应用：logger 和 show_text 收到相同消息和级别。
        self.assertEqual([call.args[0] for call in platform.call_args_list],
                         [record.getMessage() for record in logs.records])
        self.assertEqual([call.kwargs["logger_level"] for call in platform.call_args_list],
                         [record.levelno for record in logs.records])
        events = " / ".join(record.getMessage().split(" |")[0] for record in logs.records)
        self.assertEqual(events, "Nacos configuration unchanged / Nacos configuration poll failed; retaining current "
                                 "config / Nacos configuration poll recovered / Nacos configuration change detected "
                                 "by polling / Nacos configuration update applied")

    def test_nacos_text_is_decoded_as_utf8_without_a_charset_header(self):
        text = "CLIP_CONFIG:\n  detect_range: [皮皮虾, 鱿鱼]\n"
        reply = requests.Response()
        reply.status_code = 200
        reply.headers["Content-Type"] = "text/plain"
        reply.encoding = requests.utils.get_encoding_from_headers(reply.headers)  # 与 requests 适配器一致
        reply._content = text.encode("utf-8")
        session = Mock()
        session.request.return_value = reply
        anonymous = {**PROFILE, "username": "", "password": ""}
        client = NacosConfigClient(anonymous, NacosSettings(profile="test", enabled=True, profiles=(anonymous,)),
                                   session)
        self.assertEqual(client.get_config("model"), text)

    def _reload_fixture(self):
        original = SimpleNamespace()
        app_settings = Settings()
        app_settings = replace(app_settings, warmup=replace(app_settings.warmup, enabled=False))
        provider = Mock()
        provider.source_info.return_value = {"source": "nacos", "revision": "new"}
        original_service = SimpleNamespace(dino=original, clip=SimpleNamespace(ini_flag=True))
        runtime = ApplicationState(app_settings, self.config, {"source": "yaml"}, provider, original,
                                   food_service=original_service)
        app = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
        changed = copy.deepcopy(self.config)
        changed["GROUNDINGDINO_CONFIG"]["short_side"] = self.config["GROUNDINGDINO_CONFIG"].get("short_side", 800) + 1
        snapshot = NacosConfigSnapshot(changed, PROFILE, {"model": "new", "runtime": "new"})
        replacement = SimpleNamespace(dino=SimpleNamespace(), clip=SimpleNamespace(ini_flag=True))
        return app, runtime, original, snapshot, replacement

    def test_rejected_model_reload_keeps_previous_predictor(self):
        app, runtime, original, snapshot, _ = self._reload_fixture()
        with patch("app.bootstrap.lifespan.Detector", side_effect=RuntimeError("invalid replacement")):
            self.assertFalse(apply_nacos_config_update(app, snapshot))
        self.assertIs(runtime.predictor, original)
        self.assertIs(runtime.config, self.config)
        self.assertEqual(runtime.config_info, {"source": "yaml"})
        self.assertIn("invalid replacement", runtime.reload_error)

    def test_successful_reload_publishes_model_and_configuration_together(self):
        app, runtime, _, snapshot, replacement = self._reload_fixture()
        runtime.reload_error = "previous failure"
        loaded_clip = runtime.food_service.clip
        with patch("app.bootstrap.lifespan.Detector", return_value=replacement) as build:
            self.assertTrue(apply_nacos_config_update(app, snapshot))
        # 只改了 GroundingDINO 参数：重建 DINO，复用已加载的 CLIP。
        build.assert_called_once_with(snapshot.config, dino=None, clip=loaded_clip)
        self.assertIs(runtime.predictor, replacement.dino)
        self.assertIs(runtime.food_service, replacement)
        self.assertIs(runtime.config, snapshot.config)
        self.assertEqual(runtime.config_info, {"source": "nacos", "revision": "new"})
        self.assertIsNone(runtime.reload_error)

    def test_invalid_reload_parameters_keep_previous_model_and_configuration(self):
        app, runtime, original, snapshot, _ = self._reload_fixture()
        snapshot.config["GROUNDINGDINO_CONFIG"]["std"] = [1, 0, 1]
        with patch.object(GdinoDetect, "init_model") as build:
            self.assertFalse(apply_nacos_config_update(app, snapshot))
        build.assert_not_called()
        self.assertIs(runtime.predictor, original)
        self.assertIs(runtime.config, self.config)
        self.assertEqual(runtime.config_info, {"source": "yaml"})

    def test_unrelated_configuration_update_keeps_loaded_predictor(self):
        app, runtime, original, _, _ = self._reload_fixture()
        changed = copy.deepcopy(self.config)
        changed["TOTAL_CONFIG"]["machine_list"] = ["demo"]
        changed["OTHER_CONFIG"] = {"threshold": 0.5}
        snapshot = NacosConfigSnapshot(changed, PROFILE, {"model": "same", "runtime": "new"})
        with patch("app.bootstrap.lifespan.Detector") as build:
            self.assertTrue(apply_nacos_config_update(app, snapshot))
        build.assert_not_called()
        self.assertIs(runtime.predictor, original)
        self.assertIs(runtime.config, changed)
        self.assertEqual(runtime.config_info, {"source": "nacos", "revision": "new"})

    def test_type_equal_invalid_reload_values_are_validated(self):
        for section, field, value in (("GROUNDINGDINO_CONFIG", "use_gpu", 1),
                                      ("GROUNDINGDINO_CONFIG", "threads", 4.0),
                                      ("TOTAL_CONFIG", "device_id", False)):
            with self.subTest(section=section, field=field, value=value):
                app, runtime, original, _, _ = self._reload_fixture()
                changed = copy.deepcopy(self.config)
                changed[section][field] = value
                snapshot = NacosConfigSnapshot(changed, PROFILE, {"model": "new", "runtime": "new"})
                with patch.object(GdinoDetect, "init_model") as initialize:
                    self.assertFalse(apply_nacos_config_update(app, snapshot))
                initialize.assert_not_called()
                self.assertIs(runtime.predictor, original)
                self.assertIs(runtime.config, self.config)
                self.assertIsNotNone(runtime.reload_error)

    def test_stopping_prevents_publication_before_or_during_model_loading(self):
        for already_stopping in (True, False):
            with self.subTest(already_stopping=already_stopping):
                app, runtime, original, snapshot, replacement = self._reload_fixture()
                runtime.stopping = already_stopping

                def load_while_stopping(config, **reused):
                    runtime.stopping = True
                    return replacement

                with patch("app.bootstrap.lifespan.Detector", side_effect=load_while_stopping) as build:
                    self.assertFalse(apply_nacos_config_update(app, snapshot))
                self.assertEqual(build.call_count, 0 if already_stopping else 1)
                self.assertIsNone(runtime.reload_error, "rejected by an error instead of the stop flag")
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

    def test_http_middleware_logs_no_request_summary_but_keeps_trace_and_errors(self):
        async def endpoint(scope, receive, send):
            show_text("业务平台日志")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def failing(scope, receive, send):
            raise RuntimeError("intentional middleware error")

        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {"type": "http", "method": "GET", "path": "/openapi.json"}
        with self.assertLogs("platform", level="DEBUG") as platform, self.assertNoLogs("http", level="DEBUG"):
            asyncio.run(RequestLoggingMiddleware(endpoint)(scope, receive, send))
        trace_id = dict(sent[0]["headers"])[b"x-trace-id"].decode()
        # 不输出请求摘要；请求内的业务平台日志仍带本次请求的 traceId（与响应头一致）。
        self.assertEqual([record.getMessage() for record in platform.records], ["业务平台日志"])
        self.assertEqual(platform.records[0].traceId, trace_id)

        with self.assertLogs("http", level="ERROR") as http, self.assertRaises(RuntimeError):
            asyncio.run(RequestLoggingMiddleware(failing)(scope, receive, send))
        self.assertIn("HTTP 请求异常 | method=GET | path=/openapi.json", http.records[0].getMessage())

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
