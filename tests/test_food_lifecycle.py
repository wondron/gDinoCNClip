"""Check eager food-model initialization and atomic configuration updates."""

import asyncio
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.bootstrap.lifespan import apply_nacos_config_update, create_lifespan
from app.bootstrap.state import ApplicationState
from app.core.config.loader import load_config
from app.core.config.nacos import NacosConfigSnapshot
from app.core.config.settings import Settings


class FoodLifecycleChecks(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.settings = Settings()
        self.settings = replace(self.settings, warmup=replace(self.settings.warmup, enabled=False))

    @staticmethod
    def _detector(*, clip_ready=True):
        return SimpleNamespace(dino=Mock(), clip=SimpleNamespace(ini_flag=clip_ready))

    def _runtime(self):
        service = self._detector()
        provider = Mock()
        provider.source_info.return_value = {"source": "nacos", "revision": "next"}
        runtime = ApplicationState(self.settings, self.config, {"source": "yaml"}, provider,
                                   predictor=service.dino, food_service=service)
        return SimpleNamespace(state=SimpleNamespace(runtime=runtime)), runtime

    @staticmethod
    def _snapshot(config):
        return NacosConfigSnapshot(config, {"profile": "test"}, {"model": "next"})

    def test_startup_builds_one_complete_detector_and_shares_its_gdino(self):
        service = self._detector()
        provider = Mock()
        provider.load.return_value = (self.config, {"source": "yaml"})
        app = SimpleNamespace(state=SimpleNamespace())

        async def exercise(settings):
            async with create_lifespan(settings)(app):
                runtime = app.state.runtime
                self.assertIs(runtime.food_service, service)
                self.assertIs(runtime.predictor, service.dino)
                self.assertTrue(runtime.food_service.clip.ini_flag)
            self.assertTrue(runtime.stopping)
            self.assertIsNone(runtime.predictor)
            self.assertIsNone(runtime.food_service)

        with tempfile.TemporaryDirectory() as directory:
            settings = replace(self.settings, paths=replace(self.settings.paths,
                               save_images_dir=str(Path(directory) / "uploads")))
            with patch("app.bootstrap.lifespan.NacosConfigProvider", return_value=provider), \
                    patch("app.bootstrap.lifespan.Detector", return_value=service) as build, \
                    patch("app.bootstrap.lifespan.setup_logger"):
                asyncio.run(exercise(settings))
            build.assert_called_once_with(self.config, dino=None, clip=None)
        provider.close.assert_called_once()

    def test_startup_rejects_failed_clip_before_serving(self):
        provider = Mock()
        provider.load.return_value = (self.config, {"source": "yaml"})
        app = SimpleNamespace(state=SimpleNamespace())

        async def exercise(settings):
            with self.assertRaisesRegex(RuntimeError, "CLIP model initialization failed"):
                async with create_lifespan(settings)(app):
                    self.fail("The application must not serve with an uninitialized CLIP model")

        with tempfile.TemporaryDirectory() as directory:
            settings = replace(self.settings, paths=replace(self.settings.paths,
                               save_images_dir=str(Path(directory) / "uploads")))
            with patch("app.bootstrap.lifespan.NacosConfigProvider", return_value=provider), \
                    patch("app.bootstrap.lifespan.Detector", return_value=self._detector(clip_ready=False)), \
                    patch("app.bootstrap.lifespan.setup_logger"):
                asyncio.run(exercise(settings))
        provider.close.assert_called_once()
        self.assertIsNone(app.state.runtime.food_service)
        self.assertIsNone(app.state.runtime.predictor)

    def test_clip_configuration_reload_rebuilds_only_clip_and_reuses_the_loaded_gdino(self):
        app, runtime = self._runtime()
        old_service = runtime.food_service
        changed = copy.deepcopy(self.config)
        changed["CLIP_CONFIG"]["food_range"] = [*changed["CLIP_CONFIG"]["food_range"], "豆腐"]
        new_clip = SimpleNamespace(ini_flag=True)
        with patch("app.domain.food_service.GdinoDetect") as build_dino, \
                patch("app.domain.food_service.ClipDetect", return_value=new_clip) as build_clip:
            self.assertTrue(apply_nacos_config_update(app, self._snapshot(changed)))
        build_dino.assert_not_called()
        build_clip.assert_called_once_with(changed)
        self.assertIsNot(runtime.food_service, old_service)
        self.assertIs(runtime.food_service.clip, new_clip)
        self.assertIs(runtime.food_service.dino, old_service.dino)
        self.assertIs(runtime.predictor, old_service.dino)
        self.assertIs(runtime.config, changed)

    def test_failed_clip_replacement_retains_the_complete_previous_detector(self):
        app, runtime = self._runtime()
        old_service, predictor = runtime.food_service, runtime.predictor
        changed = copy.deepcopy(self.config)
        changed["CLIP_CONFIG"]["model_path"] = "missing-replacement.onnx"
        with patch("app.bootstrap.lifespan.Detector", return_value=self._detector(clip_ready=False)):
            self.assertFalse(apply_nacos_config_update(app, self._snapshot(changed)))
        self.assertIs(runtime.food_service, old_service)
        self.assertIs(runtime.predictor, predictor)
        self.assertIs(runtime.config, self.config)
        self.assertEqual(runtime.config_info, {"source": "yaml"})
        self.assertIn("CLIP model initialization failed", runtime.reload_error)

    def test_unrelated_configuration_change_preserves_the_loaded_detector(self):
        app, runtime = self._runtime()
        service, predictor = runtime.food_service, runtime.predictor
        changed = copy.deepcopy(self.config)
        changed["TOTAL_CONFIG"]["machine_list"] = ["demo"]
        with patch("app.bootstrap.lifespan.Detector") as build:
            self.assertTrue(apply_nacos_config_update(app, self._snapshot(changed)))
        build.assert_not_called()
        self.assertIs(runtime.food_service, service)
        self.assertIs(runtime.predictor, predictor)
        self.assertIs(runtime.config, changed)

    def test_gdino_reload_keeps_both_endpoints_on_the_same_predictor(self):
        app, runtime = self._runtime()
        old_service = runtime.food_service
        changed = copy.deepcopy(self.config)
        changed["GROUNDINGDINO_CONFIG"]["short_side"] += 1
        new_dino = Mock()
        with patch("app.domain.food_service.GdinoDetect", return_value=new_dino) as build_dino, \
                patch("app.domain.food_service.ClipDetect") as build_clip:
            self.assertTrue(apply_nacos_config_update(app, self._snapshot(changed)))
        build_dino.assert_called_once_with(changed)
        build_clip.assert_not_called()
        self.assertIs(runtime.predictor, new_dino)
        self.assertIs(runtime.food_service.dino, new_dino)
        self.assertIs(runtime.food_service.clip, old_service.clip)
        self.assertIs(runtime.config, changed)

    def test_device_change_rebuilds_both_models(self):
        # GroundingDINO 和 CLIP 都在 TOTAL_CONFIG.device_id 指定的 GPU 上。
        app, runtime = self._runtime()
        changed = copy.deepcopy(self.config)
        changed["TOTAL_CONFIG"]["device_id"] += 1
        new_dino, new_clip = Mock(), SimpleNamespace(ini_flag=True)
        with patch("app.domain.food_service.GdinoDetect", return_value=new_dino) as build_dino, \
                patch("app.domain.food_service.ClipDetect", return_value=new_clip) as build_clip:
            self.assertTrue(apply_nacos_config_update(app, self._snapshot(changed)))
        build_dino.assert_called_once_with(changed)
        build_clip.assert_called_once_with(changed)
        self.assertIs(runtime.food_service.dino, new_dino)
        self.assertIs(runtime.food_service.clip, new_clip)
        self.assertIs(runtime.config, changed)


if __name__ == "__main__":
    unittest.main()
