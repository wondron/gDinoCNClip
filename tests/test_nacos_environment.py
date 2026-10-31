"""Check Nacos environment selection and local fallback without remote services."""

import os
import unittest
from unittest.mock import Mock, patch

from app.core.config import constant
from app.core.config.nacos import NacosConfigError, NacosConfigProvider, NacosConfigSnapshot
from app.core.config.settings import NacosSettings


def profile(name):
    return {
        "profile": name,
        "server_addresses": f"http://{name}.invalid:8848",
        "namespace": name,
        "username": "",
        "password": "",
        "group": "TEST_GROUP",
        "data_id_list": ["model"],
    }


PROFILES = (profile("prod"), profile("test"))


class NacosEnvironmentChecks(unittest.TestCase):
    def setUp(self):
        self.local_config = {"source_marker": "local"}
        for target in ("app.core.config.nacos.logger", "app.core.config.nacos.show_text"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def settings(self, marker=None, *, enabled=True, **kwargs):
        environment = {} if marker is None else {"NACOS_ENV": marker}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(constant, "NACOS_ENABLED", enabled):
            return NacosSettings(**{"profiles": PROFILES, **kwargs})

    def provider(self, settings, factory):
        provider = NacosConfigProvider(settings, client_factory=factory)
        self.addCleanup(provider.close)
        return provider

    def test_missing_or_blank_environment_selects_prod(self):
        self.assertEqual(constant.NACOS_DEFAULT_PROFILE, "prod")
        for marker in (None, "", " ", "\t\n"):
            with self.subTest(marker=marker):
                settings = self.settings(marker)
                self.assertEqual(settings.profile, "prod")
                self.assertTrue(settings.enabled)

    def test_any_nonempty_environment_selects_test(self):
        for marker in ("test", "prod", "true", "1", " TEST ", "anything"):
            with self.subTest(marker=marker):
                self.assertEqual(self.settings(marker).profile, "test")

    def test_constant_is_the_enable_switch_regardless_of_environment(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                    patch.dict(os.environ, {"NACOS_ENV": "test", "NACOS_ENABLED": str(not enabled)}, clear=True), \
                    patch.object(constant, "NACOS_ENABLED", enabled):
                self.assertIs(NacosSettings(profiles=PROFILES).enabled, enabled)

    def test_disabled_loads_yaml_without_creating_a_client_or_polling(self):
        for marker in (None, "test", "prod"):
            with self.subTest(marker=marker):
                settings = self.settings(marker, enabled=False)
                factory = Mock()
                provider = self.provider(settings, factory)
                on_update = Mock()
                with patch("app.core.config.nacos.load_config", return_value=self.local_config) as load_local:
                    loaded, source = provider.load("local-only.yaml")
                    provider.start_polling(on_update)
                self.assertIs(loaded, self.local_config)
                self.assertEqual(source["source"], "yaml")
                self.assertEqual(source["nacos_status"], "disabled")
                self.assertFalse(source["nacos_enabled"])
                self.assertFalse(source["nacos_polling"])
                self.assertIsNone(provider._thread)
                load_local.assert_called_once_with("local-only.yaml")
                factory.assert_not_called()
                on_update.assert_not_called()

    def test_disabled_ignores_remote_only_profile_and_timeout_validation(self):
        settings = self.settings(
            "test", enabled=False, profile="unconfigured", profiles=(),
            connect_timeout_seconds=0, request_timeout_seconds=0,
            poll_interval_seconds=0, retry_interval_seconds=0,
        )
        factory = Mock()
        provider = self.provider(settings, factory)
        with patch("app.core.config.nacos.load_config", return_value=self.local_config):
            loaded, source = provider.load("local-only.yaml")
            provider.start_polling(Mock())
        self.assertIs(loaded, self.local_config)
        self.assertEqual(source["source"], "yaml")
        self.assertFalse(source["nacos_polling"])
        self.assertIsNone(provider._thread)
        factory.assert_not_called()

    def test_enabled_fetches_only_the_selected_profile_without_local_read(self):
        for marker, selected in ((None, PROFILES[0]), ("test", PROFILES[1]), ("prod", PROFILES[1])):
            with self.subTest(marker=marker):
                settings = self.settings(marker, poll_enabled=False)
                remote_config = {"source_marker": selected["profile"]}
                snapshot = NacosConfigSnapshot(remote_config, selected, {"model": "remote yaml"})
                client = Mock()
                client.fetch_snapshot.return_value = snapshot
                factory = Mock(return_value=client)
                provider = self.provider(settings, factory)
                with patch("app.core.config.nacos.load_config") as load_local:
                    loaded, source = provider.load("unused.yaml")
                    provider.start_polling(Mock())
                self.assertIs(loaded, remote_config)
                self.assertEqual(source["source"], "nacos")
                self.assertEqual(source["nacos_profile"], selected["profile"])
                self.assertFalse(source["nacos_polling"])
                factory.assert_called_once_with(selected, settings)
                client.fetch_snapshot.assert_called_once_with()
                load_local.assert_not_called()

    def test_selected_profile_failure_falls_back_to_yaml_without_trying_other_profile(self):
        for marker, selected in ((None, PROFILES[0]), ("configured", PROFILES[1])):
            with self.subTest(marker=marker):
                settings = self.settings(marker)
                client = Mock()
                client.fetch_snapshot.side_effect = NacosConfigError("offline")
                factory = Mock(return_value=client)
                provider = self.provider(settings, factory)
                on_update = Mock()
                with patch("app.core.config.nacos.load_config", return_value=self.local_config) as load_local:
                    loaded, source = provider.load("fallback.yaml")
                    provider.start_polling(on_update)
                self.assertIs(loaded, self.local_config)
                self.assertEqual(source["source"], "yaml")
                self.assertEqual(source["nacos_status"], "fallback")
                self.assertEqual(source["nacos_profile"], selected["profile"])
                self.assertFalse(source["nacos_polling"])
                factory.assert_called_once_with(selected, settings)
                load_local.assert_called_once_with("fallback.yaml")
                self.assertIsNone(provider._thread)
                on_update.assert_not_called()

    def test_enabled_still_validates_selected_profile_and_timeouts(self):
        for values in ({"profiles": ()}, {"connect_timeout_seconds": 0},
                       {"poll_interval_seconds": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.settings("test", **values)


if __name__ == "__main__":
    unittest.main()
