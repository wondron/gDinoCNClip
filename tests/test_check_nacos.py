"""Check the Nacos diagnostic's result, environment and output contracts offline."""

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

import requests
import yaml

from app.core.config.nacos import NacosConfigError, NacosConfigSnapshot
from app.core.config.settings import NacosSettings
from scripts import check_nacos


def profile(name):
    return {
        "profile": name,
        "server_addresses": f"http://{name}.invalid:8848",
        "namespace": name,
        "username": "diagnostic-user",
        "password": "profile-password",
        "group": "CHECK_GROUP",
        "data_id_list": ["model"],
    }


PROFILES = (profile("prod"), profile("test"))
RAW_CONFIG = "GROUNDINGDINO_CONFIG:\n  private_value: private-config-value\n"


class CheckNacosChecks(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.fetch_snapshot.return_value = NacosConfigSnapshot(
            {"GROUNDINGDINO_CONFIG": {"private_value": "private-config-value"}},
            PROFILES[1], {"model": RAW_CONFIG},
        )

    def run_check(self, args=(), *, enabled=True):
        current = NacosSettings(profile="test", profiles=PROFILES, enabled=enabled)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(check_nacos, "NacosSettings", return_value=current), \
                patch.object(check_nacos, "NacosConfigClient", return_value=self.client) as factory, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            result = check_nacos.main(list(args))
        return result, stdout.getvalue(), stderr.getvalue(), factory

    def test_success_uses_selected_environment_and_prints_summary_without_secrets(self):
        result, stdout, stderr, factory = self.run_check()
        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertIn("[SUCCESS]", stdout)
        self.assertIn("dataId=model", stdout)
        self.assertIn("GROUNDINGDINO_CONFIG", stdout)
        self.assertNotIn("private-config-value", stdout)
        self.assertNotIn("profile-password", stdout)
        selected, settings = factory.call_args.args
        self.assertEqual(selected, PROFILES[1])
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.poll_enabled)
        self.client.fetch_snapshot.assert_called_once_with()
        self.client.close.assert_called_once_with()

    def test_explicit_profile_overrides_the_service_environment(self):
        result, stdout, stderr, factory = self.run_check(("--profile", "prod"))
        self.assertEqual(result, 0)
        self.assertEqual(factory.call_args.args[0], PROFILES[0])
        self.assertEqual(factory.call_args.args[1].profile, "prod")

    def test_local_only_service_mode_still_checks_remote_and_explains_difference(self):
        with patch.dict(os.environ, {"NACOS_USE_LOCAL": ""}):
            result, stdout, stderr, factory = self.run_check(enabled=False)
        self.assertEqual(result, 0)
        self.assertIn("NACOS_USE_LOCAL", stdout)
        self.assertTrue(factory.call_args.args[1].enabled)
        self.client.fetch_snapshot.assert_called_once_with()

    def test_failures_exit_nonzero_without_local_fallback_and_close_client(self):
        errors = (
            NacosConfigError("Nacos dataId 'model' is empty or missing"),
            ValueError("GROUNDINGDINO_CONFIG must be a YAML mapping"),
            yaml.YAMLError("private-config-value"),
            requests.Timeout("http://test.invalid?accessToken=private-access-token"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.client.reset_mock()
                self.client.fetch_snapshot.side_effect = error
                with patch("app.core.config.nacos.load_config") as load_local:
                    result, stdout, stderr, factory = self.run_check()
                self.assertEqual(result, 1)
                self.assertIn("[FAIL]", stderr)
                self.assertNotIn("[SUCCESS]", stdout)
                self.assertNotIn("private-config-value", stderr)
                self.assertNotIn("private-access-token", stderr)
                load_local.assert_not_called()
                self.client.close.assert_called_once_with()

    def test_show_config_prints_the_original_remote_payload(self):
        result, stdout, stderr, factory = self.run_check(("--show-config",))
        self.assertEqual(result, 0)
        self.assertIn(RAW_CONFIG, stdout)


if __name__ == "__main__":
    unittest.main()
