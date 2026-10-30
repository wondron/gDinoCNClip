"""Environment-selected Nacos configuration with local YAML fallback and GET polling.

This module intentionally does not use the Nacos HTTP long-poll listener API.
Configuration changes are detected by periodically fetching the configured
dataIds through GET /nacos/v1/cs/configs and comparing snapshot fingerprints.
"""
from __future__ import annotations
import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import requests
from app.core.observability.log_helper import show_text
from app.core.config.constant import NacosProfile
from app.core.config.loader import load_config, parse_config_text, prepare_config
from app.core.config.settings import NacosSettings

logger = logging.getLogger("main.config.nacos")
_MARKDOWN_URL = re.compile(r"\[(https?://[^\]]+)\]\([^)]+\)")

class NacosConfigError(RuntimeError):
    """Raised when a Nacos operation cannot produce a usable configuration."""

def _server_address(value: str) -> str:
    """Return the first address and tolerate an accidentally pasted Markdown URL."""
    value = value.strip()
    match = _MARKDOWN_URL.fullmatch(value)
    if match:
        value = match.group(1)

    value = re.split(r"[,;]", value, maxsplit=1)[0].strip()
    if not value:
        raise NacosConfigError("Nacos server address is empty")

    return value.rstrip("/")


def _deep_merge(target: dict, incoming: dict) -> dict:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


@dataclass(frozen=True, slots=True)
class NacosConfigSnapshot:
    config: dict
    profile: NacosProfile
    raw_configs: dict[str, str]

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint including profile and every dataId payload."""
        digest = hashlib.sha256()
        digest.update(self.profile["profile"].encode("utf-8"))

        for data_id in self.profile["data_id_list"]:
            digest.update(b"\0")
            digest.update(data_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self.raw_configs[data_id].encode("utf-8"))

        return digest.hexdigest()


class NacosConfigClient:
    """Small client using only the Nacos authentication and config GET APIs."""

    def __init__(
        self,
        profile: NacosProfile,
        settings: NacosSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings
        self.session = session or requests.Session()
        self.base_url = _server_address(profile["server_addresses"])
        self._access_token: str | None = None

    def _url(self, endpoint: str) -> str:
        prefix = "" if self.base_url.endswith("/nacos") else "/nacos"
        return f"{self.base_url}{prefix}/v1/{endpoint.lstrip('/')}"

    @property
    def _timeout(self) -> tuple[float, float]:
        return (
            float(self.settings.connect_timeout_seconds),
            float(self.settings.request_timeout_seconds),
        )

    def _authenticate(self) -> None:
        username = self.profile["username"]
        password = self.profile["password"]

        if not username and not password:
            return

        try:
            response = self.session.post(
                self._url("auth/login"),
                data={"username": username, "password": password},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise NacosConfigError("Nacos authentication network request failed") from exc

        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise NacosConfigError(
                "Nacos authentication failed with "
                f"HTTP {response.status_code}"
            ) from exc

        token = payload.get("accessToken") or payload.get("access_token")
        if not token:
            raise NacosConfigError(
                "Nacos authentication response has no access token"
            )

        self._access_token = str(token)

    def _auth_params(self) -> dict[str, str]:
        if self.profile["username"] or self.profile["password"]:
            if self._access_token is None:
                self._authenticate()
            return {"accessToken": self._access_token or ""}
        return {}

    def _request(self, method: str, url: str, **kwargs):
        """Send an authenticated request and re-authenticate once on 401/403."""
        base_params = dict(kwargs.pop("params", {}))
        has_credentials = bool(self.profile["username"] or self.profile["password"])
        response = None
        for attempt in range(2):
            auth_params = self._auth_params()
            params = {**base_params, **auth_params,}
            request_kwargs = dict(kwargs)
            headers = dict(request_kwargs.pop("headers", {}))
            access_token = auth_params.get("accessToken")
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"
            if headers:
                request_kwargs["headers"] = headers
            response = self.session.request(method, url, params=params, **request_kwargs,)
            should_retry_auth = (response.status_code in {401, 403} and has_credentials and attempt == 0)
            if not should_retry_auth:
                return response
            self._access_token = None
        return response

    def get_config(self, data_id: str) -> str | None:
        """Fetch one current configuration value through the normal GET API."""
        try:
            response = self._request(
                "GET", self._url("cs/configs"),
                params={"dataId": data_id, "group": self.profile["group"],
                        "tenant": self.profile["namespace"]},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise NacosConfigError("Nacos config network request failed") from exc

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NacosConfigError(f"Nacos config request failed with HTTP {response.status_code}") from exc

        text = response.text
        return text if text.strip() else None

    def fetch_snapshot(self) -> NacosConfigSnapshot:
        """Fetch and merge all configured dataIds into one snapshot."""
        if not self.profile["data_id_list"]:
            raise NacosConfigError("Nacos data_id_list must not be empty")
        raw_configs: dict[str, str] = {}
        merged: dict = {}
        for data_id in self.profile["data_id_list"]:
            text = self.get_config(data_id)
            if text is None:
                raise NacosConfigError(f"Nacos dataId {data_id!r} is empty or missing")
            document = parse_config_text(text, f"Nacos dataId {data_id!r}",)
            _deep_merge(merged, document)
            raw_configs[data_id] = text
        return NacosConfigSnapshot(config=prepare_config(merged), profile=self.profile, raw_configs=raw_configs)

    def close(self) -> None:
        self.session.close()


ClientFactory = Callable[[NacosProfile, NacosSettings], NacosConfigClient]
UpdateCallback = Callable[[NacosConfigSnapshot], bool]


class NacosConfigProvider:
    """Load the selected environment, fall back locally, and poll through normal GET."""

    def __init__(self, settings: NacosSettings, client_factory: ClientFactory = NacosConfigClient,) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._clients: dict[str, NacosConfigClient] = {}
        self._active: NacosConfigSnapshot | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def poll_enabled(self) -> bool:
        return bool(self.settings.poll_enabled)

    @property
    def poll_interval_seconds(self) -> float:
        interval = float(self.settings.poll_interval_seconds)
        if interval <= 0:
            raise ValueError("Nacos poll_interval_seconds must be greater than 0")
        return interval

    @property
    def retry_interval_seconds(self) -> float:
        interval = float(self.settings.retry_interval_seconds)
        return max(interval, 0.1)

    def _client(self, profile: NacosProfile) -> NacosConfigClient:
        profile_name = profile["profile"]

        with self._lock:
            client = self._clients.get(profile_name)
            if client is None:
                client = self._client_factory(profile, self.settings)
                self._clients[profile_name] = client
            return client

    def _fetch_selected_profile(self, *, log_success: bool = True) -> NacosConfigSnapshot | None:
        profile = next(
            (item for item in self.settings.profiles if item["profile"] == self.settings.profile),
            None,
        )
        if profile is None:
            message = f"Nacos profile not configured | profile={self.settings.profile}"
            logger.warning(message)
            show_text(message, logger_level=logging.WARNING)
            return None

        try:
            snapshot = self._client(profile).fetch_snapshot()
            if log_success:
                logger.info("Nacos configuration loaded | profile=%s | data_ids=%s", profile["profile"], profile["data_id_list"])
                show_text(
                    f"Nacos configuration loaded | profile={profile['profile']} | "
                    f"data_ids={profile['data_id_list']}",
                    profile=profile["profile"],
                    data_ids=list(profile["data_id_list"]),
                )
            return snapshot
        except Exception as exc:
            logger.warning("Nacos profile unavailable | profile=%s | error=%s: %s", profile["profile"], type(exc).__name__, exc)
            show_text(
                f"Nacos profile unavailable | profile={profile['profile']} | "
                f"error={type(exc).__name__}: {exc}",
                logger_level=logging.WARNING,
                profile=profile["profile"],
                data_ids=list(profile["data_id_list"]),
            )
            return None

    def source_info(self, snapshot: NacosConfigSnapshot) -> dict:
        return {
            "source": "nacos",
            "nacos_enabled": True,
            "nacos_polling": self.poll_enabled,
            "nacos_poll_interval_seconds": self.poll_interval_seconds,
            "nacos_profile": snapshot.profile["profile"],
            "nacos_namespace": snapshot.profile["namespace"],
            "nacos_group": snapshot.profile["group"],
            "nacos_data_ids": list(snapshot.profile["data_id_list"]),
            "loaded_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        }

    def load(self, yaml_path: str) -> tuple[dict, dict]:
        """Load once during startup, falling back to local YAML."""
        if self.settings.enabled:
            snapshot = self._fetch_selected_profile()
            if snapshot is not None:
                with self._lock:
                    self._active = snapshot
                return snapshot.config, self.source_info(snapshot)

        config = load_config(yaml_path)
        status = "fallback" if self.settings.enabled else "disabled"
        message = (
            f"Local YAML configuration loaded | path={yaml_path} | "
            f"nacos_profile={self.settings.profile} | nacos_status={status}"
        )
        logger.info(message)
        show_text(message, profile=self.settings.profile, yaml_path=yaml_path)

        return config, {
            "source": "yaml",
            "nacos_enabled": self.settings.enabled,
            "nacos_profile": self.settings.profile,
            "nacos_polling": False,
            "nacos_poll_interval_seconds": self.poll_interval_seconds,
            "nacos_status": status,
        }

    def start_polling(self, on_update: UpdateCallback) -> None:
        """Start periodic normal-GET polling."""
        if not self.settings.enabled or not self.poll_enabled:
            logger.info(
                "Nacos configuration polling disabled | "
                "enabled=%s | poll_enabled=%s",
                self.settings.enabled,
                self.poll_enabled,
            )
            show_text(
                f"Nacos configuration polling disabled | enabled={self.settings.enabled} | "
                f"poll_enabled={self.poll_enabled}",
            )
            return

        with self._lock:
            has_active_snapshot = self._active is not None

        if not has_active_snapshot:
            logger.info("Nacos configuration polling skipped | reason=no_available_profile")
            show_text("Nacos configuration polling skipped | reason=no_available_profile")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, args=(on_update,), name="nacos-config-poller", daemon=True)
        self._thread.start()
        logger.info("Nacos configuration polling started | interval=%.3fs", self.poll_interval_seconds,)
        show_text(f"Nacos configuration polling started | interval={self.poll_interval_seconds:.3f}s")

    def _poll_loop(self, on_update: UpdateCallback) -> None:
        try:
            while not self._stop_event.is_set():
                if self._stop_event.wait(self.poll_interval_seconds):
                    break

                try:
                    candidate = self._fetch_selected_profile(log_success=False)
                except Exception as exc:
                    logger.warning("Nacos configuration poll failed | error=%s: %s", type(exc).__name__,exc)
                    show_text(
                        f"Nacos configuration poll failed | error={type(exc).__name__}: {exc}",
                        logger_level=logging.WARNING,
                    )
                    if self._stop_event.wait(self.retry_interval_seconds):
                        break
                    continue
                if candidate is None:
                    logger.warning("Nacos configuration unavailable; retaining current config and retrying")
                    show_text(
                        "Nacos configuration unavailable; retaining current config and retrying",
                        logger_level=logging.WARNING,
                    )
                    if self._stop_event.wait(self.retry_interval_seconds):
                        break
                    continue
                with self._lock:
                    active = self._active
                if active is None:
                    logger.warning("Nacos configuration polling stopped | reason=no_active_snapshot")
                    show_text(
                        "Nacos configuration polling stopped | reason=no_active_snapshot",
                        logger_level=logging.WARNING,
                    )
                    break

                if candidate.fingerprint == active.fingerprint:
                    logger.debug("Nacos configuration unchanged | profile=%s", candidate.profile["profile"])
                    show_text(
                        f"Nacos configuration unchanged | profile={candidate.profile['profile']}",
                        logger_level=logging.DEBUG,
                    )
                    continue

                logger.info(
                    "Nacos configuration change detected by polling | "
                    "old_profile=%s | new_profile=%s",
                    active.profile["profile"],
                    candidate.profile["profile"],
                )
                show_text(
                    "Nacos configuration change detected by polling | "
                    f"old_profile={active.profile['profile']} | new_profile={candidate.profile['profile']}",
                )

                if on_update(candidate):
                    with self._lock:
                        self._active = candidate

                    logger.info(
                        "Nacos configuration update applied | profile=%s",
                        candidate.profile["profile"],
                    )
                    show_text(f"Nacos configuration update applied | profile={candidate.profile['profile']}")
                else:
                    logger.warning(
                        "Nacos configuration update rejected; "
                        "keeping last working snapshot | profile=%s",
                        candidate.profile["profile"],
                    )
                    show_text(
                        "Nacos configuration update rejected; "
                        f"keeping last working snapshot | profile={candidate.profile['profile']}",
                        logger_level=logging.WARNING,
                    )
        finally:
            self._close_clients()

    def _close_clients(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()

        for client in clients:
            try:
                client.close()
            except Exception:
                logger.exception("Failed to close Nacos client")
                show_text("Failed to close Nacos client", logger_level=logging.ERROR, exc_info=True)

    def close(self) -> None:
        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            join_timeout = max(
                2.0,
                float(self.settings.connect_timeout_seconds)
                + float(self.settings.request_timeout_seconds)
                + 1.0,
            )
            thread.join(timeout=join_timeout)

        if thread is None or not thread.is_alive():
            self._close_clients()

        logger.info("Nacos configuration polling stopped")
        show_text("Nacos configuration polling stopped")
