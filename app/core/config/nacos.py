from __future__ import annotations
import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import requests
from app.core.observability.log_helper import show_text
from app.core.config.constant import NacosProfile
from app.core.config.loader import load_config, parse_config_text, prepare_config
from app.core.config.settings import NacosSettings

logger = logging.getLogger("nacos")
_MARKDOWN_URL = re.compile(r"\[(https?://[^\]]+)\]\([^)]+\)")
# 连续读取失败时，只在开始和之后每隔该时长记一次 WARNING，其余记 DEBUG，避免断网期间刷屏。
_OUTAGE_LOG_INTERVAL_SECONDS = 300.0

class NacosConfigError(RuntimeError):
    """Raised when a Nacos operation cannot produce a usable configuration."""


def _log(level: int, message: str, *, exc_info: bool = False, **fields) -> None:
    """Write one event to both the service log and the platform JSON log."""
    logger.log(level, message, exc_info=exc_info)
    show_text(message, logger_level=level, exc_info=exc_info or None, **fields)


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


def _fingerprint(profile: NacosProfile, raw_configs: dict[str, str]) -> str:
    """Stable fingerprint including profile and every dataId payload."""
    digest = hashlib.sha256()
    digest.update(profile["profile"].encode("utf-8"))

    for data_id in profile["data_id_list"]:
        digest.update(b"\0")
        digest.update(data_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw_configs[data_id].encode("utf-8"))

    return digest.hexdigest()


def _changed_data_ids(previous: dict[str, str], current: dict[str, str]) -> list[str]:
    """Return dataIds whose raw payload differs; without a previous payload every dataId differs."""
    return [
        data_id
        for data_id in dict.fromkeys([*previous, *current])
        if previous.get(data_id) != current.get(data_id)
    ]


@dataclass(frozen=True, slots=True)
class NacosConfigSnapshot:
    config: dict
    profile: NacosProfile
    raw_configs: dict[str, str]

    @classmethod
    def from_raw(cls, profile: NacosProfile, raw_configs: dict[str, str]) -> NacosConfigSnapshot:
        """Parse and merge dataIds in list order; later dataIds override earlier keys."""
        merged: dict = {}
        for data_id in profile["data_id_list"]:
            _deep_merge(merged, parse_config_text(raw_configs[data_id], f"Nacos dataId {data_id!r}"))
        return cls(config=prepare_config(merged), profile=profile, raw_configs=raw_configs)

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint including profile and every dataId payload."""
        return _fingerprint(self.profile, self.raw_configs)


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

        # Nacos 按 UTF-8 保存配置；响应头缺少 charset 时 requests 会按 ISO-8859-1 解码，中文会乱码。
        response.encoding = "utf-8"
        text = response.text
        return text if text.strip() else None

    def fetch_raw_configs(self) -> dict[str, str]:
        """Fetch the raw text of every configured dataId, in list order, without parsing it."""
        if not self.profile["data_id_list"]:
            raise NacosConfigError("Nacos data_id_list must not be empty")
        raw_configs: dict[str, str] = {}
        for data_id in self.profile["data_id_list"]:
            text = self.get_config(data_id)
            if text is None:
                raise NacosConfigError(f"Nacos dataId {data_id!r} is empty or missing")
            raw_configs[data_id] = text
        return raw_configs

    def fetch_snapshot(self) -> NacosConfigSnapshot:
        """Fetch and merge all configured dataIds into one snapshot."""
        return NacosConfigSnapshot.from_raw(self.profile, self.fetch_raw_configs())

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
        # 应用失败的内容不再反复解析和加载模型；Nacos 内容再次变化后才重新尝试。
        self._rejected_fingerprint: str | None = None
        self._failures = 0
        self._failing_since = 0.0
        self._failure_logged_at = 0.0
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

    @property
    def retry_max_interval_seconds(self) -> float:
        return max(float(self.settings.retry_max_interval_seconds), self.retry_interval_seconds)

    def _next_delay(self) -> float:
        """Poll interval normally; after failures, back off exponentially from the retry interval."""
        if not self._failures:
            return self.poll_interval_seconds
        backoff = self.retry_interval_seconds * 2 ** min(self._failures - 1, 16)
        return min(backoff, self.retry_max_interval_seconds)

    def _record_failure(self, now: float) -> bool:
        """Count one failed fetch; return whether it should be logged as a WARNING."""
        self._failures += 1
        if self._failures == 1:
            self._failing_since = now
        elif now - self._failure_logged_at < _OUTAGE_LOG_INTERVAL_SECONDS:
            return False
        self._failure_logged_at = now
        return True

    def _selected_profile(self) -> NacosProfile:
        for profile in self.settings.profiles:
            if profile["profile"] == self.settings.profile:
                return profile
        raise NacosConfigError(f"Nacos profile not configured | profile={self.settings.profile}")

    def _client(self, profile: NacosProfile) -> NacosConfigClient:
        profile_name = profile["profile"]

        with self._lock:
            client = self._clients.get(profile_name)
            if client is None:
                client = self._client_factory(profile, self.settings)
                self._clients[profile_name] = client
            return client

    def _fetch_selected_profile(self) -> NacosConfigSnapshot | None:
        """Fetch the startup snapshot; failures are logged and return None."""
        name = self.settings.profile
        try:
            profile = self._selected_profile()
            snapshot = self._client(profile).fetch_snapshot()
        except Exception as exc:
            _log(logging.WARNING, f"Nacos profile unavailable | profile={name} | error={type(exc).__name__}: {exc}",
                 profile=name)
            return None
        data_ids = list(profile["data_id_list"])
        _log(logging.INFO, f"Nacos configuration loaded | profile={name} | data_ids={data_ids}",
             profile=name, data_ids=data_ids)
        return snapshot

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
            # 启动读取失败计入连续失败，轮询线程按重试间隔尽快尝试恢复。
            self._record_failure(time.monotonic())

        config = load_config(yaml_path)
        status = "fallback" if self.settings.enabled else "disabled"
        reason = "Nacos 不可用" if self.settings.enabled else "Nacos 未启用"
        _log(logging.INFO, f"{reason}，已加载本地 YAML 配置 | path={yaml_path} | "
                           f"nacos_profile={self.settings.profile} | nacos_status={status}",
             profile=self.settings.profile, yaml_path=yaml_path)

        return config, {
            "source": "yaml",
            "nacos_enabled": self.settings.enabled,
            "nacos_profile": self.settings.profile,
            "nacos_polling": self.settings.enabled and self.poll_enabled,
            "nacos_poll_interval_seconds": (
                self.poll_interval_seconds if self.settings.enabled else self.settings.poll_interval_seconds
            ),
            "nacos_status": status,
        }

    def start_polling(self, on_update: UpdateCallback) -> None:
        """Start normal-GET polling; after a startup fallback it keeps retrying until Nacos is usable."""
        if not self.settings.enabled:
            _log(logging.INFO, "Nacos 未启用，使用本地 YAML 配置，不启动远程轮询")
            return

        if not self.poll_enabled:
            _log(logging.INFO, "Nacos 配置轮询未启用，继续使用启动时已加载的配置")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        with self._lock:
            recovering = self._active is None

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, args=(on_update,), name="nacos-config-poller", daemon=True)
        self._thread.start()
        if recovering:
            _log(logging.INFO, "Nacos 不可用，暂用本地 YAML 配置；后台继续重试，恢复后自动切换到 Nacos 配置 | "
                               f"retry_interval={self.retry_interval_seconds:g}s | "
                               f"max_retry_interval={self.retry_max_interval_seconds:g}s")
        else:
            _log(logging.INFO, f"Nacos configuration polling started | interval={self.poll_interval_seconds:.3f}s")

    def _poll_loop(self, on_update: UpdateCallback) -> None:
        try:
            while not self._stop_event.wait(self._next_delay()):
                try:
                    profile = self._selected_profile()
                    raw_configs = self._client(profile).fetch_raw_configs()
                    fingerprint = _fingerprint(profile, raw_configs)
                except Exception as exc:
                    now = time.monotonic()
                    level = logging.WARNING if self._record_failure(now) else logging.DEBUG
                    _log(level, "Nacos configuration poll failed; retaining current config | "
                                f"failures={self._failures} | outage={now - self._failing_since:.0f}s | "
                                f"next_retry={self._next_delay():g}s | error={type(exc).__name__}: {exc}")
                    continue

                if self._failures:
                    _log(logging.INFO, "Nacos configuration poll recovered | "
                                       f"failures={self._failures} | outage={time.monotonic() - self._failing_since:.0f}s")
                    self._failures = 0
                self._apply_if_changed(profile, raw_configs, fingerprint, on_update)
        finally:
            self._close_clients()

    def _apply_if_changed(
        self,
        profile: NacosProfile,
        raw_configs: dict[str, str],
        fingerprint: str,
        on_update: UpdateCallback,
    ) -> None:
        """Parse and apply fetched content only when it differs from the active and the rejected content."""
        name = profile["profile"]
        with self._lock:
            active = self._active

        if active is not None and fingerprint == active.fingerprint:
            # 内容回到当前生效版本后，之前被拒绝的内容再次发布时允许重新尝试。
            self._rejected_fingerprint = None
            _log(logging.DEBUG, f"Nacos configuration unchanged | profile={name}", profile=name)
            return
        if fingerprint == self._rejected_fingerprint:
            _log(logging.DEBUG, f"Nacos configuration still matches the rejected content | profile={name}",
                 profile=name)
            return

        changed_data_ids = _changed_data_ids(active.raw_configs if active is not None else {}, raw_configs)
        detail = f"profile={name} | changed_data_ids={changed_data_ids}"
        _log(logging.INFO, f"Nacos configuration change detected by polling | {detail}", profile=name)

        try:
            candidate = NacosConfigSnapshot.from_raw(profile, raw_configs)
        except Exception as exc:
            self._rejected_fingerprint = fingerprint
            _log(logging.WARNING, "Nacos configuration invalid; keeping last working config until the content changes | "
                                  f"{detail} | error={type(exc).__name__}: {exc}", profile=name)
            return

        try:
            applied = bool(on_update(candidate))
        except Exception as exc:
            self._rejected_fingerprint = fingerprint
            _log(logging.ERROR, "Nacos configuration update callback failed; "
                                "keeping last working snapshot until the content changes | "
                                f"{detail} | error={type(exc).__name__}: {exc}", exc_info=True, profile=name)
            return

        if not applied:
            self._rejected_fingerprint = fingerprint
            _log(logging.WARNING, "Nacos configuration update rejected; "
                                  f"keeping last working snapshot until the content changes | {detail}", profile=name)
            return

        with self._lock:
            self._active = candidate
        self._rejected_fingerprint = None
        _log(logging.INFO, f"Nacos configuration update applied | {detail}", profile=name)

    def _close_clients(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()

        for client in clients:
            try:
                client.close()
            except Exception:
                _log(logging.ERROR, "Failed to close Nacos client", exc_info=True)

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

        if thread is not None and thread.is_alive():
            # 轮询线程仍在等待请求或加载模型；它退出时会自行关闭客户端。
            _log(logging.WARNING, "Nacos configuration poller is still finishing; it will close its clients on exit")
            return

        self._close_clients()
        _log(logging.INFO, "Nacos configuration polling stopped")
