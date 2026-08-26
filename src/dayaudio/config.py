"""Portable configuration and workspace layout."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from dayaudio.cas import atomic_write_text

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    profile: str = "auto"
    asr_backend: str = "sensevoice"
    strong_backend: str | None = None
    core_seconds: float = 300.0
    context_seconds: float = 1.0
    summary_chunk_seconds: float = 900.0
    task_lease_seconds: float = 900.0
    max_attempts: int = 3
    offline: bool = False
    log_text: bool = False

    def __post_init__(self) -> None:
        if self.core_seconds <= 0:
            raise ValueError("core_seconds must be positive")
        if not 0 <= self.context_seconds < self.core_seconds:
            raise ValueError("context_seconds must be non-negative and smaller than core")
        if self.summary_chunk_seconds <= 0:
            raise ValueError("summary_chunk_seconds must be positive")
        if self.task_lease_seconds <= 0:
            raise ValueError("task_lease_seconds must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    @property
    def db_path(self) -> Path:
        return self.home / "dayaudio.sqlite3"

    @property
    def cas_dir(self) -> Path:
        return self.home / "cas"

    @property
    def work_dir(self) -> Path:
        return self.home / "work"

    @property
    def export_dir(self) -> Path:
        return self.home / "exports"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    def ensure_layout(self) -> "Settings":
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in (self.cas_dir, self.work_dir, self.export_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            for path in (self.home, self.cas_dir, self.work_dir, self.export_dir, self.models_dir):
                os.chmod(path, 0o700)
        return self

    def portable_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["home"] = str(self.home)
        return value

    def digest(self) -> str:
        payload = json.dumps(
            self.portable_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def default_home() -> Path:
    configured = os.environ.get("DAYAUDIO_HOME")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".dayaudio"


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] must be a table")
    return value


def load_settings(
    path: Path | None = None,
    *,
    home: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    settings = Settings(home=(home or default_home()).expanduser().resolve())
    if path is not None:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        workspace = _table(data, "workspace")
        pipeline = _table(data, "pipeline")
        runtime = _table(data, "runtime")
        privacy = _table(data, "privacy")
        values: dict[str, Any] = {}
        if "home" in workspace and home is None:
            values["home"] = Path(str(workspace["home"])).expanduser().resolve()
        for key in (
            "profile",
            "asr_backend",
            "strong_backend",
            "core_seconds",
            "context_seconds",
            "summary_chunk_seconds",
        ):
            if key in pipeline:
                values[key] = pipeline[key]
        for key in ("task_lease_seconds", "max_attempts", "offline"):
            if key in runtime:
                values[key] = runtime[key]
        if "log_text" in privacy:
            values["log_text"] = privacy["log_text"]
        settings = replace(settings, **values)
    env_profile = os.environ.get("DAYAUDIO_PROFILE")
    env_offline = os.environ.get("DAYAUDIO_OFFLINE")
    env_values: dict[str, Any] = {}
    if env_profile:
        env_values["profile"] = env_profile
    if env_offline is not None:
        env_values["offline"] = env_offline.lower() in {"1", "true", "yes", "on"}
    if env_values:
        settings = replace(settings, **env_values)
    if home is not None:
        settings = replace(settings, home=home.expanduser().resolve())
    if overrides:
        clean = {key: value for key, value in overrides.items() if value is not None}
        settings = replace(settings, **clean)
    return settings


def write_default_config(path: Path, settings: Settings) -> None:
    """Write a small TOML file without requiring a TOML serialization package."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = f'''[workspace]
home = {json.dumps(str(settings.home))}

[pipeline]
profile = {json.dumps(settings.profile)}
asr_backend = {json.dumps(settings.asr_backend)}
core_seconds = {settings.core_seconds}
context_seconds = {settings.context_seconds}
summary_chunk_seconds = {settings.summary_chunk_seconds}

[runtime]
task_lease_seconds = {settings.task_lease_seconds}
max_attempts = {settings.max_attempts}
offline = {str(settings.offline).lower()}

[privacy]
log_text = {str(settings.log_text).lower()}
'''
    atomic_write_text(path, text, mode=0o600)
