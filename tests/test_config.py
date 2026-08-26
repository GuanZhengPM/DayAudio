from pathlib import Path

import pytest

from dayaudio.config import Settings, load_settings, write_default_config


def test_settings_layout_and_digest(tmp_path: Path):
    settings = Settings(home=tmp_path / "home").ensure_layout()
    assert settings.db_path.parent == settings.home
    assert settings.cas_dir.is_dir()
    assert settings.digest() == settings.digest()


def test_config_round_trip(tmp_path: Path):
    path = tmp_path / "config.toml"
    original = Settings(
        home=tmp_path / "custom",
        profile="cpu",
        core_seconds=120,
        offline=True,
    )
    write_default_config(path, original)
    loaded = load_settings(path)
    assert loaded.home == original.home.resolve()
    assert loaded.profile == "cpu"
    assert loaded.core_seconds == 120
    assert loaded.offline is True


def test_invalid_context_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        Settings(home=tmp_path, core_seconds=1, context_seconds=1)


def test_explicit_home_overrides_toml_home(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[workspace]\nhome = "/from-toml"\n', encoding="utf-8")
    explicit = tmp_path / "explicit"
    assert load_settings(path, home=explicit).home == explicit.resolve()
