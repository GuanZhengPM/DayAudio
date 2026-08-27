from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dayaudio.paths import filesystem_path, filesystem_tree_path
from dayaudio.storage import SCHEMA_VERSION, Storage
from dayaudio.types import SourceRecord


def _utf16_length(value: str | os.PathLike[str]) -> int:
    return len(os.path.abspath(value).encode("utf-16-le")) // 2


@pytest.mark.skipif(os.name != "nt", reason="extended path namespaces are Windows-specific")
def test_windows_path_conversion_handles_unc_and_prefixed_paths() -> None:
    unc = r"\\server\share\folder"
    assert str(filesystem_path(unc, force_extended=True)) == (
        r"\\?\UNC\server\share\folder"
    )

    prefixed = r"\\?\C:\folder\file.bin"
    assert str(filesystem_path(prefixed)) == prefixed
    assert str(filesystem_tree_path(prefixed)) == prefixed


@pytest.mark.skipif(os.name != "nt", reason="extended path namespaces are Windows-specific")
def test_tree_path_forces_extended_namespace_below_threshold(
    near_path_root: Path,
) -> None:
    assert _utf16_length(near_path_root) < 248
    assert not str(filesystem_path(near_path_root)).startswith("\\\\?\\")
    assert str(filesystem_tree_path(near_path_root)).startswith("\\\\?\\")


@pytest.mark.skipif(os.name != "nt", reason="Windows limits paths in UTF-16 code units")
def test_path_threshold_counts_non_bmp_utf16_units(near_path_root: Path) -> None:
    prefix_units = _utf16_length(near_path_root) + 1
    prefix_characters = len(os.path.abspath(near_path_root)) + 1
    count = max(1, (248 - prefix_units - len(".bin") + 1) // 2)
    count = min(count, 120)
    path = near_path_root / (("\U0001f600" * count) + ".bin")

    assert _utf16_length(path) >= 248
    assert len(os.path.abspath(path)) < 248
    assert len(path.name.encode("utf-16-le")) // 2 < 255
    assert prefix_characters + len(path.name) < 248
    filesystem = filesystem_path(path)
    assert str(filesystem).startswith("\\\\?\\")
    filesystem.write_bytes(b"emoji path")
    assert filesystem.read_bytes() == b"emoji path"


def test_sqlite_reopens_and_uses_wal_under_deep_home(long_path_root: Path) -> None:
    database_path = long_path_root / "sqlite" / "dayaudio.sqlite3"
    payload = b"deep source"
    source = SourceRecord(
        source_id="source-" + hashlib.sha256(payload).hexdigest()[:32],
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_path=str(long_path_root / "source.wav"),
        source_name="source.wav",
        size_bytes=len(payload),
        duration_seconds=1.0,
    )

    database = Storage(database_path)
    assert database.path == database_path.resolve()
    assert database.schema_version == SCHEMA_VERSION
    assert database.journal_mode().casefold() == "wal"
    database.upsert_source(source)
    database.close()

    reopened = Storage(database_path)
    assert reopened.require_source(source.source_id) == source
    assert reopened.schema_version == SCHEMA_VERSION
    assert reopened.journal_mode().casefold() == "wal"
    reopened.close()
