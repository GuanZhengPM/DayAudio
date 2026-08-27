"""Internal path helpers for portable filesystem access."""

from __future__ import annotations

import os
from pathlib import Path

_WINDOWS_LEGACY_DIRECTORY_LIMIT = 248


def filesystem_path(
    path: str | os.PathLike[str], *, force_extended: bool = False
) -> Path:
    """Return *path* in a form accepted by long-path Windows APIs.

    Persisted and user-facing paths intentionally remain conventional paths.
    Only long values passed to filesystem operations receive the
    extended-length prefix, so persisted paths and ordinary short-path
    subprocess arguments keep their conventional spelling.  Tree walkers can
    set ``force_extended`` because descendants may cross the legacy limit even
    when the starting directory does not.
    """

    value = Path(path)
    if os.name != "nt":
        return value
    absolute = os.path.abspath(value)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    utf16_length = len(absolute.encode("utf-16-le")) // 2
    if not force_extended and utf16_length < _WINDOWS_LEGACY_DIRECTORY_LIMIT:
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{absolute[2:]}")
    return Path(f"\\\\?\\{absolute}")


def filesystem_tree_path(path: str | os.PathLike[str]) -> Path:
    """Return a root suitable for recursively walking a Windows path tree."""

    return filesystem_path(path, force_extended=True)


__all__ = ["filesystem_path", "filesystem_tree_path"]
