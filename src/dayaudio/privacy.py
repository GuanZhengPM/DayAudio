"""Privacy-safe logging helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_HOME_PATTERN = re.compile(r"(?:/Users/|/home/)[^/\s]+")
_WINDOWS_USER_PATTERN = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I)


def stable_private_id(value: str, *, prefix: str = "private") -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def redact_path(value: str | Path) -> str:
    text = str(value)
    text = _HOME_PATTERN.sub("<home>", text)
    return _WINDOWS_USER_PATTERN.sub("<home>", text)


def safe_metadata(
    metadata: dict[str, Any],
    *,
    allow_text: bool = False,
) -> dict[str, Any]:
    """Return a recursively redacted log payload.

    Transcript-like fields are removed by default. This is intentionally
    conservative: artifacts retain text; logs do not.
    """

    blocked = {"text", "transcript", "prompt", "hotwords", "voice_embedding"}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if key.lower() in blocked and not allow_text:
            result[key] = "<redacted>"
        elif isinstance(value, dict):
            result[key] = safe_metadata(value, allow_text=allow_text)
        elif isinstance(value, list):
            result[key] = [
                safe_metadata(item, allow_text=allow_text)
                if isinstance(item, dict)
                else redact_path(item)
                if isinstance(item, (str, Path))
                else item
                for item in value
            ]
        elif isinstance(value, (str, Path)):
            result[key] = redact_path(value)
        else:
            result[key] = value
    return result
