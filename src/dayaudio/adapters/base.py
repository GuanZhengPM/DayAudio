"""Adapter protocols used by the portable core."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from dayaudio.types import AsrSegment


class VadBackend(Protocol):
    name: str

    def speech_regions(self, audio_path: Path) -> list[tuple[float, float]]: ...

    def close(self) -> None: ...


class AsrBackend(Protocol):
    name: str
    model_id: str
    model_revision: str | None

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float = 0.0,
    ) -> list[AsrSegment]: ...

    def close(self) -> None: ...


class SpeakerEmbeddingBackend(Protocol):
    name: str
    model_digest: str

    def embed(self, audio_path: Path) -> list[float]: ...

    def close(self) -> None: ...


class SummaryBackend(Protocol):
    name: str

    def summarize(self, packet: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


def close_all(items: Iterable[object]) -> None:
    for item in items:
        close = getattr(item, "close", None)
        if close is not None:
            close()
