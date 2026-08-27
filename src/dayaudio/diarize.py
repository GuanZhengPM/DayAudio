"""End-to-end file-local speaker diarization orchestration.

This module deliberately emits anonymous, source-scoped speaker IDs.  Owner
matching is a separate explicit step in :mod:`dayaudio.identity`.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from dayaudio.paths import filesystem_path
from dayaudio.speaker import (
    EmbeddingBackend,
    SpeakerClusteringResult,
    SpeakerWindow,
    cluster_speaker_windows,
    orchestrate_embeddings,
)


class VadLike(Protocol):
    def speech_regions(self, audio_path: Path) -> list[tuple[float, float]]: ...


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    source_id: str
    model_digest: str
    speech_regions: tuple[tuple[float, float], ...]
    windows: tuple[SpeakerWindow, ...]
    clustering: SpeakerClusteringResult | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "model_digest": self.model_digest,
            "speech_regions": [list(item) for item in self.speech_regions],
            "windows": [
                {
                    key: value
                    for key, value in asdict(window).items()
                    if key != "payload"
                }
                for window in self.windows
            ],
            "status": self.status,
            "clusters": [cluster.to_dict() for cluster in self.clustering.clusters]
            if self.clustering
            else [],
            "turns": [turn.to_dict() for turn in self.clustering.turns]
            if self.clustering
            else [],
        }


def _window_id(source_id: str, start: float, end: float) -> str:
    material = f"{source_id}\0{round(start * 1000)}\0{round(end * 1000)}"
    return "speaker-window-" + hashlib.sha256(material.encode()).hexdigest()[:20]


def make_speaker_windows(
    source_id: str,
    regions: Iterable[tuple[float, float]],
    *,
    min_seconds: float = 1.5,
    max_seconds: float = 8.0,
) -> tuple[SpeakerWindow, ...]:
    """Split VAD regions into bounded windows suitable for speaker embeddings."""

    if min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError("speaker window bounds are invalid")
    windows: list[SpeakerWindow] = []
    for raw_start, raw_end in sorted(regions):
        start = max(0.0, float(raw_start))
        end = float(raw_end)
        if end - start < min_seconds:
            continue
        cursor = start
        while end - cursor >= min_seconds:
            piece_end = min(end, cursor + max_seconds)
            # Do not leave a final fragment that is too short: extend this
            # piece to the region end instead.
            if 0 < end - piece_end < min_seconds:
                piece_end = end
            windows.append(
                SpeakerWindow(
                    window_id=_window_id(source_id, cursor, piece_end),
                    source_id=source_id,
                    start=cursor,
                    end=piece_end,
                )
            )
            cursor = piece_end
    return tuple(windows)


def write_wav_slice(
    source: str | Path,
    destination: str | Path,
    *,
    start: float,
    end: float,
) -> Path:
    """Copy a sample-aligned interval from an uncompressed WAV file."""

    if end <= start or start < 0:
        raise ValueError("WAV slice bounds are invalid")
    source_path = Path(source)
    target = Path(destination)
    filesystem_source = filesystem_path(source_path)
    # The generated sibling can cross MAX_PATH even when the final target has
    # not, so keep the entire atomic slice operation in one namespace.
    filesystem_target = filesystem_path(target, force_extended=True)
    filesystem_target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".slice-", suffix=".tmp", dir=filesystem_target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with wave.open(str(filesystem_source), "rb") as reader:
            if reader.getcomptype() != "NONE":
                raise ValueError("speaker diarization requires an uncompressed WAV")
            rate = reader.getframerate()
            start_frame = min(reader.getnframes(), max(0, round(start * rate)))
            end_frame = min(reader.getnframes(), max(start_frame, round(end * rate)))
            if end_frame <= start_frame:
                raise ValueError("WAV slice does not contain any samples")
            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)
            params = reader.getparams()
        with wave.open(str(temporary), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
        os.replace(temporary, filesystem_target)
        return target
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def diarize_file(
    pcm_wav_path: str | Path,
    *,
    source_id: str,
    vad_backend: VadLike,
    embedding_backend: EmbeddingBackend,
    min_window_seconds: float = 1.5,
    max_window_seconds: float = 8.0,
    similarity_threshold: float = 0.72,
    embedding_batch_size: int = 32,
) -> DiarizationResult:
    """Run VAD, isolated-clip embedding, and deterministic file-local clustering."""

    source = Path(pcm_wav_path)
    if not filesystem_path(source).is_file():
        raise FileNotFoundError(source)
    regions = tuple(vad_backend.speech_regions(source))
    bare_windows = make_speaker_windows(
        source_id,
        regions,
        min_seconds=min_window_seconds,
        max_seconds=max_window_seconds,
    )
    if not bare_windows:
        return DiarizationResult(
            source_id,
            embedding_backend.model_digest,
            regions,
            (),
            None,
            "no_speech_windows",
        )

    with tempfile.TemporaryDirectory(prefix="dayaudio-speaker-") as directory:
        root = Path(directory)
        windows: list[SpeakerWindow] = []
        for window in bare_windows:
            clip = write_wav_slice(
                source,
                root / f"{window.window_id}.wav",
                start=window.start,
                end=window.end,
            )
            windows.append(
                SpeakerWindow(
                    window.window_id,
                    window.source_id,
                    window.start,
                    window.end,
                    payload=clip,
                )
            )
        embedded = orchestrate_embeddings(
            embedding_backend, windows, batch_size=embedding_batch_size
        )
        clustering = cluster_speaker_windows(
            embedded,
            model_digest=embedding_backend.model_digest,
            similarity_threshold=similarity_threshold,
        )

    # Never expose now-deleted temporary paths in the returned value.
    return DiarizationResult(
        source_id,
        embedding_backend.model_digest,
        regions,
        bare_windows,
        clustering,
        "complete",
    )


__all__ = [
    "DiarizationResult",
    "diarize_file",
    "make_speaker_windows",
    "write_wav_slice",
]
