"""Hash, probe, deduplicate, and register user-supplied audio files."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .cas import ContentAddressedStore, sha256_file
from .storage import Storage
from .types import SourceRecord

AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aiff",
        ".alac",
        ".amr",
        ".caf",
        ".flac",
        ".m4a",
        ".mka",
        ".mp3",
        ".mp4",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)


class IngestError(RuntimeError):
    pass


class ProbeError(IngestError):
    pass


class SourceChangedError(IngestError):
    pass


@dataclass(frozen=True, slots=True)
class AudioProbe:
    duration_seconds: float | None
    codec: str | None
    sample_rate: int | None
    channels: int | None
    recording_start: str | None = None
    recording_time_basis: str | None = None
    raw: Mapping[str, Any] | None = None


Runner = Callable[..., Any]


def _positive_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _positive_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


_ISO_DATETIME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[Tt ](?P<time>\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?)"
    r"(?P<zone>Z|[+-]\d{2}:?\d{2})?$"
)


def _normalize_container_time(value: Any) -> tuple[str, bool] | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    match = _ISO_DATETIME.fullmatch(raw)
    if not match:
        return None
    candidate = raw.replace("t", "T")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    zone = match.group("zone")
    if zone and zone != "Z" and ":" not in zone:
        candidate = candidate[: -len(zone)] + zone[:3] + ":" + zone[3:]
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return normalized, True
    return parsed.isoformat(), False


def _container_recording_time(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    format_data = payload.get("format")
    if isinstance(format_data, Mapping):
        tags = format_data.get("tags")
        if isinstance(tags, Mapping):
            for key, value in tags.items():
                if str(key).lower() == "creation_time":
                    normalized = _normalize_container_time(value)
                    if normalized:
                        timestamp, timezone_known = normalized
                        basis = "container:format.tags.creation_time"
                        if not timezone_known:
                            basis += ":timezone-unspecified"
                        return timestamp, basis
    streams = payload.get("streams")
    if isinstance(streams, list):
        for index, stream in enumerate(streams):
            if not isinstance(stream, Mapping) or stream.get("codec_type") != "audio":
                continue
            tags = stream.get("tags")
            if not isinstance(tags, Mapping):
                continue
            for key, value in tags.items():
                if str(key).lower() == "creation_time":
                    normalized = _normalize_container_time(value)
                    if normalized:
                        timestamp, timezone_known = normalized
                        basis = f"container:audio_stream[{index}].tags.creation_time"
                        if not timezone_known:
                            basis += ":timezone-unspecified"
                        return timestamp, basis
    return None


_FILENAME_DATETIME_PATTERNS = (
    re.compile(
        r"(?<!\d)(?P<y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})"
        r"(?:[T _-])(?P<h>\d{2})[-_.:](?P<minute>\d{2})[-_.:](?P<s>\d{2})(?!\d)"
    ),
    re.compile(
        r"(?<!\d)(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})"
        r"[_-](?P<h>\d{2})(?P<minute>\d{2})(?P<s>\d{2})(?!\d)"
    ),
)
_FILENAME_DATE_PATTERNS = (
    re.compile(r"(?<!\d)(?P<y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})(?!\d)"),
)


def filename_recording_time(path: str | os.PathLike[str]) -> tuple[str, str] | None:
    """Return only unambiguous, calendar-valid date evidence from a filename."""

    stem = Path(path).stem
    datetimes: set[str] = set()
    for pattern in _FILENAME_DATETIME_PATTERNS:
        for match in pattern.finditer(stem):
            try:
                value = datetime(
                    int(match["y"]),
                    int(match["m"]),
                    int(match["d"]),
                    int(match["h"]),
                    int(match["minute"]),
                    int(match["s"]),
                ).isoformat()
            except ValueError:
                continue
            datetimes.add(value)
    if len(datetimes) == 1:
        return next(iter(datetimes)), "filename:datetime:timezone-unspecified"
    if len(datetimes) > 1:
        return None

    dates: set[str] = set()
    for pattern in _FILENAME_DATE_PATTERNS:
        for match in pattern.finditer(stem):
            try:
                value = datetime(
                    int(match["y"]), int(match["m"]), int(match["d"])
                ).date().isoformat()
            except ValueError:
                continue
            dates.add(value)
    if len(dates) == 1:
        return next(iter(dates)), "filename:date-only"
    return None


def parse_ffprobe(payload: Mapping[str, Any], *, source_path: str | os.PathLike[str]) -> AudioProbe:
    streams = payload.get("streams")
    audio_streams = (
        [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"]
        if isinstance(streams, list)
        else []
    )
    if not audio_streams:
        raise ProbeError(f"ffprobe found no audio stream: {source_path}")
    stream = audio_streams[0]
    format_data = payload.get("format")
    if not isinstance(format_data, Mapping):
        format_data = {}
    duration = _positive_float(stream.get("duration"))
    if duration is None:
        duration = _positive_float(format_data.get("duration"))
    recording = _container_recording_time(payload) or filename_recording_time(source_path)
    return AudioProbe(
        duration_seconds=duration,
        codec=str(stream["codec_name"]) if stream.get("codec_name") else None,
        sample_rate=_positive_int(stream.get("sample_rate")),
        channels=_positive_int(stream.get("channels")),
        recording_start=recording[0] if recording else None,
        recording_time_basis=recording[1] if recording else None,
        raw=payload,
    )


def probe_audio(
    path: str | os.PathLike[str],
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Runner = subprocess.run,
    timeout: float | None = 120.0,
) -> AudioProbe:
    source = Path(path)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(source),
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise ProbeError(f"ffprobe executable not found: {ffprobe_bin}") from error
    except subprocess.TimeoutExpired as error:
        raise ProbeError(f"ffprobe timed out: {source}") from error
    except UnicodeDecodeError as error:
        raise ProbeError(f"ffprobe returned output that is not valid UTF-8: {source}") from error
    if completed.returncode != 0:
        stderr = str(getattr(completed, "stderr", "")).strip()
        raise ProbeError(f"ffprobe failed for {source}: {stderr or f'exit {completed.returncode}'}")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ProbeError(f"ffprobe returned invalid JSON for {source}") from error
    if not isinstance(payload, dict):
        raise ProbeError(f"ffprobe returned a non-object JSON payload for {source}")
    return parse_ffprobe(payload, source_path=source)


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def ingest_file(
    path: str | os.PathLike[str],
    storage: Storage,
    *,
    cas: ContentAddressedStore | None = None,
    ffprobe_bin: str = "ffprobe",
    runner: Runner = subprocess.run,
    probe_timeout: float | None = 120.0,
) -> SourceRecord:
    """Register one source, deduplicating byte-identical paths by SHA-256."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    before = _stat_signature(source)
    digest = sha256_file(source)

    existing = storage.find_source_by_sha256(digest)
    if existing is not None:
        after = _stat_signature(source)
        if before != after:
            raise SourceChangedError(f"source changed while it was being hashed: {source}")
        obj = None
        if cas is not None:
            obj = cas.put_file(source)
            if obj.sha256 != digest or _stat_signature(source) != before:
                raise SourceChangedError(f"source changed while it was copied: {source}")
        storage.add_source_location(existing.source_id, str(source), source.name)
        if obj is not None:
            storage.add_artifact(
                kind="source-audio",
                sha256=obj.sha256,
                path=obj.path,
                size_bytes=obj.size_bytes,
                source_id=existing.source_id,
                metadata={"original_name": source.name},
            )
        return existing

    probe = probe_audio(source, ffprobe_bin=ffprobe_bin, runner=runner, timeout=probe_timeout)
    after_probe = _stat_signature(source)
    if before != after_probe:
        raise SourceChangedError(f"source changed while it was being probed: {source}")
    source_id = f"source-{digest[:32]}"
    record = SourceRecord(
        source_id=source_id,
        source_sha256=digest,
        source_path=str(source),
        source_name=source.name,
        size_bytes=before[2],
        duration_seconds=probe.duration_seconds,
        codec=probe.codec,
        sample_rate=probe.sample_rate,
        channels=probe.channels,
        recording_start=probe.recording_start,
        recording_time_basis=probe.recording_time_basis,
    )
    obj = None
    if cas is not None:
        obj = cas.put_file(source)
        if obj.sha256 != digest or _stat_signature(source) != before:
            raise SourceChangedError(f"source changed while it was copied: {source}")
    stored = storage.upsert_source(record)
    if obj is not None:
        storage.add_artifact(
            kind="source-audio",
            sha256=obj.sha256,
            path=obj.path,
            size_bytes=obj.size_bytes,
            source_id=stored.source_id,
            metadata={"original_name": source.name},
        )
    return stored


def discover_audio_files(
    paths: Iterable[str | os.PathLike[str]], *, recursive: bool = True
) -> list[Path]:
    discovered: dict[str, Path] = {}
    for supplied in paths:
        path = Path(supplied).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                discovered[str(path)] = path
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        iterator = path.rglob("*") if recursive else path.glob("*")
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
                resolved = candidate.resolve()
                discovered[str(resolved)] = resolved
    return sorted(discovered.values(), key=lambda item: str(item).casefold())


def ingest_files(
    paths: Iterable[str | os.PathLike[str]],
    storage: Storage,
    *,
    cas: ContentAddressedStore | None = None,
    recursive: bool = True,
    ffprobe_bin: str = "ffprobe",
    runner: Runner = subprocess.run,
) -> list[SourceRecord]:
    return [
        ingest_file(
            path,
            storage,
            cas=cas,
            ffprobe_bin=ffprobe_bin,
            runner=runner,
        )
        for path in discover_audio_files(paths, recursive=recursive)
    ]


class Ingestor:
    """Bound convenience API used by CLI and pipeline layers."""

    def __init__(
        self,
        storage: Storage,
        *,
        cas: ContentAddressedStore | None = None,
        ffprobe_bin: str = "ffprobe",
        runner: Runner = subprocess.run,
    ) -> None:
        self.storage = storage
        self.cas = cas
        self.ffprobe_bin = ffprobe_bin
        self.runner = runner

    def ingest(self, path: str | os.PathLike[str]) -> SourceRecord:
        return ingest_file(
            path,
            self.storage,
            cas=self.cas,
            ffprobe_bin=self.ffprobe_bin,
            runner=self.runner,
        )

    def ingest_many(
        self, paths: Iterable[str | os.PathLike[str]], *, recursive: bool = True
    ) -> list[SourceRecord]:
        return ingest_files(
            paths,
            self.storage,
            cas=self.cas,
            recursive=recursive,
            ffprobe_bin=self.ffprobe_bin,
            runner=self.runner,
        )


__all__ = [
    "AUDIO_EXTENSIONS",
    "AudioProbe",
    "IngestError",
    "Ingestor",
    "ProbeError",
    "SourceChangedError",
    "discover_audio_files",
    "filename_recording_time",
    "ingest_file",
    "ingest_files",
    "parse_ffprobe",
    "probe_audio",
]
