"""Subprocess ASR adapter for CrispASR, TurnAlign, and other JSON CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from dayaudio.types import AsrSegment

OutputFormat = Literal["auto", "json", "jsonl"]
TimeUnit = Literal["seconds", "milliseconds"]
RequireEnd = Literal["auto"] | bool


class CommandAdapterError(RuntimeError):
    """Safe public error with private subprocess streams retained as attributes."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True, slots=True)
class CommandAdapterConfig:
    command: tuple[str, ...]
    name: str = "command-asr"
    model_id: str = "external-command"
    model_revision: str | None = None
    output_format: OutputFormat = "auto"
    time_unit: TimeUnit = "seconds"
    timestamps_are_absolute: bool = False
    require_end_event: RequireEnd = "auto"
    timeout_seconds: float | None = None
    cwd: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    offline: bool = True
    max_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must contain at least one argument")
        if any(not isinstance(argument, str) or not argument for argument in self.command):
            raise ValueError("command arguments must be non-empty strings")
        if self.output_format not in {"auto", "json", "jsonl"}:
            raise ValueError("output_format must be auto, json, or jsonl")
        if self.time_unit not in {"seconds", "milliseconds"}:
            raise ValueError("time_unit must be seconds or milliseconds")
        if self.require_end_event not in {"auto", True, False}:
            raise ValueError("require_end_event must be auto, true, or false")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


def _json_documents(raw: str, output_format: OutputFormat) -> list[Any]:
    text = raw.strip()
    if not text:
        return []
    if output_format != "jsonl":
        try:
            return [json.loads(text)]
        except json.JSONDecodeError:
            if output_format == "json":
                raise CommandAdapterError("command output is not valid JSON") from None

    documents: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CommandAdapterError(
                f"command JSONL contains invalid JSON on line {line_number}"
            ) from exc
    return documents


def _candidate_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        result: list[Mapping[str, Any]] = []
        for item in value:
            result.extend(_candidate_items(item))
        return result
    if not isinstance(value, Mapping):
        return []

    kind = str(value.get("kind") or value.get("type") or "").lower()
    if kind in {"end", "progress", "vad", "silence", "speech"}:
        return []
    for key in ("segments", "results", "events", "transcription", "transcript", "output"):
        nested = value.get(key)
        if isinstance(nested, (list, Mapping)):
            return _candidate_items(nested)
    for key in ("segment", "result", "data", "payload"):
        nested = value.get(key)
        if isinstance(nested, (list, Mapping)):
            if isinstance(nested, Mapping) and "text" in nested:
                merged = dict(nested)
                for inherited in ("kind", "type", "segment_id", "revision"):
                    if inherited in value and inherited not in merged:
                        merged[inherited] = value[inherited]
                return [merged]
            candidates = _candidate_items(nested)
            if candidates:
                return candidates
    if "text" in value:
        return [value]
    for key in ("transcription", "transcript", "result", "output"):
        nested_text = value.get(key)
        if isinstance(nested_text, str) and nested_text.strip():
            merged = dict(value)
            merged["text"] = nested_text
            return [merged]
    return []


def _event_kinds(value: Any) -> set[str]:
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_event_kinds(item))
        return result
    if not isinstance(value, Mapping):
        return set()
    result = {
        str(value.get("kind") or value.get("type") or "").casefold()
    }
    for key in ("events", "results"):
        nested = value.get(key)
        if isinstance(nested, (list, Mapping)):
            result.update(_event_kinds(nested))
    result.discard("")
    return result


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounds(
    item: Mapping[str, Any],
    time_unit: TimeUnit,
    fallback_duration: float | None,
) -> tuple[float, float]:
    milliseconds = time_unit == "milliseconds"
    if "start_ms" in item or "end_ms" in item:
        start = _number(item.get("start_ms"))
        end = _number(item.get("end_ms"))
        milliseconds = True
    else:
        start = _number(item.get("start", item.get("start_time")))
        end = _number(item.get("end", item.get("end_time")))
    timestamp_value = item.get("timestamp")
    if (start is None or end is None) and isinstance(timestamp_value, (list, tuple)):
        timestamp = timestamp_value
        if len(timestamp) >= 2:
            start = _number(timestamp[0])
            end = _number(timestamp[1])
    start = 0.0 if start is None else start
    if milliseconds:
        start /= 1000.0
        end = None if end is None else end / 1000.0
    if end is None or end <= start:
        end = (
            fallback_duration
            if fallback_duration is not None and fallback_duration > start
            else start + 0.001
        )
    return start, max(start + 0.001, end)


def _external_segment_id(
    source_id: str,
    block_id: str,
    raw_id: str | None,
    start: float,
    end: float,
    index: int,
) -> str:
    material = (
        f"{source_id}\0{block_id}\0external\0{raw_id}"
        if raw_id is not None
        else f"{source_id}\0{block_id}\0anonymous\0{start:.6f}\0{end:.6f}\0{index}"
    )
    return "seg-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def parse_command_output(
    raw: str,
    *,
    source_id: str,
    block_id: str,
    offset_seconds: float = 0.0,
    model_id: str = "external-command",
    model_revision: str | None = None,
    output_format: OutputFormat = "auto",
    time_unit: TimeUnit = "seconds",
    timestamps_are_absolute: bool = False,
    fallback_duration: float | None = None,
    require_end_event: RequireEnd = "auto",
) -> list[AsrSegment]:
    """Parse JSON/JSONL and retain the highest external revision per segment."""

    documents = _json_documents(raw, output_format)
    event_kinds: set[str] = set()
    for document in documents:
        event_kinds.update(_event_kinds(document))
    streaming = bool(event_kinds & {"commit", "replace", "end"})
    if (require_end_event is True or (require_end_event == "auto" and streaming)) and "end" not in event_kinds:
        raise CommandAdapterError("streaming ASR output is missing its terminal end event")

    candidates: list[Mapping[str, Any]] = []
    for document in documents:
        candidates.extend(_candidate_items(document))

    by_key: dict[str, tuple[int, int, Mapping[str, Any]]] = {}
    anonymous: list[tuple[int, Mapping[str, Any]]] = []
    for index, item in enumerate(candidates):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        raw_id_value = item.get("segment_id", item.get("id"))
        raw_id = str(raw_id_value) if raw_id_value is not None else None
        try:
            revision = max(1, int(item.get("revision", 1)))
        except (TypeError, ValueError):
            revision = 1
        if raw_id is None:
            anonymous.append((index, item))
        else:
            prior = by_key.get(raw_id)
            if prior is None or revision >= prior[0]:
                by_key[raw_id] = (revision, index, item)

    selected: list[tuple[int, Mapping[str, Any]]] = anonymous + [
        (index, item) for _, index, item in by_key.values()
    ]
    selected.sort(key=lambda pair: pair[0])

    segments: list[AsrSegment] = []
    for output_index, (raw_index, item) in enumerate(selected):
        relative_start, relative_end = _bounds(item, time_unit, fallback_duration)
        if timestamps_are_absolute:
            start, end = relative_start, relative_end
        else:
            start = offset_seconds + relative_start
            end = offset_seconds + relative_end
        metadata_value = item.get("metadata")
        external_metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        confidence = _number(
            item.get("confidence", external_metadata.get("confidence"))
        )
        language_value = item.get("language", external_metadata.get("language"))
        raw_id_value = item.get("segment_id", item.get("id"))
        raw_id = str(raw_id_value) if raw_id_value is not None else None
        try:
            revision = max(1, int(item.get("revision", 1)))
        except (TypeError, ValueError):
            revision = 1
        segments.append(
            AsrSegment(
                segment_id=_external_segment_id(
                    source_id,
                    block_id,
                    raw_id,
                    start,
                    end,
                    output_index,
                ),
                source_id=source_id,
                start=start,
                end=end,
                text=str(item.get("text") or "").strip(),
                revision=revision,
                model_id=model_id,
                model_revision=model_revision,
                confidence=confidence,
                language=str(language_value) if language_value else None,
                block_id=block_id,
                metadata={
                    "adapter": "command",
                    "external_segment_id": raw_id,
                    "external_kind": item.get("kind") or item.get("type"),
                    "raw_index": raw_index,
                    "external_metadata": external_metadata,
                },
            )
        )
    return segments


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            return source.getnframes() / rate if rate else None
    except (OSError, EOFError, wave.Error):
        return None


class CommandAsrBackend:
    """Run a local command without a shell and normalize its JSON output."""

    def __init__(self, config: CommandAdapterConfig) -> None:
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self._last_raw_output: bytes | None = None

    def _arguments(
        self,
        audio_path: Path,
        output_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float,
    ) -> list[str]:
        replacements = {
            "audio": str(audio_path),
            "output": str(output_path),
            "source_id": source_id,
            "block_id": block_id,
            "offset": f"{offset_seconds:.6f}",
        }
        args: list[str] = []
        has_audio = False
        for value in self.config.command:
            has_audio = has_audio or "{audio}" in value
            unknown = [
                name
                for name in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
                if name not in replacements
            ]
            if unknown:
                raise ValueError(f"unknown command placeholder: {unknown[0]}")
            rendered = value
            for name, replacement in replacements.items():
                rendered = rendered.replace("{" + name + "}", replacement)
            args.append(rendered)
        if not has_audio:
            args.append(str(audio_path))
        return args

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float = 0.0,
    ) -> list[AsrSegment]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory(prefix="dayaudio-command-") as directory:
            output_path = Path(directory) / "output.jsonl"
            args = self._arguments(
                path,
                output_path,
                source_id=source_id,
                block_id=block_id,
                offset_seconds=offset_seconds,
            )
            environment = os.environ.copy()
            environment.update(self.config.environment)
            environment.setdefault("PYTHONIOENCODING", "utf-8")
            environment.setdefault("PYTHONUTF8", "1")
            if self.config.offline:
                environment.setdefault("HF_HUB_OFFLINE", "1")
                environment.setdefault("TRANSFORMERS_OFFLINE", "1")
                environment.setdefault("HF_DATASETS_OFFLINE", "1")
            try:
                completed = subprocess.run(
                    args,
                    cwd=self.config.cwd,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise CommandAdapterError(
                    "ASR command timed out",
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                ) from exc
            except OSError as exc:
                raise CommandAdapterError(
                    f"ASR command could not start: {type(exc).__name__}"
                ) from exc
            if completed.returncode != 0:
                raise CommandAdapterError(
                    f"ASR command failed with exit code {completed.returncode}",
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            if not raw.strip():
                raw = completed.stdout
            if len(raw.encode("utf-8")) > self.config.max_output_bytes:
                raise CommandAdapterError("ASR command output exceeded configured size limit")
            self._last_raw_output = raw.encode("utf-8")
            return parse_command_output(
                raw,
                source_id=source_id,
                block_id=block_id,
                offset_seconds=offset_seconds,
                model_id=self.model_id,
                model_revision=self.model_revision,
                output_format=self.config.output_format,
                time_unit=self.config.time_unit,
                timestamps_are_absolute=self.config.timestamps_are_absolute,
                fallback_duration=_wav_duration(path),
                require_end_event=self.config.require_end_event,
            )

    def consume_raw_output(self) -> bytes | None:
        value, self._last_raw_output = self._last_raw_output, None
        return value

    def close(self) -> None:
        # Each command owns its own process.  The method satisfies AsrBackend.
        self._last_raw_output = None
        return None


CommandAdapter = CommandAsrBackend
CommandSpec = CommandAdapterConfig


__all__ = [
    "CommandAdapter",
    "CommandAdapterConfig",
    "CommandAdapterError",
    "CommandAsrBackend",
    "CommandSpec",
    "parse_command_output",
]
