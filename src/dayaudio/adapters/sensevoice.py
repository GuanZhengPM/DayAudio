"""Lazy resident SenseVoice and FSMN-VAD adapters.

``funasr`` is an optional dependency.  Merely importing DayAudio or creating an
adapter does not import it, construct a model, or access the network.  Model
construction occurs on the first inference call and the instance is then kept
resident until :meth:`close`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dayaudio.types import AsrSegment

DEFAULT_SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
DEFAULT_FSMN_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

_CONTROL_TOKEN = re.compile(r"<\|[^|>]+\|>")


class OptionalDependencyError(RuntimeError):
    pass


class OfflineModelUnavailableError(OptionalDependencyError):
    pass


@dataclass(frozen=True, slots=True)
class SenseVoiceConfig:
    model_id: str = DEFAULT_SENSEVOICE_MODEL
    model_revision: str | None = None
    vad_model_id: str | None = DEFAULT_FSMN_VAD_MODEL
    device: str = "cpu"
    language: str = "auto"
    use_itn: bool = True
    batch_size_seconds: int = 300
    disable_update: bool = True
    offline: bool = False
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    generate_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.batch_size_seconds <= 0:
            raise ValueError("batch_size_seconds must be positive")


@dataclass(frozen=True, slots=True)
class FsmnVadConfig:
    model_id: str = DEFAULT_FSMN_VAD_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    disable_update: bool = True
    offline: bool = False
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    generate_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")


ModelFactory = Callable[..., Any]

_WEIGHT_SUFFIXES = {
    ".bin",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


def _enable_offline_environment() -> None:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "MODELSCOPE_OFFLINE",
    ):
        os.environ[name] = "1"


def _verified_model_directory(path: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix.casefold() in _WEIGHT_SUFFIXES
            for item in path.rglob("*")
        )
    except OSError:
        return False


def _resolve_offline_model(model_id: str) -> Path | None:
    direct = Path(model_id).expanduser()
    if direct.is_file():
        return direct.resolve()
    if direct.is_dir() and _verified_model_directory(direct):
        return direct.resolve()

    roots = [
        Path(value).expanduser()
        for value in (
            os.environ.get("MODELSCOPE_CACHE"),
            os.environ.get("MODELSCOPE_HOME"),
            os.environ.get("HF_HOME"),
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            os.environ.get("TRANSFORMERS_CACHE"),
        )
        if value
    ]
    roots.extend(root / "hub" for root in tuple(roots))
    roots.extend(
        (
            Path.home() / ".cache" / "modelscope" / "hub",
            Path.home() / ".cache" / "modelscope",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    )
    organization, _, name = model_id.partition("/")
    relative_candidates = (
        Path(model_id),
        Path("models") / model_id,
        Path(f"models--{organization}--{name}"),
    )
    for root in roots:
        for relative in relative_candidates:
            candidate = root / relative
            if candidate.is_dir() and _verified_model_directory(candidate):
                snapshots = candidate / "snapshots"
                if snapshots.is_dir():
                    valid = [
                        item
                        for item in snapshots.iterdir()
                        if item.is_dir() and _verified_model_directory(item)
                    ]
                    if valid:
                        return sorted(valid, key=lambda item: item.name)[-1].resolve()
                return candidate.resolve()
    return None


def _raw_json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def _serialize_raw_payload(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_raw_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        return json.dumps(
            _raw_json_default(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


def _default_factory(**kwargs: Any) -> Any:
    try:
        from funasr import AutoModel  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise OptionalDependencyError(
            "SenseVoice requires the optional 'fast' dependencies; "
            "install DayAudio with `pip install dayaudio[fast]` and cache the "
            "model before offline use"
        ) from exc
    return AutoModel(**kwargs)


def _clean_text(value: Any) -> str:
    text = _CONTROL_TOKEN.sub("", str(value or ""))
    return " ".join(text.split()).strip()


def _stable_segment_id(
    source_id: str, block_id: str, start: float, end: float, index: int
) -> str:
    material = f"{source_id}\0{block_id}\0{start:.6f}\0{end:.6f}\0{index}"
    return "seg-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, EOFError, wave.Error):
        return None


def _seconds(value: Any, *, milliseconds: bool) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number / 1000.0 if milliseconds else number


def _time_bounds(item: Mapping[str, Any]) -> tuple[float | None, float | None]:
    for start_key, end_key, milliseconds in (
        ("start_ms", "end_ms", True),
        ("begin_ms", "finish_ms", True),
        ("start_time", "end_time", True),
        # FunASR sentence_info uses integer millisecond start/end fields.
        ("start", "end", True),
    ):
        if start_key in item and end_key in item:
            return (
                _seconds(item[start_key], milliseconds=milliseconds),
                _seconds(item[end_key], milliseconds=milliseconds),
            )

    timestamps = item.get("timestamp") or item.get("timestamps")
    if isinstance(timestamps, (list, tuple)) and timestamps:
        first = timestamps[0]
        last = timestamps[-1]
        if (
            isinstance(first, (list, tuple))
            and len(first) >= 2
            and isinstance(last, (list, tuple))
            and len(last) >= 2
        ):
            return (
                _seconds(first[0], milliseconds=True),
                _seconds(last[1], milliseconds=True),
            )
        if len(timestamps) >= 2 and not isinstance(first, (list, tuple, dict)):
            return (
                _seconds(timestamps[0], milliseconds=True),
                _seconds(timestamps[1], milliseconds=True),
            )
    return None, None


def _result_items(payload: Any) -> list[Mapping[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        for key in ("sentence_info", "sentences", "segments"):
            nested = payload.get(key)
            if isinstance(nested, list) and nested:
                return [item for item in nested if isinstance(item, Mapping)]
        return [payload]
    if isinstance(payload, list):
        items: list[Mapping[str, Any]] = []
        for value in payload:
            if not isinstance(value, Mapping):
                continue
            nested = _result_items(value)
            items.extend(nested)
        return items
    return []


def parse_sensevoice_result(
    payload: Any,
    *,
    source_id: str,
    block_id: str,
    offset_seconds: float,
    model_id: str,
    model_revision: str | None,
    fallback_duration: float | None = None,
) -> list[AsrSegment]:
    """Normalize common FunASR result envelopes into stable segments."""

    result: list[AsrSegment] = []
    items = _result_items(payload)
    for index, item in enumerate(items):
        text = _clean_text(item.get("text") or item.get("value_text"))
        if not text:
            continue
        relative_start, relative_end = _time_bounds(item)
        if relative_start is None:
            relative_start = 0.0
        if relative_end is None or relative_end <= relative_start:
            if fallback_duration is not None and fallback_duration > relative_start:
                relative_end = fallback_duration
            else:
                relative_end = relative_start + 0.001
        start = offset_seconds + max(0.0, relative_start)
        end = offset_seconds + max(relative_start + 0.001, relative_end)
        confidence: float | None = None
        for key in ("confidence", "score", "probability"):
            if key in item:
                try:
                    confidence = float(item[key])
                except (TypeError, ValueError):
                    confidence = None
                break
        result.append(
            AsrSegment(
                segment_id=_stable_segment_id(source_id, block_id, start, end, index),
                source_id=source_id,
                start=start,
                end=end,
                text=text,
                model_id=model_id,
                model_revision=model_revision,
                confidence=confidence,
                language=str(item.get("lang") or item.get("language") or "") or None,
                block_id=block_id,
                metadata={
                    "adapter": "sensevoice-fsmn",
                    "raw_index": index,
                },
            )
        )
    return result


class SenseVoiceBackend:
    """Resident ASR backend with optional built-in FSMN segmentation."""

    name = "sensevoice-fsmn"

    def __init__(
        self,
        config: SenseVoiceConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or SenseVoiceConfig()
        self.model_id = self.config.model_id
        self.model_revision = self.config.model_revision
        self._factory = model_factory or _default_factory
        self._model: Any | None = None
        self._last_raw_payload: Any | None = None
        self.model_path = (
            _resolve_offline_model(self.config.model_id)
            if self.config.offline
            else None
        )
        self.vad_model_path = (
            _resolve_offline_model(self.config.vad_model_id)
            if self.config.offline and self.config.vad_model_id
            else None
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is None:
            model_reference: str = self.config.model_id
            vad_reference: str | None = self.config.vad_model_id
            if self.config.offline:
                _enable_offline_environment()
                resolved_model = self.model_path or _resolve_offline_model(
                    self.config.model_id
                )
                if resolved_model is None:
                    raise OfflineModelUnavailableError(
                        f"offline model is not cached locally: {self.config.model_id}"
                    )
                model_reference = str(resolved_model)
                if self.config.vad_model_id:
                    resolved_vad = self.vad_model_path or _resolve_offline_model(
                        self.config.vad_model_id
                    )
                    if resolved_vad is None:
                        raise OfflineModelUnavailableError(
                            f"offline VAD model is not cached locally: {self.config.vad_model_id}"
                        )
                    vad_reference = str(resolved_vad)
            kwargs: dict[str, Any] = dict(self.config.model_kwargs)
            kwargs.update({
                "model": model_reference,
                "device": self.config.device,
                "disable_update": self.config.disable_update,
            })
            if self.config.model_revision is not None:
                kwargs["model_revision"] = self.config.model_revision
            if vad_reference:
                kwargs["vad_model"] = vad_reference
            self._model = self._factory(**kwargs)
        return self._model

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
        model = self._ensure_model()
        kwargs: dict[str, Any] = dict(self.config.generate_kwargs)
        kwargs.update({
            "input": str(path),
            "cache": {},
            "language": self.config.language,
            "use_itn": self.config.use_itn,
            "batch_size_s": self.config.batch_size_seconds,
        })
        payload = model.generate(**kwargs)
        self._last_raw_payload = payload
        return parse_sensevoice_result(
            payload,
            source_id=source_id,
            block_id=block_id,
            offset_seconds=offset_seconds,
            model_id=self.model_id,
            model_revision=self.model_revision,
            fallback_duration=_wav_duration(path),
        )

    def consume_raw_output(self) -> bytes | None:
        payload, self._last_raw_payload = self._last_raw_payload, None
        return None if payload is None else _serialize_raw_payload(payload)

    def close(self) -> None:
        model, self._model = self._model, None
        self._last_raw_payload = None
        close = getattr(model, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "SenseVoiceBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _iter_vad_regions(payload: Any) -> Iterable[tuple[float, float]]:
    values: Any = payload
    if isinstance(payload, list) and payload and all(
        isinstance(item, Mapping) for item in payload
    ):
        for item in payload:
            yield from _iter_vad_regions(item)
        return
    elif isinstance(payload, Mapping):
        values = payload.get("value") or payload.get("segments") or []
    if not isinstance(values, list):
        return
    for item in values:
        if isinstance(item, Mapping):
            start, end = _time_bounds(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = _seconds(item[0], milliseconds=True)
            end = _seconds(item[1], milliseconds=True)
        else:
            continue
        if start is not None and end is not None and end > start >= 0:
            yield start, end


def _merge_regions(regions: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(regions):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


class FsmnVadBackend:
    name = "fsmn-vad"

    def __init__(
        self,
        config: FsmnVadConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or FsmnVadConfig()
        self._factory = model_factory or _default_factory
        self._model: Any | None = None
        self._last_raw_payload: Any | None = None
        self.model_path = (
            _resolve_offline_model(self.config.model_id)
            if self.config.offline
            else None
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is None:
            model_reference = self.config.model_id
            if self.config.offline:
                _enable_offline_environment()
                resolved = self.model_path or _resolve_offline_model(
                    self.config.model_id
                )
                if resolved is None:
                    raise OfflineModelUnavailableError(
                        f"offline VAD model is not cached locally: {self.config.model_id}"
                    )
                model_reference = str(resolved)
            kwargs: dict[str, Any] = dict(self.config.model_kwargs)
            kwargs.update({
                "model": model_reference,
                "device": self.config.device,
                "disable_update": self.config.disable_update,
            })
            if self.config.model_revision is not None:
                kwargs["model_revision"] = self.config.model_revision
            self._model = self._factory(**kwargs)
        return self._model

    def speech_regions(self, audio_path: Path) -> list[tuple[float, float]]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs: dict[str, Any] = dict(self.config.generate_kwargs)
        kwargs.update({"input": str(path), "cache": {}})
        payload = self._ensure_model().generate(**kwargs)
        self._last_raw_payload = payload
        return _merge_regions(_iter_vad_regions(payload))

    def consume_raw_output(self) -> bytes | None:
        payload, self._last_raw_payload = self._last_raw_payload, None
        return None if payload is None else _serialize_raw_payload(payload)

    def close(self) -> None:
        model, self._model = self._model, None
        self._last_raw_payload = None
        close = getattr(model, "close", None)
        if close is not None:
            close()


class SenseVoiceFsmnBackend(SenseVoiceBackend):
    """Convenience backend exposing both ASR and an independently resident VAD."""

    def __init__(
        self,
        config: SenseVoiceConfig | None = None,
        *,
        vad_config: FsmnVadConfig | None = None,
        model_factory: ModelFactory | None = None,
        vad_model_factory: ModelFactory | None = None,
    ) -> None:
        super().__init__(config, model_factory=model_factory)
        effective = self.config
        self.vad = FsmnVadBackend(
            vad_config
            or FsmnVadConfig(
                model_id=effective.vad_model_id or DEFAULT_FSMN_VAD_MODEL,
                device="cpu",
                disable_update=effective.disable_update,
                offline=effective.offline,
            ),
            model_factory=vad_model_factory or model_factory,
        )

    def speech_regions(self, audio_path: Path) -> list[tuple[float, float]]:
        return self.vad.speech_regions(audio_path)

    def close(self) -> None:
        super().close()
        self.vad.close()


# Friendly aliases used by older experiment scripts and third-party examples.
SenseVoiceAdapter = SenseVoiceBackend
SenseVoiceFsmnAdapter = SenseVoiceFsmnBackend


__all__ = [
    "DEFAULT_FSMN_VAD_MODEL",
    "DEFAULT_SENSEVOICE_MODEL",
    "FsmnVadBackend",
    "FsmnVadConfig",
    "OfflineModelUnavailableError",
    "OptionalDependencyError",
    "SenseVoiceAdapter",
    "SenseVoiceBackend",
    "SenseVoiceConfig",
    "SenseVoiceFsmnAdapter",
    "SenseVoiceFsmnBackend",
    "parse_sensevoice_result",
]
