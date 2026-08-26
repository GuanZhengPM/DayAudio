"""Lazy CAM++ speaker-embedding adapter.

The adapter accepts already isolated WAV clips.  Diarization window selection
and clustering stay in :mod:`dayaudio.speaker`, which keeps identity logic
testable without importing a model runtime.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dayaudio.speaker import SpeakerWindow

DEFAULT_CAMPLUS_MODEL = "cam++"


class OptionalSpeakerDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CampPlusConfig:
    model_id: str = DEFAULT_CAMPLUS_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    hub: str = "ms"
    disable_update: bool = True
    offline: bool = False
    weight_sha256: str | None = None
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    generate_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight_sha256 is not None:
            value = self.weight_sha256.lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("weight_sha256 must be 64 hexadecimal characters")


ModelFactory = Callable[..., Any]


def _default_factory(**kwargs: Any) -> Any:
    try:
        from funasr import AutoModel  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise OptionalSpeakerDependencyError(
            "CAM++ requires the optional 'speaker' dependencies; install "
            "DayAudio with `pip install dayaudio[speaker]` and cache the model "
            "before offline use"
        ) from exc
    return AutoModel(**kwargs)


def _plain_vector(value: Any) -> tuple[float, ...]:
    current = value
    for method in ("detach", "flatten", "float", "cpu"):
        operation = getattr(current, method, None)
        if operation is not None:
            current = operation()
    if hasattr(current, "tolist"):
        current = current.tolist()
    while isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
        current = current[0]
    if not isinstance(current, (list, tuple)):
        raise RuntimeError("CAM++ returned an unsupported embedding value")
    vector = tuple(float(item) for item in current)
    if not vector or not all(math.isfinite(item) for item in vector):
        raise RuntimeError("CAM++ returned an empty or non-finite embedding")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        raise RuntimeError("CAM++ returned a zero embedding")
    return tuple(item / norm for item in vector)


def _embedding_from_result(value: Any) -> tuple[float, ...]:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, Mapping):
        for key in ("spk_embedding", "speaker_embedding", "embedding"):
            if key in value:
                return _plain_vector(value[key])
    raise RuntimeError("CAM++ returned no speaker embedding")


class CampPlusBackend:
    """Resident FunASR CAM++ backend for clip-level embeddings."""

    name = "campplus"

    def __init__(
        self,
        config: CampPlusConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or CampPlusConfig()
        self._factory = model_factory or _default_factory
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_digest(self) -> str:
        if self.config.weight_sha256:
            return self.config.weight_sha256.lower()
        material = (
            f"{self.config.model_id}\0{self.config.model_revision or ''}\0"
            f"{self.config.hub}\0{self.config.device}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _ensure_model(self) -> Any:
        if self._model is None:
            if self.config.offline:
                local = Path(self.config.model_id).expanduser()
                if not local.exists():
                    raise FileNotFoundError(
                        "offline CAM++ requires --speaker-model to reference local cached weights"
                    )
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
                os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
            kwargs: dict[str, Any] = {
                "model": self.config.model_id,
                "device": self.config.device,
                "hub": self.config.hub,
                "disable_update": self.config.disable_update,
            }
            if self.config.model_revision is not None:
                kwargs["model_revision"] = self.config.model_revision
            kwargs.update(dict(self.config.model_kwargs))
            self._model = self._factory(**kwargs)
        return self._model

    def embed_audio(self, audio_path: str | Path) -> tuple[float, ...]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs: dict[str, Any] = {"input": str(path), "disable_pbar": True}
        kwargs.update(dict(self.config.generate_kwargs))
        return _embedding_from_result(self._ensure_model().generate(**kwargs))

    def embed(self, windows: Sequence[SpeakerWindow]) -> list[tuple[float, ...]]:
        embeddings: list[tuple[float, ...]] = []
        for window in windows:
            payload = window.payload
            if not isinstance(payload, (str, Path)):
                raise ValueError(
                    "CAM++ windows must carry a path to an isolated audio clip in payload"
                )
            embeddings.append(self.embed_audio(payload))
        return embeddings

    def close(self) -> None:
        model, self._model = self._model, None
        close = getattr(model, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "CampPlusBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "CampPlusBackend",
    "CampPlusConfig",
    "OptionalSpeakerDependencyError",
]
