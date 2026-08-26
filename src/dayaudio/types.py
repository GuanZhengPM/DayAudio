"""Stable cross-module data contracts.

The SQLite layer may add storage-only columns, but these public records remain
portable and JSON serializable. Times are always source-relative seconds unless
an explicitly named absolute timestamp field is used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    source_sha256: str
    source_path: str
    source_name: str
    size_bytes: int
    duration_seconds: float | None = None
    decoded_duration_seconds: float | None = None
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    recording_start: str | None = None
    recording_time_basis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AudioBlock:
    block_id: str
    source_id: str
    source_sha256: str
    core_start: float
    core_end: float
    context_start: float
    context_end: float
    pcm_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AsrSegment:
    segment_id: str
    source_id: str
    start: float
    end: float
    text: str
    revision: int = 1
    model_id: str | None = None
    model_revision: str | None = None
    confidence: float | None = None
    language: str | None = None
    block_id: str | None = None
    anomaly_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("ASR segment end must be greater than start")
        if not self.text.strip():
            raise ValueError("ASR segment text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["anomaly_flags"] = list(self.anomaly_flags)
        return result


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    turn_id: str
    source_id: str
    local_speaker_id: str
    start: float
    end: float
    model_digest: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("speaker turn end must be greater than start")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EvidenceConfidence = Literal["high", "medium", "review"]
ParticipantRole = Literal["owner", "anonymous", "mixed", "unknown"]


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    evidence_window_id: str
    source_id: str
    start: float
    end: float
    text: str
    confidence: EvidenceConfidence
    model_state: str
    summary_sensitive: bool = False
    segment_ids: tuple[str, ...] = ()
    participant_role: ParticipantRole = "unknown"
    identity_id: str | None = None
    review_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_window_id or not self.source_id:
            raise ValueError("evidence and source ids must not be empty")
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.end <= self.start:
            raise ValueError("evidence range must be finite with end greater than start")
        if self.start < 0:
            raise ValueError("evidence start must not be negative")
        if not self.text.strip():
            raise ValueError("evidence text must not be empty")
        if self.confidence not in {"high", "medium", "review"}:
            raise ValueError("unsupported evidence confidence")
        if self.participant_role not in {"owner", "anonymous", "mixed", "unknown"}:
            raise ValueError("unsupported evidence participant role")
        if self.identity_id is not None and self.participant_role != "owner":
            raise ValueError("only owner evidence may carry an identity_id")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["segment_ids"] = list(self.segment_ids)
        result["review_reasons"] = list(self.review_reasons)
        return result
