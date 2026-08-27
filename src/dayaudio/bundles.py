"""Conservative day grouping and fixed-duration summary packetization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .cas import atomic_write_bytes
from .paths import filesystem_path
from .types import EvidenceWindow, SourceRecord

TRUSTED_RECORDING_TIME_BASES = frozenset(
    {
        "user",
        "user_provided",
        "explicit",
        "embedded",
        "container",
        "container_metadata",
        "metadata",
        "recording_metadata",
        "exif",
        "creation_time",
    }
)


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid recording_start timestamp: {value!r}") from exc
    return parsed


def _zone(timezone_name: str | None) -> ZoneInfo | None:
    if timezone_name is None:
        return None
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {timezone_name}") from exc


@dataclass(frozen=True, slots=True)
class DayAssignment:
    source_id: str
    day_key: str
    trusted: bool
    recording_start: datetime | None
    reason: str


def conservative_day_assignment(
    source: SourceRecord,
    *,
    timezone_name: str | None = None,
    trusted_time_bases: frozenset[str] = TRUSTED_RECORDING_TIME_BASES,
) -> DayAssignment:
    """Use explicit recording metadata only; never infer a day from a filename."""

    if not source.recording_start:
        return DayAssignment(
            source.source_id,
            f"undated-{source.source_id}",
            False,
            None,
            "missing_recording_start",
        )
    basis = (source.recording_time_basis or "").strip().casefold().replace("-", "_")
    trusted_basis = basis in trusted_time_bases or any(
        basis.startswith(prefix + ":") for prefix in trusted_time_bases
    )
    if not trusted_basis:
        return DayAssignment(
            source.source_id,
            f"undated-{source.source_id}",
            False,
            None,
            f"untrusted_recording_time_basis:{basis or 'missing'}",
        )
    try:
        parsed = _parse_datetime(source.recording_start)
    except ValueError:
        return DayAssignment(
            source.source_id,
            f"undated-{source.source_id}",
            False,
            None,
            "invalid_recording_start",
        )
    target_zone = _zone(timezone_name)
    if target_zone is not None:
        if parsed.tzinfo is None:
            # The supplied timezone is an explicit interpretation, not a guess.
            parsed = parsed.replace(tzinfo=target_zone)
        else:
            parsed = parsed.astimezone(target_zone)
    return DayAssignment(
        source.source_id,
        f"day-{parsed.date().isoformat()}",
        True,
        parsed,
        "trusted_recording_start",
    )


# Short alias for call sites that need only a key.
def conservative_day_key(source: SourceRecord, *, timezone_name: str | None = None) -> str:
    return conservative_day_assignment(source, timezone_name=timezone_name).day_key


@dataclass(frozen=True, slots=True)
class BundleSource:
    source_id: str
    source_name: str
    sequence: int
    timeline_start: float
    timeline_end: float
    absolute_recording_start: str | None
    time_basis: str | None
    timing_trusted: bool

    def __post_init__(self) -> None:
        if self.timeline_end < self.timeline_start:
            raise ValueError("bundle source timeline end precedes start")


@dataclass(frozen=True, slots=True)
class DayBundle:
    bundle_id: str
    day_key: str
    sources: tuple[BundleSource, ...]
    evidence_window_ids: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)

    @property
    def duration_seconds(self) -> float:
        if not self.sources:
            return 0.0
        return max(source.timeline_end for source in self.sources)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sources"] = [asdict(source) for source in self.sources]
        result["evidence_window_ids"] = list(self.evidence_window_ids)
        result["review_reasons"] = list(self.review_reasons)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DayBundle":
        return cls(
            bundle_id=str(data["bundle_id"]),
            day_key=str(data["day_key"]),
            sources=tuple(BundleSource(**dict(item)) for item in data.get("sources", ())),
            evidence_window_ids=tuple(str(item) for item in data.get("evidence_window_ids", ())),
            review_reasons=tuple(str(item) for item in data.get("review_reasons", ())),
        )


def _source_duration(source: SourceRecord) -> float:
    duration = (
        source.decoded_duration_seconds
        if source.decoded_duration_seconds is not None
        else source.duration_seconds
    )
    return max(0.0, float(duration or 0.0))


def _seconds_from_day_start(value: datetime) -> float:
    if value.tzinfo is None:
        midnight = datetime.combine(value.date(), time.min)
    else:
        midnight = datetime.combine(value.date(), time.min, tzinfo=value.tzinfo)
    return (value - midnight).total_seconds()


def _bundle_id(day_key: str, source_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(day_key.encode())
    for source_id in source_ids:
        digest.update(b"\0")
        digest.update(source_id.encode())
    return f"bundle-{digest.hexdigest()[:20]}"


def build_day_bundles(
    sources: Iterable[SourceRecord],
    evidence_windows: Iterable[EvidenceWindow] = (),
    *,
    timezone_name: str | None = None,
) -> tuple[DayBundle, ...]:
    """Group only sources with trusted, same-day recording timestamps.

    Every undated or weakly dated source receives its own bundle.  This avoids
    silently merging files based on filenames, modification time, or input
    order.
    """

    source_rows = list(sources)
    source_by_id: dict[str, SourceRecord] = {}
    for source in source_rows:
        if source.source_id in source_by_id:
            raise ValueError(f"duplicate source id: {source.source_id}")
        source_by_id[source.source_id] = source
    evidence_by_source: dict[str, list[EvidenceWindow]] = {}
    evidence_ids: set[str] = set()
    for window in evidence_windows:
        if window.source_id not in source_by_id:
            raise ValueError(f"evidence references an unknown source: {window.source_id}")
        if window.evidence_window_id in evidence_ids:
            raise ValueError(f"duplicate evidence window id: {window.evidence_window_id}")
        evidence_ids.add(window.evidence_window_id)
        evidence_by_source.setdefault(window.source_id, []).append(window)

    assignments = {
        source.source_id: conservative_day_assignment(source, timezone_name=timezone_name)
        for source in source_rows
    }
    grouped: dict[tuple[str, str], list[SourceRecord]] = {}
    for source in source_rows:
        assignment = assignments[source.source_id]
        parsed = assignment.recording_start
        if not assignment.trusted or parsed is None:
            zone_key = source.source_id
        elif timezone_name is not None:
            zone_key = timezone_name
        elif parsed.tzinfo is None:
            zone_key = "naive-local-time"
        else:
            zone_key = str(parsed.utcoffset())
        grouped.setdefault((assignment.day_key, zone_key), []).append(source)

    bundles: list[DayBundle] = []
    for (day_key, _zone_key), group in grouped.items():
        group.sort(
            key=lambda source: (
                (
                    _seconds_from_day_start(assignments[source.source_id].recording_start)
                    if assignments[source.source_id].recording_start is not None
                    else math.inf
                ),
                (
                    assignments[source.source_id].recording_start.isoformat()
                    if assignments[source.source_id].recording_start is not None
                    else ""
                ),
                source.source_id,
            )
        )
        bundle_sources: list[BundleSource] = []
        review_reasons: list[str] = []
        for sequence, source in enumerate(group):
            assignment = assignments[source.source_id]
            duration = _source_duration(source)
            if assignment.trusted and assignment.recording_start is not None:
                timeline_start = _seconds_from_day_start(assignment.recording_start)
            else:
                # Undated bundles contain one source, making a zero origin safe.
                timeline_start = 0.0
                review_reasons.append(assignment.reason)
            timeline_end = timeline_start + duration
            if timeline_end > 24 * 60 * 60 and assignment.trusted:
                review_reasons.append(f"source_crosses_day_boundary:{source.source_id}")
            bundle_sources.append(
                BundleSource(
                    source_id=source.source_id,
                    source_name=source.source_name,
                    sequence=sequence,
                    timeline_start=timeline_start,
                    timeline_end=timeline_end,
                    absolute_recording_start=(
                        assignment.recording_start.isoformat()
                        if assignment.recording_start is not None
                        else None
                    ),
                    time_basis=source.recording_time_basis,
                    timing_trusted=assignment.trusted,
                )
            )
        for previous, current in zip(bundle_sources, bundle_sources[1:]):
            if current.timeline_start < previous.timeline_end:
                review_reasons.append(
                    f"source_timeline_overlap:{previous.source_id}:{current.source_id}"
                )
        ordered_ids = tuple(source.source_id for source in group)
        bundle_evidence = sorted(
            (
                window
                for source_id in ordered_ids
                for window in evidence_by_source.get(source_id, ())
            ),
            key=lambda window: (
                next(
                    item.timeline_start
                    for item in bundle_sources
                    if item.source_id == window.source_id
                )
                + window.start,
                window.evidence_window_id,
            ),
        )
        bundles.append(
            DayBundle(
                bundle_id=_bundle_id(day_key, ordered_ids),
                day_key=day_key,
                sources=tuple(bundle_sources),
                evidence_window_ids=tuple(window.evidence_window_id for window in bundle_evidence),
                review_reasons=tuple(dict.fromkeys(review_reasons)),
            )
        )
    return tuple(sorted(bundles, key=lambda bundle: (bundle.day_key, bundle.bundle_id)))


@dataclass(frozen=True, slots=True)
class SummaryPacket:
    packet_id: str
    bundle_id: str
    day_key: str
    packet_index: int
    start: float
    end: float
    source_ids: tuple[str, ...]
    evidence_window_ids: tuple[str, ...]
    text: str
    owner_evidence_count: int = 0
    review_evidence_count: int = 0

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("summary packet end must be greater than start")
        if not self.evidence_window_ids:
            raise ValueError("summary packet must contain evidence")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_ids"] = list(self.source_ids)
        result["evidence_window_ids"] = list(self.evidence_window_ids)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SummaryPacket":
        return cls(
            packet_id=str(data["packet_id"]),
            bundle_id=str(data["bundle_id"]),
            day_key=str(data["day_key"]),
            packet_index=int(data["packet_index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            source_ids=tuple(str(item) for item in data.get("source_ids", ())),
            evidence_window_ids=tuple(str(item) for item in data.get("evidence_window_ids", ())),
            text=str(data.get("text", "")),
            owner_evidence_count=int(data.get("owner_evidence_count", 0)),
            review_evidence_count=int(data.get("review_evidence_count", 0)),
        )


def _packet_id(bundle_id: str, packet_index: int, evidence_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(f"{bundle_id}\0{packet_index}".encode())
    for evidence_id in evidence_ids:
        digest.update(b"\0")
        digest.update(evidence_id.encode())
    return f"packet-{digest.hexdigest()[:20]}"


def build_summary_packets(
    bundle: DayBundle,
    evidence_windows: Iterable[EvidenceWindow],
    *,
    packet_seconds: float = 15 * 60,
) -> tuple[SummaryPacket, ...]:
    """Partition one bundle into deterministic 15-minute evidence packets."""

    if packet_seconds <= 0:
        raise ValueError("packet_seconds must be positive")
    source_offsets = {source.source_id: source.timeline_start for source in bundle.sources}
    allowed_ids = set(bundle.evidence_window_ids)
    evidence_rows: list[tuple[float, float, EvidenceWindow]] = []
    seen: set[str] = set()
    for window in evidence_windows:
        if window.source_id not in source_offsets:
            continue
        if allowed_ids and window.evidence_window_id not in allowed_ids:
            continue
        if window.evidence_window_id in seen:
            raise ValueError(f"duplicate evidence window id: {window.evidence_window_id}")
        seen.add(window.evidence_window_id)
        offset = source_offsets[window.source_id]
        evidence_rows.append((offset + window.start, offset + window.end, window))
    evidence_rows.sort(key=lambda row: (row[0], row[1], row[2].evidence_window_id))
    if allowed_ids and seen != allowed_ids:
        missing = sorted(allowed_ids - seen)
        raise ValueError(f"bundle evidence is missing from packet input: {missing[0]}")

    grouped: dict[int, list[tuple[float, float, EvidenceWindow]]] = {}
    for absolute_start, absolute_end, window in evidence_rows:
        packet_index = math.floor(absolute_start / packet_seconds)
        grouped.setdefault(packet_index, []).append((absolute_start, absolute_end, window))

    packets: list[SummaryPacket] = []
    for packet_index, rows in sorted(grouped.items()):
        windows = [row[2] for row in rows]
        evidence_ids = tuple(window.evidence_window_id for window in windows)
        packet_start = packet_index * packet_seconds
        packet_end = packet_start + packet_seconds
        packets.append(
            SummaryPacket(
                packet_id=_packet_id(bundle.bundle_id, packet_index, evidence_ids),
                bundle_id=bundle.bundle_id,
                day_key=bundle.day_key,
                packet_index=packet_index,
                start=packet_start,
                end=packet_end,
                source_ids=tuple(dict.fromkeys(window.source_id for window in windows)),
                evidence_window_ids=evidence_ids,
                text="\n".join(
                    f"[{window.evidence_window_id}] {window.text}" for window in windows
                ),
                owner_evidence_count=sum(
                    window.participant_role == "owner" for window in windows
                ),
                review_evidence_count=sum(window.confidence == "review" for window in windows),
            )
        )
    return tuple(packets)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(destination, encoded)


def write_day_bundles(path: str | Path, bundles: Iterable[DayBundle]) -> None:
    rows = tuple(bundles)
    if len({bundle.bundle_id for bundle in rows}) != len(rows):
        raise ValueError("cannot write duplicate day bundle ids")
    _atomic_write_json(
        path,
        {
            "schema_version": "dayaudio.day_bundles.v1",
            "bundles": [bundle.to_dict() for bundle in rows],
        },
    )


def read_day_bundles(path: str | Path) -> tuple[DayBundle, ...]:
    with filesystem_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "dayaudio.day_bundles.v1":
        raise ValueError("unsupported day bundle document")
    rows = payload.get("bundles")
    if not isinstance(rows, list):
        raise ValueError("day bundle document must contain a bundles list")
    bundles = tuple(DayBundle.from_dict(item) for item in rows)
    if len({bundle.bundle_id for bundle in bundles}) != len(bundles):
        raise ValueError("day bundle document contains duplicate ids")
    return bundles


def write_summary_packets(path: str | Path, packets: Iterable[SummaryPacket]) -> None:
    rows = tuple(packets)
    if len({packet.packet_id for packet in rows}) != len(rows):
        raise ValueError("cannot write duplicate summary packet ids")
    _atomic_write_json(
        path,
        {
            "schema_version": "dayaudio.summary_packets.v1",
            "packets": [packet.to_dict() for packet in rows],
        },
    )


def read_summary_packets(path: str | Path) -> tuple[SummaryPacket, ...]:
    with filesystem_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "dayaudio.summary_packets.v1":
        raise ValueError("unsupported summary packet document")
    rows = payload.get("packets")
    if not isinstance(rows, list):
        raise ValueError("summary packet document must contain a packets list")
    packets = tuple(SummaryPacket.from_dict(item) for item in rows)
    if len({packet.packet_id for packet in packets}) != len(packets):
        raise ValueError("summary packet document contains duplicate ids")
    return packets


packetize_summary = build_summary_packets
group_day_bundles = build_day_bundles


__all__ = [
    "BundleSource",
    "DayAssignment",
    "DayBundle",
    "SummaryPacket",
    "TRUSTED_RECORDING_TIME_BASES",
    "build_day_bundles",
    "build_summary_packets",
    "conservative_day_assignment",
    "conservative_day_key",
    "group_day_bundles",
    "packetize_summary",
    "read_day_bundles",
    "read_summary_packets",
    "write_day_bundles",
    "write_summary_packets",
]
