"""Deterministic, file-local speaker utilities.

This module intentionally does not decode audio or import a model runtime.  A
speaker backend receives already selected VAD windows and returns one embedding
per window.  The remaining clustering and overlap logic is dependency free so
it can be tested, cached, and resumed independently from ASR.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from .types import ParticipantRole, SpeakerTurn

Vector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SpeakerWindow:
    """A VAD-selected interval submitted to an embedding backend.

    ``payload`` is deliberately opaque.  An adapter may place PCM samples, a
    content-addressed audio reference, or its own lazy loader in it.
    """

    window_id: str
    source_id: str
    start: float
    end: float
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("speaker window id must not be empty")
        if not self.source_id:
            raise ValueError("speaker window source id must not be empty")
        if self.end <= self.start:
            raise ValueError("speaker window end must be greater than start")


@dataclass(frozen=True, slots=True)
class EmbeddedSpeakerWindow:
    window_id: str
    source_id: str
    start: float
    end: float
    embedding: Vector
    quality: float | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("embedded window end must be greater than start")
        if not self.embedding:
            raise ValueError("speaker embedding must not be empty")
        if not all(math.isfinite(value) for value in self.embedding):
            raise ValueError("speaker embedding contains a non-finite value")
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError("embedding quality must be between 0 and 1")


class EmbeddingBackend(Protocol):
    """Minimal adapter contract for CAM++, ECAPA, WeSpeaker, and similar models."""

    @property
    def model_digest(self) -> str:
        """Stable digest binding the model weights and embedding configuration."""

    def embed(self, windows: Sequence[SpeakerWindow]) -> Sequence[Sequence[float]]:
        """Return exactly one fixed-dimensional embedding per input window."""


def vad_intervals_to_speaker_windows(
    source_id: str,
    intervals: Iterable[tuple[float, float]],
    *,
    window_seconds: float = 1.5,
    hop_seconds: float = 0.75,
    minimum_seconds: float = 0.5,
) -> tuple[SpeakerWindow, ...]:
    """Split VAD intervals into deterministic embedding windows.

    No waveform operation occurs here.  Adapters can use the returned source
    ranges to fetch PCM independently, which keeps speaker work resumable from
    existing VAD artifacts.
    """

    if not source_id:
        raise ValueError("source_id must not be empty")
    if window_seconds <= 0 or hop_seconds <= 0 or minimum_seconds <= 0:
        raise ValueError("speaker window, hop, and minimum durations must be positive")
    if minimum_seconds > window_seconds:
        raise ValueError("minimum_seconds must not exceed window_seconds")
    ranges = sorted((float(start), float(end)) for start, end in intervals)
    windows: list[SpeakerWindow] = []
    seen_ranges: set[tuple[int, int]] = set()
    for start, end in ranges:
        if start < 0 or end <= start:
            raise ValueError("VAD intervals must satisfy 0 <= start < end")
        duration = end - start
        if duration < minimum_seconds:
            continue
        candidates: list[tuple[float, float]] = []
        if duration <= window_seconds:
            candidates.append((start, end))
        else:
            cursor = start
            while cursor + window_seconds <= end + 1e-9:
                candidates.append((cursor, min(end, cursor + window_seconds)))
                cursor += hop_seconds
            # Ensure the tail is represented even when the hop does not land on
            # the final complete window.
            tail = (max(start, end - window_seconds), end)
            if not candidates or abs(candidates[-1][0] - tail[0]) > 1e-6:
                candidates.append(tail)
        for window_start, window_end in candidates:
            range_key = (round(window_start * 1000), round(window_end * 1000))
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)
            digest = hashlib.sha256(
                f"{source_id}\0{range_key[0]}\0{range_key[1]}".encode()
            ).hexdigest()[:20]
            windows.append(
                SpeakerWindow(
                    window_id=f"speaker-window-{digest}",
                    source_id=source_id,
                    start=window_start,
                    end=window_end,
                )
            )
    return tuple(windows)


def _normalized(values: Sequence[float]) -> Vector:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("speaker embedding must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("speaker embedding contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("speaker embedding must not be a zero vector")
    return tuple(value / norm for value in vector)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
    *,
    use_numpy: bool = False,
) -> float:
    """Return cosine similarity without requiring NumPy.

    NumPy is imported only when explicitly requested.  This keeps importing the
    DayAudio core inexpensive and makes a missing optional dependency harmless.
    """

    if len(left) != len(right) or not left:
        raise ValueError("speaker vectors must have the same non-zero dimension")
    if use_numpy:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("NumPy is not installed; use the standard backend") from exc
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1e-12:
            raise ValueError("speaker vectors must not be zero vectors")
        return max(-1.0, min(1.0, float(np.dot(a, b)) / denominator))

    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm * right_norm <= 1e-12:
        raise ValueError("speaker vectors must not be zero vectors")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def orchestrate_embeddings(
    backend: EmbeddingBackend,
    windows: Iterable[SpeakerWindow],
    *,
    batch_size: int = 32,
) -> tuple[EmbeddedSpeakerWindow, ...]:
    """Batch a backend call and validate its one-to-one output contract."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not str(backend.model_digest).strip():
        raise ValueError("embedding backend model_digest must not be empty")
    ordered = sorted(windows, key=lambda item: (item.source_id, item.start, item.end, item.window_id))
    if not ordered:
        return ()
    source_ids = {window.source_id for window in ordered}
    if len(source_ids) != 1:
        raise ValueError("one embedding run may contain windows from only one source")

    results: list[EmbeddedSpeakerWindow] = []
    dimension: int | None = None
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        embeddings = list(backend.embed(batch))
        if len(embeddings) != len(batch):
            raise ValueError(
                f"embedding backend returned {len(embeddings)} vectors for {len(batch)} windows"
            )
        for window, raw_embedding in zip(batch, embeddings):
            embedding = _normalized(raw_embedding)
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError("embedding backend returned inconsistent vector dimensions")
            results.append(
                EmbeddedSpeakerWindow(
                    window_id=window.window_id,
                    source_id=window.source_id,
                    start=window.start,
                    end=window.end,
                    embedding=embedding,
                )
            )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class SpeakerCluster:
    local_speaker_id: str
    source_id: str
    member_window_ids: tuple[str, ...]
    centroid: Vector
    mean_similarity: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["member_window_ids"] = list(self.member_window_ids)
        result["centroid"] = list(self.centroid)
        return result


@dataclass(frozen=True, slots=True)
class SpeakerClusteringResult:
    source_id: str
    model_digest: str
    clusters: tuple[SpeakerCluster, ...]
    turns: tuple[SpeakerTurn, ...]
    window_assignments: tuple[tuple[str, str], ...]

    def assignment_map(self) -> dict[str, str]:
        return dict(self.window_assignments)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        raise ValueError("cannot calculate an empty centroid")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("speaker vectors have inconsistent dimensions")
    return _normalized(
        tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension))
    )


def _stable_local_id(
    source_id: str,
    model_digest: str,
    members: Sequence[EmbeddedSpeakerWindow],
) -> str:
    """Derive an ID from file-local evidence, never from an arbitrary label."""

    digest = hashlib.sha256()
    digest.update(source_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model_digest.encode("utf-8"))
    # Anchor the identifier to the earliest member rather than the complete
    # membership list.  Processing later windows can then extend a cluster
    # without renaming an already emitted file-local speaker.
    anchor = min(members, key=lambda item: (item.start, item.end, item.window_id))
    digest.update(
        f"\0{anchor.window_id}:{round(anchor.start * 1000)}:{round(anchor.end * 1000)}".encode(
            "utf-8"
        )
    )
    return f"local-speaker-{digest.hexdigest()[:16]}"


def cluster_speaker_windows(
    windows: Iterable[EmbeddedSpeakerWindow],
    *,
    model_digest: str,
    similarity_threshold: float = 0.72,
    merge_gap_seconds: float = 0.25,
) -> SpeakerClusteringResult:
    """Cluster VAD-window embeddings within exactly one source.

    Clustering is deterministic online-centroid assignment in timeline order.
    It intentionally makes no cross-file identity claim.  Cross-file identity
    belongs to :mod:`dayaudio.identity` and requires an explicit profile.
    """

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds must not be negative")
    ordered = sorted(windows, key=lambda item: (item.start, item.end, item.window_id))
    if not ordered:
        raise ValueError("at least one embedded window is required")
    source_ids = {window.source_id for window in ordered}
    if len(source_ids) != 1:
        raise ValueError("speaker clustering is file-local and accepts one source at a time")
    source_id = ordered[0].source_id
    dimension = len(ordered[0].embedding)
    if any(len(window.embedding) != dimension for window in ordered):
        raise ValueError("speaker embeddings have inconsistent dimensions")

    groups: list[list[EmbeddedSpeakerWindow]] = []
    centroids: list[Vector] = []
    for window in ordered:
        vector = _normalized(window.embedding)
        if not groups:
            groups.append([window])
            centroids.append(vector)
            continue
        similarities = [cosine_similarity(vector, centroid) for centroid in centroids]
        best_index = max(range(len(similarities)), key=lambda index: (similarities[index], -index))
        if similarities[best_index] >= similarity_threshold:
            groups[best_index].append(window)
            centroids[best_index] = _mean_vector(
                [member.embedding for member in groups[best_index]]
            )
        else:
            groups.append([window])
            centroids.append(vector)

    cluster_rows: list[tuple[SpeakerCluster, list[EmbeddedSpeakerWindow]]] = []
    assignment_by_window: dict[str, str] = {}
    for members, centroid in zip(groups, centroids):
        local_id = _stable_local_id(source_id, model_digest, members)
        similarities = [cosine_similarity(member.embedding, centroid) for member in members]
        cluster = SpeakerCluster(
            local_speaker_id=local_id,
            source_id=source_id,
            member_window_ids=tuple(member.window_id for member in members),
            centroid=centroid,
            mean_similarity=sum(similarities) / len(similarities),
        )
        cluster_rows.append((cluster, members))
        for member in members:
            if member.window_id in assignment_by_window:
                raise ValueError(f"duplicate speaker window id: {member.window_id}")
            assignment_by_window[member.window_id] = local_id

    # Sort by stable ID so cluster order does not leak an implementation label.
    clusters = tuple(sorted((row[0] for row in cluster_rows), key=lambda item: item.local_speaker_id))
    raw_turns = sorted(
        (
            member.start,
            member.end,
            assignment_by_window[member.window_id],
            cluster.mean_similarity,
        )
        for cluster, members in cluster_rows
        for member in members
    )
    merged: list[tuple[float, float, str, list[float]]] = []
    for start, end, local_id, confidence in raw_turns:
        if (
            merged
            and merged[-1][2] == local_id
            and start <= merged[-1][1] + merge_gap_seconds
        ):
            previous = merged[-1]
            merged[-1] = (
                previous[0],
                max(previous[1], end),
                local_id,
                previous[3] + [confidence],
            )
        else:
            merged.append((start, end, local_id, [confidence]))
    turns: list[SpeakerTurn] = []
    for start, end, local_id, confidences in merged:
        turn_digest = hashlib.sha256(
            f"{source_id}\0{local_id}\0{round(start * 1000)}\0{round(end * 1000)}".encode()
        ).hexdigest()[:20]
        turns.append(
            SpeakerTurn(
                turn_id=f"speaker-turn-{turn_digest}",
                source_id=source_id,
                local_speaker_id=local_id,
                start=start,
                end=end,
                model_digest=model_digest,
                confidence=sum(confidences) / len(confidences),
            )
        )
    return SpeakerClusteringResult(
        source_id=source_id,
        model_digest=model_digest,
        clusters=clusters,
        turns=tuple(turns),
        window_assignments=tuple(sorted(assignment_by_window.items())),
    )


def embed_and_cluster(
    backend: EmbeddingBackend,
    windows: Iterable[SpeakerWindow],
    *,
    batch_size: int = 32,
    similarity_threshold: float = 0.72,
    merge_gap_seconds: float = 0.25,
) -> SpeakerClusteringResult:
    """Run a resident embedding backend and deterministic clustering together."""

    embedded = orchestrate_embeddings(backend, windows, batch_size=batch_size)
    return cluster_speaker_windows(
        embedded,
        model_digest=backend.model_digest,
        similarity_threshold=similarity_threshold,
        merge_gap_seconds=merge_gap_seconds,
    )


@dataclass(frozen=True, slots=True)
class SpeakerOverlap:
    local_speaker_id: str
    seconds: float
    interval_fraction: float


@dataclass(frozen=True, slots=True)
class SpeakerOverlapDistribution:
    start: float
    end: float
    overlaps: tuple[SpeakerOverlap, ...]
    covered_seconds: float
    unassigned_seconds: float
    concurrent_seconds: float

    @property
    def coverage_fraction(self) -> float:
        return self.covered_seconds / (self.end - self.start)

    def as_mapping(self) -> dict[str, float]:
        return {item.local_speaker_id: item.interval_fraction for item in self.overlaps}


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def overlap_distribution(
    start: float,
    end: float,
    turns: Iterable[SpeakerTurn],
    *,
    source_id: str | None = None,
) -> SpeakerOverlapDistribution:
    """Return all speaker overlap, uncovered time, and concurrent overlap.

    Per-speaker intervals are unioned before measuring, so overlapping windows
    from one speaker are not double counted.  Different speakers may overlap;
    that excess is reported as ``concurrent_seconds`` instead of being hidden by
    normalization.
    """

    if end <= start:
        raise ValueError("overlap interval end must be greater than start")
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    all_intervals: list[tuple[float, float]] = []
    for turn in turns:
        if source_id is not None and turn.source_id != source_id:
            continue
        clipped_start = max(start, turn.start)
        clipped_end = min(end, turn.end)
        if clipped_end <= clipped_start:
            continue
        interval = (clipped_start, clipped_end)
        by_speaker.setdefault(turn.local_speaker_id, []).append(interval)
        all_intervals.append(interval)
    duration = end - start
    measured = [
        SpeakerOverlap(
            local_speaker_id=local_id,
            seconds=_union_length(intervals),
            interval_fraction=_union_length(intervals) / duration,
        )
        for local_id, intervals in by_speaker.items()
    ]
    measured.sort(key=lambda item: (-item.seconds, item.local_speaker_id))
    covered = min(duration, _union_length(all_intervals))
    summed = sum(item.seconds for item in measured)
    return SpeakerOverlapDistribution(
        start=start,
        end=end,
        overlaps=tuple(measured),
        covered_seconds=covered,
        unassigned_seconds=max(0.0, duration - covered),
        concurrent_seconds=max(0.0, summed - covered),
    )


@dataclass(frozen=True, slots=True)
class SpeakerAssignment:
    local_speaker_id: str | None
    participant_role: ParticipantRole
    distribution: SpeakerOverlapDistribution
    reasons: tuple[str, ...] = ()


def assign_speaker_fail_closed(
    distribution: SpeakerOverlapDistribution,
    *,
    min_coverage: float = 0.60,
    min_dominance: float = 0.80,
    min_margin: float = 0.20,
) -> SpeakerAssignment:
    """Assign one local speaker only when coverage, dominance, and margin pass."""

    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    if not 0.0 <= min_dominance <= 1.0:
        raise ValueError("min_dominance must be between 0 and 1")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be between 0 and 1")
    overlaps = distribution.overlaps
    if not overlaps:
        return SpeakerAssignment(None, "unknown", distribution, ("no_speaker_overlap",))

    reasons: list[str] = []
    if distribution.coverage_fraction < min_coverage:
        reasons.append("insufficient_speaker_coverage")
    winner = overlaps[0]
    total_speaker_seconds = sum(item.seconds for item in overlaps)
    dominance = winner.seconds / total_speaker_seconds if total_speaker_seconds else 0.0
    runner_fraction = overlaps[1].interval_fraction if len(overlaps) > 1 else 0.0
    margin = winner.interval_fraction - runner_fraction
    if dominance < min_dominance:
        reasons.append("no_dominant_speaker")
    if margin < min_margin:
        reasons.append("speaker_margin_too_small")
    if distribution.concurrent_seconds > 0 and len(overlaps) > 1:
        reasons.append("overlapping_speakers")

    # Concurrent overlap is retained as a warning, but a long clean dominant
    # interval may still pass the quantitative gates.  Near ties fail closed.
    blocking = {
        "insufficient_speaker_coverage",
        "no_dominant_speaker",
        "speaker_margin_too_small",
    }
    if not blocking.intersection(reasons):
        return SpeakerAssignment(winner.local_speaker_id, "anonymous", distribution, tuple(reasons))
    role: ParticipantRole = "mixed" if len(overlaps) > 1 else "unknown"
    return SpeakerAssignment(None, role, distribution, tuple(reasons))


def assign_speaker(
    start: float,
    end: float,
    turns: Iterable[SpeakerTurn],
    *,
    source_id: str | None = None,
    min_coverage: float = 0.60,
    min_dominance: float = 0.80,
    min_margin: float = 0.20,
) -> SpeakerAssignment:
    """Convenience wrapper combining distribution and fail-closed assignment."""

    return assign_speaker_fail_closed(
        overlap_distribution(start, end, turns, source_id=source_id),
        min_coverage=min_coverage,
        min_dominance=min_dominance,
        min_margin=min_margin,
    )


# Concise aliases retained for adapters and exploratory notebooks.
cluster_windows = cluster_speaker_windows
speaker_overlap_distribution = overlap_distribution


__all__ = [
    "EmbeddingBackend",
    "EmbeddedSpeakerWindow",
    "SpeakerAssignment",
    "SpeakerCluster",
    "SpeakerClusteringResult",
    "SpeakerOverlap",
    "SpeakerOverlapDistribution",
    "SpeakerWindow",
    "assign_speaker",
    "assign_speaker_fail_closed",
    "cluster_speaker_windows",
    "cluster_windows",
    "cosine_similarity",
    "embed_and_cluster",
    "orchestrate_embeddings",
    "overlap_distribution",
    "speaker_overlap_distribution",
    "vad_intervals_to_speaker_windows",
]
