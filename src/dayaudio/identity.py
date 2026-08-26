"""Explicit, append-only speaker identity and owner enrollment.

Speaker clusters are file-local.  This module is the only place that may map an
embedding to a cross-file identity, and it always exposes an uncertain band.
Mixed-speaker samples are retained as audit evidence but never train a profile.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from .cas import atomic_write_bytes
from .speaker import Vector, cosine_similarity

SampleStatus = Literal["positive", "negative", "mixed"]
IdentityRole = Literal["owner", "identity"]
IdentityMatchStatus = Literal["owner", "match", "non_owner", "uncertain"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_embedding(values: Sequence[float]) -> Vector:
    embedding = tuple(float(value) for value in values)
    if not embedding:
        raise ValueError("identity sample embedding must not be empty")
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("identity sample embedding contains a non-finite value")
    if math.sqrt(sum(value * value for value in embedding)) <= 1e-12:
        raise ValueError("identity sample embedding must not be a zero vector")
    return embedding


@dataclass(frozen=True, slots=True)
class EnrollmentSample:
    """One immutable human judgment about one speaker sample."""

    sample_id: str
    status: SampleStatus
    embedding: Vector
    source_id: str | None = None
    start: float | None = None
    end: float | None = None
    local_speaker_id: str | None = None
    created_at: str = field(default_factory=_utc_now)
    supersedes_sample_id: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("identity sample id must not be empty")
        if self.status not in {"positive", "negative", "mixed"}:
            raise ValueError(f"unsupported identity sample status: {self.status}")
        object.__setattr__(self, "embedding", _validate_embedding(self.embedding))
        if (self.start is None) != (self.end is None):
            raise ValueError("identity sample start and end must be provided together")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("identity sample end must be greater than start")
        if self.supersedes_sample_id == self.sample_id:
            raise ValueError("an identity sample cannot supersede itself")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["embedding"] = list(self.embedding)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnrollmentSample":
        return cls(
            sample_id=str(data["sample_id"]),
            status=data["status"],
            embedding=tuple(float(value) for value in data["embedding"]),
            source_id=data.get("source_id"),
            start=data.get("start"),
            end=data.get("end"),
            local_speaker_id=data.get("local_speaker_id"),
            created_at=data.get("created_at") or _utc_now(),
            supersedes_sample_id=data.get("supersedes_sample_id"),
            note=data.get("note"),
        )


@dataclass(frozen=True, slots=True)
class IdentityProfileRevision:
    """A complete immutable snapshot in an append-only profile history."""

    identity_id: str
    revision_id: str
    revision: int
    role: IdentityRole
    model_digest: str
    samples: tuple[EnrollmentSample, ...] = ()
    display_name: str | None = None
    previous_revision_id: str | None = None
    created_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity_id or not self.revision_id:
            raise ValueError("identity and revision ids must not be empty")
        if self.revision < 1:
            raise ValueError("identity revision must be positive")
        if self.role not in {"owner", "identity"}:
            raise ValueError(f"unsupported identity role: {self.role}")
        if not self.model_digest:
            raise ValueError("identity model digest must not be empty")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("identity profile contains duplicate sample ids")
        known: set[str] = set()
        superseded: set[str] = set()
        dimension: int | None = None
        for sample in self.samples:
            if dimension is None:
                dimension = len(sample.embedding)
            elif len(sample.embedding) != dimension:
                raise ValueError("identity samples have inconsistent embedding dimensions")
            if sample.supersedes_sample_id:
                if sample.supersedes_sample_id not in known:
                    raise ValueError("identity sample supersedes an unknown or future sample")
                if sample.supersedes_sample_id in superseded:
                    raise ValueError("an identity sample may be superseded only once")
                superseded.add(sample.supersedes_sample_id)
            known.add(sample.sample_id)

    @property
    def effective_samples(self) -> tuple[EnrollmentSample, ...]:
        superseded = {
            sample.supersedes_sample_id
            for sample in self.samples
            if sample.supersedes_sample_id is not None
        }
        return tuple(sample for sample in self.samples if sample.sample_id not in superseded)

    @property
    def positive_samples(self) -> tuple[EnrollmentSample, ...]:
        return tuple(sample for sample in self.effective_samples if sample.status == "positive")

    @property
    def negative_samples(self) -> tuple[EnrollmentSample, ...]:
        return tuple(sample for sample in self.effective_samples if sample.status == "negative")

    @property
    def mixed_samples(self) -> tuple[EnrollmentSample, ...]:
        return tuple(sample for sample in self.effective_samples if sample.status == "mixed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["samples"] = [sample.to_dict() for sample in self.samples]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityProfileRevision":
        return cls(
            identity_id=str(data["identity_id"]),
            revision_id=str(data["revision_id"]),
            revision=int(data["revision"]),
            role=data["role"],
            model_digest=str(data["model_digest"]),
            samples=tuple(EnrollmentSample.from_dict(item) for item in data.get("samples", ())),
            display_name=data.get("display_name"),
            previous_revision_id=data.get("previous_revision_id"),
            created_at=data.get("created_at") or _utc_now(),
            metadata=dict(data.get("metadata", {})),
        )


# Publicly friendly aliases.
IdentityProfile = IdentityProfileRevision
OwnerProfile = IdentityProfileRevision


def _revision_digest(
    identity_id: str,
    revision: int,
    previous_revision_id: str | None,
    samples: Sequence[EnrollmentSample],
    *,
    role: IdentityRole,
    model_digest: str,
    display_name: str | None,
    created_at: str,
    metadata: dict[str, Any],
) -> str:
    payload = {
        "identity_id": identity_id,
        "revision": revision,
        "previous_revision_id": previous_revision_id,
        "role": role,
        "model_digest": model_digest,
        "display_name": display_name,
        "created_at": created_at,
        "metadata": metadata,
        "samples": [sample.to_dict() for sample in samples],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"identity-revision-{digest[:32]}"


def create_identity_profile(
    identity_id: str,
    *,
    model_digest: str,
    role: IdentityRole = "identity",
    display_name: str | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IdentityProfileRevision:
    if not identity_id:
        raise ValueError("identity id must not be empty")
    created_at_value = created_at or _utc_now()
    metadata_value = dict(metadata or {})
    revision_id = _revision_digest(
        identity_id,
        1,
        None,
        (),
        role=role,
        model_digest=model_digest,
        display_name=display_name,
        created_at=created_at_value,
        metadata=metadata_value,
    )
    return IdentityProfileRevision(
        identity_id=identity_id,
        revision_id=revision_id,
        revision=1,
        role=role,
        model_digest=model_digest,
        display_name=display_name,
        created_at=created_at_value,
        metadata=metadata_value,
    )


def create_owner_profile(
    identity_id: str,
    *,
    model_digest: str,
    display_name: str | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IdentityProfileRevision:
    return create_identity_profile(
        identity_id,
        model_digest=model_digest,
        role="owner",
        display_name=display_name,
        created_at=created_at,
        metadata=metadata,
    )


def append_enrollment_sample(
    profile: IdentityProfileRevision,
    sample: EnrollmentSample,
    *,
    created_at: str | None = None,
) -> IdentityProfileRevision:
    """Return a new revision; the prior profile and judgments stay untouched."""

    if any(existing.sample_id == sample.sample_id for existing in profile.samples):
        raise ValueError(f"identity sample already exists: {sample.sample_id}")
    samples = profile.samples + (sample,)
    revision = profile.revision + 1
    created_at_value = created_at or _utc_now()
    revision_id = _revision_digest(
        profile.identity_id,
        revision,
        profile.revision_id,
        samples,
        role=profile.role,
        model_digest=profile.model_digest,
        display_name=profile.display_name,
        created_at=created_at_value,
        metadata=dict(profile.metadata),
    )
    return IdentityProfileRevision(
        identity_id=profile.identity_id,
        revision_id=revision_id,
        revision=revision,
        role=profile.role,
        model_digest=profile.model_digest,
        samples=samples,
        display_name=profile.display_name,
        previous_revision_id=profile.revision_id,
        created_at=created_at_value,
        metadata=dict(profile.metadata),
    )


@contextmanager
def _profile_lock(destination: Path):
    """Cross-platform advisory lock on a non-sensitive sibling file."""

    lock_path = destination.with_name(destination.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":  # pragma: no cover - exercised in Windows CI
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no branch - one platform per run
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_profile_revision(path: str | Path, profile: IdentityProfileRevision) -> None:
    """Append one JSONL snapshot while verifying the on-disk revision chain."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _profile_lock(destination):
        revisions = load_profile_revisions(destination) if destination.exists() else ()
        if revisions:
            latest = revisions[-1]
            if latest.identity_id != profile.identity_id:
                raise ValueError("cannot append a different identity to a profile log")
            if profile.previous_revision_id != latest.revision_id:
                raise ValueError("identity revision does not extend the latest on-disk revision")
            if profile.revision != latest.revision + 1:
                raise ValueError("identity revision number is not sequential")
            if (
                profile.role != latest.role
                or profile.model_digest != latest.model_digest
                or profile.display_name != latest.display_name
                or profile.metadata != latest.metadata
            ):
                raise ValueError("identity profile immutable fields changed")
            if (
                len(profile.samples) != len(latest.samples) + 1
                or profile.samples[: len(latest.samples)] != latest.samples
            ):
                raise ValueError("identity revision must append one judgment without rewriting history")
        elif profile.revision != 1 or profile.previous_revision_id is not None:
            raise ValueError("a new profile log must begin at revision 1")
        previous = destination.read_bytes() if destination.exists() else b""
        if previous and not previous.endswith(b"\n"):
            previous += b"\n"
        record = (
            json.dumps(profile.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        atomic_write_bytes(destination, previous + record, mode=0o600)


def load_profile_revisions(path: str | Path) -> tuple[IdentityProfileRevision, ...]:
    source = Path(path)
    if not source.exists():
        return ()
    revisions: list[IdentityProfileRevision] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                revision = IdentityProfileRevision.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid identity profile log at line {line_number}") from exc
            expected_revision_id = _revision_digest(
                revision.identity_id,
                revision.revision,
                revision.previous_revision_id,
                revision.samples,
                role=revision.role,
                model_digest=revision.model_digest,
                display_name=revision.display_name,
                created_at=revision.created_at,
                metadata=revision.metadata,
            )
            if revision.revision_id != expected_revision_id:
                raise ValueError(f"identity profile content digest mismatch at line {line_number}")
            if revisions:
                previous = revisions[-1]
                if (
                    revision.identity_id != previous.identity_id
                    or revision.previous_revision_id != previous.revision_id
                    or revision.revision != previous.revision + 1
                ):
                    raise ValueError(f"broken identity revision chain at line {line_number}")
            elif revision.revision != 1 or revision.previous_revision_id is not None:
                raise ValueError("identity profile log does not begin at revision 1")
            revisions.append(revision)
    return tuple(revisions)


def load_latest_profile(path: str | Path) -> IdentityProfileRevision | None:
    """Load the newest valid revision, or ``None`` for a missing/empty log."""

    revisions = load_profile_revisions(path)
    return revisions[-1] if revisions else None


def save_owner_profile(path: str | Path, profile: IdentityProfileRevision) -> None:
    """Append one owner revision without rewriting prior biometric judgments."""

    if profile.role != "owner":
        raise ValueError("save_owner_profile accepts only an owner profile")
    append_profile_revision(path, profile)


def load_owner_profile(path: str | Path) -> IdentityProfileRevision | None:
    profile = load_latest_profile(path)
    if profile is not None and profile.role != "owner":
        raise ValueError("profile log contains a non-owner identity")
    return profile


def append_owner_sample(
    path: str | Path,
    sample: EnrollmentSample,
    *,
    created_at: str | None = None,
) -> IdentityProfileRevision:
    """Append one enrollment judgment to an existing on-disk owner log."""

    profile = load_owner_profile(path)
    if profile is None:
        raise ValueError("owner profile log does not exist; create and save revision 1 first")
    revision = append_enrollment_sample(profile, sample, created_at=created_at)
    save_owner_profile(path, revision)
    return revision


def _centroid(samples: Sequence[EnrollmentSample]) -> Vector:
    if not samples:
        raise ValueError("at least one positive identity sample is required")
    dimension = len(samples[0].embedding)
    if any(len(sample.embedding) != dimension for sample in samples):
        raise ValueError("identity samples have inconsistent embedding dimensions")
    values = tuple(
        sum(sample.embedding[index] for sample in samples) / len(samples)
        for index in range(dimension)
    )
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("positive identity samples cancel to a zero centroid")
    return tuple(value / norm for value in values)


def score_identity(
    profile: IdentityProfileRevision,
    embedding: Sequence[float],
    *,
    model_digest: str | None = None,
) -> float:
    """Score a candidate only against effective positive samples."""

    if model_digest is not None and model_digest != profile.model_digest:
        raise ValueError("candidate embedding model does not match the identity profile")
    positives = profile.positive_samples
    if not positives:
        raise ValueError("identity profile has no positive enrollment sample")
    candidate = _validate_embedding(embedding)
    if len(candidate) != len(positives[0].embedding):
        raise ValueError("candidate embedding dimension does not match the identity profile")
    return cosine_similarity(candidate, _centroid(positives))


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    """Three-state thresholds; scores between the bounds remain uncertain."""

    non_match_max: float
    match_min: float
    provisional: bool
    positive_floor: float | None = None
    negative_ceiling: float | None = None
    positive_count: int = 0
    negative_count: int = 0
    excluded_mixed_count: int = 0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not -1.0 <= self.non_match_max < self.match_min <= 1.0:
            raise ValueError("identity thresholds must define a non-empty uncertain band")

    @property
    def uncertain_band(self) -> tuple[float, float]:
        return (self.non_match_max, self.match_min)

    # Owner-oriented names used by the CLI and report layer.
    @property
    def non_owner_max(self) -> float:
        return self.non_match_max

    @property
    def owner_min(self) -> float:
        return self.match_min

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["uncertain_band"] = list(self.uncertain_band)
        return result


def calibrate_thresholds(
    profile: IdentityProfileRevision,
    *,
    default_non_match_max: float = 0.35,
    default_match_min: float = 0.75,
) -> ThresholdCalibration:
    """Derive conservative provisional thresholds from explicit labels.

    With separated positives and negatives, one third of the observed gap is
    left on each side, reproducing a broad uncertain band.  Sparse or
    overlapping labels retain conservative defaults and remain provisional.
    """

    if not -1.0 <= default_non_match_max < default_match_min <= 1.0:
        raise ValueError("default thresholds must define an uncertain band")
    positives = profile.positive_samples
    negatives = profile.negative_samples
    mixed = profile.mixed_samples
    reasons: list[str] = []

    positive_scores: list[float] = []
    if len(positives) >= 2:
        for index, sample in enumerate(positives):
            other_samples = positives[:index] + positives[index + 1 :]
            positive_scores.append(cosine_similarity(sample.embedding, _centroid(other_samples)))
    elif positives:
        # A single sample establishes an identity seed, not an empirical floor.
        reasons.append("single_positive_sample")
    else:
        reasons.append("no_positive_samples")

    negative_scores: list[float] = []
    if positives:
        positive_centroid = _centroid(positives)
        negative_scores = [
            cosine_similarity(sample.embedding, positive_centroid) for sample in negatives
        ]
    if not negatives:
        reasons.append("no_negative_samples")
    if mixed:
        reasons.append("mixed_samples_excluded")

    positive_floor = min(positive_scores) if positive_scores else None
    negative_ceiling = max(negative_scores) if negative_scores else None
    sufficiently_labeled = len(positives) >= 2 and len(negatives) >= 1
    separated = (
        positive_floor is not None
        and negative_ceiling is not None
        and negative_ceiling < positive_floor
    )
    if sufficiently_labeled and separated:
        gap = positive_floor - negative_ceiling
        non_match_max = negative_ceiling + gap / 3.0
        match_min = positive_floor - gap / 3.0
        # Five positive and five negative conditions are a useful beginning,
        # but still not a population-level calibration claim.
        provisional = len(positives) < 5 or len(negatives) < 5
        if provisional:
            reasons.append("limited_calibration_samples")
    else:
        non_match_max = default_non_match_max
        match_min = default_match_min
        provisional = True
        if sufficiently_labeled and not separated:
            reasons.append("positive_negative_score_overlap")
            # Make collisions uncertain; never move match_min downward merely
            # to fit overlapping enrollment data.
            assert negative_ceiling is not None
            match_min = min(1.0, max(match_min, negative_ceiling + 0.05))
            if match_min >= 1.0:
                match_min = 0.999999
            non_match_max = min(non_match_max, match_min - 0.05)

    return ThresholdCalibration(
        non_match_max=max(-1.0, non_match_max),
        match_min=min(1.0, match_min),
        provisional=provisional,
        positive_floor=positive_floor,
        negative_ceiling=negative_ceiling,
        positive_count=len(positives),
        negative_count=len(negatives),
        excluded_mixed_count=len(mixed),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    identity_id: str | None
    status: IdentityMatchStatus
    score: float
    calibration: ThresholdCalibration
    profile_revision_id: str
    reasons: tuple[str, ...] = ()

    @property
    def is_uncertain(self) -> bool:
        return self.status == "uncertain"


def classify_identity_score(
    score: float,
    calibration: ThresholdCalibration,
    *,
    role: IdentityRole = "owner",
) -> IdentityMatchStatus:
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("identity score must be finite and between -1 and 1")
    if score >= calibration.match_min:
        return "owner" if role == "owner" else "match"
    if score <= calibration.non_match_max:
        return "non_owner"
    return "uncertain"


def match_identity(
    profile: IdentityProfileRevision,
    embedding: Sequence[float],
    *,
    calibration: ThresholdCalibration | None = None,
    model_digest: str | None = None,
) -> IdentityDecision:
    active_calibration = calibration or calibrate_thresholds(profile)
    score = score_identity(profile, embedding, model_digest=model_digest)
    empirically_separated = (
        active_calibration.positive_count >= 2
        and active_calibration.negative_count >= 1
        and active_calibration.positive_floor is not None
        and active_calibration.negative_ceiling is not None
        and active_calibration.negative_ceiling < active_calibration.positive_floor
    )
    status = (
        classify_identity_score(score, active_calibration, role=profile.role)
        if empirically_separated
        else "uncertain"
    )
    reasons: list[str] = []
    if not empirically_separated:
        reasons.append("insufficient_empirical_calibration")
    if active_calibration.provisional:
        reasons.append("provisional_thresholds")
    if status == "uncertain":
        reasons.append("score_in_uncertain_band")
    return IdentityDecision(
        identity_id=profile.identity_id if status in {"owner", "match"} else None,
        status=status,
        score=score,
        calibration=active_calibration,
        profile_revision_id=profile.revision_id,
        reasons=tuple(reasons),
    )


def make_enrollment_sample(
    *,
    status: SampleStatus,
    embedding: Sequence[float],
    source_id: str | None = None,
    start: float | None = None,
    end: float | None = None,
    local_speaker_id: str | None = None,
    created_at: str | None = None,
    supersedes_sample_id: str | None = None,
    note: str | None = None,
) -> EnrollmentSample:
    """Construct a stable sample ID when callers do not already have one."""

    vector = _validate_embedding(embedding)
    digest_input = {
        "status": status,
        "embedding": [round(value, 8) for value in vector],
        "source_id": source_id,
        "start": start,
        "end": end,
        "local_speaker_id": local_speaker_id,
        "supersedes_sample_id": supersedes_sample_id,
    }
    digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return EnrollmentSample(
        sample_id=f"identity-sample-{digest}",
        status=status,
        embedding=vector,
        source_id=source_id,
        start=start,
        end=end,
        local_speaker_id=local_speaker_id,
        created_at=created_at or _utc_now(),
        supersedes_sample_id=supersedes_sample_id,
        note=note,
    )


__all__ = [
    "EnrollmentSample",
    "IdentityDecision",
    "IdentityMatchStatus",
    "IdentityProfile",
    "IdentityProfileRevision",
    "IdentityRole",
    "OwnerProfile",
    "SampleStatus",
    "ThresholdCalibration",
    "append_enrollment_sample",
    "append_owner_sample",
    "append_profile_revision",
    "calibrate_thresholds",
    "classify_identity_score",
    "create_identity_profile",
    "create_owner_profile",
    "load_profile_revisions",
    "load_latest_profile",
    "load_owner_profile",
    "make_enrollment_sample",
    "match_identity",
    "score_identity",
    "save_owner_profile",
]
