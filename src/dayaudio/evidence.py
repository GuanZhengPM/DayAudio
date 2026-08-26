"""Evidence-window construction and conservative ASR revision selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .identity import IdentityDecision
from .speaker import SpeakerAssignment
from .types import AsrSegment, EvidenceConfidence, EvidenceWindow, ParticipantRole

_REFUSAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:i (?:cannot|can't|am unable to)|unable to) (?:transcribe|process|hear)\b",
        r"\b(?:as an ai|language model)\b",
        r"(?:无法|不能|抱歉).{0,12}(?:转录|识别|处理|听清)",
        r"(?:请提供|需要提供).{0,12}(?:音频|录音)",
        r"(?:没有|未提供).{0,8}(?:音频|录音)",
    )
)

_SENSITIVE_PATTERNS = (
    re.compile(r"\d"),
    re.compile(r"(?:决定|确定|同意|拒绝|不要|不能|不会|必须|应该|需要|承诺|答应|行动项|截止|发布|上线)"),
    re.compile(r"\b(?:decid(?:e|ed)|agree(?:d)?|refus(?:e|ed)|must|should|need|will not|won't|deadline|launch|ship|action item)\b", re.I),
    re.compile(r"(?:公司|先生|女士|老师|总监|经理|项目|产品|模型)"),
)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_is_summary_sensitive(text: str) -> bool:
    """Flag claims for which review-only evidence is insufficient."""

    compact = _compact_text(text)
    return any(pattern.search(compact) for pattern in _SENSITIVE_PATTERNS)


def _repetition_score(text: str) -> float:
    """Estimate degenerate repetition for both spaced and CJK text."""

    compact = _compact_text(text)
    if len(compact) < 12:
        return 0.0
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\s]", compact.casefold())
    if len(tokens) < 6:
        return 0.0
    # A dominant token alone catches common decoder loops.
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    unigram = max(counts.values()) / len(tokens)
    # Repeated 2-4 token phrases catch "thank you thank you ..." and CJK loops.
    best_ngram = 0.0
    for size in (2, 3, 4):
        if len(tokens) < size * 3:
            continue
        ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
        ngram_counts: dict[tuple[str, ...], int] = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        repeats = max(ngram_counts.values())
        best_ngram = max(best_ngram, repeats * size / len(tokens))
    return max(unigram, best_ngram)


def transcription_anomaly_flags(text: str) -> tuple[str, ...]:
    compact = _compact_text(text)
    flags: list[str] = []
    if not compact:
        flags.append("empty_text")
    if any(pattern.search(compact) for pattern in _REFUSAL_PATTERNS):
        flags.append("model_refusal")
    if _repetition_score(compact) >= 0.58:
        flags.append("decoder_repetition")
    if len(compact) >= 240 and len(set(compact)) <= 5:
        flags.append("low_character_diversity")
    return tuple(flags)


@dataclass(frozen=True, slots=True)
class AlternativeAssessment:
    candidate: AsrSegment
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


def assess_alternative(
    fast_segment: AsrSegment,
    candidate: AsrSegment,
    *,
    minimum_length_ratio: float = 0.35,
    maximum_length_ratio: float = 4.0,
) -> AlternativeAssessment:
    """Reject unsafe alternatives before considering any model replacement."""

    if not 0.0 < minimum_length_ratio <= 1.0:
        raise ValueError("minimum_length_ratio must be between 0 and 1")
    if maximum_length_ratio < 1.0:
        raise ValueError("maximum_length_ratio must be at least 1")
    reasons: list[str] = []
    if candidate.source_id != fast_segment.source_id:
        reasons.append("source_mismatch")
    tolerance = 0.050
    if candidate.start > fast_segment.end + tolerance or candidate.end < fast_segment.start - tolerance:
        reasons.append("timeline_mismatch")
    reasons.extend(candidate.anomaly_flags)
    reasons.extend(transcription_anomaly_flags(candidate.text))
    fast_length = len(_compact_text(fast_segment.text))
    candidate_length = len(_compact_text(candidate.text))
    if fast_length >= 8 and candidate_length < max(2, math.floor(fast_length * minimum_length_ratio)):
        reasons.append("suspicious_under_transcription")
    if fast_length >= 4 and candidate_length > max(80, math.ceil(fast_length * maximum_length_ratio)):
        reasons.append("suspicious_over_expansion")
    # A backend may mark output as non-speech or an explicit hallucination.
    if candidate.metadata.get("hallucinated") is True:
        reasons.append("backend_hallucination_flag")
    if candidate.metadata.get("no_speech") is True and fast_length:
        reasons.append("candidate_claims_no_speech")
    deduplicated = tuple(dict.fromkeys(str(reason) for reason in reasons if reason))
    return AlternativeAssessment(candidate, not deduplicated, deduplicated)


@dataclass(frozen=True, slots=True)
class RevisionSelection:
    fast_segment: AsrSegment
    selected_segment: AsrSegment
    alternatives: tuple[AlternativeAssessment, ...]
    selection_reason: str

    @property
    def preserved_fast(self) -> bool:
        return self.selected_segment is self.fast_segment or (
            self.selected_segment.segment_id == self.fast_segment.segment_id
            and self.selected_segment.revision == self.fast_segment.revision
        )

    @property
    def rejected_alternatives(self) -> tuple[AlternativeAssessment, ...]:
        return tuple(item for item in self.alternatives if not item.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_segment": self.fast_segment.to_dict(),
            "selected_segment": self.selected_segment.to_dict(),
            "alternatives": [item.to_dict() for item in self.alternatives],
            "selection_reason": self.selection_reason,
            "preserved_fast": self.preserved_fast,
        }


def select_segment_revision(
    fast_segment: AsrSegment,
    alternatives: Iterable[AsrSegment] = (),
    *,
    allow_replacement: bool = False,
    minimum_confidence_gain: float = 0.05,
) -> RevisionSelection:
    """Preserve the fast transcript unless a safe alternative earns promotion.

    ``allow_replacement`` is intentionally false by default.  Even when true,
    a candidate must pass anomaly gates and either be explicitly
    ``review_approved``, repair an anomalous fast result, or provide a measured
    confidence gain.  Raw alternatives remain attached to the decision.
    """

    assessments = tuple(assess_alternative(fast_segment, item) for item in alternatives)
    if not allow_replacement:
        reason = "fast_default_no_replacement_requested"
        if any(not item.accepted for item in assessments):
            reason = "fast_preserved_after_rejected_alternative"
        return RevisionSelection(fast_segment, fast_segment, assessments, reason)

    eligible = [item.candidate for item in assessments if item.accepted]
    if not eligible:
        return RevisionSelection(
            fast_segment,
            fast_segment,
            assessments,
            "fast_preserved_no_safe_alternative",
        )
    fast_flags = tuple(dict.fromkeys(fast_segment.anomaly_flags + transcription_anomaly_flags(fast_segment.text)))

    def promotion_rank(candidate: AsrSegment) -> tuple[int, float, int, int]:
        approved = bool(candidate.metadata.get("review_approved") or candidate.metadata.get("validated"))
        candidate_confidence = candidate.confidence if candidate.confidence is not None else -1.0
        gain = (
            candidate_confidence - fast_segment.confidence
            if fast_segment.confidence is not None and candidate.confidence is not None
            else -1.0
        )
        promotable = approved or bool(fast_flags) or gain >= minimum_confidence_gain
        return (
            int(promotable),
            candidate_confidence,
            candidate.revision,
            len(_compact_text(candidate.text)),
        )

    selected = max(eligible, key=promotion_rank)
    rank = promotion_rank(selected)
    if not rank[0]:
        return RevisionSelection(
            fast_segment,
            fast_segment,
            assessments,
            "fast_preserved_alternative_not_verified",
        )
    return RevisionSelection(fast_segment, selected, assessments, "safe_alternative_promoted")


def _selection_confidence(selection: RevisionSelection) -> tuple[EvidenceConfidence, tuple[str, ...]]:
    segment = selection.selected_segment
    reasons: list[str] = list(segment.anomaly_flags)
    reasons.extend(transcription_anomaly_flags(segment.text))
    if str(segment.metadata.get("cascade_action", "")).casefold() == "review":
        reasons.extend(str(item) for item in segment.metadata.get("cascade_reasons", ()))
        reasons.append("cascade_review_required")
        return "review", tuple(dict.fromkeys(reasons))
    if reasons:
        return "review", tuple(dict.fromkeys(reasons))
    if segment.confidence is not None and segment.confidence < 0.35:
        return "review", ("low_asr_confidence",)
    if selection.rejected_alternatives:
        reasons.append("unsafe_alternative_rejected")
    if segment.confidence is None or segment.confidence < 0.75 or reasons:
        return "medium", tuple(dict.fromkeys(reasons))
    return "high", ()


def _lowest_confidence(values: Iterable[EvidenceConfidence]) -> EvidenceConfidence:
    rank = {"high": 0, "medium": 1, "review": 2}
    return max(values, key=rank.__getitem__)


def _participant_for_selection(
    selection: RevisionSelection,
    speaker_assignments: Mapping[str, SpeakerAssignment],
    identity_decisions: Mapping[str, IdentityDecision],
) -> tuple[ParticipantRole, str | None]:
    segment_id = selection.fast_segment.segment_id
    assignment = speaker_assignments.get(segment_id)
    if assignment is None:
        assignment = speaker_assignments.get(selection.selected_segment.segment_id)
    if assignment is None:
        return "unknown", None
    if assignment.participant_role in {"mixed", "unknown"} or not assignment.local_speaker_id:
        return assignment.participant_role, None
    identity = identity_decisions.get(assignment.local_speaker_id)
    if identity is not None and identity.status == "owner" and identity.identity_id:
        return "owner", identity.identity_id
    return "anonymous", None


def _stable_evidence_id(
    source_id: str,
    start: float,
    end: float,
    selections: Sequence[RevisionSelection],
) -> str:
    digest = hashlib.sha256()
    digest.update(source_id.encode())
    digest.update(f"\0{round(start * 1000)}\0{round(end * 1000)}".encode())
    for selection in selections:
        segment = selection.selected_segment
        digest.update(
            f"\0{segment.segment_id}:{segment.revision}:{segment.model_id or ''}".encode()
        )
    return f"evidence-{digest.hexdigest()[:20]}"


def build_evidence_windows(
    segments: Iterable[AsrSegment | RevisionSelection],
    *,
    window_seconds: float = 30.0,
    speaker_assignments: Mapping[str, SpeakerAssignment] | None = None,
    identity_decisions: Mapping[str, IdentityDecision] | None = None,
) -> tuple[EvidenceWindow, ...]:
    """Build deterministic, source-local evidence windows.

    Inputs may be raw fast segments or completed revision decisions.  Raw
    segments are treated as immutable fast defaults.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    assignments = speaker_assignments or {}
    identities = identity_decisions or {}
    selections = [
        item
        if isinstance(item, RevisionSelection)
        else RevisionSelection(item, item, (), "fast_default")
        for item in segments
    ]
    selections.sort(
        key=lambda item: (
            item.selected_segment.source_id,
            item.selected_segment.start,
            item.selected_segment.end,
            item.selected_segment.segment_id,
        )
    )
    if not selections:
        return ()
    logical_keys = [
        (item.fast_segment.source_id, item.fast_segment.segment_id) for item in selections
    ]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("evidence input contains duplicate logical ASR segments")

    grouped: list[list[RevisionSelection]] = []
    active: list[RevisionSelection] = []
    active_source: str | None = None
    anchor = 0.0
    for selection in selections:
        segment = selection.selected_segment
        if (
            not active
            or segment.source_id != active_source
            or (
                segment.start >= anchor + window_seconds
                and segment.start >= max(item.selected_segment.end for item in active)
            )
        ):
            if active:
                grouped.append(active)
            active = [selection]
            active_source = segment.source_id
            anchor = math.floor(segment.start / window_seconds) * window_seconds
        else:
            active.append(selection)
    if active:
        grouped.append(active)

    windows: list[EvidenceWindow] = []
    for group in grouped:
        source_id = group[0].selected_segment.source_id
        start = min(item.selected_segment.start for item in group)
        end = max(item.selected_segment.end for item in group)
        text = " ".join(_compact_text(item.selected_segment.text) for item in group).strip()
        confidence_rows = [_selection_confidence(item) for item in group]
        confidence = _lowest_confidence(row[0] for row in confidence_rows)
        review_reasons = tuple(
            dict.fromkeys(reason for _, reasons in confidence_rows for reason in reasons)
        )
        participants = [
            _participant_for_selection(item, assignments, identities) for item in group
        ]
        roles = {role for role, _ in participants}
        identity_ids = {identity_id for _, identity_id in participants if identity_id is not None}
        if roles == {"owner"} and len(identity_ids) == 1:
            role: ParticipantRole = "owner"
            identity_id = next(iter(identity_ids))
        elif roles == {"anonymous"}:
            role = "anonymous"
            identity_id = None
        elif roles == {"unknown"}:
            role = "unknown"
            identity_id = None
        else:
            role = "mixed"
            identity_id = None
        model_states = {item.selection_reason for item in group}
        model_state = next(iter(model_states)) if len(model_states) == 1 else "mixed_revisions"
        windows.append(
            EvidenceWindow(
                evidence_window_id=_stable_evidence_id(source_id, start, end, group),
                source_id=source_id,
                start=start,
                end=end,
                text=text,
                confidence=confidence,
                model_state=model_state,
                summary_sensitive=text_is_summary_sensitive(text),
                segment_ids=tuple(item.selected_segment.segment_id for item in group),
                participant_role=role,
                identity_id=identity_id,
                review_reasons=review_reasons,
            )
        )
    return tuple(windows)


def evidence_by_id(windows: Iterable[EvidenceWindow]) -> dict[str, EvidenceWindow]:
    result: dict[str, EvidenceWindow] = {}
    for window in windows:
        if window.evidence_window_id in result:
            raise ValueError(f"duplicate evidence window id: {window.evidence_window_id}")
        result[window.evidence_window_id] = window
    return result


class SegmentStore(Protocol):
    """Structural subset of :class:`dayaudio.storage.Storage` used here."""

    def list_segments(
        self,
        *,
        source_id: str | None = None,
        block_id: str | None = None,
        latest_only: bool = True,
    ) -> list[AsrSegment]: ...


def select_revisions_from_storage(
    storage: SegmentStore,
    *,
    source_id: str | None = None,
    allow_replacement: bool = False,
) -> tuple[RevisionSelection, ...]:
    """Select authoritative revisions only from completed pipeline tasks.

    Revisions sharing ``segment_id`` (or an explicit ``base_segment_id`` in
    metadata) are evaluated together.  The earliest revision, or a segment
    marked ``is_fast``/``stage=fast``, is the fast default.
    """

    stored = storage.list_segments(source_id=source_id, latest_only=False)
    all_revisions_by_task: dict[str, list[AsrSegment]] = {}
    for stored_segment in stored:
        stored_task_key = stored_segment.metadata.get("pipeline_task_key")
        if stored_task_key:
            all_revisions_by_task.setdefault(str(stored_task_key), []).append(stored_segment)

    queue_factory = getattr(storage, "task_queue", None)
    queue = queue_factory() if callable(queue_factory) else None

    def task_is_complete(task_key: str | None, task_revisions: Sequence[AsrSegment]) -> bool:
        if not task_key or queue is None or not hasattr(queue, "get"):
            return True
        task = queue.get(task_key)
        if task is None:
            return False
        status = getattr(task, "status", None)
        if str(getattr(status, "value", status)).casefold() != "complete":
            return False
        list_artifacts = getattr(storage, "list_artifacts", None)
        if not callable(list_artifacts):
            return True
        artifacts = list(
            list_artifacts(kind="asr-block-manifest", task_key=task_key) or ()
        )
        finals = {
            (item.segment_id, item.revision): item
            for item in task_revisions
            if str(item.metadata.get("pipeline_stage", "")).casefold() == "final"
        }
        for artifact in reversed(artifacts):
            metadata = getattr(artifact, "metadata", None)
            if not isinstance(metadata, Mapping):
                continue
            if (
                metadata.get("manifest_version") != 1
                or metadata.get("complete") is not True
                or metadata.get("task_key") != task_key
            ):
                continue
            canonical = {key: value for key, value in metadata.items() if key != "metadata_only"}
            encoded = json.dumps(
                canonical,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            if getattr(artifact, "sha256", None) != digest:
                continue
            if metadata.get("metadata_only") is not True:
                try:
                    marker = Path(str(getattr(artifact, "path"))).read_bytes()
                except OSError:
                    continue
                if marker != encoded or hashlib.sha256(marker).hexdigest() != digest:
                    continue
            expected_value = metadata.get("final_revisions")
            if not isinstance(expected_value, list):
                continue
            try:
                expected = [(str(item[0]), int(item[1])) for item in expected_value]
            except (TypeError, ValueError, IndexError):
                continue
            if metadata.get("final_count") != len(expected):
                continue
            if all(item in finals for item in expected) and len(expected) == len(finals):
                return True
        return False

    grouped: dict[tuple[str, str], list[AsrSegment]] = {}
    for segment in stored:
        base_id = str(segment.metadata.get("base_segment_id") or segment.segment_id)
        grouped.setdefault((segment.source_id, base_id), []).append(segment)
    selections: list[RevisionSelection] = []
    for _, revisions in sorted(
        grouped.items(),
        key=lambda item: (
            min(segment.start for segment in item[1]),
            item[0][0],
            item[0][1],
        ),
    ):
        revisions.sort(key=lambda segment: (segment.revision, segment.segment_id))
        revisions_by_task: dict[str, list[AsrSegment]] = {}
        legacy_revisions: list[AsrSegment] = []
        for segment in revisions:
            task_key = segment.metadata.get("pipeline_task_key")
            if task_key:
                revisions_by_task.setdefault(str(task_key), []).append(segment)
            else:
                legacy_revisions.append(segment)
        if revisions_by_task:
            completed = [
                values
                for key, values in revisions_by_task.items()
                if task_is_complete(key, all_revisions_by_task.get(key, values))
            ]
            if completed:
                # A pending reprocess must not hide a prior completed result.
                # Storage revision numbers monotonically increase on conflict,
                # so the highest completed revision is the newest safe run.
                revisions = max(
                    completed,
                    key=lambda values: max(item.revision for item in values),
                )
                revisions.sort(key=lambda segment: (segment.revision, segment.segment_id))
            elif legacy_revisions:
                revisions = legacy_revisions
            else:
                # A killed worker can leave raw or partial final rows.  They
                # remain provenance until the durable task is complete.
                continue
        explicitly_fast = [
            segment
            for segment in revisions
            if segment.metadata.get("is_fast") is True
            or str(segment.metadata.get("stage", "")).casefold() == "fast"
        ]
        fast = explicitly_fast[0] if explicitly_fast else revisions[0]
        alternatives = tuple(segment for segment in revisions if segment is not fast)
        pipeline_finals = [
            segment
            for segment in alternatives
            if str(segment.metadata.get("pipeline_stage", "")).casefold() == "final"
            or str(segment.metadata.get("stage", "")).casefold() == "final"
        ]
        if pipeline_finals:
            selected = max(
                pipeline_finals,
                key=lambda segment: (segment.revision, segment.segment_id),
            )
            selections.append(
                RevisionSelection(
                    fast,
                    selected,
                    tuple(assess_alternative(fast, item) for item in alternatives),
                    "pipeline_final_" + str(selected.metadata.get("cascade_action") or "selected"),
                )
            )
        else:
            selections.append(
                select_segment_revision(
                    fast,
                    alternatives,
                    allow_replacement=allow_replacement,
                )
            )
    return tuple(selections)


def build_evidence_from_storage(
    storage: SegmentStore,
    *,
    source_id: str | None = None,
    allow_replacement: bool = False,
    window_seconds: float = 30.0,
    speaker_assignments: Mapping[str, SpeakerAssignment] | None = None,
    identity_decisions: Mapping[str, IdentityDecision] | None = None,
) -> tuple[EvidenceWindow, ...]:
    """Load completed immutable revisions and build evidence windows."""

    return build_evidence_windows(
        select_revisions_from_storage(
            storage,
            source_id=source_id,
            allow_replacement=allow_replacement,
        ),
        window_seconds=window_seconds,
        speaker_assignments=speaker_assignments,
        identity_decisions=identity_decisions,
    )


# Familiar aliases for callers that use refinement terminology.
choose_revision = select_segment_revision
make_evidence_windows = build_evidence_windows


__all__ = [
    "AlternativeAssessment",
    "RevisionSelection",
    "assess_alternative",
    "build_evidence_windows",
    "build_evidence_from_storage",
    "choose_revision",
    "evidence_by_id",
    "make_evidence_windows",
    "select_revisions_from_storage",
    "select_segment_revision",
    "text_is_summary_sensitive",
    "transcription_anomaly_flags",
]
