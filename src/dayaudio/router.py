"""Deterministic anomaly detection and conservative ASR cascade routing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Literal

from dayaudio.types import AsrSegment

RouteAction = Literal["keep-fast", "accept-strong", "review"]

_INFORMATIONAL_FLAGS = frozenset(
    {"boundary_timestamp_clipped", "overlap_timestamp_clipped"}
)

_REFUSAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi am unable to transcribe(?: this)? audio(?: into text)?\b",
        r"\bi cannot transcribe(?: this)? audio(?: into text)?\b",
        r"\bsorry[,，]?\s*i (?:cannot|can't) transcribe\b",
        r"\bas an ai language model\b",
        r"(?:抱歉|对不起).{0,12}(?:无法|不能).{0,12}(?:转录|识别).{0,8}(?:音频|语音)",
        r"(?:无法|不能).{0,10}(?:转录|识别)(?:这段|该|此)?.{0,6}(?:音频|语音)",
    )
)


@dataclass(frozen=True, slots=True)
class AnomalyPolicy:
    min_chars_per_second: float = 0.05
    max_chars_per_second: float = 25.0
    min_duration_for_low_rate: float = 8.0
    min_repetition_chars: int = 12
    repeated_character_fraction: float = 0.55
    ngram_diversity_threshold: float = 0.20


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    flags: tuple[str, ...]
    normalized_characters: int
    chars_per_second: float | None

    @property
    def anomalous(self) -> bool:
        return bool(self.flags)

    @property
    def severe(self) -> bool:
        return any(
            flag
            in {
                "punctuation_only",
                "model_refusal",
                "repetition_loop",
                "chars_per_second_high",
            }
            for flag in self.flags
        )


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    escalate: bool
    reasons: tuple[str, ...]
    anomalies: AnomalyReport


@dataclass(frozen=True, slots=True)
class CascadePolicy:
    anomaly: AnomalyPolicy = AnomalyPolicy()
    low_confidence_threshold: float = 0.55
    strong_confidence_threshold: float = 0.45
    min_text_coverage: float = 0.65
    max_text_expansion: float = 2.50
    min_consensus_similarity: float = 0.55
    escalate_summary_sensitive: bool = True


@dataclass(frozen=True, slots=True)
class CascadeDecision:
    selected: AsrSegment
    action: RouteAction
    reasons: tuple[str, ...]
    fast_anomalies: AnomalyReport
    strong_anomalies: AnomalyReport | None
    text_coverage: float | None
    text_expansion: float | None
    consensus_similarity: float | None

    @property
    def accepted_strong(self) -> bool:
        return self.action == "accept-strong"


def normalize_text(text: str) -> str:
    """Normalize for structural comparison without rewriting stored output."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def text_similarity(left: str, right: str) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > 10_000:
        a = a[:5_000] + a[-5_000:]
    if len(b) > 10_000:
        b = b[:5_000] + b[-5_000:]
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _has_repetition(text: str, policy: AnomalyPolicy) -> bool:
    if len(text) < policy.min_repetition_chars:
        return False
    sample = text if len(text) <= 10_000 else text[:5_000] + text[-5_000:]
    most_common = max(Counter(sample).values(), default=0)
    if most_common / len(sample) >= policy.repeated_character_fraction:
        return True

    # Four consecutive repeats of a short unit catch loops such as ``you just``
    # while remaining deterministic and cheap for long transcripts.
    maximum_unit = min(12, max(1, len(sample) // 4))
    for unit_length in range(1, maximum_unit + 1):
        for start in range(0, min(len(sample), 48)):
            unit = sample[start : start + unit_length]
            if len(unit) < unit_length:
                break
            repeated = unit * 4
            if repeated in sample:
                return True

    if len(sample) >= 40:
        trigrams = [sample[index : index + 3] for index in range(len(sample) - 2)]
        diversity = len(set(trigrams)) / len(trigrams)
        if diversity < policy.ngram_diversity_threshold:
            return True
    return False


def detect_anomalies(
    value: AsrSegment | str,
    *,
    duration_seconds: float | None = None,
    policy: AnomalyPolicy | None = None,
) -> AnomalyReport:
    policy = policy or AnomalyPolicy()
    if isinstance(value, AsrSegment):
        text = value.text
        duration = value.end - value.start if duration_seconds is None else duration_seconds
        existing = set(value.anomaly_flags)
    else:
        text = value
        duration = duration_seconds
        existing = set()

    flags = existing
    structural = normalize_text(text)
    if text.strip() and not structural:
        flags.add("punctuation_only")
    if any(pattern.search(text) for pattern in _REFUSAL_PATTERNS):
        flags.add("model_refusal")
    if _has_repetition(structural, policy):
        flags.add("repetition_loop")

    rate: float | None = None
    if duration is not None and duration > 0:
        rate = len(structural) / duration
        if rate > policy.max_chars_per_second:
            flags.add("chars_per_second_high")
        if (
            duration >= policy.min_duration_for_low_rate
            and structural
            and rate < policy.min_chars_per_second
        ):
            flags.add("chars_per_second_low")
    return AnomalyReport(tuple(sorted(flags)), len(structural), rate)


def annotate_anomalies(
    segment: AsrSegment, policy: AnomalyPolicy | None = None
) -> AsrSegment:
    report = detect_anomalies(segment, policy=policy)
    if report.flags == segment.anomaly_flags:
        return segment
    metadata = dict(segment.metadata)
    metadata["anomaly_chars_per_second"] = report.chars_per_second
    return replace(segment, anomaly_flags=report.flags, metadata=metadata)


class CascadeRouter:
    """Select revisions conservatively; disagreement is evidence for review."""

    def __init__(self, policy: CascadePolicy | None = None) -> None:
        self.policy = policy or CascadePolicy()

    def should_escalate(
        self,
        segment: AsrSegment,
        *,
        summary_sensitive: bool = False,
        force: bool = False,
    ) -> EscalationDecision:
        report = detect_anomalies(segment, policy=self.policy.anomaly)
        reasons: list[str] = []
        if force:
            reasons.append("forced")
        quality_flags = [
            flag for flag in report.flags if flag not in _INFORMATIONAL_FLAGS
        ]
        if quality_flags:
            reasons.extend(f"fast:{flag}" for flag in quality_flags)
        if (
            segment.confidence is not None
            and segment.confidence < self.policy.low_confidence_threshold
        ):
            reasons.append("fast:low_confidence")
        if summary_sensitive and self.policy.escalate_summary_sensitive:
            reasons.append("summary_sensitive")
        return EscalationDecision(bool(reasons), tuple(reasons), report)

    def evaluate(
        self,
        fast: AsrSegment,
        strong: AsrSegment,
        *,
        consensus: AsrSegment | str | None = None,
    ) -> CascadeDecision:
        """Evaluate one proposed strong revision without mutating raw outputs."""

        fast_report = detect_anomalies(fast, policy=self.policy.anomaly)
        strong_report = detect_anomalies(strong, policy=self.policy.anomaly)
        fast_text = normalize_text(fast.text)
        strong_text = normalize_text(strong.text)
        fast_length = max(1, len(fast_text))
        coverage = len(strong_text) / fast_length
        expansion = len(strong_text) / fast_length
        comparison_text = (
            consensus.text if isinstance(consensus, AsrSegment) else consensus
        )
        if comparison_text is None:
            comparison_text = fast.text
        similarity = text_similarity(strong.text, comparison_text)

        rejection: list[str] = []
        strong_quality_flags = [
            flag for flag in strong_report.flags if flag not in _INFORMATIONAL_FLAGS
        ]
        if strong_quality_flags:
            rejection.extend(f"strong:{flag}" for flag in strong_quality_flags)
        if coverage < self.policy.min_text_coverage:
            rejection.append("strong:insufficient_text_coverage")
        if expansion > self.policy.max_text_expansion:
            rejection.append("strong:excessive_text_expansion")
        if (
            strong.confidence is not None
            and strong.confidence < self.policy.strong_confidence_threshold
        ):
            rejection.append("strong:low_confidence")

        # A structurally broken fast result cannot corroborate a replacement.
        # It therefore requires a third result (or human review) rather than
        # allowing the strong model to overwrite it on its own.
        if fast_report.severe and consensus is None:
            rejection.append("strong:independent_consensus_required")
        elif similarity < self.policy.min_consensus_similarity:
            rejection.append("strong:consensus_below_threshold")

        if rejection:
            selected = annotate_anomalies(fast, self.policy.anomaly)
            return CascadeDecision(
                selected=selected,
                action="review",
                reasons=tuple(dict.fromkeys(rejection)),
                fast_anomalies=fast_report,
                strong_anomalies=strong_report,
                text_coverage=coverage,
                text_expansion=expansion,
                consensus_similarity=similarity,
            )

        if strong.text == fast.text:
            return CascadeDecision(
                selected=annotate_anomalies(fast, self.policy.anomaly),
                action="keep-fast",
                reasons=("equivalent_text",),
                fast_anomalies=fast_report,
                strong_anomalies=strong_report,
                text_coverage=coverage,
                text_expansion=expansion,
                consensus_similarity=similarity,
            )

        metadata: dict[str, Any] = dict(strong.metadata)
        metadata.update(
            {
                "cascade": {
                    "fast_model_id": fast.model_id,
                    "strong_model_id": strong.model_id,
                    "strong_segment_id": strong.segment_id,
                    "strong_bounds": [strong.start, strong.end],
                    "text_coverage": coverage,
                    "text_expansion": expansion,
                    "consensus_similarity": similarity,
                }
            }
        )
        selected = replace(
            strong,
            segment_id=fast.segment_id,
            source_id=fast.source_id,
            start=fast.start,
            end=fast.end,
            revision=max(fast.revision + 1, strong.revision),
            block_id=fast.block_id,
            anomaly_flags=strong_report.flags,
            metadata=metadata,
        )
        return CascadeDecision(
            selected=selected,
            action="accept-strong",
            reasons=("coverage_and_consensus_passed",),
            fast_anomalies=fast_report,
            strong_anomalies=strong_report,
            text_coverage=coverage,
            text_expansion=expansion,
            consensus_similarity=similarity,
        )

    route_segment = evaluate


def route_revision(
    fast: AsrSegment,
    strong: AsrSegment,
    *,
    consensus: AsrSegment | str | None = None,
    policy: CascadePolicy | None = None,
) -> CascadeDecision:
    return CascadeRouter(policy).evaluate(fast, strong, consensus=consensus)


__all__ = [
    "AnomalyPolicy",
    "AnomalyReport",
    "CascadeDecision",
    "CascadePolicy",
    "CascadeRouter",
    "EscalationDecision",
    "annotate_anomalies",
    "detect_anomalies",
    "normalize_text",
    "route_revision",
    "text_similarity",
]
