"""Structured summary backends and evidence-citation validation.

The same JSON contract is used by the dependency-free extractive fallback,
local command adapters, and opt-in HTTP adapters.  Network access is never
required by importing or using the core fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.request import Request, urlopen

from .bundles import SummaryPacket
from .evidence import evidence_by_id, text_is_summary_sensitive
from .types import EvidenceWindow


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    request_id: str
    scope_id: str
    evidence: tuple[EvidenceWindow, ...]
    language: str = "zh-CN"
    max_claims: int = 12
    instructions: str | None = None
    packet_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.scope_id:
            raise ValueError("summary request and scope ids must not be empty")
        if self.max_claims < 1:
            raise ValueError("summary max_claims must be positive")
        evidence_by_id(self.evidence)  # duplicate check

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dayaudio.summary.request.v1",
            "request_id": self.request_id,
            "scope_id": self.scope_id,
            "language": self.language,
            "max_claims": self.max_claims,
            "instructions": self.instructions,
            "packet_ids": list(self.packet_ids),
            "evidence": [window.to_dict() for window in self.evidence],
            "metadata": self.metadata,
            "output_contract": {
                "claims": [
                    {
                        "claim_id": "string",
                        "text": "string",
                        "evidence_ids": ["evidence-id"],
                        "category": "observation|decision|action|owner|other",
                        "state": "observed|suggested|intended|completed|unknown",
                    }
                ]
            },
        }


def make_summary_request(
    scope_id: str,
    evidence: Iterable[EvidenceWindow],
    *,
    language: str = "zh-CN",
    max_claims: int = 12,
    instructions: str | None = None,
    packet_ids: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> SummaryRequest:
    windows = tuple(evidence)
    packet_id_rows = tuple(packet_ids)
    metadata_value = dict(metadata or {})
    request_material = {
        "scope_id": scope_id,
        "language": language,
        "max_claims": max_claims,
        "instructions": instructions,
        "packet_ids": list(packet_id_rows),
        "metadata": metadata_value,
        "evidence": [window.to_dict() for window in windows],
    }
    digest = hashlib.sha256(
        json.dumps(
            request_material,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return SummaryRequest(
        request_id=f"summary-request-{digest.hexdigest()[:20]}",
        scope_id=scope_id,
        evidence=windows,
        language=language,
        max_claims=max_claims,
        instructions=instructions,
        packet_ids=packet_id_rows,
        metadata=metadata_value,
    )


def request_from_packet(
    packet: SummaryPacket,
    evidence: Iterable[EvidenceWindow],
    *,
    language: str = "zh-CN",
    max_claims: int = 8,
    instructions: str | None = None,
) -> SummaryRequest:
    by_id = evidence_by_id(evidence)
    missing = [identifier for identifier in packet.evidence_window_ids if identifier not in by_id]
    if missing:
        raise ValueError(f"summary packet references unknown evidence: {missing[0]}")
    windows = tuple(by_id[identifier] for identifier in packet.evidence_window_ids)
    return make_summary_request(
        packet.packet_id,
        windows,
        language=language,
        max_claims=max_claims,
        instructions=instructions,
        packet_ids=(packet.packet_id,),
        metadata={"bundle_id": packet.bundle_id, "day_key": packet.day_key},
    )


@dataclass(frozen=True, slots=True)
class SummaryClaim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]
    category: str = "observation"
    state: str = "observed"

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("summary claim id must not be empty")
        if not self.text.strip():
            raise ValueError("summary claim text must not be empty")
        if self.state not in {"observed", "suggested", "intended", "completed", "unknown"}:
            raise ValueError("unsupported summary claim state")

    @property
    def sensitive(self) -> bool:
        return self.category in {"decision", "action", "name", "number", "negation"} or text_is_summary_sensitive(self.text)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


@dataclass(frozen=True, slots=True)
class SummaryResult:
    summary_id: str
    request_id: str
    scope_id: str
    backend_id: str
    claims: tuple[SummaryClaim, ...]
    created_at: str = field(default_factory=_utc_now)
    model_id: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary_id or not self.request_id or not self.scope_id or not self.backend_id:
            raise ValueError("summary result identifiers must not be empty")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("summary result contains duplicate claim ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dayaudio.summary.result.v1",
            "summary_id": self.summary_id,
            "request_id": self.request_id,
            "scope_id": self.scope_id,
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "claims": [claim.to_dict() for claim in self.claims],
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        lines = [f"# Summary: {self.scope_id}", ""]
        for claim in self.claims:
            citations = " ".join(f"[{identifier}]" for identifier in claim.evidence_ids)
            lines.append(f"- {claim.text} {citations}".rstrip())
        return "\n".join(lines) + "\n"


def summary_result_from_dict(
    data: Mapping[str, Any],
    *,
    request: SummaryRequest,
    backend_id: str,
) -> SummaryResult:
    claims_data = data.get("claims")
    if not isinstance(claims_data, list):
        raise ValueError("summary backend response must contain a claims list")
    claims: list[SummaryClaim] = []
    for index, item in enumerate(claims_data):
        if not isinstance(item, Mapping):
            raise ValueError(f"summary claim {index} must be an object")
        text = str(item.get("text", "")).strip()
        evidence_ids_value = item.get("evidence_ids", ())
        if not isinstance(evidence_ids_value, (list, tuple)):
            raise ValueError(f"summary claim {index} evidence_ids must be a list")
        claim_id = str(item.get("claim_id") or _claim_id(request.scope_id, text, evidence_ids_value))
        claims.append(
            SummaryClaim(
                claim_id=claim_id,
                text=text,
                evidence_ids=tuple(str(identifier) for identifier in evidence_ids_value),
                category=str(item.get("category", "observation")),
                state=str(item.get("state", "observed")),
            )
        )
    model_id = str(data["model_id"]) if data.get("model_id") is not None else None
    summary_id = _summary_id(request.request_id, backend_id, claims, model_id=model_id)
    metadata = dict(data.get("metadata", {}))
    if data.get("summary_id") is not None and str(data["summary_id"]) != summary_id:
        metadata["external_summary_id"] = str(data["summary_id"])
    return SummaryResult(
        summary_id=summary_id,
        request_id=str(data.get("request_id") or request.request_id),
        scope_id=str(data.get("scope_id") or request.scope_id),
        backend_id=backend_id,
        claims=tuple(claims),
        created_at=str(data.get("created_at") or _utc_now()),
        model_id=model_id,
        raw_text=str(data["raw_text"]) if data.get("raw_text") is not None else None,
        metadata=metadata,
    )


def _claim_id(scope_id: str, text: str, evidence_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(f"{scope_id}\0{text}".encode())
    for identifier in evidence_ids:
        digest.update(b"\0")
        digest.update(str(identifier).encode())
    return f"claim-{digest.hexdigest()[:20]}"


def _summary_id(
    request_id: str,
    backend_id: str,
    claims: Iterable[SummaryClaim],
    *,
    model_id: str | None = None,
) -> str:
    digest = hashlib.sha256(f"{request_id}\0{backend_id}\0{model_id or ''}".encode())
    for claim in claims:
        digest.update(b"\0")
        digest.update(
            json.dumps(
                claim.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return f"summary-{digest.hexdigest()[:20]}"


class SummaryBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def summarize(self, request: SummaryRequest) -> SummaryResult: ...


def _first_sentence(text: str, max_characters: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    pieces = re.split(r"(?<=[。！？!?])\s*|(?<=[.;])\s+", compact, maxsplit=1)
    sentence = pieces[0].strip()
    if len(sentence) <= max_characters:
        return sentence
    return sentence[: max_characters - 1].rstrip() + "…"


def _claim_state(text: str) -> str:
    if re.search(r"建议|提议|可以考虑|should|suggest|recommend", text, re.IGNORECASE):
        return "suggested"
    if re.search(r"计划|准备|打算|将要|intend|plan(?:ned)? to", text, re.IGNORECASE):
        return "intended"
    if re.search(r"已经|已完成|完成了|决定|done|completed|decided", text, re.IGNORECASE):
        return "completed"
    return "observed"


class ExtractiveSummaryBackend:
    """Deterministic offline fallback that never invents an uncited claim."""

    backend_id = "extractive-local-v1"

    def summarize(self, request: SummaryRequest) -> SummaryResult:
        scored: list[tuple[tuple[int, int, int, int], EvidenceWindow]] = []
        for index, window in enumerate(request.evidence):
            sentence = _first_sentence(window.text)
            if not sentence:
                continue
            sensitive = window.summary_sensitive or text_is_summary_sensitive(sentence)
            # A review-only sensitive statement cannot enter even the local
            # fallback, because it would fail the final citation gate.
            if window.confidence == "review" and sensitive:
                continue
            score = (
                int(window.confidence != "review"),
                int(window.summary_sensitive),
                int(window.participant_role == "owner"),
                -index,
            )
            scored.append((score, window))
        scored.sort(key=lambda item: item[0], reverse=True)

        claims: list[SummaryClaim] = []
        normalized_seen: set[str] = set()
        for _, window in scored:
            text = _first_sentence(window.text)
            normalized = re.sub(r"\W+", "", text).casefold()
            if not normalized or normalized in normalized_seen:
                continue
            normalized_seen.add(normalized)
            category = "owner" if window.participant_role == "owner" else "observation"
            state = _claim_state(text)
            if text_is_summary_sensitive(text):
                if re.search(r"决定|确定|decid", text, re.IGNORECASE):
                    category = "decision"
                elif state != "observed":
                    category = "action"
            claim = SummaryClaim(
                claim_id=_claim_id(request.scope_id, text, (window.evidence_window_id,)),
                text=text,
                evidence_ids=(window.evidence_window_id,),
                category=category,
                state=state,
            )
            claims.append(claim)
            if len(claims) >= request.max_claims:
                break
        if not claims and request.evidence:
            review_ids = tuple(window.evidence_window_id for window in request.evidence)
            text = "该范围仅包含待复核的转录证据，暂不提取事实性结论。"
            claims.append(
                SummaryClaim(
                    claim_id=_claim_id(request.scope_id, text, review_ids),
                    text=text,
                    evidence_ids=review_ids,
                    category="review_notice",
                    state="unknown",
                )
            )
        result = SummaryResult(
            summary_id="pending",
            request_id=request.request_id,
            scope_id=request.scope_id,
            backend_id=self.backend_id,
            claims=tuple(claims),
            metadata={"fallback": True},
        )
        return SummaryResult(
            summary_id=_summary_id(request.request_id, self.backend_id, result.claims),
            request_id=result.request_id,
            scope_id=result.scope_id,
            backend_id=result.backend_id,
            claims=result.claims,
            created_at=result.created_at,
            metadata=result.metadata,
        )


# Common concise name.
ExtractiveSummarizer = ExtractiveSummaryBackend


class CommandSummaryBackend:
    """Run a local executable speaking the summary JSON contract over stdio."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        backend_id: str = "command",
        timeout_seconds: float = 300.0,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("summary command must not be empty")
        self.command = tuple(command)
        descriptor = hashlib.sha256(
            json.dumps(list(self.command), separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self.backend_id = f"command-{descriptor}" if backend_id == "command" else backend_id
        self.timeout_seconds = timeout_seconds
        self.environment = dict(environment or {})

    def summarize(self, request: SummaryRequest) -> SummaryResult:
        environment = os.environ.copy()
        environment.update(self.environment)
        completed = subprocess.run(
            self.command,
            input=json.dumps(request.to_dict(), ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            # Subprocess streams may contain transcript text, prompts, or user
            # paths.  Keep the default exception safe for terminals and logs.
            raise RuntimeError(
                f"summary command failed with exit code {completed.returncode}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("summary command did not return valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("summary command response must be a JSON object")
        return summary_result_from_dict(payload, request=request, backend_id=self.backend_id)


class HttpSummaryBackend:
    """Opt-in HTTP adapter using the same contract as local commands."""

    def __init__(
        self,
        endpoint: str,
        *,
        backend_id: str = "http",
        timeout_seconds: float = 300.0,
        headers: Mapping[str, str] | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("summary endpoint must be HTTP or HTTPS")
        if endpoint.startswith("http://") and not allow_insecure_http:
            raise ValueError(
                "plain HTTP summary endpoints require explicit allow_insecure_http=True"
            )
        self.endpoint = endpoint
        descriptor = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
        self.backend_id = f"http-{descriptor}" if backend_id == "http" else backend_id
        self.timeout_seconds = timeout_seconds
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}

    def summarize(self, request: SummaryRequest) -> SummaryResult:
        body = json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8")
        http_request = Request(self.endpoint, data=body, headers=self.headers, method="POST")
        with urlopen(http_request, timeout=self.timeout_seconds) as response:  # noqa: S310
            payload_bytes = response.read()
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("summary HTTP endpoint did not return valid UTF-8 JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("summary HTTP response must be a JSON object")
        return summary_result_from_dict(payload, request=request, backend_id=self.backend_id)


@dataclass(frozen=True, slots=True)
class CitationViolation:
    code: str
    message: str
    claim_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CitationValidationReport:
    valid: bool
    violations: tuple[CitationViolation, ...]
    checked_claims: int
    checked_citations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_claims": self.checked_claims,
            "checked_citations": self.checked_citations,
            "violations": [
                {
                    "code": item.code,
                    "message": item.message,
                    "claim_id": item.claim_id,
                    "evidence_ids": list(item.evidence_ids),
                }
                for item in self.violations
            ],
        }


class CitationValidationError(ValueError):
    def __init__(self, report: CitationValidationReport) -> None:
        self.report = report
        super().__init__(
            "; ".join(violation.message for violation in report.violations)
            or "summary citation validation failed"
        )


def validate_summary_citations(
    result: SummaryResult,
    evidence: Iterable[EvidenceWindow] | Mapping[str, EvidenceWindow],
    *,
    expected_request_id: str | None = None,
    raise_on_error: bool = False,
) -> CitationValidationReport:
    """Enforce known evidence IDs and review-evidence restrictions."""

    if isinstance(evidence, Mapping):
        evidence_map = dict(evidence)
        mismatched = [
            key
            for key, window in evidence_map.items()
            if key != window.evidence_window_id
        ]
        if mismatched:
            raise ValueError("evidence mapping key does not match its evidence_window_id")
    else:
        evidence_map = evidence_by_id(evidence)
    violations: list[CitationViolation] = []
    checked_citations = 0
    if expected_request_id is not None and result.request_id != expected_request_id:
        violations.append(
            CitationViolation(
                "request_mismatch",
                "summary result does not belong to the expected request",
            )
        )
    for claim in result.claims:
        if not claim.evidence_ids:
            violations.append(
                CitationViolation(
                    "missing_citation",
                    "summary claim has no evidence citation",
                    claim.claim_id,
                )
            )
            continue
        if len(claim.evidence_ids) != len(set(claim.evidence_ids)):
            violations.append(
                CitationViolation(
                    "duplicate_citation",
                    "summary claim repeats an evidence citation",
                    claim.claim_id,
                    claim.evidence_ids,
                )
            )
        checked_citations += len(claim.evidence_ids)
        unknown = tuple(identifier for identifier in claim.evidence_ids if identifier not in evidence_map)
        if unknown:
            violations.append(
                CitationViolation(
                    "unknown_evidence",
                    "summary claim cites an unknown evidence id",
                    claim.claim_id,
                    unknown,
                )
            )
            continue
        cited = tuple(evidence_map[identifier] for identifier in claim.evidence_ids)
        sensitive = claim.sensitive
        if sensitive and all(window.confidence == "review" for window in cited):
            violations.append(
                CitationViolation(
                    "review_only_sensitive_claim",
                    "review evidence cannot be the sole support for a sensitive claim",
                    claim.claim_id,
                    claim.evidence_ids,
                )
            )
        if claim.category in {"action", "decision"} and claim.state in {"observed", "unknown"}:
            violations.append(
                CitationViolation(
                    "ambiguous_action_state",
                    "action and decision claims must distinguish suggested, intended, or completed state",
                    claim.claim_id,
                    claim.evidence_ids,
                )
            )
    report = CitationValidationReport(
        valid=not violations,
        violations=tuple(violations),
        checked_claims=len(result.claims),
        checked_citations=checked_citations,
    )
    if raise_on_error and not report.valid:
        raise CitationValidationError(report)
    return report


validate_citations = validate_summary_citations


__all__ = [
    "CitationValidationError",
    "CitationValidationReport",
    "CitationViolation",
    "CommandSummaryBackend",
    "ExtractiveSummarizer",
    "ExtractiveSummaryBackend",
    "HttpSummaryBackend",
    "SummaryBackend",
    "SummaryClaim",
    "SummaryRequest",
    "SummaryResult",
    "make_summary_request",
    "request_from_packet",
    "summary_result_from_dict",
    "validate_citations",
    "validate_summary_citations",
]
