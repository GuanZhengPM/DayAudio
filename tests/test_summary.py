from __future__ import annotations

import sys

from dayaudio.summary import (
    CommandSummaryBackend,
    ExtractiveSummaryBackend,
    SummaryClaim,
    SummaryResult,
    make_summary_request,
    validate_summary_citations,
)
from dayaudio.types import EvidenceWindow


def _evidence(identifier: str, text: str, confidence="high", *, sensitive=False):
    return EvidenceWindow(
        identifier,
        "source-1",
        0,
        5,
        text,
        confidence,
        "fast_default",
        summary_sensitive=sensitive,
    )


def _result(claim: SummaryClaim) -> SummaryResult:
    return SummaryResult("summary-1", "request-1", "scope-1", "test", (claim,))


def test_extractive_fallback_is_cited_and_skips_sensitive_review_evidence() -> None:
    evidence = (
        _evidence("e-review", "决定周五发布版本。", "review", sensitive=True),
        _evidence("e-good", "团队讨论了当前进度。", "high"),
    )
    request = make_summary_request("scope", evidence)
    result = ExtractiveSummaryBackend().summarize(request)
    assert len(result.claims) == 1
    assert result.claims[0].evidence_ids == ("e-good",)
    assert validate_summary_citations(result, evidence, expected_request_id=request.request_id).valid


def test_citation_validator_rejects_unknown_and_review_only_sensitive_claims() -> None:
    review = _evidence("e-review", "决定周五发布。", "review", sensitive=True)
    sensitive = SummaryClaim("claim-1", "决定周五发布。", ("e-review",), "decision")
    report = validate_summary_citations(_result(sensitive), (review,))
    assert not report.valid
    assert report.violations[0].code == "review_only_sensitive_claim"

    unknown = SummaryClaim("claim-2", "普通观察。", ("missing",))
    report = validate_summary_citations(_result(unknown), (review,))
    assert {item.code for item in report.violations} == {"unknown_evidence"}


def test_review_notice_without_fact_is_allowed() -> None:
    review = _evidence("e-review", "决定周五发布。", "review", sensitive=True)
    notice = SummaryClaim(
        "claim-review",
        "该范围仅包含待复核的转录证据，暂不提取事实性结论。",
        ("e-review",),
        "review_notice",
    )
    assert validate_summary_citations(_result(notice), (review,)).valid


def test_command_backend_uses_json_contract() -> None:
    evidence = (_evidence("e1", "讨论了项目进度。"),)
    request = make_summary_request("scope", evidence)
    script = (
        "import json,sys; r=json.load(sys.stdin); "
        "json.dump({'request_id':r['request_id'],'scope_id':r['scope_id'],"
        "'claims':[{'text':'讨论了项目进度。','evidence_ids':['e1']}]},sys.stdout)"
    )
    backend = CommandSummaryBackend((sys.executable, "-c", script), backend_id="fixture")
    result = backend.summarize(request)
    assert result.backend_id == "fixture"
    assert result.claims[0].evidence_ids == ("e1",)
    assert validate_summary_citations(result, evidence, expected_request_id=request.request_id).valid
