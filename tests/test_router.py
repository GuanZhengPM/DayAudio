from __future__ import annotations

from dayaudio.router import CascadeRouter, detect_anomalies
from dayaudio.types import AsrSegment


def _segment(
    text: str,
    *,
    segment_id: str = "seg",
    model: str = "fast",
    confidence: float | None = None,
    duration: float = 2.0,
) -> AsrSegment:
    return AsrSegment(
        segment_id=segment_id,
        source_id="src",
        start=0,
        end=duration,
        text=text,
        model_id=model,
        confidence=confidence,
        block_id="block",
    )


def test_anomaly_detector_covers_refusal_repetition_rate_and_punctuation() -> None:
    assert "model_refusal" in detect_anomalies(
        _segment("I am unable to transcribe audio into text.")
    ).flags
    assert "repetition_loop" in detect_anomalies(
        _segment("吹" * 80, duration=2.0)
    ).flags
    punctuation = detect_anomalies(_segment("……！！！"))
    assert "punctuation_only" in punctuation.flags
    assert "chars_per_second_high" in detect_anomalies(
        _segment("正常语句" * 100, duration=1.0)
    ).flags


def test_similar_strong_revision_passes_coverage_and_consensus() -> None:
    router = CascadeRouter()
    fast = _segment("今天讨论产品发布计划", confidence=0.2)
    strong = _segment(
        "今天讨论产品版本发布计划",
        segment_id="strong",
        model="strong",
        confidence=0.9,
    )
    assert router.should_escalate(fast).escalate
    decision = router.evaluate(fast, strong)
    assert decision.accepted_strong
    assert decision.selected.segment_id == fast.segment_id
    assert decision.selected.revision == fast.revision + 1
    assert decision.selected.model_id == "strong"


def test_under_transcription_never_overwrites_fast() -> None:
    fast = _segment("这是一个完整且包含许多信息的讨论内容")
    strong = _segment("好的", segment_id="strong", model="strong")
    decision = CascadeRouter().evaluate(fast, strong)
    assert decision.action == "review"
    assert decision.selected.text == fast.text
    assert "strong:insufficient_text_coverage" in decision.reasons


def test_anomalous_fast_requires_independent_consensus() -> None:
    router = CascadeRouter()
    fast = _segment("谢谢谢谢谢谢谢谢谢谢谢谢")
    strong = _segment(
        "今天我们讨论产品发布安排",
        segment_id="strong",
        model="strong",
    )
    without_third = router.evaluate(fast, strong)
    assert without_third.action == "review"
    assert "strong:independent_consensus_required" in without_third.reasons

    third = _segment(
        "今天我们讨论产品发布的安排",
        segment_id="third",
        model="third",
    )
    with_third = router.evaluate(fast, strong, consensus=third)
    assert with_third.action == "accept-strong"


def test_strong_refusal_is_rejected_even_when_fast_is_low_confidence() -> None:
    fast = _segment("今天讨论项目安排", confidence=0.1)
    strong = _segment(
        "I cannot transcribe this audio into text.",
        segment_id="strong",
        model="strong",
    )
    decision = CascadeRouter().evaluate(fast, strong)
    assert decision.action == "review"
    assert "strong:model_refusal" in decision.reasons
