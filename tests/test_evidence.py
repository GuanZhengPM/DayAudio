from __future__ import annotations

from dayaudio.evidence import (
    build_evidence_from_storage,
    build_evidence_windows,
    select_segment_revision,
    transcription_anomaly_flags,
)
from dayaudio.storage import Storage
from dayaudio.types import AsrSegment, SourceRecord


def _segment(
    segment_id: str,
    text: str,
    *,
    start: float = 0,
    end: float = 5,
    revision: int = 1,
    confidence: float | None = 0.9,
    metadata=None,
) -> AsrSegment:
    return AsrSegment(
        segment_id,
        "source-1",
        start,
        end,
        text,
        revision=revision,
        confidence=confidence,
        metadata=metadata or {},
    )


def test_refusal_detection_is_repeatable() -> None:
    text = "抱歉，无法转录这段音频。"
    assert "model_refusal" in transcription_anomaly_flags(text)
    assert "model_refusal" in transcription_anomaly_flags(text)


def test_hallucinated_or_short_alternative_never_overwrites_fast() -> None:
    fast = _segment("seg", "我们决定周五发布测试版本并补充成本测算。", confidence=0.7)
    refusal = _segment(
        "seg",
        "抱歉，我无法转录。",
        revision=2,
        confidence=0.99,
        metadata={"review_approved": True},
    )
    selection = select_segment_revision(fast, [refusal], allow_replacement=True)
    assert selection.preserved_fast
    assert selection.selection_reason == "fast_preserved_no_safe_alternative"
    assert "model_refusal" in selection.rejected_alternatives[0].reasons


def test_clean_review_approved_alternative_can_be_promoted() -> None:
    fast = _segment("seg", "周五测试。", confidence=0.6)
    strong = _segment(
        "seg",
        "我们决定周五发布测试版本。",
        revision=2,
        confidence=0.92,
        metadata={"review_approved": True},
    )
    selection = select_segment_revision(fast, [strong], allow_replacement=True)
    assert not selection.preserved_fast
    assert selection.selected_segment.revision == 2


def test_evidence_window_defaults_to_fast_and_marks_sensitive_content() -> None:
    segments = (
        _segment("s1", "今天讨论了项目进度。", start=1, end=5, confidence=0.92),
        _segment("s2", "决定在8月30日发布。", start=10, end=15, confidence=0.81),
        _segment("s3", "另一个话题。", start=35, end=40, confidence=None),
    )
    windows = build_evidence_windows(segments)
    assert len(windows) == 2
    assert windows[0].confidence == "high"
    assert windows[0].summary_sensitive
    assert windows[0].model_state == "fast_default"
    assert windows[1].confidence == "medium"
    assert windows[0].evidence_window_id.startswith("evidence-")


def test_build_evidence_from_storage_groups_revisions() -> None:
    fast = _segment("seg", "快速结果。", confidence=0.6, metadata={"stage": "fast"})
    strong = _segment(
        "seg",
        "人工确认后的准确结果。",
        revision=2,
        confidence=0.95,
        metadata={"review_approved": True},
    )

    class Store:
        def list_segments(self, **kwargs):
            assert kwargs["latest_only"] is False
            return [fast, strong]

    windows = build_evidence_from_storage(Store(), allow_replacement=True)
    assert windows[0].text == strong.text
    assert windows[0].model_state == "safe_alternative_promoted"


def test_pending_reprocess_keeps_prior_completed_evidence() -> None:
    old_fast = _segment(
        "seg",
        "旧快速结果。",
        revision=1,
        metadata={
            "base_segment_id": "seg",
            "stage": "fast",
            "is_fast": True,
            "pipeline_task_key": "old",
        },
    )
    old_final = _segment(
        "seg",
        "旧任务的完整最终结果。",
        revision=2,
        metadata={
            "base_segment_id": "seg",
            "stage": "final",
            "pipeline_stage": "final",
            "pipeline_task_key": "old",
        },
    )
    new_partial = _segment(
        "seg",
        "新任务尚未完成。",
        revision=3,
        metadata={
            "base_segment_id": "seg",
            "stage": "fast",
            "is_fast": True,
            "pipeline_task_key": "new",
        },
    )

    class Task:
        def __init__(self, status):
            self.status = status

    class Queue:
        def get(self, key):
            return Task("complete" if key == "old" else "pending")

    class Store:
        def list_segments(self, **_):
            return [old_fast, old_final, new_partial]

        def task_queue(self):
            return Queue()

    windows = build_evidence_from_storage(Store())
    assert len(windows) == 1
    assert windows[0].text == old_final.text


def test_pipeline_review_action_forces_review_confidence() -> None:
    fast = _segment(
        "seg",
        "决定在8月30日发布。",
        confidence=0.99,
        metadata={"stage": "fast", "is_fast": True, "base_segment_id": "seg"},
    )
    final = _segment(
        "seg",
        "决定在8月30日发布。",
        revision=2,
        confidence=0.99,
        metadata={
            "stage": "final",
            "pipeline_stage": "final",
            "base_segment_id": "seg",
            "cascade_action": "review",
            "cascade_reasons": ["consensus_below_threshold"],
        },
    )

    class Store:
        def list_segments(self, **_):
            return [fast, final]

    window = build_evidence_from_storage(Store())[0]
    assert window.confidence == "review"
    assert "consensus_below_threshold" in window.review_reasons


def test_complete_task_without_manifest_is_not_authoritative(tmp_path) -> None:
    storage = Storage(tmp_path / "state.sqlite3")
    source = storage.upsert_source(
        SourceRecord(
            source_id="source-1",
            source_sha256="a" * 64,
            source_path=str(tmp_path / "source.wav"),
            source_name="source.wav",
            size_bytes=1,
        )
    )
    queue = storage.task_queue()
    task = queue.enqueue(
        kind="asr-block",
        source_id=source.source_id,
        source_sha256=source.source_sha256,
        range_start=0,
        range_end=1,
        model_digest="model",
        config_digest="config",
    )
    claimed = queue.claim("worker")
    assert claimed is not None and claimed.lease_token
    fast = _segment(
        "seg",
        "快速结果。",
        metadata={
            "stage": "fast",
            "is_fast": True,
            "base_segment_id": "seg",
            "pipeline_task_key": task.task_key,
        },
    )
    final = _segment(
        "seg",
        "最终结果。",
        revision=2,
        metadata={
            "stage": "final",
            "pipeline_stage": "final",
            "base_segment_id": "seg",
            "pipeline_task_key": task.task_key,
        },
    )
    storage.add_segment(fast)
    storage.add_segment(final)
    queue.complete(task.task_id, "worker", claimed.lease_token)
    assert build_evidence_from_storage(storage) == ()
