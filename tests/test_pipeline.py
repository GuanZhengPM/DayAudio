from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from dayaudio.evidence import build_evidence_from_storage
from dayaudio.paths import filesystem_path
from dayaudio.pipeline import (
    PipelineConfig,
    PipelineResult,
    ResumablePipeline,
    make_block_config_digest,
    make_resume_key,
)
from dayaudio.storage import Storage
from dayaudio.tasks import TaskQueue, make_task_key
from dayaudio.types import AsrSegment, AudioBlock, SourceRecord, TaskStatus


class StubBackend:
    name = "stub"
    model_revision = "r1"

    def __init__(
        self,
        model_id: str,
        text: str,
        confidence: float | None = None,
        config: object | None = None,
    ) -> None:
        self.model_id = model_id
        self.text = text
        self.confidence = confidence
        self.config = config
        self.calls = 0
        self.closed = False

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float = 0.0,
    ) -> list[AsrSegment]:
        self.calls += 1
        return [
            AsrSegment(
                segment_id=f"{self.model_id}-{block_id}-segment",
                source_id=source_id,
                start=offset_seconds,
                end=offset_seconds + 2,
                text=self.text,
                model_id=self.model_id,
                model_revision=self.model_revision,
                confidence=self.confidence,
                block_id=block_id,
            )
        ]

    def close(self) -> None:
        self.closed = True


class StubStorage:
    def __init__(self) -> None:
        self.values: dict[tuple[str, int], AsrSegment] = {}
        self.artifacts: list[dict[str, Any]] = []

    def add_segment(self, segment: AsrSegment) -> AsrSegment:
        key = (segment.segment_id, segment.revision)
        existing = self.values.get(key)
        if existing is not None and existing != segment:
            raise RuntimeError("conflict")
        self.values[key] = segment
        return segment

    def get_segment(
        self, segment_id: str, revision: int | None = None
    ) -> AsrSegment | None:
        matches = [
            item for (item_id, _), item in self.values.items() if item_id == segment_id
        ]
        if revision is not None:
            return self.values.get((segment_id, revision))
        return max(matches, key=lambda item: item.revision) if matches else None

    def list_segments(
        self,
        *,
        source_id: str | None = None,
        block_id: str | None = None,
        latest_only: bool = True,
    ) -> list[AsrSegment]:
        values = [
            item
            for item in self.values.values()
            if (block_id is None or item.block_id == block_id)
            and (source_id is None or item.source_id == source_id)
        ]
        if not latest_only:
            return values
        latest: dict[str, AsrSegment] = {}
        for item in values:
            if item.segment_id not in latest or item.revision > latest[item.segment_id].revision:
                latest[item.segment_id] = item
        return list(latest.values())

    def add_artifact(self, **values: Any) -> dict[str, Any]:
        record = dict(values)
        record.setdefault("artifact_id", f"artifact-{len(self.artifacts) + 1}")
        self.artifacts.append(record)
        return record

    def list_artifacts(
        self,
        *,
        source_id: str | None = None,
        kind: str | None = None,
        task_key: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in self.artifacts
            if (source_id is None or item.get("source_id") == source_id)
            and (kind is None or item.get("kind") == kind)
            and (task_key is None or item.get("task_key") == task_key)
        ]

    def delete_artifacts(self, *, kind: str, task_key: str) -> list[dict[str, Any]]:
        removed = [
            item
            for item in self.artifacts
            if item.get("kind") == kind and item.get("task_key") == task_key
        ]
        self.artifacts = [item for item in self.artifacts if item not in removed]
        return removed


def _block() -> AudioBlock:
    return AudioBlock(
        block_id="block",
        source_id="src",
        source_sha256="a" * 64,
        core_start=0,
        core_end=10,
        context_start=0,
        context_end=10,
    )


def _extend_to_utf16_units(base: Path, units: int) -> Path:
    current = len(os.path.abspath(base).encode("utf-16-le")) // 2
    while current >= units - 1:
        base = base.parent
        current = len(os.path.abspath(base).encode("utf-16-le")) // 2
    component_units = units - current - 1
    assert 0 < component_units < 255
    return base / ("p" * component_units)


def test_resume_key_matches_task_queue_contract() -> None:
    block = _block()
    expected = make_task_key(
        source_sha256=block.source_sha256,
        range_start=block.core_start,
        range_end=block.core_end,
        model_digest="model",
        config_digest=make_block_config_digest(block, config_digest="config"),
        kind="asr-block",
    )
    assert make_resume_key(
        block, model_digest="model", config_digest="config"
    ) == expected


def test_resume_key_binds_context_pcm_and_routing_inputs() -> None:
    base = _block()
    changed_context = replace(base, context_end=11)
    changed_pcm = replace(base, pcm_sha256="b" * 64)
    normal = make_resume_key(base, model_digest="model", config_digest="config")
    assert make_resume_key(
        changed_context, model_digest="model", config_digest="config"
    ) != normal
    assert make_resume_key(
        changed_pcm, model_digest="model", config_digest="config"
    ) != normal
    assert make_resume_key(
        base,
        model_digest="model",
        config_digest="config",
        force_strong=True,
    ) != normal
    assert make_resume_key(
        base,
        model_digest="model",
        config_digest="config",
        summary_sensitive_segment_ids=("seg-b", "seg-a"),
    ) != normal


def test_backend_configuration_is_bound_into_task_identity() -> None:
    first = ResumablePipeline(
        StubBackend(
            "same-model",
            "text",
            config={"language": "zh", "environment": {"TOKEN": "secret-a"}},
        )
    )
    second = ResumablePipeline(
        StubBackend(
            "same-model",
            "text",
            config={"language": "en", "environment": {"TOKEN": "secret-b"}},
        )
    )
    assert first.model_digest != second.model_digest
    assert first.task_key(_block()) != second.task_key(_block())
    assert "secret-a" not in first.model_digest


def test_local_model_bytes_are_bound_into_backend_digest(tmp_path: Path) -> None:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"first weights")
    first = ResumablePipeline(StubBackend(str(weights), "text")).model_digest
    weights.write_bytes(b"changed weights")
    second = ResumablePipeline(StubBackend(str(weights), "text")).model_digest
    assert first != second


def test_local_model_tree_hashes_descendants_beyond_max_path(
    near_path_root: Path,
) -> None:
    weights = near_path_root / ("nested-" + "w" * 40) / "weights.bin"
    filesystem_weights = filesystem_path(weights)
    filesystem_weights.parent.mkdir(parents=True, exist_ok=True)
    filesystem_weights.write_bytes(b"first weights")
    assert len(str(weights)) > 260

    first = ResumablePipeline(StubBackend(str(near_path_root), "text")).model_digest
    filesystem_weights.write_bytes(b"changed weights")
    second = ResumablePipeline(StubBackend(str(near_path_root), "text")).model_digest
    assert first != second


def test_local_weight_hash_is_cached_once_per_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dayaudio.cas as cas_module

    weights = tmp_path / "large-model.gguf"
    filesystem_path(weights).write_bytes(b"weights")
    real = cas_module.sha256_file
    calls = 0

    def counted(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return real(path, *args, **kwargs)

    monkeypatch.setattr(cas_module, "sha256_file", counted)
    pipeline = ResumablePipeline(StubBackend(str(weights), "text"))
    first = pipeline.model_digest
    assert pipeline.model_digest == first
    pipeline.task_key(_block())
    pipeline.task_key(_block(), force_strong=True)
    assert calls == 1


def test_pipeline_cascades_persists_and_resumes(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = StubStorage()
    storage.path = tmp_path / "dayaudio.sqlite3"  # type: ignore[attr-defined]
    fast = StubBackend("fast", "今天讨论产品发布计划", confidence=0.2)
    strong = StubBackend("strong", "今天讨论产品版本发布计划", confidence=0.9)
    pipeline = ResumablePipeline(
        fast,
        strong_backend=strong,
        storage=storage,
        config=PipelineConfig(config_digest="config", model_digest="model"),
    )
    result = pipeline.process_block(_block(), audio)
    assert result.accepted_strong_count == 1
    assert result.final_segments[0].text == strong.text
    assert result.final_segments[0].metadata["pipeline_stage"] == "final"
    assert result.fast_segments[0].metadata["is_fast"] is True
    assert result.strong_segments[0].metadata["base_segment_id"] == result.fast_segments[0].segment_id
    assert fast.calls == strong.calls == 1

    # Raw candidate IDs must remain alternatives to one logical fast segment,
    # never become duplicate transcript rows in evidence.
    default_evidence = build_evidence_from_storage(
        storage, source_id="src", allow_replacement=False
    )
    promoted = build_evidence_from_storage(
        storage, source_id="src", allow_replacement=True
    )
    assert len(default_evidence) == len(promoted) == 1
    assert default_evidence[0].text == strong.text
    assert promoted[0].text == strong.text
    marker = storage.list_artifacts(kind="asr-block-manifest")[0]
    assert marker["metadata"]["metadata_only"] is False
    assert filesystem_path(marker["path"]).is_file()

    resumed = pipeline.process_block(_block(), audio)
    assert resumed.resumed
    assert resumed.final_segments == result.final_segments
    assert fast.calls == strong.calls == 1
    pipeline.close()
    assert fast.closed and strong.closed


def test_pipeline_completion_marker_supports_long_storage_path(
    tmp_path: Path, long_path_root: Path
) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = StubStorage()
    storage.path = long_path_root / "dayaudio.sqlite3"  # type: ignore[attr-defined]
    backend = StubBackend("fast", "长路径完成标记", confidence=0.9)
    pipeline = ResumablePipeline(
        backend,
        storage=storage,
        config=PipelineConfig(config_digest="config", model_digest="model"),
    )

    first = pipeline.process_block(_block(), audio)
    marker = storage.list_artifacts(kind="asr-block-manifest")[0]
    marker_path = Path(marker["path"])
    assert len(str(marker_path)) > 260
    assert filesystem_path(marker_path).read_bytes()
    resumed = pipeline.process_block(_block(), audio)
    assert resumed.resumed
    assert resumed.final_segments == first.final_segments
    assert backend.calls == 1


def test_completion_marker_is_case_safe_and_cleans_mixed_namespace_path(
    near_path_root: Path,
) -> None:
    storage_parent = _extend_to_utf16_units(near_path_root.parent, 220)
    storage = StubStorage()
    storage.path = storage_parent / "dayaudio.sqlite3"  # type: ignore[attr-defined]
    pipeline = ResumablePipeline(
        StubBackend("fast", "text"),
        storage=storage,
        config=PipelineConfig(config_digest="config", model_digest="model"),
    )

    first_segment = AsrSegment(
        "segment", "src", 0, 1, "first", revision=1, block_id="block"
    )
    second_segment = replace(first_segment, text="second", revision=2)
    first = PipelineResult(
        "task", "block", "src", (first_segment,), (), (), (first_segment,), ()
    )
    second = PipelineResult(
        "task", "block", "src", (second_segment,), (), (), (second_segment,), ()
    )

    pipeline._write_completion_marker(first)
    old_path = Path(storage.artifacts[0]["path"])
    marker_root_units = len(
        os.path.abspath(old_path.parent).encode("utf-16-le")
    ) // 2
    assert marker_root_units < 248
    assert len(os.path.abspath(old_path).encode("utf-16-le")) // 2 > 260
    assert len(old_path.stem) == 64
    assert old_path.stem == old_path.stem.lower()

    pipeline._write_completion_marker(second)
    new_path = Path(storage.artifacts[0]["path"])
    assert new_path != old_path
    assert not filesystem_path(old_path).exists()
    assert filesystem_path(new_path).is_file()


def test_completion_marker_cleanup_preserves_same_file_alias(tmp_path: Path) -> None:
    storage = StubStorage()
    storage.path = tmp_path / "dayaudio.sqlite3"  # type: ignore[attr-defined]
    pipeline = ResumablePipeline(
        StubBackend("fast", "text"),
        storage=storage,
        config=PipelineConfig(config_digest="config", model_digest="model"),
    )
    segment = AsrSegment(
        "segment", "src", 0, 1, "text", revision=1, block_id="block"
    )
    result = PipelineResult(
        "task", "block", "src", (segment,), (), (), (segment,), ()
    )

    pipeline._write_completion_marker(result)
    marker_path = Path(storage.artifacts[0]["path"])
    alias_directory = marker_path.parent / "alias"
    filesystem_path(alias_directory).mkdir()
    storage.artifacts[0]["path"] = str(alias_directory / ".." / marker_path.name)

    pipeline._write_completion_marker(result)
    replacement = Path(storage.artifacts[0]["path"])
    assert replacement == marker_path
    assert filesystem_path(replacement).is_file()


def test_pipeline_keeps_fast_when_strong_fails_gate(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    fast = StubBackend(
        "fast", "这是一个完整且包含许多信息的讨论内容", confidence=0.1
    )
    strong = StubBackend("strong", "好的", confidence=0.9)
    result = ResumablePipeline(fast, strong_backend=strong).process_block(
        _block(), audio, force_strong=True
    )
    assert result.review_count == 1
    assert result.final_segments[0].text == fast.text


def test_summary_sensitive_text_escalates_without_explicit_ids(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    fast = StubBackend("fast", "决定周五发布产品版本", confidence=0.9)
    strong = StubBackend("strong", "决定在周五发布产品版本", confidence=0.95)
    result = ResumablePipeline(fast, strong_backend=strong).process_block(
        _block(), audio
    )
    assert strong.calls == 1
    assert result.accepted_strong_count == 1


def test_required_escalation_without_strong_backend_is_review(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    fast = StubBackend("fast", "决定周五发布产品版本", confidence=0.99)
    result = ResumablePipeline(fast).process_block(_block(), audio)
    assert result.review_count == 1
    assert result.final_segments[0].text == fast.text
    assert "strong_not_configured" in result.final_segments[0].anomaly_flags


class FailingBackend(StubBackend):
    def transcribe(self, *args: Any, **kwargs: Any) -> list[AsrSegment]:
        self.calls += 1
        raise RuntimeError("private backend detail")


def test_strong_backend_failure_preserves_fast_as_review(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    fast = StubBackend("fast", "决定周五发布产品版本", confidence=0.99)
    strong = FailingBackend("strong", "unused")
    result = ResumablePipeline(fast, strong_backend=strong).process_block(
        _block(), audio
    )
    assert result.review_count == 1
    assert result.final_segments[0].text == fast.text
    assert "strong_backend_error_RuntimeError" in result.final_segments[0].anomaly_flags


def test_partial_final_without_completion_marker_is_repaired(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = StubStorage()
    fast = StubBackend("fast", "完整文本", confidence=0.9)
    pipeline = ResumablePipeline(
        fast,
        storage=storage,
        config=PipelineConfig(config_digest="config", model_digest="model"),
    )
    key = pipeline.task_key(_block())
    storage.add_segment(
        AsrSegment(
            segment_id="partial",
            source_id="src",
            start=0,
            end=1,
            text="部分结果",
            revision=2,
            block_id="block",
            metadata={"pipeline_stage": "final", "pipeline_task_key": key},
        )
    )
    repaired = pipeline.process_block(_block(), audio)
    assert not repaired.resumed
    assert fast.calls == 1
    assert storage.list_artifacts(kind="asr-block-manifest", task_key=key)


def test_repair_supersedes_corrupt_manifest_artifact(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = Storage(tmp_path / "dayaudio.sqlite3")
    storage.upsert_source(
        SourceRecord("src", "a" * 64, str(audio), audio.name, audio.stat().st_size)
    )
    backend = StubBackend("fast", "正常文本", confidence=0.9)
    pipeline = ResumablePipeline(
        backend,
        storage=storage,
        config=PipelineConfig(model_digest="model", config_digest="config"),
    )
    first = pipeline.process_block(_block(), audio)
    marker = storage.list_artifacts(
        kind="asr-block-manifest", task_key=first.task_key
    )[0]
    filesystem_path(marker.path).write_bytes(b"corrupt")
    repaired = pipeline.process_block(_block(), audio)
    assert not repaired.resumed
    assert backend.calls == 2
    markers = storage.list_artifacts(
        kind="asr-block-manifest", task_key=first.task_key
    )
    assert len(markers) == 1
    from dayaudio.cas import sha256_file

    assert sha256_file(markers[0].path) == markers[0].sha256


@dataclass
class StubTask:
    task_id: str
    task_key: str
    payload: dict[str, Any]
    lease_token: str = "lease"


class StubQueue:
    def __init__(self, task: StubTask) -> None:
        self.task = task
        self.completed = False
        self.failed = False

    def claim(self, worker_id: str, **_: object) -> StubTask | None:
        task, self.task = self.task, None  # type: ignore[assignment]
        return task

    def is_cancel_requested(self, task_id: str) -> bool:
        return False

    def heartbeat(self, *args: object, **kwargs: object) -> None:
        return None

    def complete(self, *args: object, **kwargs: object) -> None:
        self.completed = True

    def fail(self, *args: object, **kwargs: object) -> None:
        self.failed = True


def test_pipeline_processes_duck_typed_task_queue(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    block = _block()
    pipeline = ResumablePipeline(
        StubBackend("fast", "正常文本", confidence=0.9),
    )
    task = StubTask(
        "task-id",
        pipeline.task_key(block),
        {"audio_path": str(audio), "block": block.to_dict()},
    )
    queue = StubQueue(task)
    pipeline.task_queue = queue
    result = pipeline.process_next()
    assert result is not None
    assert result.task_key == task.task_key
    assert queue.completed
    assert not queue.failed


def test_pipeline_rejects_foreign_claim_before_inference(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    backend = StubBackend("model-b", "不应运行", confidence=0.9)
    pipeline = ResumablePipeline(backend)
    foreign = StubTask(
        "foreign-task",
        "task-foreign-model-a",
        {"audio_path": str(audio), "block": _block().to_dict()},
    )
    queue = StubQueue(foreign)
    pipeline.task_queue = queue
    with pytest.raises(Exception, match="does not belong"):
        pipeline.process_next()
    assert backend.calls == 0
    assert queue.failed


def test_real_queue_does_not_cross_claim_another_model(tmp_path: Path) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = Storage(tmp_path / "dayaudio.sqlite3")
    storage.upsert_source(
        SourceRecord("src", "a" * 64, str(audio), audio.name, audio.stat().st_size)
    )
    queue = TaskQueue(storage)
    model_a = ResumablePipeline(
        StubBackend("model-a", "A", config={"language": "zh"}),
        storage=storage,
        task_queue=queue,
    )
    queued = model_a.enqueue_block(_block(), audio)
    backend_b = StubBackend("model-b", "B", config={"language": "en"})
    model_b = ResumablePipeline(
        backend_b,
        storage=storage,
        task_queue=queue,
    )
    assert model_b.process_next() is None
    assert backend_b.calls == 0
    assert queue.require(queued.task_id).status is TaskStatus.PENDING


class BoundaryBackend(StubBackend):
    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float = 0.0,
    ) -> list[AsrSegment]:
        self.calls += 1
        return [
            AsrSegment(
                segment_id=f"boundary-{block_id}",
                source_id=source_id,
                start=9.5,
                end=10.5,
                text="跨块语句只应出现一次",
                model_id=self.model_id,
                block_id=block_id,
            )
        ]


class SegmentListBackend(StubBackend):
    def __init__(self, model_id: str, rows: list[tuple[float, float, str]]) -> None:
        super().__init__(model_id, "unused")
        self.rows = rows

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float = 0.0,
    ) -> list[AsrSegment]:
        self.calls += 1
        return [
            AsrSegment(
                segment_id=f"{self.model_id}-{block_id}-{index}",
                source_id=source_id,
                start=start,
                end=end,
                text=text,
                model_id=self.model_id,
                confidence=0.2 if self.model_id == "fast" else 0.9,
                block_id=block_id,
            )
            for index, (start, end, text) in enumerate(self.rows)
        ]


class RawStubBackend(StubBackend):
    def transcribe(self, *args: Any, **kwargs: Any) -> list[AsrSegment]:
        self._raw = b'{"text":"full raw with control <|zh|>"}'
        return super().transcribe(*args, **kwargs)

    def consume_raw_output(self) -> bytes | None:
        value = getattr(self, "_raw", None)
        self._raw = None
        return value


class SlowBackend(StubBackend):
    def transcribe(self, *args: Any, **kwargs: Any) -> list[AsrSegment]:
        time.sleep(0.25)
        return super().transcribe(*args, **kwargs)


def test_adjacent_blocks_use_midpoint_core_ownership_and_clip(tmp_path: Path) -> None:
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"stub")
    second_audio.write_bytes(b"stub")
    storage = StubStorage()
    pipeline = ResumablePipeline(
        BoundaryBackend("fast", "unused"),
        storage=storage,
        config=PipelineConfig(config_digest="config", model_digest="model"),
    )
    first = AudioBlock(
        block_id="block-0",
        source_id="src",
        source_sha256="a" * 64,
        core_start=0,
        core_end=10,
        context_start=0,
        context_end=11,
    )
    second = AudioBlock(
        block_id="block-1",
        source_id="src",
        source_sha256="a" * 64,
        core_start=10,
        core_end=20,
        context_start=9,
        context_end=20,
    )
    first_result = pipeline.process_block(first, first_audio)
    second_result = pipeline.process_block(second, second_audio)
    assert first_result.final_segments == ()
    assert len(second_result.final_segments) == 1
    owned = second_result.final_segments[0]
    assert owned.start == 10
    assert owned.end == 10.5
    assert "boundary_timestamp_clipped" in owned.anomaly_flags
    assert owned.metadata["boundary_timestamp_clipped"]["original_start"] == 9.5


def test_pipeline_retains_full_backend_payload_in_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = StubStorage()
    storage.path = tmp_path / "dayaudio.sqlite3"  # type: ignore[attr-defined]
    result = ResumablePipeline(
        RawStubBackend("fast", "normalized text", confidence=0.9),
        storage=storage,
    ).process_block(_block(), audio)
    raw_artifacts = storage.list_artifacts(kind="asr-raw-fast")
    assert len(raw_artifacts) == 1
    assert filesystem_path(raw_artifacts[0]["path"]).read_bytes() == (
        b'{"text":"full raw with control <|zh|>"}'
    )
    assert result.final_segments[0].metadata["raw_output_retained"] is True
    assert result.final_segments[0].metadata["raw_artifact_id"] == raw_artifacts[0]["artifact_id"]


def test_lost_lease_during_inference_fences_all_durable_writes(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    storage = StubStorage()
    heartbeats = 0

    def heartbeat() -> None:
        nonlocal heartbeats
        heartbeats += 1
        if heartbeats >= 2:
            raise RuntimeError("lease lost")

    pipeline = ResumablePipeline(
        SlowBackend("fast", "text", confidence=0.9),
        storage=storage,
        config=PipelineConfig(lease_seconds=0.15),
    )
    with pytest.raises(Exception, match="lease"):
        pipeline.process_block(
            _block(),
            audio,
            heartbeat=heartbeat,
            cancelled=lambda: False,
        )
    assert heartbeats >= 2
    assert storage.values == {}
    assert storage.artifacts == []


@pytest.mark.parametrize(
    ("fast_rows", "strong_rows"),
    (
        (
            [(0.0, 1.0, "第一段文本"), (1.0, 2.0, "第二段文本")],
            [(0.0, 2.0, "第一段和第二段合并文本")],
        ),
        (
            [(0.0, 2.0, "合并的快速文本")],
            [(0.0, 1.0, "第一段"), (1.0, 2.0, "第二段")],
        ),
    ),
)
def test_segmentation_cardinality_mismatch_never_duplicates_strong_text(
    tmp_path: Path,
    fast_rows: list[tuple[float, float, str]],
    strong_rows: list[tuple[float, float, str]],
) -> None:
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"stub")
    fast = SegmentListBackend("fast", fast_rows)
    strong = SegmentListBackend("strong", strong_rows)
    result = ResumablePipeline(fast, strong_backend=strong).process_block(
        _block(), audio, force_strong=True
    )
    assert [item.text for item in result.final_segments] == [
        item[2] for item in fast_rows
    ]
    assert result.review_count == len(fast_rows)
    assert all(
        "strong:segmentation_cardinality_mismatch" in decision.reasons
        for decision in result.decisions
    )
