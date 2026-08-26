from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dayaudio.pipeline import PipelineConfig, ResumablePipeline
from dayaudio.storage import Storage
from dayaudio.tasks import TaskLeaseError, TaskQueue, make_task_key
from dayaudio.types import AsrSegment, AudioBlock, SourceRecord, TaskStatus

SHA = hashlib.sha256(b"source").hexdigest()


def queue_with_source(tmp_path: Path) -> tuple[Storage, TaskQueue, SourceRecord]:
    storage = Storage(tmp_path / "state.sqlite3")
    source = storage.upsert_source(
        SourceRecord(
            source_id="source-test",
            source_sha256=SHA,
            source_path=str(tmp_path / "source.m4a"),
            source_name="source.m4a",
            size_bytes=6,
        )
    )
    return storage, TaskQueue(storage), source


def enqueue(queue: TaskQueue, source: SourceRecord, **overrides: object):
    values = {
        "kind": "fast-asr",
        "source_id": source.source_id,
        "source_sha256": source.source_sha256,
        "range_start": 0,
        "range_end": 16000,
        "model_digest": "sensevoice@abc",
        "config_digest": "cfg@123",
    }
    values.update(overrides)
    return queue.enqueue(**values)  # type: ignore[arg-type]


def test_task_key_is_canonical_and_binds_every_input() -> None:
    base = make_task_key(
        source_sha256=SHA,
        range_start=1,
        range_end=2.0,
        model_digest="model-a",
        config_digest="config-a",
        kind="asr",
    )
    equivalent = make_task_key(
        source_sha256=SHA,
        range_start="1.00",
        range_end="2",
        model_digest="model-a",
        config_digest="config-a",
        kind="asr",
    )
    assert equivalent == base
    changes = [
        {"source_sha256": hashlib.sha256(b"other").hexdigest()},
        {"range_start": 0},
        {"range_end": 3},
        {"model_digest": "model-b"},
        {"config_digest": "config-b"},
        {"kind": "speaker"},
    ]
    defaults = dict(
        source_sha256=SHA,
        range_start=1,
        range_end=2,
        model_digest="model-a",
        config_digest="config-a",
        kind="asr",
    )
    for change in changes:
        assert make_task_key(**(defaults | change)) != base


def test_enqueue_is_idempotent_and_claims_priority(tmp_path: Path) -> None:
    _, queue, source = queue_with_source(tmp_path)
    low = enqueue(queue, source, range_start=0, range_end=10, priority=0)
    duplicate = enqueue(queue, source, range_start="0.0", range_end="10.00", priority=99)
    high = enqueue(queue, source, range_start=10, range_end=20, priority=10)
    assert duplicate.task_id == low.task_id
    assert len(queue.list()) == 2

    claimed = queue.claim("worker-1", lease_seconds=30)
    assert claimed is not None
    assert claimed.task_id == high.task_id
    assert claimed.status is TaskStatus.RUNNING
    assert claimed.attempts == 1
    assert claimed.lease_token


def test_claim_filters_model_and_pipeline_config(tmp_path: Path) -> None:
    _, queue, source = queue_with_source(tmp_path)
    enqueue(
        queue,
        source,
        range_start=0,
        range_end=10,
        model_digest="model-a",
        payload={"pipeline_config_digest": "family-a"},
    )
    enqueue(
        queue,
        source,
        range_start=10,
        range_end=20,
        model_digest="model-b",
        payload={"pipeline_config_digest": "family-b"},
    )
    claimed = queue.claim(
        "worker-b",
        model_digest="model-b",
        payload_filters={"pipeline_config_digest": "family-b"},
    )
    assert claimed is not None
    assert claimed.model_digest == "model-b"
    assert claimed.payload["pipeline_config_digest"] == "family-b"


def test_crashed_worker_stale_lease_is_recovered_and_old_token_rejected(tmp_path: Path) -> None:
    _, queue, source = queue_with_source(tmp_path)
    start = datetime(2026, 8, 26, tzinfo=timezone.utc)
    task = enqueue(queue, source, max_attempts=3, available_at=start)
    first = queue.claim("crashed-worker", lease_seconds=5, now=start)
    assert first is not None and first.lease_token

    recovered = queue.recover_stale(now=start + timedelta(seconds=6))
    assert recovered == 1
    assert queue.require(task.task_id).status is TaskStatus.PENDING

    second = queue.claim("new-worker", lease_seconds=30, now=start + timedelta(seconds=6))
    assert second is not None and second.lease_token != first.lease_token
    with pytest.raises(TaskLeaseError):
        queue.complete(
            task.task_id,
            "crashed-worker",
            first.lease_token,
            now=start + timedelta(seconds=7),
        )
    completed = queue.complete(
        task.task_id,
        "new-worker",
        second.lease_token,
        now=start + timedelta(seconds=7),
    )
    assert completed.status is TaskStatus.COMPLETE


def test_retry_limit_and_cancel_running_task(tmp_path: Path) -> None:
    _, queue, source = queue_with_source(tmp_path)
    start = datetime(2026, 8, 26, tzinfo=timezone.utc)
    task = enqueue(queue, source, max_attempts=2, available_at=start)
    first = queue.claim("w", lease_seconds=30, now=start)
    assert first is not None and first.lease_token
    pending = queue.fail(task.task_id, "w", first.lease_token, "oom", now=start)
    assert pending.status is TaskStatus.PENDING

    second = queue.claim("w", lease_seconds=30, now=start + timedelta(seconds=1))
    assert second is not None and second.lease_token
    failed = queue.fail(
        task.task_id, "w", second.lease_token, "oom again", now=start + timedelta(seconds=1)
    )
    assert failed.status is TaskStatus.FAILED
    assert queue.claim("w", now=start + timedelta(seconds=2)) is None

    other = enqueue(
        queue, source, range_start=16000, range_end=32000, available_at=start
    )
    running = queue.claim("w", lease_seconds=30, now=start + timedelta(seconds=3))
    assert running is not None and running.task_id == other.task_id and running.lease_token
    requested = queue.cancel(other.task_id, now=start + timedelta(seconds=4))
    assert requested.status is TaskStatus.RUNNING
    assert requested.cancel_requested
    cancelled = queue.complete(
        other.task_id,
        "w",
        running.lease_token,
        now=start + timedelta(seconds=5),
    )
    assert cancelled.status is TaskStatus.CANCELLED


def test_terminal_task_requires_explicit_retry(tmp_path: Path) -> None:
    _, queue, source = queue_with_source(tmp_path)
    started = datetime(2026, 8, 26, tzinfo=timezone.utc)
    task = enqueue(queue, source, max_attempts=1, available_at=started)
    claim = queue.claim("worker", now=started)
    assert claim is not None and claim.lease_token
    queue.fail(
        task.task_id,
        "worker",
        claim.lease_token,
        error="safe failure",
        retry=False,
        now=started,
    )
    assert queue.require(task.task_id).status is TaskStatus.FAILED
    retried = queue.retry(task.task_id)
    assert retried.status is TaskStatus.PENDING
    assert retried.attempts == 0


def test_last_stale_attempt_becomes_failed(tmp_path: Path) -> None:
    _, queue, source = queue_with_source(tmp_path)
    start = datetime(2026, 8, 26, tzinfo=timezone.utc)
    task = enqueue(queue, source, max_attempts=1, available_at=start)
    assert queue.claim("w", lease_seconds=1, now=start)
    assert queue.recover_stale(now=start + timedelta(seconds=2)) == 1
    failed = queue.require(task.task_id)
    assert failed.status is TaskStatus.FAILED
    assert "lease expired" in (failed.error or "")


def test_two_queue_instances_cannot_claim_the_same_task(tmp_path: Path) -> None:
    storage, queue, source = queue_with_source(tmp_path)
    enqueue(queue, source)
    second_queue = TaskQueue(Storage(storage.path))

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(
                lambda item: item[0].claim(item[1], lease_seconds=30),
                ((queue, "worker-a"), (second_queue, "worker-b")),
            )
        )
    assert sum(item is not None for item in claims) == 1


def test_real_queue_and_storage_drive_resumable_pipeline(tmp_path: Path) -> None:
    storage, queue, source = queue_with_source(tmp_path)
    audio = tmp_path / "block.wav"
    audio.write_bytes(b"adapter fixture")
    block = AudioBlock(
        block_id="block-real-queue",
        source_id=source.source_id,
        source_sha256=source.source_sha256,
        core_start=0,
        core_end=10,
        context_start=0,
        context_end=10,
    )

    class Backend:
        name = "fixture"
        model_id = "fixture"
        model_revision = "r1"

        def transcribe(self, path, *, source_id, block_id, offset_seconds=0.0):
            return [
                AsrSegment(
                    segment_id="segment-real-queue",
                    source_id=source_id,
                    block_id=block_id,
                    start=offset_seconds,
                    end=offset_seconds + 1,
                    text="durable pipeline",
                    confidence=0.9,
                )
            ]

        def close(self):
            return None

    pipeline = ResumablePipeline(
        Backend(),
        storage=storage,
        task_queue=queue,
        config=PipelineConfig(model_digest="model", config_digest="config"),
    )
    enqueued = pipeline.enqueue_block(block, audio)
    result = pipeline.process_next()
    assert result is not None and result.final_segments
    assert queue.require(enqueued.task_id).status is TaskStatus.COMPLETE
    assert storage.list_segments(block_id=block.block_id, latest_only=False)
    assert pipeline.process_next() is None
