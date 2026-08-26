from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dayaudio.cas import ContentAddressedStore, atomic_write_text
from dayaudio.storage import SCHEMA_VERSION, Storage, StorageConflictError
from dayaudio.types import AsrSegment, SourceRecord


def source_record(path: Path, data: bytes = b"audio") -> SourceRecord:
    return SourceRecord(
        source_id=f"source-{hashlib.sha256(data).hexdigest()[:32]}",
        source_sha256=hashlib.sha256(data).hexdigest(),
        source_path=str(path),
        source_name=path.name,
        size_bytes=len(data),
        duration_seconds=12.5,
        codec="aac",
        sample_rate=48_000,
        channels=1,
    )


def test_schema_migrates_and_uses_wal(tmp_path: Path) -> None:
    database = Storage(tmp_path / "state.sqlite3")
    assert database.schema_version == SCHEMA_VERSION
    assert database.journal_mode() == "wal"

    # Re-opening proves migrations are idempotent.
    reopened = Storage(tmp_path / "state.sqlite3")
    assert reopened.schema_version == SCHEMA_VERSION


def test_sources_deduplicate_content_but_retain_locations(tmp_path: Path) -> None:
    database = Storage(tmp_path / "state.sqlite3")
    first = source_record(tmp_path / "first.m4a")
    stored = database.upsert_source(first)
    duplicate = SourceRecord(
        **{
            **first.to_dict(),
            "source_id": "source-another-id",
            "source_path": str(tmp_path / "copy.m4a"),
            "source_name": "copy.m4a",
        }
    )
    again = database.upsert_source(duplicate)

    assert again == stored
    assert database.list_sources() == [stored]
    assert set(database.source_locations(stored.source_id)) == {
        str(tmp_path / "first.m4a"),
        str(tmp_path / "copy.m4a"),
    }


def test_immutable_artifacts_and_cas(tmp_path: Path) -> None:
    database = Storage(tmp_path / "state.sqlite3")
    source = database.upsert_source(source_record(tmp_path / "a.m4a"))
    cas = ContentAddressedStore(tmp_path / "objects")
    obj = cas.put_bytes(b"immutable transcript")
    same = cas.put_bytes(b"immutable transcript")
    assert same == obj
    assert cas.verify(obj.sha256)

    artifact = database.add_artifact(
        artifact_id="artifact-fixed",
        kind="raw-asr",
        sha256=obj.sha256,
        path=obj.path,
        size_bytes=obj.size_bytes,
        source_id=source.source_id,
        metadata={"model": "test"},
    )
    assert database.get_artifact(artifact.artifact_id) == artifact
    with pytest.raises(StorageConflictError):
        database.add_artifact(
            artifact_id="artifact-fixed",
            kind="cleaned-asr",
            sha256=obj.sha256,
            path=obj.path,
            size_bytes=obj.size_bytes,
            source_id=source.source_id,
        )


def test_atomic_write_replaces_only_final_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text() == "two"
    assert not list(target.parent.glob("*.tmp"))


def test_segment_revisions_are_append_only(tmp_path: Path) -> None:
    database = Storage(tmp_path / "state.sqlite3")
    source = database.upsert_source(source_record(tmp_path / "a.m4a"))
    first = AsrSegment(
        segment_id="seg-1",
        source_id=source.source_id,
        start=1.0,
        end=2.0,
        text="first pass",
        revision=1,
        anomaly_flags=("low-confidence",),
        metadata={"raw": True},
    )
    second = AsrSegment(
        segment_id="seg-1",
        source_id=source.source_id,
        start=1.0,
        end=2.0,
        text="reviewed pass",
        revision=2,
    )
    database.add_segment(first)
    # JSON-equivalent containers remain idempotent after a database round trip.
    with_tuple_metadata = AsrSegment(
        segment_id="seg-json",
        source_id=source.source_id,
        start=2.0,
        end=3.0,
        text="json semantics",
        metadata={"values": (1, 2)},
    )
    database.add_segment(with_tuple_metadata)
    database.add_segment(with_tuple_metadata)
    database.add_segment(second)

    assert database.get_segment("seg-1") == second
    latest = database.list_segments(source_id=source.source_id)
    assert latest[0] == second
    assert latest[1].segment_id == "seg-json"
    all_revisions = database.list_segments(source_id=source.source_id, latest_only=False)
    assert all_revisions[:2] == [first, second]

    conflicting = AsrSegment(
        segment_id="seg-1",
        source_id=source.source_id,
        start=1.0,
        end=2.0,
        text="mutated",
        revision=1,
    )
    with pytest.raises(StorageConflictError):
        database.add_segment(conflicting)
