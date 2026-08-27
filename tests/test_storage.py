from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from dayaudio.cas import (
    CASObject,
    ContentAddressedStore,
    atomic_write_bytes,
    atomic_write_text,
)
from dayaudio.paths import filesystem_path, filesystem_tree_path
from dayaudio.storage import SCHEMA_VERSION, Storage, StorageConflictError
from dayaudio.types import AsrSegment, SourceRecord


def _extend_to_utf16_units(base: Path, units: int, *, fill: str = "n") -> Path:
    current = len(os.path.abspath(base).encode("utf-16-le")) // 2
    component_units = units - current - 1
    assert 0 < component_units < 255
    return base / (fill * component_units)


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
    assert filesystem_path(target).read_text() == "two"
    assert not list(filesystem_path(target.parent).glob("*.tmp"))


def test_atomic_write_and_cas_support_paths_beyond_max_path(long_path_root: Path) -> None:
    target = long_path_root / "nested" / "state.json"
    assert len(str(target)) > 260
    assert atomic_write_text(target, "one") == target
    atomic_write_text(target, "two")
    assert filesystem_path(target).read_text(encoding="utf-8") == "two"

    source = long_path_root / "source.bin"
    atomic_write_bytes(source, b"source payload")
    cas = ContentAddressedStore(long_path_root / "objects")
    from_bytes = cas.put_bytes(b"bytes payload")
    from_file = cas.put_file(source)
    assert cas.get(from_bytes.sha256, verify=True) == from_bytes
    assert cas.get(from_file.sha256, verify=True) == from_file
    assert {item.sha256 for item in cas.iter_objects()} == {
        from_bytes.sha256,
        from_file.sha256,
    }


def test_atomic_write_and_cas_cross_max_path_from_a_near_root(
    near_path_root: Path,
) -> None:
    atomic_parent = _extend_to_utf16_units(near_path_root, 245, fill="a")
    target = atomic_parent / "x"
    target_units = len(os.path.abspath(target).encode("utf-16-le")) // 2
    assert target_units == 247
    assert 245 + 1 + len(".atomic-xxxxxxxx.tmp") >= 260
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert filesystem_path(target).read_text(encoding="utf-8") == "two"

    cas_root = _extend_to_utf16_units(near_path_root.parent, 234, fill="c")
    cas = ContentAddressedStore(cas_root)
    stored = cas.put_bytes(b"near-threshold object")
    assert len(str(stored.path)) > 260
    assert cas.get(stored.sha256, verify=True) == stored
    assert tuple(cas.iter_objects()) == (stored,)


def test_iter_objects_skips_malformed_names_and_shards(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    stored = cas.put_bytes(b"valid object")

    malformed_name = cas.object_root / "00" / "00" / ("z" * 64)
    wrong_first = "00" if stored.sha256[:2] != "00" else "ff"
    wrong_second = "00" if stored.sha256[2:4] != "00" else "ff"
    misplaced = cas.object_root / wrong_first / wrong_second / stored.sha256
    for path in (malformed_name, misplaced):
        filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
        filesystem_path(path).write_bytes(b"not a canonical CAS object")

    assert tuple(cas.iter_objects()) == (stored,)
    assert filesystem_tree_path(malformed_name).is_file()
    assert filesystem_tree_path(misplaced).is_file()


@pytest.mark.parametrize("method", ["put_bytes", "put_file"])
def test_same_content_concurrent_cas_puts_converge(
    tmp_path: Path, method: str
) -> None:
    payload = b"same immutable payload" * 4096
    source = tmp_path / f"{method}.bin"
    source.write_bytes(payload)

    for round_index in range(5):
        cas = ContentAddressedStore(tmp_path / method / str(round_index))
        barrier = Barrier(4)

        def put() -> CASObject:
            barrier.wait()
            if method == "put_bytes":
                return cas.put_bytes(payload)
            return cas.put_file(source, chunk_size=4096)

        with ThreadPoolExecutor(max_workers=4) as executor:
            objects = list(executor.map(lambda _: put(), range(8)))

        assert len({item.sha256 for item in objects}) == 1
        assert len({item.path for item in objects}) == 1
        assert cas.verify(objects[0].sha256)
        assert len(tuple(cas.iter_objects())) == 1
        assert not list(filesystem_path(cas.object_root).rglob("*.tmp"))


def test_cas_rejects_a_corrupt_existing_winner(tmp_path: Path) -> None:
    payload = b"immutable winner"
    cas = ContentAddressedStore(tmp_path / "cas")
    installed = cas.put_bytes(payload)
    installed_path = filesystem_path(installed.path)
    os.chmod(installed_path, 0o600)
    installed_path.write_bytes(b"corrupt")
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    with pytest.raises(OSError, match="integrity check failed"):
        cas.put_bytes(payload)
    with pytest.raises(OSError, match="integrity check failed"):
        cas.put_file(source)


@pytest.mark.skipif(os.name != "nt", reason="Windows uses create-if-absent CAS rename")
def test_windows_cas_losing_writer_validates_the_winner(tmp_path: Path) -> None:
    payload = b"winning immutable bytes"
    cas = ContentAddressedStore(tmp_path / "cas")
    winner = cas.put_bytes(payload)
    staging = tmp_path / "loser.tmp"
    staging.write_bytes(payload)

    result, consumed = cas._install_staged_file(
        str(staging),
        winner.path,
        digest=winner.sha256,
        size=len(payload),
    )
    assert result == winner
    assert consumed is False
    assert staging.is_file()

    filesystem_path(winner.path).write_bytes(b"corrupt")
    with pytest.raises(OSError, match="integrity check failed"):
        cas._install_staged_file(
            str(staging),
            winner.path,
            digest=winner.sha256,
            size=len(payload),
        )


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
