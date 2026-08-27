from __future__ import annotations

from dayaudio.bundles import (
    build_day_bundles,
    build_summary_packets,
    conservative_day_key,
    read_day_bundles,
    read_summary_packets,
    write_day_bundles,
    write_summary_packets,
)
from dayaudio.types import EvidenceWindow, SourceRecord


def _source(source_id: str, recording_start: str | None, basis: str | None = "metadata"):
    return SourceRecord(
        source_id=source_id,
        source_sha256=(source_id[-1] * 64),
        source_path=f"/{source_id}.wav",
        source_name=f"{source_id}.wav",
        size_bytes=1,
        decoded_duration_seconds=1200,
        recording_start=recording_start,
        recording_time_basis=basis,
    )


def _evidence(identifier: str, source_id: str, start: float, text: str = "文本"):
    return EvidenceWindow(
        identifier,
        source_id,
        start,
        start + 5,
        text,
        "high",
        "fast_default",
    )


def test_conservative_grouping_never_uses_untrusted_timestamp() -> None:
    sources = (
        _source("source-1", "2026-08-26T09:00:00+08:00"),
        _source("source-2", "2026-08-26T10:00:00+08:00"),
        _source("source-3", "2026-08-26T11:00:00+08:00", "filesystem_mtime"),
        _source("source-4", None),
    )
    bundles = build_day_bundles(sources)
    assert len(bundles) == 3
    dated = next(bundle for bundle in bundles if bundle.day_key == "day-2026-08-26")
    assert dated.source_ids == ("source-1", "source-2")
    assert conservative_day_key(sources[2]) == "undated-source-3"


def test_timezone_conversion_happens_before_day_grouping() -> None:
    source = _source("source-1", "2026-08-25T23:30:00Z")
    assert conservative_day_key(source, timezone_name="Asia/Shanghai") == "day-2026-08-26"


def test_container_provenance_detail_is_trusted() -> None:
    source = _source(
        "source-1",
        "2026-08-26T09:30:00+08:00",
        "container:format.tags.creation_time",
    )
    assert conservative_day_key(source) == "day-2026-08-26"


def test_different_explicit_offsets_are_not_implicitly_merged() -> None:
    utc = _source("source-1", "2026-08-26T01:00:00Z")
    shanghai = _source("source-2", "2026-08-26T09:00:00+08:00")
    assert len(build_day_bundles((utc, shanghai))) == 2
    assert len(build_day_bundles((utc, shanghai), timezone_name="Asia/Shanghai")) == 1


def test_packetization_is_day_aligned_and_round_trips(tmp_path) -> None:
    source = _source("source-1", "2026-08-26T00:00:00+08:00")
    evidence = (
        _evidence("e1", "source-1", 10),
        _evidence("e2", "source-1", 899),
        _evidence("e3", "source-1", 901),
    )
    bundle = build_day_bundles((source,), evidence)[0]
    packets = build_summary_packets(bundle, evidence)
    assert len(packets) == 2
    assert packets[0].evidence_window_ids == ("e1", "e2")
    assert packets[1].evidence_window_ids == ("e3",)
    assert packets[0].end == 900

    bundle_path = tmp_path / "bundles.json"
    packet_path = tmp_path / "packets.json"
    write_day_bundles(bundle_path, (bundle,))
    write_summary_packets(packet_path, packets)
    assert read_day_bundles(bundle_path) == (bundle,)
    assert read_summary_packets(packet_path) == packets


def test_bundle_documents_support_paths_beyond_max_path(long_path_root) -> None:
    source = _source("source-long", "2026-08-26T00:00:00+08:00")
    evidence = (_evidence("e-long", "source-long", 10),)
    bundle = build_day_bundles((source,), evidence)[0]
    packets = build_summary_packets(bundle, evidence)
    bundle_path = long_path_root / "bundles" / "day-bundles.json"
    packet_path = long_path_root / "bundles" / "summary-packets.json"
    assert len(str(bundle_path)) > 260

    write_day_bundles(bundle_path, (bundle,))
    write_summary_packets(packet_path, packets)
    assert read_day_bundles(bundle_path) == (bundle,)
    assert read_summary_packets(packet_path) == packets
