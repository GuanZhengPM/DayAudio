from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dayaudio.cas import ContentAddressedStore, atomic_write_bytes
from dayaudio.ingest import (
    ProbeError,
    discover_audio_files,
    filename_recording_time,
    ingest_file,
    parse_ffprobe,
    probe_audio,
)
from dayaudio.paths import filesystem_path
from dayaudio.storage import Storage


def ffprobe_payload(*, creation_time: str | None = None) -> dict:
    tags = {"creation_time": creation_time} if creation_time else {}
    return {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "65.25",
            }
        ],
        "format": {"duration": "65.30", "tags": tags},
    }


class ProbeRunner:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(self.payload), stderr="")


def test_ingest_hashes_probes_copies_and_deduplicates(tmp_path: Path) -> None:
    first = tmp_path / "recording_2026-08-25_10-20-30.m4a"
    atomic_write_bytes(first, b"same audio bytes")
    copy = tmp_path / "copy.m4a"
    atomic_write_bytes(copy, filesystem_path(first).read_bytes())
    storage = Storage(tmp_path / "state.sqlite3")
    cas = ContentAddressedStore(tmp_path / "cas")
    runner = ProbeRunner(ffprobe_payload())

    stored = ingest_file(first, storage, cas=cas, runner=runner)
    duplicate = ingest_file(copy, storage, cas=cas, runner=runner)

    assert duplicate.source_id == stored.source_id
    assert len(storage.list_sources()) == 1
    assert len(runner.calls) == 1  # duplicate bytes do not require another probe
    assert set(storage.source_locations(stored.source_id)) == {str(first), str(copy)}
    assert stored.recording_start == "2026-08-25T10:20:30"
    assert stored.recording_time_basis == "filename:datetime:timezone-unspecified"
    artifacts = storage.list_artifacts(source_id=stored.source_id, kind="source-audio")
    assert len(artifacts) == 1
    assert cas.verify(stored.source_sha256)


def test_container_time_wins_and_is_normalized_to_utc(tmp_path: Path) -> None:
    payload = ffprobe_payload(creation_time="2026-08-25T10:20:30+08:00")
    probe = parse_ffprobe(payload, source_path=tmp_path / "recording_20200101_000000.m4a")
    assert probe.recording_start == "2026-08-25T02:20:30Z"
    assert probe.recording_time_basis == "container:format.tags.creation_time"


def test_filename_date_evidence_is_conservative() -> None:
    assert filename_recording_time("meeting_2026-02-30.m4a") is None
    assert filename_recording_time("mix_2026-08-01_and_2026-08-02.m4a") is None
    assert filename_recording_time("voice_20260825.m4a") == (
        "2026-08-25",
        "filename:date-only",
    )
    assert filename_recording_time("random_12345678.m4a") is None


def test_probe_rejects_non_audio_payload(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        parse_ffprobe(
            {"streams": [{"codec_type": "video", "codec_name": "h264"}]},
            source_path=tmp_path / "video.mp4",
        )


def test_probe_uses_utf8_for_chinese_metadata(tmp_path: Path) -> None:
    payload = ffprobe_payload(creation_time="2026-08-25T10:20:30+08:00")
    payload["format"]["tags"]["title"] = "中文录音"

    def runner(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "strict"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload, ensure_ascii=False),
            stderr="",
        )

    probe = probe_audio(tmp_path / "中文录音.m4a", runner=runner)
    assert probe.recording_start == "2026-08-25T02:20:30Z"


def test_probe_rejects_non_utf8_output(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(ProbeError, match="not valid UTF-8"):
        probe_audio(tmp_path / "recording.m4a", runner=runner)


def test_discover_audio_files_walks_descendants_beyond_max_path(
    near_path_root: Path,
) -> None:
    nested = near_path_root / ("deep-" + "x" * 35)
    audio = nested / ("recording-" + "y" * 30 + ".wav")
    ignored = nested / ("notes-" + "z" * 30 + ".txt")
    filesystem_path(nested).mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(audio, b"audio")
    atomic_write_bytes(ignored, b"text")

    assert discover_audio_files([near_path_root]) == [audio.resolve()]
