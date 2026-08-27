from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

from dayaudio.diarize import diarize_file, make_speaker_windows, write_wav_slice
from dayaudio.paths import filesystem_path


def make_wav(path: Path, seconds: float = 5.0) -> None:
    rate = 100
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"".join(struct.pack("<h", index % 100) for index in range(round(seconds * rate))))


class FakeVad:
    def speech_regions(self, _: Path) -> list[tuple[float, float]]:
        return [(0.0, 2.0), (2.5, 4.5)]


class FakeEmbeddings:
    model_digest = "model-digest"

    def embed(self, windows):
        return [(1.0, 0.0) if window.start < 2.5 else (0.0, 1.0) for window in windows]


def test_window_split_does_not_leave_short_tail() -> None:
    windows = make_speaker_windows("source-1", [(0.0, 9.0)], min_seconds=2.0, max_seconds=8.0)
    assert [(item.start, item.end) for item in windows] == [(0.0, 9.0)]


def test_write_slice_and_diarize(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    make_wav(source)
    sliced = write_wav_slice(source, tmp_path / "slice.wav", start=1.0, end=3.0)
    with wave.open(str(sliced), "rb") as input_audio:
        assert input_audio.getnframes() == 200

    result = diarize_file(
        source,
        source_id="source-1",
        vad_backend=FakeVad(),
        embedding_backend=FakeEmbeddings(),
        similarity_threshold=0.8,
    )
    assert result.status == "complete"
    assert result.clustering is not None
    assert len(result.clustering.clusters) == 2
    assert len(result.clustering.turns) == 2


def test_write_slice_supports_path_beyond_max_path(
    tmp_path: Path, long_path_root: Path
) -> None:
    source = tmp_path / "source.wav"
    make_wav(source)
    destination = long_path_root / "slices" / "slice.wav"
    assert len(str(destination)) > 260

    sliced = write_wav_slice(source, destination, start=1.0, end=3.0)
    assert sliced == destination
    with wave.open(str(filesystem_path(sliced)), "rb") as input_audio:
        assert input_audio.getnframes() == 200


def test_write_slice_temp_file_crosses_max_path_from_near_root(
    tmp_path: Path, near_path_root: Path
) -> None:
    source = tmp_path / "source.wav"
    make_wav(source)
    root_units = len(os.path.abspath(near_path_root).encode("utf-16-le")) // 2
    parent = near_path_root / ("s" * (245 - root_units - 1))
    destination = parent / "x"
    assert len(os.path.abspath(destination).encode("utf-16-le")) // 2 == 247

    write_wav_slice(source, destination, start=1.0, end=3.0)
    with wave.open(str(filesystem_path(destination)), "rb") as input_audio:
        assert input_audio.getnframes() == 200
