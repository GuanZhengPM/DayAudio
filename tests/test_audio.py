from __future__ import annotations

import hashlib
import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dayaudio.audio import (
    AudioDecodeError,
    block_owns_time,
    build_blocks_for_wav,
    decode_audio,
    ffmpeg_decode_command,
    iter_block_ranges,
    read_wav_info,
)
from dayaudio.paths import filesystem_path


def write_wav(path: Path, *, sample_rate: int, frames: int, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        samples = (struct.pack("<h", (index % 100) - 50) for index in range(frames * channels))
        handle.writeframes(b"".join(samples))


def test_block_ranges_have_contiguous_cores_and_bounded_context() -> None:
    ranges = list(
        iter_block_ranges(25, 10, core_seconds=1, context_seconds=0.2)
    )
    assert [
        (
            item.core_start_sample,
            item.core_end_sample,
            item.context_start_sample,
            item.context_end_sample,
        )
        for item in ranges
    ] == [(0, 10, 0, 12), (10, 20, 8, 22), (20, 25, 18, 25)]
    assert sum(item.core_sample_count for item in ranges) == 25


def test_audio_blocks_are_deterministic_and_hash_context_pcm(tmp_path: Path) -> None:
    wav = tmp_path / "pcm.wav"
    write_wav(wav, sample_rate=10, frames=25)
    source_sha = hashlib.sha256(b"original").hexdigest()
    first = build_blocks_for_wav(
        wav,
        source_id="source-1",
        source_sha256=source_sha,
        core_seconds=1,
        context_seconds=0.2,
    )
    second = build_blocks_for_wav(
        wav,
        source_id="source-1",
        source_sha256=source_sha,
        core_seconds=1,
        context_seconds=0.2,
    )
    assert first == second
    assert len({block.block_id for block in first}) == 3
    assert all(block.pcm_sha256 for block in first)
    assert block_owns_time(first[0], 0.999)
    assert not block_owns_time(first[0], 1.0)


def test_decode_is_mockable_and_installs_only_valid_pcm(tmp_path: Path) -> None:
    source = tmp_path / "input.fake"
    source.write_bytes(b"container")
    output = tmp_path / "decoded.wav"
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        write_wav(Path(command[-1]), sample_rate=16_000, frames=1_600)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    decoded = decode_audio(source, output, runner=runner)
    assert decoded.path == output
    assert decoded.sample_count == 1_600
    assert decoded.duration_seconds == 0.1
    assert read_wav_info(output).sample_width == 2
    command = calls[0][0]
    assert command[:5] == ["ffmpeg", "-v", "error", "-nostdin", "-y"]
    assert "-map_metadata" in command and "pcm_s16le" in command


def test_failed_decode_does_not_replace_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "input.fake"
    source.write_bytes(b"container")
    output = tmp_path / "decoded.wav"
    output.write_bytes(b"existing")

    def runner(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stdout="", stderr="decoder crashed")

    with pytest.raises(AudioDecodeError):
        decode_audio(source, output, runner=runner, overwrite=True)
    assert output.read_bytes() == b"existing"


def test_decode_rejects_hard_link_to_source(tmp_path: Path) -> None:
    source = tmp_path / "input.fake"
    source.write_bytes(b"container")
    alias = tmp_path / "alias.fake"
    os.link(source, alias)

    with pytest.raises(ValueError, match="must differ"):
        decode_audio(source, alias, runner=lambda *_args, **_kwargs: None, overwrite=True)

    assert source.read_bytes() == b"container"


@pytest.mark.skipif(os.name != "nt", reason="extended path aliases are Windows-specific")
def test_decode_rejects_extended_path_alias_to_source(tmp_path: Path) -> None:
    source = tmp_path / "input.fake"
    source.write_bytes(b"container")
    alias = filesystem_path(source, force_extended=True)

    with pytest.raises(ValueError, match="must differ"):
        decode_audio(source, alias, runner=lambda *_args, **_kwargs: None, overwrite=True)

    assert source.read_bytes() == b"container"


def test_decode_installs_to_path_beyond_max_path(
    tmp_path: Path, long_path_root: Path
) -> None:
    source = tmp_path / "input.fake"
    source.write_bytes(b"container")
    output = long_path_root / "audio" / "decoded.wav"
    assert len(str(output)) > 260

    def runner(command, **kwargs):
        write_wav(Path(command[-1]), sample_rate=16_000, frames=1_600)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    decoded = decode_audio(source, output, runner=runner)
    assert decoded.path == output
    assert decoded.sample_count == 1_600
    assert read_wav_info(output).duration_seconds == 0.1


def test_decode_forces_extended_parent_for_near_threshold_temporary_file(
    tmp_path: Path,
    near_path_root: Path,
) -> None:
    source = tmp_path / "input.fake"
    source.write_bytes(b"container")
    root_units = len(os.path.abspath(near_path_root).encode("utf-16-le")) // 2
    output_parent = near_path_root / ("p" * (241 - root_units - 1))
    output = output_parent / "o.wav"
    assert len(os.path.abspath(output).encode("utf-16-le")) // 2 < 248
    assert (
        len(os.path.abspath(output_parent).encode("utf-16-le")) // 2
        + len("\\.audio-xxxxxxxx.tmp")
        > 260
    )

    def runner(command, **kwargs):
        write_wav(Path(command[-1]), sample_rate=16_000, frames=1_600)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    decoded = decode_audio(source, output, runner=runner)
    assert decoded.path == output
    assert read_wav_info(output).duration_seconds == 0.1


def test_decode_command_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        ffmpeg_decode_command("in", "out", sample_rate=0)
