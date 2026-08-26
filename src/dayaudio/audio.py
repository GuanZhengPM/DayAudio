"""Canonical PCM decoding and deterministic sample-based block boundaries."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Iterator

from .cas import digest_json, sha256_file
from .types import AudioBlock


class AudioDecodeError(RuntimeError):
    pass


Runner = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    path: Path
    sha256: str
    size_bytes: int
    sample_rate: int
    channels: int
    sample_width: int
    sample_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class BlockRange:
    index: int
    sample_rate: int
    core_start_sample: int
    core_end_sample: int
    context_start_sample: int
    context_end_sample: int

    @property
    def core_start(self) -> float:
        return self.core_start_sample / self.sample_rate

    @property
    def core_end(self) -> float:
        return self.core_end_sample / self.sample_rate

    @property
    def context_start(self) -> float:
        return self.context_start_sample / self.sample_rate

    @property
    def context_end(self) -> float:
        return self.context_end_sample / self.sample_rate

    @property
    def core_sample_count(self) -> int:
        return self.core_end_sample - self.core_start_sample

    @property
    def context_sample_count(self) -> int:
        return self.context_end_sample - self.context_start_sample


def read_wav_info(path: str | os.PathLike[str]) -> WavInfo:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise AudioDecodeError(f"compressed WAV is not canonical PCM: {source}")
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
    except (wave.Error, EOFError) as error:
        raise AudioDecodeError(f"invalid WAV file: {source}") from error
    if sample_rate <= 0 or channels <= 0 or sample_width <= 0 or frame_count < 0:
        raise AudioDecodeError(f"invalid WAV parameters: {source}")
    return WavInfo(
        path=source,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
    )


def ffmpeg_decode_command(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16_000,
    channels: int = 1,
) -> list[str]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    return [
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "-f",
        "wav",
        str(output),
    ]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def decode_audio(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16_000,
    channels: int = 1,
    runner: Runner = subprocess.run,
    timeout: float | None = None,
    overwrite: bool = False,
) -> DecodedAudio:
    """Decode one input to metadata-free signed 16-bit little-endian PCM WAV.

    FFmpeg writes a temporary sibling.  The final name changes only after a
    successful exit and a structural WAV check, so a killed worker cannot
    leave a file that looks complete.
    """

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"not a regular file: {source_path}")
    if source_path == output_path:
        raise ValueError("source and output paths must differ")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    command = ffmpeg_decode_command(
        source_path,
        temporary_path,
        ffmpeg_bin=ffmpeg_bin,
        sample_rate=sample_rate,
        channels=channels,
    )
    try:
        try:
            completed = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as error:
            raise AudioDecodeError(f"ffmpeg executable not found: {ffmpeg_bin}") from error
        except subprocess.TimeoutExpired as error:
            raise AudioDecodeError(f"ffmpeg timed out: {source_path}") from error
        if completed.returncode != 0:
            stderr = str(getattr(completed, "stderr", "")).strip()
            raise AudioDecodeError(
                f"ffmpeg failed for {source_path}: {stderr or f'exit {completed.returncode}'}"
            )
        info = read_wav_info(temporary_path)
        if info.sample_rate != sample_rate or info.channels != channels or info.sample_width != 2:
            raise AudioDecodeError(
                "ffmpeg output does not match requested PCM format "
                f"({info.sample_rate} Hz, {info.channels} ch, {info.sample_width} bytes)"
            )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)
        return DecodedAudio(
            path=output_path,
            sha256=sha256_file(output_path),
            size_bytes=output_path.stat().st_size,
            sample_rate=info.sample_rate,
            channels=info.channels,
            sample_width=info.sample_width,
            sample_count=info.frame_count,
            duration_seconds=info.duration_seconds,
        )
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


normalize_audio = decode_audio


def _seconds_to_samples(value: int | float | Decimal, sample_rate: int, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        seconds = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if not seconds.is_finite() or seconds < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return int((seconds * sample_rate).to_integral_value(rounding=ROUND_HALF_UP))


def iter_block_ranges(
    total_samples: int,
    sample_rate: int,
    *,
    core_seconds: int | float | Decimal = 300,
    context_seconds: int | float | Decimal = 5,
) -> Iterator[BlockRange]:
    """Yield contiguous cores with bounded context, using integer samples."""

    if isinstance(total_samples, bool) or total_samples < 0:
        raise ValueError("total_samples must be a non-negative integer")
    if not isinstance(total_samples, int):
        raise TypeError("total_samples must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    core_samples = _seconds_to_samples(core_seconds, sample_rate, "core_seconds")
    context_samples = _seconds_to_samples(context_seconds, sample_rate, "context_seconds")
    if core_samples <= 0:
        raise ValueError("core_seconds must cover at least one sample")
    index = 0
    core_start = 0
    while core_start < total_samples:
        core_end = min(total_samples, core_start + core_samples)
        yield BlockRange(
            index=index,
            sample_rate=sample_rate,
            core_start_sample=core_start,
            core_end_sample=core_end,
            context_start_sample=max(0, core_start - context_samples),
            context_end_sample=min(total_samples, core_end + context_samples),
        )
        core_start = core_end
        index += 1


def _hash_wav_frames(path: Path, start_sample: int, end_sample: int) -> str:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as handle:
        handle.setpos(start_sample)
        remaining = end_sample - start_sample
        while remaining > 0:
            count = min(remaining, 65_536)
            frames = handle.readframes(count)
            if not frames:
                raise AudioDecodeError("WAV ended before the advertised frame count")
            digest.update(frames)
            bytes_per_frame = handle.getnchannels() * handle.getsampwidth()
            frames_read = len(frames) // bytes_per_frame
            remaining -= frames_read
    return digest.hexdigest()


def build_audio_blocks(
    *,
    source_id: str,
    source_sha256: str,
    total_samples: int,
    sample_rate: int,
    core_seconds: int | float | Decimal = 300,
    context_seconds: int | float | Decimal = 5,
    pcm_wav_path: str | os.PathLike[str] | None = None,
) -> tuple[AudioBlock, ...]:
    digest = source_sha256.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("source_sha256 must be 64 hexadecimal characters")
    pcm_path = Path(pcm_wav_path) if pcm_wav_path is not None else None
    if pcm_path is not None:
        info = read_wav_info(pcm_path)
        if info.sample_rate != sample_rate or info.frame_count != total_samples:
            raise ValueError("pcm_wav_path parameters do not match total_samples/sample_rate")

    blocks: list[AudioBlock] = []
    for block_range in iter_block_ranges(
        total_samples,
        sample_rate,
        core_seconds=core_seconds,
        context_seconds=context_seconds,
    ):
        identity = {
            "context_end_sample": block_range.context_end_sample,
            "context_start_sample": block_range.context_start_sample,
            "core_end_sample": block_range.core_end_sample,
            "core_start_sample": block_range.core_start_sample,
            "sample_rate": sample_rate,
            "source_sha256": digest,
        }
        pcm_sha256 = (
            _hash_wav_frames(
                pcm_path,
                block_range.context_start_sample,
                block_range.context_end_sample,
            )
            if pcm_path is not None
            else None
        )
        blocks.append(
            AudioBlock(
                block_id=f"block-{digest_json(identity)[:32]}",
                source_id=source_id,
                source_sha256=digest,
                core_start=block_range.core_start,
                core_end=block_range.core_end,
                context_start=block_range.context_start,
                context_end=block_range.context_end,
                pcm_sha256=pcm_sha256,
            )
        )
    return tuple(blocks)


def build_blocks_for_wav(
    pcm_wav_path: str | os.PathLike[str],
    *,
    source_id: str,
    source_sha256: str,
    core_seconds: int | float | Decimal = 300,
    context_seconds: int | float | Decimal = 5,
) -> tuple[AudioBlock, ...]:
    info = read_wav_info(pcm_wav_path)
    return build_audio_blocks(
        source_id=source_id,
        source_sha256=source_sha256,
        total_samples=info.frame_count,
        sample_rate=info.sample_rate,
        core_seconds=core_seconds,
        context_seconds=context_seconds,
        pcm_wav_path=pcm_wav_path,
    )


def block_owns_time(block: AudioBlock, timestamp: float, *, is_final: bool = False) -> bool:
    """Use half-open core ownership to deduplicate context-window results."""

    if not math.isfinite(timestamp):
        return False
    if is_final:
        return block.core_start <= timestamp <= block.core_end
    return block.core_start <= timestamp < block.core_end


__all__ = [
    "AudioDecodeError",
    "BlockRange",
    "DecodedAudio",
    "WavInfo",
    "block_owns_time",
    "build_audio_blocks",
    "build_blocks_for_wav",
    "decode_audio",
    "ffmpeg_decode_command",
    "iter_block_ranges",
    "normalize_audio",
    "read_wav_info",
]
