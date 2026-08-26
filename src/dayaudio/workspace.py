"""Workspace lifecycle and durable artifact path conventions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import wave
from pathlib import Path
from typing import Any, Iterable

from dayaudio.audio import DecodedAudio, decode_audio, read_wav_info
from dayaudio.cas import ContentAddressedStore, sha256_file
from dayaudio.config import Settings
from dayaudio.diarize import write_wav_slice
from dayaudio.storage import ArtifactRecord, Storage
from dayaudio.types import AudioBlock, SourceRecord


class WorkspaceError(RuntimeError):
    pass


def atomic_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
        return destination
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


class Workspace:
    """Own the SQLite/CAS handles and stable derived-artifact locations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.ensure_layout()
        self.storage = Storage(self.settings.db_path)
        self.cas = ContentAddressedStore(self.settings.cas_dir)

    @property
    def evidence_path(self) -> Path:
        return self.settings.work_dir / "evidence.json"

    @property
    def bundles_path(self) -> Path:
        return self.settings.work_dir / "day_bundles.json"

    @property
    def packets_path(self) -> Path:
        return self.settings.work_dir / "summary_packets.json"

    @property
    def owner_profile_path(self) -> Path:
        return self.settings.work_dir / "identity" / "owner.jsonl"

    @property
    def recording_time_overrides_path(self) -> Path:
        return self.settings.work_dir / "recording_time_overrides.json"

    @property
    def speaker_dir(self) -> Path:
        return self.settings.work_dir / "speakers"

    @property
    def summary_dir(self) -> Path:
        return self.settings.work_dir / "summaries"

    def pcm_path(self, source_id: str) -> Path:
        return self.settings.work_dir / "pcm" / f"{source_id}.wav"

    def block_path(self, block: AudioBlock) -> Path:
        return self.settings.work_dir / "blocks" / block.source_id / f"{block.block_id}.wav"

    def speaker_path(self, source_id: str) -> Path:
        return self.speaker_dir / f"{source_id}.json"

    def summary_json_path(self, scope_id: str, summary_id: str | None = None) -> Path:
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in scope_id)
        suffix = f".{summary_id}" if summary_id else ""
        return self.summary_dir / f"{safe}{suffix}.json"

    def summary_markdown_path(self, scope_id: str, summary_id: str | None = None) -> Path:
        return self.summary_json_path(scope_id, summary_id).with_suffix(".md")

    def _source_container_path(self, source: SourceRecord) -> Path:
        artifacts = self.storage.list_artifacts(
            source_id=source.source_id, kind="source-audio"
        )
        for artifact in reversed(artifacts):
            candidate = Path(artifact.path)
            if candidate.is_file():
                if sha256_file(candidate) != artifact.sha256:
                    raise WorkspaceError(
                        f"source CAS integrity check failed for {source.source_id}"
                    )
                return candidate
        original = Path(source.source_path)
        if original.is_file():
            return original
        for location in reversed(self.storage.source_locations(source.source_id)):
            candidate = Path(location)
            if candidate.is_file():
                return candidate
        raise WorkspaceError(
            f"source bytes are unavailable for {source.source_id}; re-ingest the file"
        )

    def _record_decoded(self, source: SourceRecord, decoded: DecodedAudio) -> ArtifactRecord:
        self.storage.set_decoded_duration(source.source_id, decoded.duration_seconds)
        return self.storage.add_artifact(
            kind="decoded-pcm",
            sha256=decoded.sha256,
            path=decoded.path,
            size_bytes=decoded.size_bytes,
            source_id=source.source_id,
            metadata={
                "sample_rate": decoded.sample_rate,
                "channels": decoded.channels,
                "sample_width": decoded.sample_width,
                "sample_count": decoded.sample_count,
            },
        )

    def ensure_decoded(self, source: SourceRecord, *, verify: bool = True) -> Path:
        """Return canonical PCM, decoding atomically only when needed."""

        artifacts = self.storage.list_artifacts(
            source_id=source.source_id, kind="decoded-pcm"
        )
        for artifact in reversed(artifacts):
            candidate = Path(artifact.path)
            if candidate.is_file() and (not verify or sha256_file(candidate) == artifact.sha256):
                return candidate

        if artifacts:
            with self.storage.transaction(immediate=True) as connection:
                connection.execute(
                    "DELETE FROM artifacts WHERE source_id = ? AND kind = 'decoded-pcm'",
                    (source.source_id,),
                )

        destination = self.pcm_path(source.source_id)
        if destination.is_file():
            # An unregistered or hash-mismatched PCM is not authoritative.
            # It is derived and can be recreated from the verified source.
            destination.unlink()

        decoded = decode_audio(
            self._source_container_path(source),
            destination,
            overwrite=False,
        )
        self._record_decoded(source, decoded)
        return destination

    def prepare_block_clip(self, pcm_path: str | Path, block: AudioBlock) -> Path:
        """Materialize one context clip atomically for a resumable task."""

        destination = self.block_path(block)

        def frame_digest(path: Path) -> str:
            digest = hashlib.sha256()
            with wave.open(str(path), "rb") as input_audio:
                while True:
                    frames = input_audio.readframes(65_536)
                    if not frames:
                        break
                    digest.update(frames)
            return digest.hexdigest()

        if destination.is_file():
            info = read_wav_info(destination)
            expected = block.context_end - block.context_start
            hash_matches = block.pcm_sha256 is None or frame_digest(destination) == block.pcm_sha256
            if (
                abs(info.duration_seconds - expected) <= max(1 / info.sample_rate, 1e-6)
                and hash_matches
            ):
                return destination
        result = write_wav_slice(
            pcm_path,
            destination,
            start=block.context_start,
            end=block.context_end,
        )
        if block.pcm_sha256 is not None and frame_digest(result) != block.pcm_sha256:
            result.unlink(missing_ok=True)
            raise WorkspaceError("materialized block PCM hash does not match its task identity")
        return result

    def validate_artifacts(self, *, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        selected = set(kinds or ())
        results: list[dict[str, Any]] = []
        for artifact in self.storage.list_artifacts():
            if selected and artifact.kind not in selected:
                continue
            path = Path(artifact.path)
            exists = path.is_file()
            actual = sha256_file(path) if exists else None
            results.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "exists": exists,
                    "sha256_matches": actual == artifact.sha256 if exists else False,
                    "size_matches": path.stat().st_size == artifact.size_bytes if exists else False,
                }
            )
        return results

    def close(self) -> None:
        self.storage.close()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["Workspace", "WorkspaceError", "atomic_json", "read_json"]
