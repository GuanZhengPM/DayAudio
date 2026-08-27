from __future__ import annotations

import shutil
import struct
import sys
import wave
from pathlib import Path

import pytest

import dayaudio.cli as cli_module
import dayaudio.ingest as ingest_module
from dayaudio.cli import main
from dayaudio.config import Settings
from dayaudio.ingest import AudioProbe
from dayaudio.paths import filesystem_path, filesystem_tree_path
from dayaudio.types import AsrSegment
from dayaudio.workspace import Workspace


def _wav(path: Path, seconds: float = 2.0) -> None:
    rate = 16_000
    filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    with wave.open(str(filesystem_path(path)), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(
            b"".join(
                struct.pack("<h", (index % 100) - 50)
                for index in range(round(seconds * rate))
            )
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="public CLI E2E requires FFmpeg and FFprobe",
)
def test_public_command_e2e(tmp_path: Path) -> None:
    home = tmp_path / "home"
    audio = tmp_path / "fixture.wav"
    _wav(audio)
    fixture_command = Path(__file__).parents[1] / "scripts" / "fixture_asr_command.py"

    assert main(["init", "--home", str(home), "--json"]) == 0
    assert main(["ingest", "--home", str(home), str(audio), "--json"]) == 0
    assert (
        main(
            [
                "process",
                "--home",
                str(home),
                "--backend",
                "command",
                "--fast-command-arg",
                sys.executable,
                "--fast-command-arg",
                str(fixture_command),
                "--fast-command-arg",
                "{audio}",
                "--max-tasks",
                "1",
                "--json",
            ]
        )
        == 0
    )
    assert main(["build-evidence", "--home", str(home), "--json"]) == 0
    with Workspace(Settings(home=home)) as workspace:
        source_id = workspace.storage.list_sources()[0].source_id
    assert (
        main(
            [
                "set-recording-time",
                "2026-08-26T09:00:00+08:00",
                "--home",
                str(home),
                "--source-id",
                source_id,
                "--json",
            ]
        )
        == 0
    )
    assert main(["build-bundles", "--home", str(home), "--json"]) == 0
    assert main(["build-summary-packets", "--home", str(home), "--json"]) == 0
    assert main(["summarize", "--home", str(home), "--json"]) == 0
    assert main(["summarize", "--home", str(home), "--json"]) == 0
    assert main(["validate", "--home", str(home), "--json"]) == 0
    assert (
        main(
            [
                "set-recording-time",
                "2026-08-27T09:00:00+08:00",
                "--home",
                str(home),
                "--source-id",
                source_id,
                "--json",
            ]
        )
        == 0
    )
    assert main(["validate", "--home", str(home), "--json"]) == 0
    assert main(["build-bundles", "--home", str(home), "--json"]) == 0
    assert main(["build-summary-packets", "--home", str(home), "--json"]) == 0
    assert main(["summarize", "--home", str(home), "--json"]) == 0
    assert main(["validate", "--home", str(home), "--json"]) == 0
    output = tmp_path / "transcript.md"
    assert (
        main(
            [
                "export",
                "--home",
                str(home),
                "--format",
                "markdown",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    assert "公开生成" in output.read_text(encoding="utf-8")


class _CliStubBackend:
    name = "cli-stub"
    model_id = "cli-stub-model"
    model_revision = "r1"
    config = {"fixture": True}

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        block_id: str,
        offset_seconds: float = 0.0,
    ) -> list[AsrSegment]:
        assert filesystem_path(audio_path).is_file()
        return [
            AsrSegment(
                segment_id=f"segment-{block_id}",
                source_id=source_id,
                start=offset_seconds,
                end=offset_seconds + 1.0,
                text="深路径转录",
                model_id=self.model_id,
                model_revision=self.model_revision,
                confidence=0.99,
                block_id=block_id,
            )
        ]

    def close(self) -> None:
        return None


def test_cli_common_workflow_supports_deep_workspace(
    long_path_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = long_path_root / "home"
    audio = long_path_root / "inputs" / "fixture.wav"
    _wav(audio)

    def fake_probe(path: Path, **_: object) -> AudioProbe:
        return AudioProbe(
            duration_seconds=2.0,
            codec="pcm_s16le",
            sample_rate=16_000,
            channels=1,
            recording_start=None,
            recording_time_basis=None,
            raw={"path_name": Path(path).name},
        )
    monkeypatch.setattr(ingest_module, "probe_audio", fake_probe)
    assert main(["init", "--home", str(home), "--json"]) == 0
    assert filesystem_path(home / "config.toml").is_file()
    assert main(["ingest", "--home", str(home), str(audio), "--json"]) == 0
    with Workspace(Settings(home=home)) as workspace:
        source = workspace.storage.list_sources()[0]
        assert source.source_path == str(audio.resolve())
        assert not source.source_path.startswith("\\\\?\\")

    monkeypatch.setattr(cli_module, "_asr_backend", lambda **_: _CliStubBackend())
    monkeypatch.setattr(
        Workspace,
        "ensure_decoded",
        lambda _self, source, **_: Path(source.source_path),
    )
    assert (
        main(
            [
                "process",
                "--home",
                str(home),
                "--backend",
                "command",
                "--keep-blocks",
                "--json",
            ]
        )
        == 0
    )
    assert main(["validate", "--home", str(home), "--json"]) == 0
    with Workspace(Settings(home=home)) as workspace:
        assert all(
            not str(artifact.path).startswith("\\\\?\\")
            for artifact in workspace.storage.list_artifacts()
        )

    export = home / "exports" / "deep-transcript.md"
    assert (
        main(
            [
                "export",
                "--home",
                str(home),
                "--output",
                str(export),
                "--json",
            ]
        )
        == 0
    )
    assert "深路径转录" in filesystem_path(export).read_text(encoding="utf-8")
    assert main(["cleanup", "--home", str(home), "--blocks", "--yes", "--json"]) == 0
    blocks = home / "work" / "blocks"
    assert not list(filesystem_tree_path(blocks).rglob("*.wav"))


def test_local_speaker_model_digest_binds_nested_directory_content(tmp_path: Path) -> None:
    model = tmp_path / "speaker-model"
    nested = model / "nested"
    filesystem_tree_path(nested).mkdir(parents=True)
    primary = model / "campplus.bin"
    secondary = nested / "projection.bin"
    filesystem_path(primary).write_bytes(b"primary-v1")
    filesystem_path(secondary).write_bytes(b"secondary")

    first = cli_module._model_weight_digest(str(model), None)
    assert first is not None and len(first) == 64

    filesystem_path(primary).write_bytes(b"primary-v2")
    second = cli_module._model_weight_digest(str(model), None)
    assert second is not None and second != first

    filesystem_path(secondary).write_bytes(b"changed-secondary")
    third = cli_module._model_weight_digest(str(model), None)
    assert third is not None and third not in {first, second}
