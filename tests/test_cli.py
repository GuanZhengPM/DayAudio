from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

from dayaudio.cli import main
from dayaudio.config import Settings
from dayaudio.workspace import Workspace


def _wav(path: Path, seconds: float = 2.0) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(
            b"".join(
                struct.pack("<h", (index % 100) - 50)
                for index in range(round(seconds * rate))
            )
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
