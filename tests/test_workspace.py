from __future__ import annotations

import struct
import wave
from pathlib import Path

from dayaudio.audio import build_blocks_for_wav
from dayaudio.config import Settings
from dayaudio.paths import filesystem_path
from dayaudio.workspace import Workspace, atomic_json, read_json


def test_workspace_paths_and_atomic_json(tmp_path: Path) -> None:
    with Workspace(Settings(home=tmp_path / "home")) as workspace:
        assert workspace.storage.journal_mode().lower() == "wal"
        assert workspace.owner_profile_path.parent.name == "identity"
        output = atomic_json(workspace.evidence_path, {"ok": True})
        assert read_json(output) == {"ok": True}


def test_atomic_json_supports_paths_beyond_max_path(long_path_root: Path) -> None:
    target = long_path_root / "workspace" / "evidence.json"
    assert len(str(target)) > 260
    assert atomic_json(target, {"text": "长路径", "ok": True}) == target
    assert read_json(target) == {"text": "长路径", "ok": True}


def test_prepare_block_repairs_same_duration_tampering(tmp_path: Path) -> None:
    pcm = tmp_path / "source.wav"
    with wave.open(str(pcm), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(100)
        output.writeframes(struct.pack("<h", 7) * 200)
    block = build_blocks_for_wav(
        pcm,
        source_id="source-1",
        source_sha256="a" * 64,
        core_seconds=2,
        context_seconds=0,
    )[0]
    with Workspace(Settings(home=tmp_path / "home")) as workspace:
        clip = workspace.prepare_block_clip(pcm, block)
        with wave.open(str(filesystem_path(clip)), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(100)
            output.writeframes(struct.pack("<h", 0) * 200)
        workspace.prepare_block_clip(pcm, block)
        with wave.open(str(filesystem_path(clip)), "rb") as repaired:
            assert repaired.readframes(1) == struct.pack("<h", 7)
