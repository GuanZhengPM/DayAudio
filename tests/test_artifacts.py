from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayaudio.artifacts import read_evidence, validate_workspace, write_evidence
from dayaudio.config import Settings
from dayaudio.types import EvidenceWindow
from dayaudio.workspace import Workspace


def test_evidence_round_trip_and_empty_workspace_validation(tmp_path: Path) -> None:
    evidence = EvidenceWindow(
        "ev-1", "source-1", 0.0, 1.0, "hello", "high", "fast_default"
    )
    path = tmp_path / "evidence.json"
    write_evidence(path, [evidence])
    assert read_evidence(path) == (evidence,)

    with Workspace(Settings(home=tmp_path / "home")) as workspace:
        report = validate_workspace(workspace)
    assert report["valid"]


def test_evidence_reader_rejects_unknown_confidence(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "dayaudio.evidence.v1",
                "windows": [
                    {
                        "evidence_window_id": "ev-1",
                        "source_id": "source-1",
                        "start": 0,
                        "end": 1,
                        "text": "claim",
                        "confidence": "trusted",
                        "model_state": "tampered",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="confidence"):
        read_evidence(path)
