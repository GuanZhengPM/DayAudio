"""Portable evidence, validation, and export artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dayaudio.bundles import read_day_bundles, read_summary_packets
from dayaudio.cas import atomic_write_text
from dayaudio.evidence import evidence_by_id, select_revisions_from_storage
from dayaudio.summary import (
    SummaryRequest,
    summary_result_from_dict,
    validate_summary_citations,
)
from dayaudio.types import AsrSegment, EvidenceWindow
from dayaudio.workspace import Workspace, atomic_json, read_json

EVIDENCE_SCHEMA = "dayaudio.evidence.v1"
SUMMARY_ARTIFACT_SCHEMA = "dayaudio.summary.artifact.v1"


def write_evidence(path: str | Path, windows: Iterable[EvidenceWindow]) -> Path:
    rows = tuple(windows)
    evidence_by_id(rows)
    return atomic_json(
        path,
        {
            "schema_version": EVIDENCE_SCHEMA,
            "windows": [window.to_dict() for window in rows],
        },
    )


def read_evidence(path: str | Path) -> tuple[EvidenceWindow, ...]:
    payload = read_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported evidence document")
    values = payload.get("windows")
    if not isinstance(values, list):
        raise ValueError("evidence document must contain a windows list")
    rows = tuple(
        EvidenceWindow(
            evidence_window_id=str(item["evidence_window_id"]),
            source_id=str(item["source_id"]),
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item["text"]),
            confidence=str(item["confidence"]),  # type: ignore[arg-type]
            model_state=str(item["model_state"]),
            summary_sensitive=bool(item.get("summary_sensitive", False)),
            segment_ids=tuple(str(value) for value in item.get("segment_ids", ())),
            participant_role=str(item.get("participant_role", "unknown")),  # type: ignore[arg-type]
            identity_id=(
                str(item["identity_id"]) if item.get("identity_id") is not None else None
            ),
            review_reasons=tuple(str(value) for value in item.get("review_reasons", ())),
        )
        for item in values
    )
    evidence_by_id(rows)
    return rows


def write_summary_artifact(
    path: str | Path,
    *,
    request: SummaryRequest,
    result: object,
    validation: object,
) -> Path:
    result_dict = getattr(result, "to_dict")()
    validation_dict = getattr(validation, "to_dict")()
    return atomic_json(
        path,
        {
            "schema_version": SUMMARY_ARTIFACT_SCHEMA,
            "request": request.to_dict(),
            "result": result_dict,
            "validation": validation_dict,
        },
    )


def _selected_segments(workspace: Workspace) -> list[AsrSegment]:
    return sorted(
        [item.selected_segment for item in select_revisions_from_storage(workspace.storage)],
        key=lambda item: (item.source_id, item.start, item.end, item.segment_id),
    )


def export_transcripts(
    workspace: Workspace,
    *,
    format: str = "markdown",
    output: str | Path | None = None,
) -> Path:
    """Export selected final revisions without mutating source artifacts."""

    normalized = format.casefold()
    if normalized not in {"markdown", "jsonl"}:
        raise ValueError("export format must be markdown or jsonl")
    suffix = ".md" if normalized == "markdown" else ".jsonl"
    destination = Path(output) if output is not None else workspace.settings.export_dir / f"transcripts{suffix}"
    segments = _selected_segments(workspace)
    sources = {item.source_id: item for item in workspace.storage.list_sources()}
    if normalized == "jsonl":
        text = "".join(
            json.dumps(
                {
                    "schema_version": "dayaudio.transcript.segment.v1",
                    **segment.to_dict(),
                    "source_name": sources.get(segment.source_id).source_name
                    if segment.source_id in sources
                    else None,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for segment in segments
        )
    else:
        lines = ["# DayAudio transcript", ""]
        active_source: str | None = None
        for segment in segments:
            if segment.source_id != active_source:
                active_source = segment.source_id
                source = sources.get(active_source)
                lines.extend([f"## {source.source_name if source else active_source}", ""])
            lines.append(
                f"- `{segment.start:.3f}–{segment.end:.3f}` {segment.text} "
                f"`[{segment.segment_id}:r{segment.revision}]`"
            )
        text = "\n".join(lines).rstrip() + "\n"
    return atomic_write_text(destination, text)


def validate_workspace(workspace: Workspace) -> dict[str, Any]:
    """Validate durable hashes and every available evidence relationship."""

    checks: list[dict[str, Any]] = []
    artifact_rows = workspace.validate_artifacts()
    artifacts_valid = all(
        row["exists"] and row["sha256_matches"] and row["size_matches"]
        for row in artifact_rows
    )
    checks.append(
        {
            "name": "artifact_integrity",
            "valid": artifacts_valid,
            "checked": len(artifact_rows),
            "failures": [row for row in artifact_rows if not (row["exists"] and row["sha256_matches"] and row["size_matches"])],
        }
    )
    registered_by_path = {
        str(Path(item.path).resolve()): item
        for item in workspace.storage.list_artifacts()
    }
    required_paths = [
        path
        for path in (
            workspace.evidence_path,
            workspace.bundles_path,
            workspace.packets_path,
            workspace.recording_time_overrides_path,
            workspace.owner_profile_path,
        )
        if path.exists()
    ]
    if workspace.speaker_dir.exists():
        required_paths.extend(workspace.speaker_dir.glob("*.json"))
    if workspace.summary_dir.exists():
        required_paths.extend(workspace.summary_dir.glob("*.*"))
    unregistered = [
        str(path)
        for path in required_paths
        if str(path.resolve()) not in registered_by_path
    ]
    checks.append(
        {
            "name": "derived_artifact_registration",
            "valid": not unregistered,
            "checked": len(required_paths),
            "unregistered": unregistered,
        }
    )

    evidence: tuple[EvidenceWindow, ...] = ()
    if workspace.evidence_path.exists():
        try:
            evidence = read_evidence(workspace.evidence_path)
            known_segments = {
                (item.segment_id, item.source_id)
                for item in workspace.storage.list_segments(latest_only=False)
            }
            missing = [
                [window.evidence_window_id, segment_id]
                for window in evidence
                for segment_id in window.segment_ids
                if (segment_id, window.source_id) not in known_segments
            ]
            checks.append(
                {
                    "name": "evidence_segments",
                    "valid": not missing,
                    "checked": sum(len(item.segment_ids) for item in evidence),
                    "missing": missing,
                }
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            checks.append({"name": "evidence_segments", "valid": False, "error": type(exc).__name__})

    bundles = ()
    if workspace.bundles_path.exists():
        try:
            bundles = read_day_bundles(workspace.bundles_path)
            known = {item.evidence_window_id for item in evidence}
            missing = [
                [bundle.bundle_id, identifier]
                for bundle in bundles
                for identifier in bundle.evidence_window_ids
                if identifier not in known
            ]
            checks.append({"name": "bundle_evidence", "valid": not missing, "checked": len(bundles), "missing": missing})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            checks.append({"name": "bundle_evidence", "valid": False, "error": type(exc).__name__})

    if workspace.packets_path.exists():
        try:
            packets = read_summary_packets(workspace.packets_path)
            known_bundles = {item.bundle_id for item in bundles}
            known_evidence = {item.evidence_window_id for item in evidence}
            missing = [
                [packet.packet_id, "bundle", packet.bundle_id]
                for packet in packets
                if packet.bundle_id not in known_bundles
            ] + [
                [packet.packet_id, "evidence", identifier]
                for packet in packets
                for identifier in packet.evidence_window_ids
                if identifier not in known_evidence
            ]
            checks.append({"name": "packet_links", "valid": not missing, "checked": len(packets), "missing": missing})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            checks.append({"name": "packet_links", "valid": False, "error": type(exc).__name__})

    summary_count = 0
    summary_failures: list[dict[str, Any]] = []
    evidence_map = evidence_by_id(evidence)
    if workspace.summary_dir.exists():
        for path in sorted(workspace.summary_dir.glob("*.json")):
            summary_count += 1
            try:
                payload = read_json(path)
                if payload.get("schema_version") != SUMMARY_ARTIFACT_SCHEMA:
                    raise ValueError("unsupported summary artifact")
                request_data = payload["request"]
                request_evidence = tuple(
                    evidence_map[str(item["evidence_window_id"])]
                    for item in request_data.get("evidence", ())
                )
                request = SummaryRequest(
                    request_id=str(request_data["request_id"]),
                    scope_id=str(request_data["scope_id"]),
                    evidence=request_evidence,
                    language=str(request_data.get("language", "zh-CN")),
                    max_claims=int(request_data.get("max_claims", 12)),
                    instructions=request_data.get("instructions"),
                    packet_ids=tuple(str(item) for item in request_data.get("packet_ids", ())),
                    metadata=dict(request_data.get("metadata", {})),
                )
                result_data = payload["result"]
                result = summary_result_from_dict(
                    result_data,
                    request=request,
                    backend_id=str(result_data["backend_id"]),
                )
                report = validate_summary_citations(
                    result,
                    request.evidence,
                    expected_request_id=request.request_id,
                )
                if not report.valid:
                    summary_failures.append({"path": path.name, "violations": report.to_dict()["violations"]})
            except (OSError, ValueError, TypeError, KeyError) as exc:
                summary_failures.append({"path": path.name, "error": type(exc).__name__})
    checks.append({"name": "summary_citations", "valid": not summary_failures, "checked": summary_count, "failures": summary_failures})
    return {
        "schema_version": "dayaudio.validation.v1",
        "valid": all(bool(check["valid"]) for check in checks),
        "checks": checks,
        "counts": {
            "sources": len(workspace.storage.list_sources()),
            "segments": len(workspace.storage.list_segments(latest_only=False)),
            "evidence_windows": len(evidence),
            "bundles": len(bundles),
            "summaries": summary_count,
        },
    }


__all__ = [
    "export_transcripts",
    "read_evidence",
    "validate_workspace",
    "write_evidence",
    "write_summary_artifact",
]
