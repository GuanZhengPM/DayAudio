"""DayAudio v0.2 command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from dayaudio import __version__
from dayaudio.adapters.campplus import CampPlusBackend, CampPlusConfig
from dayaudio.adapters.command import CommandAdapterConfig, CommandAsrBackend
from dayaudio.adapters.sensevoice import (
    DEFAULT_FSMN_VAD_MODEL,
    FsmnVadBackend,
    FsmnVadConfig,
    SenseVoiceConfig,
    SenseVoiceFsmnBackend,
)
from dayaudio.artifacts import (
    export_transcripts,
    read_evidence,
    validate_workspace,
    write_evidence,
    write_summary_artifact,
)
from dayaudio.audio import build_blocks_for_wav
from dayaudio.bundles import (
    build_day_bundles,
    build_summary_packets,
    read_day_bundles,
    read_summary_packets,
    write_day_bundles,
    write_summary_packets,
)
from dayaudio.cas import atomic_write_text, sha256_file
from dayaudio.config import Settings, default_home, load_settings, write_default_config
from dayaudio.diarize import diarize_file
from dayaudio.doctor import run_doctor
from dayaudio.evidence import build_evidence_from_storage, select_revisions_from_storage
from dayaudio.identity import (
    IdentityDecision,
    append_enrollment_sample,
    calibrate_thresholds,
    create_owner_profile,
    load_owner_profile,
    make_enrollment_sample,
    match_identity,
    save_owner_profile,
)
from dayaudio.ingest import Ingestor
from dayaudio.pipeline import PipelineConfig, ResumablePipeline
from dayaudio.privacy import redact_path
from dayaudio.profiles import get_profile, list_profiles
from dayaudio.speaker import SpeakerAssignment, assign_speaker
from dayaudio.summary import (
    CommandSummaryBackend,
    ExtractiveSummaryBackend,
    HttpSummaryBackend,
    make_summary_request,
    request_from_packet,
    validate_summary_citations,
)
from dayaudio.types import SpeakerTurn
from dayaudio.workspace import Workspace, atomic_json, read_json


def _emit(value: Any, *, json_output: bool = False) -> None:
    if json_output or not isinstance(value, str):
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(value)


def _settings(args: argparse.Namespace) -> Settings:
    requested_home = Path(args.home).expanduser().resolve() if getattr(args, "home", None) else None
    config_value = getattr(args, "config", None)
    config_path = Path(config_value).expanduser().resolve() if config_value else None
    if config_path is None:
        candidate = (requested_home or default_home()) / "config.toml"
        config_path = candidate if candidate.is_file() else None
    return load_settings(config_path, home=requested_home)


def _command_args(value: str | None, repeated: Sequence[str] | None) -> tuple[str, ...] | None:
    if repeated:
        return tuple(repeated)
    if value:
        return tuple(shlex.split(value, posix=os.name != "nt"))
    return None


def _model_weight_digest(model: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit.lower()
    path = Path(model).expanduser()
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        candidates = list(path.glob("campplus*.bin")) + list(path.glob("*.bin"))
        if len(candidates) == 1:
            return sha256_file(candidates[0])
    return None


def _require_reproducible_model(
    model: str,
    *,
    revision: str | None = None,
    digest: str | None = None,
    label: str,
) -> None:
    if Path(model).expanduser().exists() or digest:
        return
    if revision:
        return
    raise ValueError(
        f"{label} must be a local cached path, carry an explicit digest, or use an immutable revision"
    )


def _register_file_artifact(
    workspace: Workspace,
    *,
    kind: str,
    path: Path,
    source_id: str | None = None,
    task_key: str | None = None,
    replace_current: bool = False,
) -> None:
    if replace_current:
        with workspace.storage.transaction(immediate=True) as connection:
            if source_id is None:
                connection.execute(
                    "DELETE FROM artifacts WHERE kind = ? AND source_id IS NULL AND task_key IS NULL",
                    (kind,),
                )
            else:
                connection.execute(
                    "DELETE FROM artifacts WHERE kind = ? AND source_id = ? AND task_key IS NULL",
                    (kind, source_id),
                )
    workspace.storage.add_artifact(
        kind=kind,
        sha256=sha256_file(path),
        path=path,
        size_bytes=path.stat().st_size,
        source_id=source_id,
        task_key=task_key,
        metadata={"current": replace_current},
    )


def _invalidate_derived(workspace: Workspace, *, from_stage: str) -> None:
    stage_kinds = {
        "evidence": {
            "evidence-current",
            "day-bundles-current",
            "summary-packets-current",
            "summary-json",
            "summary-markdown",
        },
        "bundles": {
            "day-bundles-current",
            "summary-packets-current",
            "summary-json",
            "summary-markdown",
        },
        "packets": {"summary-packets-current", "summary-json", "summary-markdown"},
        "summaries": {"summary-json", "summary-markdown"},
    }
    kinds = stage_kinds[from_stage]
    with workspace.storage.transaction(immediate=True) as connection:
        placeholders = ",".join("?" for _ in kinds)
        rows = connection.execute(
            f"SELECT path FROM artifacts WHERE kind IN ({placeholders})",
            tuple(sorted(kinds)),
        ).fetchall()
        connection.execute(
            f"DELETE FROM artifacts WHERE kind IN ({placeholders})",
            tuple(sorted(kinds)),
        )
    safe_root = workspace.settings.work_dir.resolve()
    for row in rows:
        path = Path(row["path"]).resolve()
        if path.is_file() and path.is_relative_to(safe_root):
            path.unlink()
    for path in (workspace.evidence_path, workspace.bundles_path, workspace.packets_path):
        kind = {
            workspace.evidence_path: "evidence-current",
            workspace.bundles_path: "day-bundles-current",
            workspace.packets_path: "summary-packets-current",
        }[path]
        if kind in kinds and path.exists():
            path.unlink()
    if {"summary-json", "summary-markdown"}.intersection(kinds) and workspace.summary_dir.exists():
        shutil.rmtree(workspace.summary_dir)


def command_init(args: argparse.Namespace) -> int:
    settings = _settings(args).ensure_layout()
    config_path = Path(args.output_config).expanduser().resolve() if args.output_config else settings.home / "config.toml"
    if config_path.exists() and not args.force:
        raise FileExistsError(f"configuration already exists: {config_path}")
    write_default_config(config_path, settings)
    with Workspace(settings) as workspace:
        result = {
            "version": __version__,
            "home": str(settings.home),
            "config": str(config_path),
            "database_schema": workspace.storage.schema_version,
            "journal_mode": workspace.storage.journal_mode(),
        }
    _emit(result, json_output=args.json)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    report = run_doctor(
        args.profile or settings.profile,
        asr_command=_command_args(args.asr_command, args.asr_command_arg),
    )
    _emit(report.to_dict(), json_output=True)
    return 0 if report.ready else 2


def command_profiles(args: argparse.Namespace) -> int:
    _emit({"profiles": [item.to_dict() for item in list_profiles()]}, json_output=True)
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with Workspace(settings) as workspace:
        records = Ingestor(
            workspace.storage,
            cas=None if args.reference else workspace.cas,
        ).ingest_many(
            args.paths, recursive=not args.no_recursive
        )
        result = {
            "ingested": len(records),
            "source_storage": "reference" if args.reference else "copied_to_cas",
            "sources": [
                {
                    "source_id": item.source_id,
                    "source_name": item.source_name,
                    "source_sha256": item.source_sha256,
                    "duration_seconds": item.duration_seconds,
                    "recording_start": item.recording_start,
                    "recording_time_basis": item.recording_time_basis,
                }
                for item in records
            ],
        }
    _emit(result, json_output=args.json)
    return 0


def _status_payload(workspace: Workspace) -> dict[str, Any]:
    sources = workspace.storage.list_sources()
    segments = workspace.storage.list_segments(latest_only=False)
    evidence = read_evidence(workspace.evidence_path) if workspace.evidence_path.exists() else ()
    bundles = read_day_bundles(workspace.bundles_path) if workspace.bundles_path.exists() else ()
    packets = read_summary_packets(workspace.packets_path) if workspace.packets_path.exists() else ()
    owner = load_owner_profile(workspace.owner_profile_path)
    artifacts = workspace.storage.list_artifacts()
    return {
        "version": __version__,
        "home": str(workspace.settings.home),
        "database": {
            "schema": workspace.storage.schema_version,
            "journal_mode": workspace.storage.journal_mode(),
        },
        "counts": {
            "sources": len(sources),
            "audio_hours": round(
                sum(float(item.decoded_duration_seconds or item.duration_seconds or 0.0) for item in sources) / 3600,
                6,
            ),
            "segment_revisions": len(segments),
            "authoritative_segments": len(
                select_revisions_from_storage(workspace.storage)
            ),
            "artifacts": len(artifacts),
            "artifact_bytes": sum(item.size_bytes for item in artifacts),
            "decoded_pcm_bytes": sum(
                item.size_bytes for item in artifacts if item.kind == "decoded-pcm"
            ),
            "evidence_windows": len(evidence),
            "day_bundles": len(bundles),
            "summary_packets": len(packets),
            "summaries": len(list(workspace.summary_dir.glob("*.json"))) if workspace.summary_dir.exists() else 0,
        },
        "tasks": workspace.storage.task_queue().counts(),
        "owner_profile": {
            "present": owner is not None,
            "revision": owner.revision if owner else None,
            "positive_samples": len(owner.positive_samples) if owner else 0,
            "negative_samples": len(owner.negative_samples) if owner else 0,
            "mixed_samples": len(owner.mixed_samples) if owner else 0,
        },
    }


def command_status(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        result = _status_payload(workspace)
    _emit(result, json_output=True)
    return 0


def _asr_backend(
    *,
    backend_name: str,
    profile: Any,
    model: str | None,
    model_revision: str | None,
    vad_model: str | None,
    device: str | None,
    command: tuple[str, ...] | None,
    offline: bool,
    label: str,
) -> object:
    normalized = backend_name.casefold()
    if normalized in {"sensevoice", "sensevoice-fsmn"}:
        return SenseVoiceFsmnBackend(
            SenseVoiceConfig(
                model_id=model or profile.fast_model_id,
                model_revision=model_revision,
                vad_model_id=vad_model or DEFAULT_FSMN_VAD_MODEL,
                device=device or profile.asr_device,
                batch_size_seconds=profile.batch_size_seconds,
                disable_update=True,
                offline=offline,
            )
        )
    if normalized == "command":
        if not command:
            raise ValueError(f"{label} command backend requires --{label}-command or --{label}-command-arg")
        return CommandAsrBackend(
            CommandAdapterConfig(
                command=command,
                name=f"{label}-command-asr",
                model_id=model or f"{label}-external-command",
                model_revision=model_revision,
                offline=offline,
            )
        )
    raise ValueError(f"unsupported {label} ASR backend: {backend_name}")


def _selected_sources(workspace: Workspace, identifiers: Sequence[str] | None) -> list[Any]:
    sources = workspace.storage.list_sources()
    if not identifiers:
        return sources
    requested = set(identifiers)
    selected = [item for item in sources if item.source_id in requested]
    missing = sorted(requested - {item.source_id for item in selected})
    if missing:
        raise KeyError(f"unknown source_id: {missing[0]}")
    return selected


def _cleanup_completed_block_files(workspace: Workspace) -> int:
    root = (workspace.settings.work_dir / "blocks").resolve()
    removed = 0
    for task in workspace.storage.task_queue().list(status="complete", kind="asr-block"):
        value = task.payload.get("audio_path") if isinstance(task.payload, dict) else None
        if not value:
            continue
        path = Path(str(value)).resolve()
        if path.is_file() and path.is_relative_to(root):
            path.unlink()
            removed += 1
    return removed


def command_process(args: argparse.Namespace) -> int:
    settings = _settings(args)
    profile = get_profile(args.profile or settings.profile)
    offline = bool(args.offline or settings.offline)
    fast_name = args.backend or profile.fast_backend
    if fast_name in {"sensevoice", "sensevoice-fsmn"}:
        _require_reproducible_model(
            args.model or profile.fast_model_id,
            revision=args.model_revision,
            label="SenseVoice model",
        )
        if not Path(args.vad_model or DEFAULT_FSMN_VAD_MODEL).expanduser().exists():
            raise ValueError("FSMN-VAD must be supplied as a local cached path in v0.2")
    fast_command = _command_args(args.fast_command, args.fast_command_arg)
    strong_command = _command_args(args.strong_command, args.strong_command_arg)
    consensus_command = _command_args(args.consensus_command, args.consensus_command_arg)
    fast = _asr_backend(
        backend_name=fast_name,
        profile=profile,
        model=args.model,
        model_revision=args.model_revision,
        vad_model=args.vad_model,
        device=args.device,
        command=fast_command,
        offline=offline,
        label="fast",
    )
    strong = (
        _asr_backend(
            backend_name="command",
            profile=profile,
            model=args.strong_model,
            model_revision=args.strong_model_revision,
            vad_model=None,
            device=None,
            command=strong_command,
            offline=offline,
            label="strong",
        )
        if strong_command
        else None
    )
    consensus = (
        _asr_backend(
            backend_name="command",
            profile=profile,
            model=args.consensus_model,
            model_revision=None,
            vad_model=None,
            device=None,
            command=consensus_command,
            offline=offline,
            label="consensus",
        )
        if consensus_command
        else None
    )
    processed = 0
    resumed = 0
    enqueued = 0
    final_segments = 0
    verified_complete = 0
    with Workspace(settings) as workspace, ResumablePipeline(
        fast,  # type: ignore[arg-type]
        storage=workspace.storage,
        task_queue=workspace.storage.task_queue(),
        strong_backend=strong,  # type: ignore[arg-type]
        consensus_backend=consensus,  # type: ignore[arg-type]
        config=PipelineConfig(
            profile_name=profile.name,
            lease_seconds=settings.task_lease_seconds,
            max_attempts=settings.max_attempts,
            enable_strong=strong is not None,
            enable_consensus=consensus is not None,
        ),
    ) as pipeline:
        queue = workspace.storage.task_queue()
        queue.recover_stale()
        if args.retry_failed:
            for task in queue.list(status="failed"):
                if (
                    task.model_digest == pipeline.model_digest
                    and task.payload.get("pipeline_config_digest") == pipeline.config_digest
                ):
                    queue.retry(task.task_id)
        sources = _selected_sources(workspace, args.source_id)
        stop = False
        for source in sources:
            if stop:
                break
            pcm = workspace.ensure_decoded(source, verify=True)
            blocks = build_blocks_for_wav(
                pcm,
                source_id=source.source_id,
                source_sha256=source.source_sha256,
                core_seconds=args.core_seconds or settings.core_seconds,
                context_seconds=args.context_seconds
                if args.context_seconds is not None
                else settings.context_seconds,
            )
            for block in blocks:
                if args.max_tasks is not None and processed >= args.max_tasks and not args.enqueue_only:
                    stop = True
                    break
                key = pipeline.task_key(block, force_strong=args.force_strong)
                existing = queue.get(key)
                if existing is not None and existing.status.value == "complete":
                    clip = workspace.prepare_block_clip(pcm, block)
                    cached = pipeline.process_block(
                        block,
                        clip,
                        task_key=key,
                        force_strong=args.force_strong,
                    )
                    resumed += int(cached.resumed)
                    processed += int(not cached.resumed)
                    verified_complete += 1
                    final_segments += len(cached.final_segments)
                    continue
                if existing is not None and existing.status.value in {"failed", "cancelled"}:
                    continue
                clip = workspace.prepare_block_clip(pcm, block)
                pipeline.enqueue_block(block, clip, force_strong=args.force_strong)
                enqueued += 1
                if not args.enqueue_only:
                    result = pipeline.process_next()
                    if result is not None:
                        processed += 1
                        resumed += int(result.resumed)
                        final_segments += len(result.final_segments)
        if not args.keep_blocks and not args.enqueue_only:
            _cleanup_completed_block_files(workspace)
        relevant_tasks = [
            task
            for task in queue.list(kind="asr-block")
            if task.model_digest == pipeline.model_digest
            and task.payload.get("pipeline_config_digest") == pipeline.config_digest
        ]
        incomplete = [
            task
            for task in relevant_tasks
            if task.status.value in {"pending", "running", "failed"}
        ]
        result_payload = {
            "profile": profile.to_dict(),
            "sources": len(sources),
            "enqueued": enqueued,
            "processed": processed,
            "verified_complete": verified_complete,
            "resumed_or_already_complete": resumed,
            "final_segments": final_segments,
            "task_counts": queue.counts(),
            "offline": offline,
            "complete": not incomplete,
            "incomplete_task_ids": [task.task_id for task in incomplete],
        }
        exit_code = 0 if args.enqueue_only or not incomplete else 4
    _emit(result_payload, json_output=True)
    return exit_code


def _owner_decisions_for_speaker_payload(
    workspace: Workspace, payload: dict[str, Any]
) -> dict[str, IdentityDecision]:
    profile = load_owner_profile(workspace.owner_profile_path)
    if profile is None or profile.model_digest != payload.get("model_digest"):
        return {}
    decisions: dict[str, IdentityDecision] = {}
    for cluster in payload.get("clusters", ()):
        local_id = str(cluster["local_speaker_id"])
        decisions[local_id] = match_identity(profile, cluster["centroid"])
    return decisions


def command_diarize(args: argparse.Namespace) -> int:
    settings = _settings(args)
    profile = get_profile(args.profile or settings.profile)
    vad = FsmnVadBackend(
        FsmnVadConfig(
            model_id=args.vad_model or DEFAULT_FSMN_VAD_MODEL,
            device="cpu",
            disable_update=True,
            offline=settings.offline,
        )
    )
    speaker_model = args.speaker_model or "cam++"
    speaker_digest = _model_weight_digest(speaker_model, args.speaker_model_sha256)
    _require_reproducible_model(
        speaker_model,
        digest=speaker_digest,
        label="speaker model",
    )
    embeddings = CampPlusBackend(
        CampPlusConfig(
            model_id=speaker_model,
            device=args.speaker_device or profile.speaker_device,
            weight_sha256=speaker_digest,
            offline=settings.offline,
        )
    )
    rows: list[dict[str, Any]] = []
    try:
        with Workspace(settings) as workspace:
            completed_payloads: list[tuple[Any, Any, dict[str, Any], dict[str, IdentityDecision]]] = []
            for source in _selected_sources(workspace, args.source_id):
                pcm = workspace.ensure_decoded(source)
                result = diarize_file(
                    pcm,
                    source_id=source.source_id,
                    vad_backend=vad,
                    embedding_backend=embeddings,
                    min_window_seconds=args.min_window_seconds,
                    max_window_seconds=args.max_window_seconds,
                    similarity_threshold=args.similarity_threshold,
                )
                payload = result.to_dict()
                decisions = _owner_decisions_for_speaker_payload(workspace, payload)
                payload["identity_decisions"] = {
                    key: {
                        "status": value.status,
                        "score": value.score,
                        "identity_id": value.identity_id,
                        "profile_revision_id": value.profile_revision_id,
                        "reasons": list(value.reasons),
                    }
                    for key, value in decisions.items()
                }
                completed_payloads.append((source, result, payload, decisions))
            _invalidate_derived(workspace, from_stage="evidence")
            for source, result, payload, decisions in completed_payloads:
                atomic_json(workspace.speaker_path(source.source_id), payload)
                _register_file_artifact(
                    workspace,
                    kind="speaker-track-current",
                    path=workspace.speaker_path(source.source_id),
                    source_id=source.source_id,
                    replace_current=True,
                )
                rows.append(
                    {
                        "source_id": source.source_id,
                        "status": result.status,
                        "clusters": len(result.clustering.clusters) if result.clustering else 0,
                        "turns": len(result.clustering.turns) if result.clustering else 0,
                        "owner_matches": sum(value.status == "owner" for value in decisions.values()),
                    }
                )
    finally:
        vad.close()
        embeddings.close()
    _emit({"sources": rows}, json_output=True)
    return 0


def _owner_backend(args: argparse.Namespace) -> CampPlusBackend:
    model = args.speaker_model or "cam++"
    settings = _settings(args)
    digest = _model_weight_digest(model, args.speaker_model_sha256)
    _require_reproducible_model(model, digest=digest, label="speaker model")
    return CampPlusBackend(
        CampPlusConfig(
            model_id=model,
            device=args.speaker_device or "cpu",
            weight_sha256=digest,
            offline=settings.offline,
        )
    )


def command_owner_enroll(args: argparse.Namespace) -> int:
    backend = _owner_backend(args)
    try:
        with Workspace(_settings(args)) as workspace:
            profile = load_owner_profile(workspace.owner_profile_path)
            if profile is None:
                profile = create_owner_profile(
                    args.identity_id,
                    model_digest=backend.model_digest,
                    display_name=args.display_name,
                )
                save_owner_profile(workspace.owner_profile_path, profile)
            elif profile.model_digest != backend.model_digest:
                raise ValueError("owner profile model digest differs from the selected speaker model")
            existing_ids = {item.sample_id for item in profile.samples}
            added = 0
            for status, paths in (
                ("positive", args.positive or ()),
                ("negative", args.negative or ()),
                ("mixed", args.mixed or ()),
            ):
                for supplied in paths:
                    path = Path(supplied).expanduser().resolve()
                    digest = sha256_file(path)
                    sample = make_enrollment_sample(
                        status=status,  # type: ignore[arg-type]
                        embedding=backend.embed_audio(path),
                        source_id=f"enrollment-{digest[:20]}",
                        note=args.note,
                    )
                    if sample.sample_id in existing_ids:
                        continue
                    profile = append_enrollment_sample(profile, sample)
                    save_owner_profile(workspace.owner_profile_path, profile)
                    existing_ids.add(sample.sample_id)
                    added += 1
            calibration = calibrate_thresholds(profile)
            _register_file_artifact(
                workspace,
                kind="owner-profile-current",
                path=workspace.owner_profile_path,
                replace_current=True,
            )
            _invalidate_derived(workspace, from_stage="evidence")
            payload = {
                "identity_id": profile.identity_id,
                "revision": profile.revision,
                "model_digest": profile.model_digest,
                "added": added,
                "positive_samples": len(profile.positive_samples),
                "negative_samples": len(profile.negative_samples),
                "mixed_samples": len(profile.mixed_samples),
                "calibration": calibration.to_dict(),
            }
    finally:
        backend.close()
    _emit(payload, json_output=True)
    return 0


def command_owner_status(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        profile = load_owner_profile(workspace.owner_profile_path)
        if profile is None:
            payload = {"present": False}
        else:
            payload = {
                "present": True,
                "identity_id": profile.identity_id,
                "display_name": profile.display_name,
                "revision": profile.revision,
                "revision_id": profile.revision_id,
                "model_digest": profile.model_digest,
                "positive_samples": len(profile.positive_samples),
                "negative_samples": len(profile.negative_samples),
                "mixed_samples": len(profile.mixed_samples),
                "calibration": calibrate_thresholds(profile).to_dict(),
            }
    _emit(payload, json_output=True)
    return 0


def command_owner_delete(args: argparse.Namespace) -> int:
    if not args.yes:
        raise ValueError("owner delete requires --yes")
    with Workspace(_settings(args)) as workspace:
        existed = workspace.owner_profile_path.exists()
        if existed:
            workspace.owner_profile_path.unlink()
        scrubbed_speaker_files = 0
        if workspace.speaker_dir.exists():
            for path in workspace.speaker_dir.glob("*.json"):
                payload = read_json(path)
                if "identity_decisions" in payload:
                    payload.pop("identity_decisions", None)
                    atomic_json(path, payload)
                    source_id = path.stem
                    _register_file_artifact(
                        workspace,
                        kind="speaker-track-current",
                        path=path,
                        source_id=source_id,
                        replace_current=True,
                    )
                    scrubbed_speaker_files += 1
        # Derived matches are now stale but anonymous speaker tracks remain.
        _invalidate_derived(workspace, from_stage="evidence")
        with workspace.storage.transaction(immediate=True) as connection:
            connection.execute("DELETE FROM artifacts WHERE kind = 'owner-profile-current'")
    _emit(
        {
            "removed_from_workspace": existed,
            "scrubbed_speaker_files": scrubbed_speaker_files,
            "secure_erasure_guaranteed": False,
        },
        json_output=True,
    )
    return 0


def _speaker_context(workspace: Workspace) -> tuple[dict[str, SpeakerAssignment], dict[str, IdentityDecision]]:
    assignments: dict[str, SpeakerAssignment] = {}
    identities: dict[str, IdentityDecision] = {}
    profile = load_owner_profile(workspace.owner_profile_path)
    for source in workspace.storage.list_sources():
        path = workspace.speaker_path(source.source_id)
        if not path.exists():
            continue
        payload = read_json(path)
        turns = tuple(
            SpeakerTurn(
                turn_id=str(item["turn_id"]),
                source_id=str(item["source_id"]),
                local_speaker_id=str(item["local_speaker_id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                model_digest=str(item["model_digest"]),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            )
            for item in payload.get("turns", ())
        )
        segments = sorted(
            workspace.storage.list_segments(source_id=source.source_id, latest_only=False),
            key=lambda item: (item.start, item.end, item.segment_id, item.revision),
        )
        ordered_turns = sorted(turns, key=lambda item: (item.start, item.end, item.turn_id))
        active_turns: list[SpeakerTurn] = []
        turn_index = 0
        for segment in segments:
            while turn_index < len(ordered_turns) and ordered_turns[turn_index].start < segment.end:
                active_turns.append(ordered_turns[turn_index])
                turn_index += 1
            active_turns = [item for item in active_turns if item.end > segment.start]
            assignments[segment.segment_id] = assign_speaker(
                segment.start,
                segment.end,
                active_turns,
                source_id=source.source_id,
            )
        if profile is not None and profile.model_digest == payload.get("model_digest"):
            for cluster in payload.get("clusters", ()):
                identities[str(cluster["local_speaker_id"])] = match_identity(
                    profile, cluster["centroid"]
                )
    return assignments, identities


def command_build_evidence(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        assignments, identities = _speaker_context(workspace)
        windows = build_evidence_from_storage(
            workspace.storage,
            allow_replacement=args.allow_replacement,
            window_seconds=args.window_seconds,
            speaker_assignments=assignments,
            identity_decisions=identities,
        )
        _invalidate_derived(workspace, from_stage="bundles")
        write_evidence(workspace.evidence_path, windows)
        _register_file_artifact(
            workspace,
            kind="evidence-current",
            path=workspace.evidence_path,
            replace_current=True,
        )
        payload = {
            "evidence_windows": len(windows),
            "high": sum(item.confidence == "high" for item in windows),
            "medium": sum(item.confidence == "medium" for item in windows),
            "review": sum(item.confidence == "review" for item in windows),
            "owner": sum(item.participant_role == "owner" for item in windows),
            "mixed_or_unknown": sum(item.participant_role in {"mixed", "unknown"} for item in windows),
            "path": str(workspace.evidence_path),
        }
    _emit(payload, json_output=True)
    return 0


def command_build_bundles(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        evidence = read_evidence(workspace.evidence_path)
        bundles = build_day_bundles(
            _sources_with_recording_time_overrides(workspace),
            evidence,
            timezone_name=args.timezone,
        )
        _invalidate_derived(workspace, from_stage="packets")
        write_day_bundles(workspace.bundles_path, bundles)
        _register_file_artifact(
            workspace,
            kind="day-bundles-current",
            path=workspace.bundles_path,
            replace_current=True,
        )
        payload = {
            "bundles": len(bundles),
            "trusted_day_bundles": sum(item.day_key.startswith("day-") for item in bundles),
            "undated_bundles": sum(item.day_key.startswith("undated-") for item in bundles),
            "path": str(workspace.bundles_path),
        }
    _emit(payload, json_output=True)
    return 0


def command_set_recording_time(args: argparse.Namespace) -> int:
    value = args.timestamp.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO 8601 date or datetime") from exc
    if args.timezone:
        zone = ZoneInfo(args.timezone)
        parsed = parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)
    timestamp = parsed.isoformat()
    with Workspace(_settings(args)) as workspace:
        sources = _selected_sources(workspace, args.source_id)
        if workspace.recording_time_overrides_path.exists():
            document = read_json(workspace.recording_time_overrides_path)
            if document.get("schema_version") != "dayaudio.recording_time_overrides.v1":
                raise ValueError("unsupported recording-time override document")
        else:
            document = {
                "schema_version": "dayaudio.recording_time_overrides.v1",
                "revisions": [],
            }
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for source in sources:
            document["revisions"].append(
                {
                    "source_id": source.source_id,
                    "recording_start": timestamp,
                    "recording_time_basis": "user_provided",
                    "created_at": created_at,
                }
            )
        atomic_json(workspace.recording_time_overrides_path, document)
        _register_file_artifact(
            workspace,
            kind="recording-time-overrides-current",
            path=workspace.recording_time_overrides_path,
            replace_current=True,
        )
        _invalidate_derived(workspace, from_stage="bundles")
    _emit(
        {
            "updated_sources": [source.source_id for source in sources],
            "recording_start": timestamp,
            "recording_time_basis": "user_provided",
        },
        json_output=True,
    )
    return 0


def _sources_with_recording_time_overrides(workspace: Workspace) -> list[Any]:
    sources = workspace.storage.list_sources()
    path = workspace.recording_time_overrides_path
    if not path.exists():
        return sources
    document = read_json(path)
    if document.get("schema_version") != "dayaudio.recording_time_overrides.v1":
        raise ValueError("unsupported recording-time override document")
    latest: dict[str, dict[str, Any]] = {}
    for revision in document.get("revisions", ()):
        latest[str(revision["source_id"])] = revision
    return [
        replace(
            source,
            recording_start=str(latest[source.source_id]["recording_start"]),
            recording_time_basis=str(latest[source.source_id]["recording_time_basis"]),
        )
        if source.source_id in latest
        else source
        for source in sources
    ]


def command_build_packets(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        evidence = read_evidence(workspace.evidence_path)
        bundles = read_day_bundles(workspace.bundles_path)
        packets = tuple(
            packet
            for bundle in bundles
            for packet in build_summary_packets(
                bundle, evidence, packet_seconds=args.packet_seconds
            )
        )
        _invalidate_derived(workspace, from_stage="summaries")
        write_summary_packets(workspace.packets_path, packets)
        _register_file_artifact(
            workspace,
            kind="summary-packets-current",
            path=workspace.packets_path,
            replace_current=True,
        )
        payload = {
            "packets": len(packets),
            "review_packets": sum(item.review_evidence_count > 0 for item in packets),
            "owner_packets": sum(item.owner_evidence_count > 0 for item in packets),
            "path": str(workspace.packets_path),
        }
    _emit(payload, json_output=True)
    return 0


def _summary_backend(args: argparse.Namespace, settings: Settings) -> object:
    if args.summary_backend == "extractive":
        return ExtractiveSummaryBackend()
    if args.summary_backend == "command":
        command = _command_args(args.summary_command, args.summary_command_arg)
        if not command:
            raise ValueError("command summary backend requires --summary-command or --summary-command-arg")
        return CommandSummaryBackend(command)
    if args.summary_backend == "http":
        if not args.allow_network or settings.offline:
            raise ValueError("HTTP summary requires --allow-network and offline mode disabled")
        if not args.summary_endpoint:
            raise ValueError("HTTP summary backend requires --summary-endpoint")
        return HttpSummaryBackend(
            args.summary_endpoint,
            allow_insecure_http=args.allow_insecure_http,
        )
    raise ValueError(f"unknown summary backend: {args.summary_backend}")


def command_summarize(args: argparse.Namespace) -> int:
    settings = _settings(args)
    backend = _summary_backend(args, settings)
    completed: list[dict[str, Any]] = []
    with Workspace(settings) as workspace:
        evidence = read_evidence(workspace.evidence_path)
        packets = read_summary_packets(workspace.packets_path)
        bundles = read_day_bundles(workspace.bundles_path)
        evidence_map = {item.evidence_window_id: item for item in evidence}
        if args.scope == "packet":
            requests = [
                request_from_packet(packet, evidence, language=args.language, max_claims=args.max_claims)
                for packet in packets
                if not args.scope_id or packet.packet_id in set(args.scope_id)
            ]
        else:
            selected_ids = set(args.scope_id or ())
            packet_by_bundle: dict[str, list[str]] = {}
            for packet in packets:
                packet_by_bundle.setdefault(packet.bundle_id, []).append(packet.packet_id)
            requests = []
            for bundle in bundles:
                if selected_ids and bundle.bundle_id not in selected_ids and bundle.day_key not in selected_ids:
                    continue
                windows = [evidence_map[item] for item in bundle.evidence_window_ids]
                requests.append(
                    make_summary_request(
                        bundle.bundle_id,
                        windows,
                        language=args.language,
                        max_claims=args.max_claims,
                        packet_ids=packet_by_bundle.get(bundle.bundle_id, ()),
                        metadata={"day_key": bundle.day_key},
                    )
                )
        for request in requests:
            result = backend.summarize(request)  # type: ignore[attr-defined]
            validation = validate_summary_citations(
                result,
                request.evidence,
                expected_request_id=request.request_id,
                raise_on_error=True,
            )
            summary_json_path = workspace.summary_json_path(request.scope_id, result.summary_id)
            summary_markdown_path = workspace.summary_markdown_path(
                request.scope_id, result.summary_id
            )
            reused = summary_json_path.exists()
            if reused:
                existing = read_json(summary_json_path)
                if (
                    existing.get("schema_version") != "dayaudio.summary.artifact.v1"
                    or existing.get("request", {}).get("request_id") != request.request_id
                    or existing.get("result", {}).get("summary_id") != result.summary_id
                    or not existing.get("validation", {}).get("valid")
                ):
                    raise ValueError("existing summary artifact does not match its content address")
                if not summary_markdown_path.is_file():
                    atomic_write_text(
                        summary_markdown_path,
                        result.to_markdown(),
                        mode=0o600,
                    )
                elif summary_markdown_path.read_text(encoding="utf-8") != result.to_markdown():
                    raise ValueError("existing summary Markdown does not match its content address")
            else:
                write_summary_artifact(
                    summary_json_path,
                    request=request,
                    result=result,
                    validation=validation,
                )
                atomic_write_text(
                    summary_markdown_path,
                    result.to_markdown(),
                    mode=0o600,
                )
            # Registration is idempotent and also repairs a crash that happened
            # after file installation but before the SQLite artifact rows.
            _register_file_artifact(
                workspace,
                kind="summary-json",
                path=summary_json_path,
                task_key=result.summary_id,
            )
            _register_file_artifact(
                workspace,
                kind="summary-markdown",
                path=summary_markdown_path,
                task_key=result.summary_id,
            )
            completed.append(
                {
                    "scope_id": request.scope_id,
                    "summary_id": result.summary_id,
                    "claims": len(result.claims),
                    "valid": validation.valid,
                    "reused": reused,
                }
            )
    _emit({"summaries": completed}, json_output=True)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        report = validate_workspace(workspace)
    _emit(report, json_output=True)
    return 0 if report["valid"] else 3


def command_export(args: argparse.Namespace) -> int:
    with Workspace(_settings(args)) as workspace:
        path = export_transcripts(
            workspace, format=args.format, output=args.output
        )
    _emit({"path": str(path), "format": args.format}, json_output=True)
    return 0


def command_cleanup(args: argparse.Namespace) -> int:
    if not args.yes:
        raise ValueError("cleanup requires --yes")
    if not args.blocks and not args.pcm:
        raise ValueError("cleanup requires --blocks and/or --pcm")
    with Workspace(_settings(args)) as workspace:
        removed_blocks = _cleanup_completed_block_files(workspace) if args.blocks else 0
        removed_pcm = 0
        removed_pcm_bytes = 0
        if args.pcm:
            root = workspace.settings.work_dir.resolve()
            pcm_artifacts = workspace.storage.list_artifacts(kind="decoded-pcm")
            artifact_ids: list[str] = []
            for artifact in pcm_artifacts:
                path = Path(artifact.path).resolve()
                if path.is_file() and path.is_relative_to(root):
                    removed_pcm_bytes += path.stat().st_size
                    path.unlink()
                    removed_pcm += 1
                artifact_ids.append(artifact.artifact_id)
            if artifact_ids:
                with workspace.storage.transaction(immediate=True) as connection:
                    connection.executemany(
                        "DELETE FROM artifacts WHERE artifact_id = ?",
                        ((identifier,) for identifier in artifact_ids),
                    )
    _emit(
        {
            "removed_completed_block_clips": removed_blocks,
            "removed_decoded_pcm_files": removed_pcm,
            "removed_decoded_pcm_bytes": removed_pcm_bytes,
        },
        json_output=True,
    )
    return 0


def command_forget_source(args: argparse.Namespace) -> int:
    if not args.yes:
        raise ValueError("forget-source requires --yes")
    with Workspace(_settings(args)) as workspace:
        source = workspace.storage.require_source(args.source_id)
        artifact_paths = [Path(item.path) for item in workspace.storage.list_artifacts(source_id=source.source_id)]
        with workspace.storage.transaction(immediate=True) as connection:
            connection.execute("DELETE FROM segments WHERE source_id = ?", (source.source_id,))
            connection.execute("DELETE FROM tasks WHERE source_id = ?", (source.source_id,))
            connection.execute("DELETE FROM artifacts WHERE source_id = ?", (source.source_id,))
            connection.execute("DELETE FROM source_locations WHERE source_id = ?", (source.source_id,))
            connection.execute("DELETE FROM sources WHERE source_id = ?", (source.source_id,))
        safe_roots = (workspace.settings.work_dir.resolve(), workspace.settings.cas_dir.resolve())
        removed_files = 0
        for path in artifact_paths + [workspace.pcm_path(source.source_id), workspace.speaker_path(source.source_id)]:
            resolved = path.resolve()
            with workspace.storage.connection() as connection:
                remaining_references = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM artifacts WHERE path = ?", (str(path),)
                    ).fetchone()[0]
                )
            if (
                remaining_references == 0
                and resolved.is_file()
                and any(resolved.is_relative_to(root) for root in safe_roots)
            ):
                if os.name == "nt":  # clear the read-only attribute from older CAS objects
                    os.chmod(resolved, 0o600)
                resolved.unlink()
                removed_files += 1
        blocks = (workspace.settings.work_dir / "blocks" / source.source_id).resolve()
        if blocks.is_dir() and blocks.is_relative_to(workspace.settings.work_dir.resolve()):
            shutil.rmtree(blocks)
        _invalidate_derived(workspace, from_stage="evidence")
        if workspace.recording_time_overrides_path.exists():
            document = read_json(workspace.recording_time_overrides_path)
            document["revisions"] = [
                item
                for item in document.get("revisions", ())
                if item.get("source_id") != source.source_id
            ]
            atomic_json(workspace.recording_time_overrides_path, document)
            _register_file_artifact(
                workspace,
                kind="recording-time-overrides-current",
                path=workspace.recording_time_overrides_path,
                replace_current=True,
            )
        if args.delete_exports and workspace.settings.export_dir.exists():
            shutil.rmtree(workspace.settings.export_dir)
            workspace.settings.export_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _emit(
        {
            "source_id": args.source_id,
            "forgotten": True,
            "removed_files": removed_files,
            "exports_deleted": bool(args.delete_exports),
            "warning": None if args.delete_exports else "existing exports may still contain this source",
        },
        json_output=True,
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", help="workspace directory (default: DAYAUDIO_HOME or ~/.dayaudio)")
    parser.add_argument("--config", help="TOML configuration path")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dayaudio",
        description="Local, resumable, speaker-aware long-audio processing",
    )
    parser.add_argument("--version", action="version", version=f"dayaudio {__version__}")
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize a workspace")
    _add_common(init)
    init.add_argument("--output-config")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=command_init)

    doctor = subparsers.add_parser("doctor", help="probe local runtime capabilities")
    _add_common(doctor)
    doctor.add_argument("--profile")
    doctor.add_argument("--asr-command")
    doctor.add_argument("--asr-command-arg", action="append")
    doctor.set_defaults(func=command_doctor)

    profiles = subparsers.add_parser("profiles", help="list hardware profiles")
    _add_common(profiles)
    profiles.set_defaults(func=command_profiles)

    ingest = subparsers.add_parser("ingest", help="hash, probe, and import audio")
    _add_common(ingest)
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument(
        "--reference",
        action="store_true",
        help="do not copy source containers into CAS; original paths must remain available",
    )
    ingest.set_defaults(func=command_ingest)

    status = subparsers.add_parser("status", help="show durable pipeline state")
    _add_common(status)
    status.set_defaults(func=command_status)

    process = subparsers.add_parser("process", help="decode and transcribe resumable blocks")
    _add_common(process)
    process.add_argument("--source-id", action="append")
    process.add_argument("--profile")
    process.add_argument("--backend", choices=("sensevoice", "sensevoice-fsmn", "command"))
    process.add_argument("--model")
    process.add_argument("--model-revision")
    process.add_argument("--vad-model")
    process.add_argument("--device")
    process.add_argument("--fast-command")
    process.add_argument("--fast-command-arg", action="append")
    process.add_argument("--strong-command")
    process.add_argument("--strong-command-arg", action="append")
    process.add_argument("--strong-model")
    process.add_argument("--strong-model-revision")
    process.add_argument("--consensus-command")
    process.add_argument("--consensus-command-arg", action="append")
    process.add_argument("--consensus-model")
    process.add_argument("--force-strong", action="store_true")
    process.add_argument("--core-seconds", type=float)
    process.add_argument("--context-seconds", type=float)
    process.add_argument("--max-tasks", type=int)
    process.add_argument("--enqueue-only", action="store_true")
    process.add_argument("--keep-blocks", action="store_true")
    process.add_argument("--offline", action="store_true")
    process.add_argument("--retry-failed", action="store_true")
    process.set_defaults(func=command_process)

    diarize = subparsers.add_parser("diarize", help="create file-local anonymous speaker tracks")
    _add_common(diarize)
    diarize.add_argument("--source-id", action="append")
    diarize.add_argument("--profile")
    diarize.add_argument("--vad-model")
    diarize.add_argument("--speaker-model")
    diarize.add_argument("--speaker-model-sha256")
    diarize.add_argument("--speaker-device")
    diarize.add_argument("--min-window-seconds", type=float, default=1.5)
    diarize.add_argument("--max-window-seconds", type=float, default=8.0)
    diarize.add_argument("--similarity-threshold", type=float, default=0.72)
    diarize.set_defaults(func=command_diarize)

    owner = subparsers.add_parser("owner", help="explicit owner voice enrollment")
    owner_sub = owner.add_subparsers(dest="owner_command", required=True)
    enroll = owner_sub.add_parser("enroll", help="append labeled owner/non-owner clips")
    _add_common(enroll)
    enroll.add_argument("--positive", nargs="+", required=True)
    enroll.add_argument("--negative", nargs="+")
    enroll.add_argument("--mixed", nargs="+")
    enroll.add_argument("--identity-id", default="owner")
    enroll.add_argument("--display-name")
    enroll.add_argument("--note")
    enroll.add_argument("--speaker-model")
    enroll.add_argument("--speaker-model-sha256")
    enroll.add_argument("--speaker-device")
    enroll.set_defaults(func=command_owner_enroll)
    owner_status = owner_sub.add_parser("status", help="show non-biometric profile metadata")
    _add_common(owner_status)
    owner_status.set_defaults(func=command_owner_status)
    owner_delete = owner_sub.add_parser("delete", help="irreversibly delete the owner profile")
    _add_common(owner_delete)
    owner_delete.add_argument("--yes", action="store_true")
    owner_delete.set_defaults(func=command_owner_delete)

    evidence = subparsers.add_parser("build-evidence", help="build confidence and identity-aware evidence")
    _add_common(evidence)
    evidence.add_argument("--allow-replacement", action="store_true")
    evidence.add_argument("--window-seconds", type=float, default=30.0)
    evidence.set_defaults(func=command_build_evidence)

    bundles = subparsers.add_parser("build-bundles", help="group evidence by trusted recording day")
    _add_common(bundles)
    bundles.add_argument("--timezone")
    bundles.set_defaults(func=command_build_bundles)

    recording_time = subparsers.add_parser(
        "set-recording-time",
        help="explicitly confirm a recording date/time for one or more sources",
    )
    _add_common(recording_time)
    recording_time.add_argument("timestamp", help="ISO 8601 date or datetime")
    recording_time.add_argument("--source-id", action="append", required=True)
    recording_time.add_argument("--timezone", help="IANA timezone for a date/naive datetime")
    recording_time.set_defaults(func=command_set_recording_time)

    packets = subparsers.add_parser("build-summary-packets", help="make fixed-duration evidence packets")
    _add_common(packets)
    packets.add_argument("--packet-seconds", type=float, default=900.0)
    packets.set_defaults(func=command_build_packets)

    summarize = subparsers.add_parser("summarize", help="create citation-validated summaries")
    _add_common(summarize)
    summarize.add_argument("--scope", choices=("bundle", "packet"), default="bundle")
    summarize.add_argument("--scope-id", action="append")
    summarize.add_argument("--summary-backend", choices=("extractive", "command", "http"), default="extractive")
    summarize.add_argument("--summary-command")
    summarize.add_argument("--summary-command-arg", action="append")
    summarize.add_argument("--summary-endpoint")
    summarize.add_argument("--allow-network", action="store_true")
    summarize.add_argument("--allow-insecure-http", action="store_true")
    summarize.add_argument("--language", default="zh-CN")
    summarize.add_argument("--max-claims", type=int, default=12)
    summarize.set_defaults(func=command_summarize)

    validate = subparsers.add_parser("validate", help="verify hashes, links, and citations")
    _add_common(validate)
    validate.set_defaults(func=command_validate)

    export = subparsers.add_parser("export", help="export selected transcript revisions")
    _add_common(export)
    export.add_argument("--format", choices=("markdown", "jsonl"), default="markdown")
    export.add_argument("--output")
    export.set_defaults(func=command_export)

    cleanup = subparsers.add_parser("cleanup", help="delete recomputable completed block clips")
    _add_common(cleanup)
    cleanup.add_argument("--blocks", action="store_true")
    cleanup.add_argument("--pcm", action="store_true")
    cleanup.add_argument("--yes", action="store_true")
    cleanup.set_defaults(func=command_cleanup)

    forget = subparsers.add_parser("forget-source", help="delete one source and derived local state")
    _add_common(forget)
    forget.add_argument("source_id")
    forget.add_argument("--delete-exports", action="store_true")
    forget.add_argument("--yes", action="store_true")
    forget.set_defaults(func=command_forget_source)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        detail = redact_path(str(exc)).replace("\n", " ")[:500]
        print(f"error: {type(exc).__name__}: {detail}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
