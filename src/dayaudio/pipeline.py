"""Resumable ASR block worker with conservative strong-model routing.

This module owns no database or audio decoder.  It consumes protocol-shaped
objects (``add_segment``, ``list_segments``, ``enqueue``, ``claim``...) so the
standard SQLite implementation and small embedders/tests can use the same
worker.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import threading
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dayaudio.adapters.base import AsrBackend, close_all
from dayaudio.paths import filesystem_path
from dayaudio.router import (
    CascadeDecision,
    CascadePolicy,
    CascadeRouter,
    annotate_anomalies,
    detect_anomalies,
)
from dayaudio.types import AsrSegment, AudioBlock


def _filesystem_identity(path: str | Path) -> str:
    """Return a stable comparison key for conventional and namespaced paths."""

    value = filesystem_path(path, force_extended=True)
    try:
        value = value.resolve()
    except (OSError, RuntimeError):
        pass
    return os.path.normcase(os.path.normpath(str(value)))


def _same_filesystem_entry(left: str | Path, right: str | Path) -> bool:
    if _filesystem_identity(left) == _filesystem_identity(right):
        return True
    try:
        return filesystem_path(left, force_extended=True).samefile(
            filesystem_path(right, force_extended=True)
        )
    except OSError:
        return False


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(PipelineError):
    pass


PIPELINE_SCHEMA_VERSION = "dayaudio.pipeline.v0.2.1"


class LeaseGuardError(PipelineError):
    pass


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    profile_name: str = "auto"
    task_kind: str = "asr-block"
    lease_seconds: float = 7200.0
    max_attempts: int = 3
    retry_delay_seconds: float = 5.0
    enable_strong: bool = True
    enable_consensus: bool = True
    model_digest: str | None = None
    config_digest: str | None = None
    worker_id: str | None = None

    def __post_init__(self) -> None:
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    task_key: str
    block_id: str
    source_id: str
    fast_segments: tuple[AsrSegment, ...]
    strong_segments: tuple[AsrSegment, ...]
    consensus_segments: tuple[AsrSegment, ...]
    final_segments: tuple[AsrSegment, ...]
    decisions: tuple[CascadeDecision, ...]
    resumed: bool = False

    @property
    def review_count(self) -> int:
        return sum(decision.action == "review" for decision in self.decisions)

    @property
    def accepted_strong_count(self) -> int:
        return sum(decision.accepted_strong for decision in self.decisions)

    def manifest(self) -> dict[str, Any]:
        """Return provenance/counts without transcript text or source paths."""

        return {
            "task_key": self.task_key,
            "block_id": self.block_id,
            "source_id": self.source_id,
            "fast_segment_ids": [item.segment_id for item in self.fast_segments],
            "strong_segment_ids": [item.segment_id for item in self.strong_segments],
            "consensus_segment_ids": [item.segment_id for item in self.consensus_segments],
            "final_revisions": [
                [item.segment_id, item.revision] for item in self.final_segments
            ],
            "review_count": self.review_count,
            "accepted_strong_count": self.accepted_strong_count,
            "resumed": self.resumed,
        }


def _digestable(value: Any) -> Any:
    """Canonicalize configuration material without ever returning it publicly."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        resolved = value.expanduser().resolve()
        descriptor: dict[str, Any] = {"path": str(resolved)}
        try:
            stat = filesystem_path(resolved).stat()
        except OSError:
            descriptor["exists"] = False
        else:
            descriptor.update(
                {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return descriptor
    if isinstance(value, Enum):
        return _digestable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _digestable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _digestable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_digestable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_digestable(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    # Avoid memory-address reprs.  Custom adapters should expose a dataclass or
    # mapping ``config`` and an explicit model/weights digest for full binding.
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _content_path_descriptor(path: Path) -> dict[str, Any] | None:
    resolved = path.expanduser().resolve()
    # A model root can be below MAX_PATH while one of its descendants is not.
    filesystem_resolved = filesystem_path(resolved, force_extended=True)
    if filesystem_resolved.is_file():
        from dayaudio.cas import sha256_file

        stat = filesystem_resolved.stat()
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": stat.st_size,
            "sha256": sha256_file(resolved),
        }
    if filesystem_resolved.is_dir():
        from dayaudio.cas import sha256_file

        digest = hashlib.sha256()
        count = 0
        total = 0
        for child in sorted(
            item for item in filesystem_resolved.rglob("*") if item.is_file()
        ):
            relative = child.relative_to(filesystem_resolved).as_posix()
            child_size = child.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(child_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\n")
            count += 1
            total += child_size
        return {
            "path": str(resolved),
            "kind": "directory",
            "file_count": count,
            "size_bytes": total,
            "tree_sha256": digest.hexdigest(),
        }
    return None


def _backend_local_inputs(backend: AsrBackend) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    cwd_value = getattr(getattr(backend, "config", None), "cwd", None)
    cwd = Path(cwd_value).expanduser() if cwd_value else Path.cwd()

    def consider(value: Any) -> None:
        if not isinstance(value, (str, Path)):
            return
        raw = str(value)
        if "{" in raw or "}" in raw:
            return
        if "=" in raw and raw.startswith("-"):
            raw = raw.split("=", 1)[1]
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = cwd / path
        filesystem_candidate = filesystem_path(path)
        if filesystem_candidate.is_file() or filesystem_candidate.is_dir():
            candidates.append(path)

    consider(getattr(backend, "model_id", None))
    for name in ("weights_path", "model_path", "vad_model_path"):
        consider(getattr(backend, name, None))
    config = getattr(backend, "config", None)
    command = getattr(config, "command", None) or getattr(backend, "command", None)
    if isinstance(command, (list, tuple)):
        for argument in command:
            consider(argument)

    descriptors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        descriptor = _content_path_descriptor(candidate)
        if descriptor is not None:
            descriptors.append(descriptor)
            seen.add(resolved)
    return descriptors


def _backend_digest(backend: AsrBackend | None) -> str:
    if backend is None:
        return "none"
    material: dict[str, Any] = {
        "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "adapter": getattr(backend, "name", type(backend).__name__),
        "model_id": getattr(backend, "model_id", "unknown"),
        "model_revision": getattr(backend, "model_revision", None),
    }
    for name in (
        "config",
        "model_digest",
        "weights_digest",
        "weights_path",
        "model_path",
        "vad_model_path",
        "command",
        "language",
        "use_itn",
        "device",
        "quantization",
        "backend_options",
    ):
        if hasattr(backend, name):
            material[name] = _digestable(getattr(backend, name))
    local_inputs = _backend_local_inputs(backend)
    if local_inputs:
        material["local_inputs"] = local_inputs
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "backend-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _digest_configuration(config: PipelineConfig, policy: CascadePolicy) -> str:
    # Lease/retry/worker settings do not affect model output and therefore must
    # not invalidate durable ASR results.
    payload = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline": {
            "profile_name": config.profile_name,
            "enable_strong": config.enable_strong,
            "enable_consensus": config.enable_consensus,
        },
        "cascade": asdict(policy),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_resume_key(
    block: AudioBlock,
    *,
    model_digest: str,
    config_digest: str,
    kind: str = "asr-block",
    summary_sensitive_segment_ids: Iterable[str] = (),
    force_strong: bool = False,
) -> str:
    """Return the exact durable task identity used by :class:`TaskQueue`."""

    # Import locally so adapter-only users do not initialize storage.  Reusing
    # the queue's canonical decimal encoding guarantees direct and queued work
    # address the same computation.
    from dayaudio.tasks import make_task_key

    block_config_digest = make_block_config_digest(
        block,
        config_digest=config_digest,
        summary_sensitive_segment_ids=summary_sensitive_segment_ids,
        force_strong=force_strong,
    )
    return make_task_key(
        source_sha256=block.source_sha256,
        range_start=block.core_start,
        range_end=block.core_end,
        model_digest=model_digest,
        config_digest=block_config_digest,
        kind=kind,
    )


def make_block_config_digest(
    block: AudioBlock,
    *,
    config_digest: str,
    summary_sensitive_segment_ids: Iterable[str] = (),
    force_strong: bool = False,
) -> str:
    """Bind context PCM and result-changing routing inputs to one block."""

    payload = {
        "base_config_digest": config_digest,
        "block_id": block.block_id,
        "context_start": format(block.context_start, ".17g"),
        "context_end": format(block.context_end, ".17g"),
        "pcm_sha256": block.pcm_sha256,
        "summary_sensitive_segment_ids": sorted(
            {str(value) for value in summary_sensitive_segment_ids}
        ),
        "force_strong": bool(force_strong),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _supported_call(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call a duck-typed API with only keyword arguments it advertises."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **kwargs)


def _assert_active(
    heartbeat: Callable[[], None] | None,
    cancelled: Callable[[], bool] | None,
) -> None:
    try:
        if cancelled is not None and cancelled():
            raise PipelineCancelled("task cancellation requested")
        if heartbeat is not None:
            heartbeat()
    except PipelineCancelled:
        raise
    except Exception as exc:
        raise LeaseGuardError("task lease is not active") from exc


def _run_with_lease_guard(
    function: Callable[[], Any],
    *,
    heartbeat: Callable[[], None] | None,
    cancelled: Callable[[], bool] | None,
    lease_seconds: float,
) -> Any:
    """Keep a lease alive during blocking inference and fence stale writes."""

    try:
        _assert_active(heartbeat, cancelled)
    except PipelineCancelled:
        raise
    except Exception as exc:
        raise LeaseGuardError("task lease is not active") from exc
    if heartbeat is None and cancelled is None:
        return function()
    stop = threading.Event()
    failures: list[Exception] = []
    interval = max(0.05, min(60.0, lease_seconds / 3.0))

    def maintain() -> None:
        while not stop.wait(interval):
            try:
                _assert_active(heartbeat, cancelled)
            except Exception as exc:
                failures.append(exc)
                stop.set()
                return

    thread = threading.Thread(
        target=maintain,
        name="dayaudio-lease-heartbeat",
        daemon=True,
    )
    thread.start()
    function_failure: BaseException | None = None
    result: Any = None
    try:
        result = function()
    except BaseException as exc:
        function_failure = exc
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval * 2.0))
    if failures:
        failure = failures[0]
        if isinstance(failure, PipelineCancelled):
            raise failure
        raise LeaseGuardError("task lease was lost during inference") from failure
    if function_failure is not None:
        raise function_failure
    try:
        _assert_active(heartbeat, cancelled)
    except PipelineCancelled:
        raise
    except Exception as exc:
        raise LeaseGuardError("task lease is not active after inference") from exc
    return result


def _metadata_stage(
    segment: AsrSegment,
    *,
    stage: str,
    task_key: str,
    revision: int | None = None,
    decision: CascadeDecision | None = None,
    base_segment_id: str | None = None,
    logical_stage: str | None = None,
    is_fast: bool | None = None,
    raw_artifact_id: str | None = None,
) -> AsrSegment:
    metadata = dict(segment.metadata)
    metadata["pipeline_stage"] = stage
    metadata["pipeline_task_key"] = task_key
    if base_segment_id is not None:
        metadata["base_segment_id"] = base_segment_id
    if logical_stage is not None:
        metadata["stage"] = logical_stage
    if is_fast is not None:
        metadata["is_fast"] = is_fast
    metadata["raw_output_retained"] = raw_artifact_id is not None
    if raw_artifact_id is not None:
        metadata["raw_artifact_id"] = raw_artifact_id
    if decision is not None:
        metadata["cascade_action"] = decision.action
        metadata["cascade_reasons"] = list(decision.reasons)
    anomaly_flags = segment.anomaly_flags
    if decision is not None and decision.action == "review":
        anomaly_flags = tuple(dict.fromkeys((*anomaly_flags, "cascade_review")))
    return replace(
        segment,
        revision=segment.revision if revision is None else revision,
        anomaly_flags=anomaly_flags,
        metadata=metadata,
    )


def _temporal_match(
    target: AsrSegment, candidates: Sequence[AsrSegment]
) -> AsrSegment | None:
    best: AsrSegment | None = None
    best_score = 0.0
    for candidate in candidates:
        overlap = max(0.0, min(target.end, candidate.end) - max(target.start, candidate.start))
        union = max(target.end, candidate.end) - min(target.start, candidate.start)
        score = overlap / union if union > 0 else 0.0
        if score > best_score:
            best, best_score = candidate, score
    return best


def _one_to_one_temporal_matches(
    targets: Sequence[AsrSegment], candidates: Sequence[AsrSegment]
) -> dict[str, AsrSegment]:
    scored: list[tuple[float, int, int]] = []
    for target_index, target in enumerate(targets):
        for candidate_index, candidate in enumerate(candidates):
            overlap = max(
                0.0,
                min(target.end, candidate.end) - max(target.start, candidate.start),
            )
            union = max(target.end, candidate.end) - min(target.start, candidate.start)
            score = overlap / union if union > 0 else 0.0
            if score > 0:
                scored.append((score, target_index, candidate_index))
    assigned_targets: set[int] = set()
    assigned_candidates: set[int] = set()
    result: dict[str, AsrSegment] = {}
    for _, target_index, candidate_index in sorted(
        scored, key=lambda item: (-item[0], item[1], item[2])
    ):
        if target_index in assigned_targets or candidate_index in assigned_candidates:
            continue
        target = targets[target_index]
        result[target.segment_id] = candidates[candidate_index]
        assigned_targets.add(target_index)
        assigned_candidates.add(candidate_index)
    return result


def _review_decision(
    fast: AsrSegment,
    strong: AsrSegment | None,
    *,
    reason: str,
    router: CascadeRouter,
) -> CascadeDecision:
    selected = annotate_anomalies(fast, router.policy.anomaly)
    review_flag = reason.replace(":", "_")
    metadata = dict(selected.metadata)
    metadata["review_reason"] = reason
    selected = replace(
        selected,
        anomaly_flags=tuple(
            dict.fromkeys((*selected.anomaly_flags, review_flag))
        ),
        metadata=metadata,
    )
    return CascadeDecision(
        selected=selected,
        action="review",
        reasons=(reason,),
        fast_anomalies=detect_anomalies(fast, policy=router.policy.anomaly),
        strong_anomalies=(
            detect_anomalies(strong, policy=router.policy.anomaly)
            if strong is not None
            else None
        ),
        text_coverage=None,
        text_expansion=None,
        consensus_similarity=None,
    )


def _candidate_base(
    candidate: AsrSegment, fast_segments: Sequence[AsrSegment]
) -> tuple[AsrSegment | None, bool]:
    """Return an evidence base and whether it has actual temporal overlap."""

    overlapping = _temporal_match(candidate, fast_segments)
    if overlapping is not None:
        return overlapping, True
    if not fast_segments:
        return None, False
    midpoint = (candidate.start + candidate.end) / 2.0
    nearest = min(
        fast_segments,
        key=lambda item: abs(((item.start + item.end) / 2.0) - midpoint),
    )
    return nearest, False


def _core_owned_segment(
    segment: AsrSegment, block: AudioBlock
) -> AsrSegment | None:
    """Apply midpoint ownership and clip timestamps to the deterministic core."""

    midpoint = (segment.start + segment.end) / 2.0
    final_block = block.core_end >= block.context_end - 1e-9
    if midpoint < block.core_start:
        return None
    if midpoint >= block.core_end and not final_block:
        return None
    if midpoint > block.core_end and final_block:
        return None
    clipped_start = max(block.core_start, segment.start)
    clipped_end = min(block.core_end, segment.end)
    if clipped_end <= clipped_start:
        return None
    if clipped_start == segment.start and clipped_end == segment.end:
        return segment
    flags = tuple(
        dict.fromkeys((*segment.anomaly_flags, "boundary_timestamp_clipped"))
    )
    metadata = dict(segment.metadata)
    metadata["boundary_timestamp_clipped"] = {
        "original_start": segment.start,
        "original_end": segment.end,
        "core_start": block.core_start,
        "core_end": block.core_end,
    }
    return replace(
        segment,
        start=clipped_start,
        end=clipped_end,
        anomaly_flags=flags,
        metadata=metadata,
    )


def _core_owned_segments(
    segments: Iterable[AsrSegment], block: AudioBlock
) -> list[AsrSegment]:
    result: list[AsrSegment] = []
    for segment in segments:
        owned = _core_owned_segment(segment, block)
        if owned is not None:
            result.append(owned)
    return result


class ResumablePipeline:
    def __init__(
        self,
        fast_backend: AsrBackend,
        *,
        storage: object | None = None,
        task_queue: object | None = None,
        strong_backend: AsrBackend | None = None,
        consensus_backend: AsrBackend | None = None,
        router: CascadeRouter | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.fast_backend = fast_backend
        self.strong_backend = strong_backend
        self.consensus_backend = consensus_backend
        self.storage = storage
        self.task_queue = task_queue
        self.router = router or CascadeRouter()
        self.config = config or PipelineConfig()
        self.worker_id = self.config.worker_id or f"{socket.gethostname()}-{id(self):x}"
        self._model_digest_value: str | None = None

    @property
    def model_digest(self) -> str:
        if self.config.model_digest:
            return self.config.model_digest
        if self._model_digest_value is not None:
            return self._model_digest_value
        material = "\0".join(
            (
                _backend_digest(self.fast_backend),
                _backend_digest(self.strong_backend),
                _backend_digest(self.consensus_backend),
            )
        )
        self._model_digest_value = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return self._model_digest_value

    @property
    def config_digest(self) -> str:
        return self.config.config_digest or _digest_configuration(
            self.config, self.router.policy
        )

    def task_key(
        self,
        block: AudioBlock,
        *,
        summary_sensitive_segment_ids: Iterable[str] = (),
        force_strong: bool = False,
    ) -> str:
        return make_resume_key(
            block,
            model_digest=self.model_digest,
            config_digest=self.config_digest,
            kind=self.config.task_kind,
            summary_sensitive_segment_ids=summary_sensitive_segment_ids,
            force_strong=force_strong,
        )

    def enqueue_block(
        self,
        block: AudioBlock,
        audio_path: str | Path,
        *,
        priority: int = 0,
        summary_sensitive_segment_ids: Iterable[str] = (),
        force_strong: bool = False,
        range_start: int | float | None = None,
        range_end: int | float | None = None,
    ) -> object:
        if self.task_queue is None or not hasattr(self.task_queue, "enqueue"):
            raise PipelineError("enqueue_block requires a task queue")
        sensitive_ids = sorted({str(value) for value in summary_sensitive_segment_ids})
        block_config_digest = make_block_config_digest(
            block,
            config_digest=self.config_digest,
            summary_sensitive_segment_ids=sensitive_ids,
            force_strong=force_strong,
        )
        payload = {
            "audio_path": str(Path(audio_path).expanduser().resolve()),
            "block": block.to_dict(),
            "pipeline_config_digest": self.config_digest,
            "summary_sensitive_segment_ids": sensitive_ids,
            "force_strong": bool(force_strong),
        }
        return _supported_call(
            getattr(self.task_queue, "enqueue"),
            kind=self.config.task_kind,
            source_id=block.source_id,
            source_sha256=block.source_sha256,
            range_start=block.core_start if range_start is None else range_start,
            range_end=block.core_end if range_end is None else range_end,
            model_digest=self.model_digest,
            config_digest=block_config_digest,
            payload=payload,
            priority=priority,
            max_attempts=self.config.max_attempts,
        )

    def _stored_segments(self, block_id: str) -> list[AsrSegment]:
        if self.storage is None:
            return []
        list_segments = getattr(self.storage, "list_segments", None)
        if list_segments is None:
            return []
        values = _supported_call(
            list_segments, block_id=block_id, latest_only=False
        )
        return [item for item in values if isinstance(item, AsrSegment)]

    @staticmethod
    def _completion_manifest(result: PipelineResult) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "complete": True,
            "task_key": result.task_key,
            "block_id": result.block_id,
            "source_id": result.source_id,
            "final_revisions": [
                [item.segment_id, item.revision] for item in result.final_segments
            ],
            "final_count": len(result.final_segments),
        }

    @staticmethod
    def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
        return json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _write_completion_marker(self, result: PipelineResult) -> None:
        if self.storage is None:
            return
        add_artifact = getattr(self.storage, "add_artifact", None)
        if add_artifact is None:
            return
        manifest = self._completion_manifest(result)
        encoded = self._manifest_bytes(manifest)
        digest = hashlib.sha256(encoded).hexdigest()
        storage_path = getattr(self.storage, "path", None)
        metadata_only = storage_path is None or str(storage_path) == ":memory:"
        if metadata_only:
            marker_path = f"dayaudio://task/{result.task_key}/{digest}.json"
        else:
            database_path = Path(storage_path).expanduser().resolve()
            marker_root = database_path.parent / "work" / "pipeline_manifests"
            # Lowercase hex remains one-to-one on case-insensitive filesystems
            # and avoids repeating the long task key in the filename.
            marker_path_value = marker_root / f"{digest}.json"
            from dayaudio.cas import atomic_write_bytes

            atomic_write_bytes(marker_path_value, encoded)
            marker_path = str(marker_path_value)
            delete_artifacts = getattr(self.storage, "delete_artifacts", None)
            if delete_artifacts is not None:
                removed = list(
                    _supported_call(
                        delete_artifacts,
                        kind="asr-block-manifest",
                        task_key=result.task_key,
                    )
                    or ()
                )
                remaining_records = []
                list_artifacts = getattr(self.storage, "list_artifacts", None)
                if list_artifacts is not None:
                    remaining_records = list(_supported_call(list_artifacts) or ())
                referenced_paths = [
                    Path(value)
                    for item in remaining_records
                    if (value := str(self._artifact_value(item, "path") or ""))
                ]
                safe_root = filesystem_path(marker_root, force_extended=True).resolve()
                for old in removed:
                    old_path_text = str(self._artifact_value(old, "path") or "")
                    if not old_path_text:
                        continue
                    old_path = Path(old_path_text)
                    if _same_filesystem_entry(old_path, marker_path_value):
                        continue
                    try:
                        safe_path = filesystem_path(old_path, force_extended=True).resolve()
                        safe = safe_path.is_relative_to(safe_root)
                    except (OSError, ValueError):
                        safe = False
                    if safe and not any(
                        _same_filesystem_entry(old_path, reference)
                        for reference in referenced_paths
                    ):
                        try:
                            safe_path.unlink()
                        except FileNotFoundError:
                            pass
        artifact_metadata = dict(manifest)
        artifact_metadata["metadata_only"] = metadata_only
        _supported_call(
            add_artifact,
            kind="asr-block-manifest",
            sha256=digest,
            path=marker_path,
            size_bytes=len(encoded),
            source_id=result.source_id,
            task_key=result.task_key,
            metadata=artifact_metadata,
        )

    @staticmethod
    def _artifact_value(artifact: object, name: str) -> Any:
        if isinstance(artifact, Mapping):
            return artifact.get(name)
        return getattr(artifact, name, None)

    def _cached_final(
        self, block: AudioBlock, task_key: str
    ) -> tuple[AsrSegment, ...] | None:
        if self.storage is None:
            return None
        list_artifacts = getattr(self.storage, "list_artifacts", None)
        if list_artifacts is None:
            return None
        artifacts = list(
            _supported_call(
                list_artifacts,
                source_id=block.source_id,
                kind="asr-block-manifest",
                task_key=task_key,
            )
            or ()
        )
        if not artifacts:
            return None
        values = [
            item
            for item in self._stored_segments(block.block_id)
            if item.metadata.get("pipeline_task_key") == task_key
            and item.metadata.get("pipeline_stage") == "final"
        ]
        by_revision = {(item.segment_id, item.revision): item for item in values}
        for artifact in reversed(artifacts):
            manifest = self._artifact_value(artifact, "metadata")
            if not isinstance(manifest, Mapping):
                continue
            if (
                manifest.get("manifest_version") != 1
                or manifest.get("complete") is not True
                or manifest.get("task_key") != task_key
                or manifest.get("block_id") != block.block_id
                or manifest.get("source_id") != block.source_id
            ):
                continue
            canonical_manifest = {
                key: value for key, value in manifest.items() if key != "metadata_only"
            }
            encoded = self._manifest_bytes(canonical_manifest)
            expected_digest = hashlib.sha256(encoded).hexdigest()
            if self._artifact_value(artifact, "sha256") != expected_digest:
                continue
            marker_path = str(self._artifact_value(artifact, "path") or "")
            metadata_only = manifest.get("metadata_only") is True
            if metadata_only:
                storage_path = getattr(self.storage, "path", None)
                if storage_path is not None and str(storage_path) != ":memory:":
                    continue
            else:
                path = filesystem_path(marker_path)
                try:
                    marker_bytes = path.read_bytes()
                except OSError:
                    continue
                if (
                    len(marker_bytes) != self._artifact_value(artifact, "size_bytes")
                    or hashlib.sha256(marker_bytes).hexdigest() != expected_digest
                    or marker_bytes != encoded
                ):
                    continue
            expected_value = manifest.get("final_revisions")
            if not isinstance(expected_value, list):
                continue
            try:
                expected = [(str(item[0]), int(item[1])) for item in expected_value]
            except (TypeError, ValueError, IndexError):
                continue
            if manifest.get("final_count") != len(expected):
                continue
            cached: list[AsrSegment] = []
            valid = True
            for segment_id, revision in expected:
                item = by_revision.get((segment_id, revision))
                if item is None:
                    valid = False
                    break
                cached.append(item)
            if valid and len(cached) == len(expected):
                return tuple(
                    sorted(cached, key=lambda item: (item.start, item.end, item.segment_id))
                )
        return None

    def _persist_backend_raw(
        self,
        backend: AsrBackend,
        *,
        stage: str,
        source_id: str,
        task_key: str,
    ) -> str | None:
        consume = getattr(backend, "consume_raw_output", None)
        if consume is None:
            return None
        raw = consume()
        if raw is None:
            return None
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise PipelineError("backend consume_raw_output must return bytes or None")
        if self.storage is None:
            return None
        add_artifact = getattr(self.storage, "add_artifact", None)
        storage_path = getattr(self.storage, "path", None)
        if (
            add_artifact is None
            or storage_path is None
            or str(storage_path) == ":memory:"
        ):
            # Private transcript bytes are never copied into metadata-only stores.
            return None
        from dayaudio.cas import ContentAddressedStore

        database_path = Path(storage_path).expanduser().resolve()
        obj = ContentAddressedStore(database_path.parent / "cas").put_bytes(bytes(raw))
        record = _supported_call(
            add_artifact,
            kind=f"asr-raw-{stage}",
            sha256=obj.sha256,
            path=str(obj.path),
            size_bytes=obj.size_bytes,
            source_id=source_id,
            task_key=task_key,
            metadata={
                "stage": stage,
                "backend": getattr(backend, "name", type(backend).__name__),
                "model_id": getattr(backend, "model_id", None),
                "model_revision": getattr(backend, "model_revision", None),
                "contains_transcript_text": True,
            },
        )
        return str(self._artifact_value(record, "artifact_id") or "") or None

    def _persist(
        self, segment: AsrSegment, *, raw_artifact_id: str | None = None
    ) -> AsrSegment:
        if self.storage is None:
            return segment
        add = getattr(self.storage, "add_segment", None) or getattr(
            self.storage, "save_segment", None
        )
        if add is None:
            return segment
        get = getattr(self.storage, "get_segment", None)
        candidate = segment
        if get is not None:
            existing = _supported_call(get, candidate.segment_id, revision=candidate.revision)
            if existing is not None and existing != candidate:
                latest = _supported_call(get, candidate.segment_id)
                next_revision = max(
                    candidate.revision,
                    int(getattr(latest, "revision", candidate.revision)) + 1,
                )
                metadata = dict(candidate.metadata)
                metadata["retry_revision"] = True
                candidate = replace(candidate, revision=next_revision, metadata=metadata)
        stored = _supported_call(add, candidate, raw_artifact_id=raw_artifact_id)
        return stored if isinstance(stored, AsrSegment) else candidate

    def process_block(
        self,
        block: AudioBlock,
        audio_path: str | Path,
        *,
        task_key: str | None = None,
        summary_sensitive_segment_ids: Iterable[str] = (),
        force_strong: bool = False,
        heartbeat: Callable[[], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> PipelineResult:
        path = Path(audio_path)
        if not filesystem_path(path).is_file():
            raise FileNotFoundError(path)
        sensitive = {str(value) for value in summary_sensitive_segment_ids}
        key = task_key or self.task_key(
            block,
            summary_sensitive_segment_ids=sensitive,
            force_strong=force_strong,
        )
        cached = self._cached_final(block, key)
        if cached is not None:
            return PipelineResult(
                task_key=key,
                block_id=block.block_id,
                source_id=block.source_id,
                fast_segments=(),
                strong_segments=(),
                consensus_segments=(),
                final_segments=cached,
                decisions=(),
                resumed=True,
            )

        if cancelled is not None and cancelled():
            raise PipelineCancelled("task cancellation requested")
        try:
            fast_backend_output = _run_with_lease_guard(
                lambda: self.fast_backend.transcribe(
                    path,
                    source_id=block.source_id,
                    block_id=block.block_id,
                    offset_seconds=block.context_start,
                ),
                heartbeat=heartbeat,
                cancelled=cancelled,
                lease_seconds=self.config.lease_seconds,
            )
        except (PipelineCancelled, LeaseGuardError):
            raise
        except Exception:
            try:
                self._persist_backend_raw(
                    self.fast_backend,
                    stage="fast-failed",
                    source_id=block.source_id,
                    task_key=key,
                )
            except Exception:
                pass
            raise
        _assert_active(heartbeat, cancelled)
        fast_raw_artifact_id = self._persist_backend_raw(
            self.fast_backend,
            stage="fast",
            source_id=block.source_id,
            task_key=key,
        )
        fast_raw = _core_owned_segments(fast_backend_output, block)
        fast_segments: list[AsrSegment] = []
        for index, item in enumerate(fast_raw):
            if index % 32 == 0:
                _assert_active(heartbeat, cancelled)
            raw = _metadata_stage(
                item,
                stage="fast-normalized",
                task_key=key,
                base_segment_id=item.segment_id,
                logical_stage="fast",
                is_fast=True,
                raw_artifact_id=fast_raw_artifact_id,
            )
            persisted = self._persist(raw, raw_artifact_id=fast_raw_artifact_id)
            fast_segments.append(
                annotate_anomalies(persisted, self.router.policy.anomaly)
            )
        if heartbeat is not None:
            heartbeat()
        if cancelled is not None and cancelled():
            raise PipelineCancelled("task cancellation requested")

        from dayaudio.evidence import text_is_summary_sensitive

        escalation = {
            item.segment_id: self.router.should_escalate(
                item,
                summary_sensitive=(
                    item.segment_id in sensitive
                    or text_is_summary_sensitive(item.text)
                ),
                force=force_strong,
            )
            for item in fast_segments
        }
        needs_strong = any(value.escalate for value in escalation.values())
        strong_segments: list[AsrSegment] = []
        consensus_segments: list[AsrSegment] = []
        strong_raw_artifact_id: str | None = None
        strong_attempted = False
        strong_unavailable_reason: str | None = None
        if needs_strong and self.strong_backend is None:
            strong_unavailable_reason = "strong:not_configured"
        elif needs_strong and not self.config.enable_strong:
            strong_unavailable_reason = "strong:disabled"
        if (
            needs_strong
            and self.config.enable_strong
            and self.strong_backend is not None
        ):
            strong_attempted = True
            try:
                strong_backend_output = _run_with_lease_guard(
                    lambda: self.strong_backend.transcribe(
                        path,
                        source_id=block.source_id,
                        block_id=block.block_id,
                        offset_seconds=block.context_start,
                    ),
                    heartbeat=heartbeat,
                    cancelled=cancelled,
                    lease_seconds=self.config.lease_seconds,
                )
                _assert_active(heartbeat, cancelled)
                strong_raw_artifact_id = self._persist_backend_raw(
                    self.strong_backend,
                    stage="strong",
                    source_id=block.source_id,
                    task_key=key,
                )
            except (PipelineCancelled, LeaseGuardError):
                raise
            except Exception as exc:
                # Fast output remains usable evidence, but never silently high
                # confidence when an explicitly requested refinement failed.
                try:
                    strong_raw_artifact_id = self._persist_backend_raw(
                        self.strong_backend,
                        stage="strong-failed",
                        source_id=block.source_id,
                        task_key=key,
                    )
                except (PipelineCancelled, LeaseGuardError):
                    raise
                except Exception:
                    strong_raw_artifact_id = None
                strong_backend_output = []
                strong_unavailable_reason = (
                    f"strong:backend_error_{type(exc).__name__}"
                )
            strong_raw = _core_owned_segments(strong_backend_output, block)
            for index, item in enumerate(strong_raw):
                if index % 32 == 0:
                    _assert_active(heartbeat, cancelled)
                base, overlaps = _candidate_base(item, fast_segments)
                if not overlaps:
                    metadata = dict(item.metadata)
                    metadata["candidate_match"] = "nearest_without_overlap"
                    item = replace(
                        item,
                        anomaly_flags=tuple(
                            dict.fromkeys((*item.anomaly_flags, "timeline_mismatch"))
                        ),
                        metadata=metadata,
                    )
                strong_segments.append(
                    self._persist(
                        _metadata_stage(
                            item,
                            stage="strong-normalized",
                            task_key=key,
                            base_segment_id=(base.segment_id if base else item.segment_id),
                            logical_stage="strong",
                            is_fast=False,
                            raw_artifact_id=strong_raw_artifact_id,
                        ),
                        raw_artifact_id=strong_raw_artifact_id,
                    )
                )
            if heartbeat is not None:
                heartbeat()
            if (
                strong_unavailable_reason is None
                and self.config.enable_consensus
                and self.consensus_backend is not None
                and any(value.anomalies.severe for value in escalation.values())
            ):
                try:
                    consensus_backend_output = _run_with_lease_guard(
                        lambda: self.consensus_backend.transcribe(
                            path,
                            source_id=block.source_id,
                            block_id=block.block_id,
                            offset_seconds=block.context_start,
                        ),
                        heartbeat=heartbeat,
                        cancelled=cancelled,
                        lease_seconds=self.config.lease_seconds,
                    )
                    _assert_active(heartbeat, cancelled)
                    consensus_raw_artifact_id = self._persist_backend_raw(
                        self.consensus_backend,
                        stage="consensus",
                        source_id=block.source_id,
                        task_key=key,
                    )
                except (PipelineCancelled, LeaseGuardError):
                    raise
                except Exception:
                    try:
                        self._persist_backend_raw(
                            self.consensus_backend,
                            stage="consensus-failed",
                            source_id=block.source_id,
                            task_key=key,
                        )
                    except (PipelineCancelled, LeaseGuardError):
                        raise
                    except Exception:
                        pass
                    consensus_backend_output = []
                    consensus_raw_artifact_id = None
                consensus_raw = _core_owned_segments(consensus_backend_output, block)
                for index, item in enumerate(consensus_raw):
                    if index % 32 == 0:
                        _assert_active(heartbeat, cancelled)
                    base, overlaps = _candidate_base(item, fast_segments)
                    if not overlaps:
                        metadata = dict(item.metadata)
                        metadata["candidate_match"] = "nearest_without_overlap"
                        item = replace(
                            item,
                            anomaly_flags=tuple(
                                dict.fromkeys((*item.anomaly_flags, "timeline_mismatch"))
                            ),
                            metadata=metadata,
                        )
                    consensus_segments.append(
                        self._persist(
                            _metadata_stage(
                                item,
                                stage="consensus-normalized",
                                task_key=key,
                                base_segment_id=(
                                    base.segment_id if base else item.segment_id
                                ),
                                logical_stage="consensus",
                                is_fast=False,
                                raw_artifact_id=consensus_raw_artifact_id,
                            ),
                            raw_artifact_id=consensus_raw_artifact_id,
                        )
                    )
        if cancelled is not None and cancelled():
            raise PipelineCancelled("task cancellation requested")

        decisions: list[CascadeDecision] = []
        final_segments: list[AsrSegment] = []
        strong_matches = _one_to_one_temporal_matches(
            fast_segments, strong_segments
        )
        consensus_matches = _one_to_one_temporal_matches(
            fast_segments, consensus_segments
        )
        strong_cardinality_matches = len(strong_segments) == len(fast_segments)
        consensus_cardinality_matches = (
            not consensus_segments or len(consensus_segments) == len(fast_segments)
        )
        for index, fast in enumerate(fast_segments):
            if index % 32 == 0:
                _assert_active(heartbeat, cancelled)
            decision: CascadeDecision | None = None
            if (
                escalation[fast.segment_id].escalate
                and strong_unavailable_reason is not None
            ):
                decision = _review_decision(
                    fast,
                    None,
                    reason=strong_unavailable_reason,
                    router=self.router,
                )
                decisions.append(decision)
            elif escalation[fast.segment_id].escalate and strong_attempted:
                strong = strong_matches.get(fast.segment_id)
                consensus = consensus_matches.get(fast.segment_id)
                if not strong_cardinality_matches:
                    decision = _review_decision(
                        fast,
                        strong,
                        reason="strong:segmentation_cardinality_mismatch",
                        router=self.router,
                    )
                elif strong is None:
                    decision = _review_decision(
                        fast,
                        None,
                        reason="strong:no_one_to_one_temporal_match",
                        router=self.router,
                    )
                elif (
                    escalation[fast.segment_id].anomalies.severe
                    and consensus_segments
                    and not consensus_cardinality_matches
                ):
                    decision = _review_decision(
                        fast,
                        strong,
                        reason="consensus:segmentation_cardinality_mismatch",
                        router=self.router,
                    )
                else:
                    decision = self.router.evaluate(fast, strong, consensus=consensus)
                decisions.append(decision)
            selected = decision.selected if decision is not None else fast
            selected_raw_artifact_id = (
                strong_raw_artifact_id
                if decision is not None and decision.accepted_strong
                else fast_raw_artifact_id
            )
            latest_revision = max(fast.revision + 1, selected.revision)
            final = _metadata_stage(
                selected,
                stage="final",
                task_key=key,
                revision=latest_revision,
                decision=decision,
                base_segment_id=fast.segment_id,
                logical_stage="final",
                is_fast=False,
                raw_artifact_id=selected_raw_artifact_id,
            )
            owned_final = _core_owned_segment(final, block)
            if owned_final is not None:
                final_segments.append(
                    self._persist(
                        owned_final,
                        raw_artifact_id=selected_raw_artifact_id,
                    )
                )

        result = PipelineResult(
            task_key=key,
            block_id=block.block_id,
            source_id=block.source_id,
            fast_segments=tuple(fast_segments),
            strong_segments=tuple(strong_segments),
            consensus_segments=tuple(consensus_segments),
            final_segments=tuple(final_segments),
            decisions=tuple(decisions),
        )
        # This marker is the commit record.  It is written only after every
        # referenced final revision is durable, so partial finals are retried.
        _assert_active(heartbeat, cancelled)
        self._write_completion_marker(result)
        return result

    @staticmethod
    def _block_from_payload(payload: Mapping[str, Any]) -> AudioBlock:
        value = payload.get("block")
        if not isinstance(value, Mapping):
            raise PipelineError("task payload is missing an audio block")
        fields = {
            name: value[name]
            for name in (
                "block_id",
                "source_id",
                "source_sha256",
                "core_start",
                "core_end",
                "context_start",
                "context_end",
            )
        }
        fields["pcm_sha256"] = value.get("pcm_sha256")
        return AudioBlock(**fields)

    def process_next(self) -> PipelineResult | None:
        if self.task_queue is None:
            raise PipelineError("process_next requires a task queue")
        claim = getattr(self.task_queue, "claim", None)
        if claim is None:
            raise PipelineError("task queue does not provide claim")
        task = _supported_call(
            claim,
            self.worker_id,
            lease_seconds=self.config.lease_seconds,
            kinds=(self.config.task_kind,),
            model_digest=self.model_digest,
            payload_filters={"pipeline_config_digest": self.config_digest},
        )
        if task is None:
            return None
        task_id = str(getattr(task, "task_id", getattr(task, "task_key", "")))
        task_key = str(getattr(task, "task_key", task_id))
        lease_token = str(getattr(task, "lease_token", ""))
        payload = getattr(task, "payload", None)

        def is_cancelled() -> bool:
            probe = getattr(self.task_queue, "is_cancel_requested", None)
            return bool(_supported_call(probe, task_id)) if probe is not None else False

        def heartbeat() -> None:
            beat = getattr(self.task_queue, "heartbeat", None)
            if beat is not None:
                _supported_call(
                    beat,
                    task_id,
                    self.worker_id,
                    lease_token,
                    lease_seconds=self.config.lease_seconds,
                )

        try:
            if not isinstance(payload, Mapping):
                raise PipelineError("claimed task has no mapping payload")
            block = self._block_from_payload(payload)
            sensitive_ids = tuple(
                sorted(
                    {
                        str(value)
                        for value in payload.get(
                            "summary_sensitive_segment_ids", ()
                        )
                    }
                )
            )
            force_strong = bool(payload.get("force_strong", False))
            expected_block_config = make_block_config_digest(
                block,
                config_digest=self.config_digest,
                summary_sensitive_segment_ids=sensitive_ids,
                force_strong=force_strong,
            )
            mismatches: list[str] = []
            claimed_model = getattr(task, "model_digest", None)
            if claimed_model is not None and str(claimed_model) != self.model_digest:
                mismatches.append("model_digest")
            claimed_config = getattr(task, "config_digest", None)
            if claimed_config is not None and str(claimed_config) != expected_block_config:
                mismatches.append("config_digest")
            payload_config = payload.get("pipeline_config_digest")
            if payload_config is not None and str(payload_config) != self.config_digest:
                mismatches.append("pipeline_config_digest")
            expected_key = self.task_key(
                block,
                summary_sensitive_segment_ids=sensitive_ids,
                force_strong=force_strong,
            )
            if task_key != expected_key:
                mismatches.append("task_key")
            claimed_source = getattr(task, "source_id", None)
            if claimed_source is not None and str(claimed_source) != block.source_id:
                mismatches.append("source_id")
            claimed_source_sha = getattr(task, "source_sha256", None)
            if (
                claimed_source_sha is not None
                and str(claimed_source_sha).lower() != block.source_sha256.lower()
            ):
                mismatches.append("source_sha256")
            if mismatches:
                raise PipelineError(
                    "claimed task does not belong to this pipeline: "
                    + ",".join(mismatches)
                )
            result = self.process_block(
                block,
                str(payload.get("audio_path") or ""),
                task_key=task_key,
                summary_sensitive_segment_ids=sensitive_ids,
                force_strong=force_strong,
                heartbeat=heartbeat,
                cancelled=is_cancelled,
            )
            complete = getattr(self.task_queue, "complete", None)
            if complete is None:
                raise PipelineError("task queue does not provide complete")
            _supported_call(
                complete,
                task_id,
                self.worker_id,
                lease_token,
                result_artifact_id=None,
            )
            return result
        except Exception as exc:
            fail = getattr(self.task_queue, "fail", None)
            if fail is not None:
                # Error text intentionally excludes source paths and transcripts.
                safe_error = f"{type(exc).__name__}: pipeline task failed"
                try:
                    _supported_call(
                        fail,
                        task_id,
                        self.worker_id,
                        lease_token,
                        error=safe_error,
                        retry=not isinstance(exc, PipelineCancelled),
                        retry_delay_seconds=self.config.retry_delay_seconds,
                    )
                except Exception:
                    # A completion/cancellation race may already have released
                    # the lease.  Preserve the original processing exception.
                    pass
            raise

    def run(self, *, max_tasks: int | None = None) -> list[PipelineResult]:
        if max_tasks is not None and max_tasks < 0:
            raise ValueError("max_tasks must not be negative")
        results: list[PipelineResult] = []
        while max_tasks is None or len(results) < max_tasks:
            result = self.process_next()
            if result is None:
                break
            results.append(result)
        return results

    def close(self) -> None:
        unique: list[object] = []
        seen: set[int] = set()
        for item in (
            self.fast_backend,
            self.strong_backend,
            self.consensus_backend,
        ):
            if item is not None and id(item) not in seen:
                unique.append(item)
                seen.add(id(item))
        close_all(unique)

    def __enter__(self) -> "ResumablePipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


Pipeline = ResumablePipeline


__all__ = [
    "PIPELINE_SCHEMA_VERSION",
    "Pipeline",
    "PipelineCancelled",
    "PipelineConfig",
    "PipelineError",
    "PipelineResult",
    "ResumablePipeline",
    "make_block_config_digest",
    "make_resume_key",
]
