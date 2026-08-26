"""Durable, lease-based task queue built on :mod:`dayaudio.storage`."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .cas import digest_json
from .storage import Storage
from .types import TaskStatus


class TaskError(RuntimeError):
    pass


class TaskLeaseError(TaskError):
    pass


class TaskCancelledError(TaskError):
    pass


def _canonical_range_value(value: int | float | str | Decimal) -> str:
    if isinstance(value, bool):
        raise TypeError("PCM range values must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid PCM range value: {value!r}") from error
    if not number.is_finite():
        raise ValueError("PCM range values must be finite")
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def make_task_key(
    *,
    source_sha256: str,
    range_start: int | float | str | Decimal,
    range_end: int | float | str | Decimal,
    model_digest: str,
    config_digest: str,
    kind: str = "asr",
) -> str:
    """Build the stable identity required for resumable computation.

    ``range_start`` and ``range_end`` may be PCM sample indexes or exact
    source-relative units selected by the caller.  Their canonical decimal
    encoding prevents ``1``, ``1.0`` and ``Decimal('1.00')`` from producing
    different tasks.
    """

    source_digest = source_sha256.lower()
    if len(source_digest) != 64 or any(c not in "0123456789abcdef" for c in source_digest):
        raise ValueError("source_sha256 must be 64 hexadecimal characters")
    start = _canonical_range_value(range_start)
    end = _canonical_range_value(range_end)
    if Decimal(end) <= Decimal(start):
        raise ValueError("task range must satisfy range_start < range_end")
    if not model_digest.strip():
        raise ValueError("model_digest must not be empty")
    if not config_digest.strip():
        raise ValueError("config_digest must not be empty")
    if not kind.strip():
        raise ValueError("kind must not be empty")
    identity = {
        "config_digest": config_digest,
        "kind": kind,
        "model_digest": model_digest,
        "range_end": end,
        "range_start": start,
        "source_sha256": source_digest,
    }
    return f"task-{digest_json(identity)}"


def task_key(
    source_sha256: str,
    range_start: int | float | str | Decimal,
    range_end: int | float | str | Decimal,
    model_digest: str,
    config_digest: str,
    kind: str = "asr",
) -> str:
    """Positional compatibility wrapper around :func:`make_task_key`."""

    return make_task_key(
        source_sha256=source_sha256,
        range_start=range_start,
        range_end=range_end,
        model_digest=model_digest,
        config_digest=config_digest,
        kind=kind,
    )


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    task_key: str
    kind: str
    source_id: str | None
    source_sha256: str
    range_start: str
    range_end: str
    model_digest: str
    config_digest: str
    payload: dict[str, Any]
    status: TaskStatus
    priority: int
    attempts: int
    max_attempts: int
    available_at: str
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: str | None
    cancel_requested: bool
    error: str | None
    result_artifact_id: str | None
    created_at: str
    updated_at: str
    completed_at: str | None

    @property
    def pcm_range(self) -> tuple[str, str]:
        return self.range_start, self.range_end


def _task_from_row(row: sqlite3.Row) -> TaskRecord:
    payload = json.loads(row["payload_json"] or "{}")
    if not isinstance(payload, dict):
        raise TaskError("stored task payload is not an object")
    return TaskRecord(
        task_id=row["task_id"],
        task_key=row["task_key"],
        kind=row["kind"],
        source_id=row["source_id"],
        source_sha256=row["source_sha256"],
        range_start=row["range_start"],
        range_end=row["range_end"],
        model_digest=row["model_digest"],
        config_digest=row["config_digest"],
        payload=payload,
        status=TaskStatus(row["status"]),
        priority=row["priority"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        available_at=row["available_at"],
        lease_owner=row["lease_owner"],
        lease_token=row["lease_token"],
        lease_expires_at=row["lease_expires_at"],
        cancel_requested=bool(row["cancel_requested"]),
        error=row["error"],
        result_artifact_id=row["result_artifact_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _coerce_now(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current.astimezone(timezone.utc)


def _time_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TaskQueue:
    """Multi-process-safe task queue with expiring, token-guarded leases."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def enqueue(
        self,
        *,
        kind: str,
        source_id: str | None,
        source_sha256: str,
        range_start: int | float | str | Decimal,
        range_end: int | float | str | Decimal,
        model_digest: str,
        config_digest: str,
        payload: Mapping[str, Any] | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        available_at: datetime | None = None,
    ) -> TaskRecord:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        start = _canonical_range_value(range_start)
        end = _canonical_range_value(range_end)
        key = make_task_key(
            source_sha256=source_sha256,
            range_start=start,
            range_end=end,
            model_digest=model_digest,
            config_digest=config_digest,
            kind=kind,
        )
        identifier = key
        now = _coerce_now()
        available = _coerce_now(available_at) if available_at is not None else now
        payload_text = json.dumps(
            dict(payload or {}), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        with self.storage.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM tasks WHERE task_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                return _task_from_row(existing)
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, task_key, kind, source_id, source_sha256,
                    range_start, range_end, model_digest, config_digest,
                    payload_json, status, priority, attempts, max_attempts,
                    available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    key,
                    kind,
                    source_id,
                    source_sha256.lower(),
                    start,
                    end,
                    model_digest,
                    config_digest,
                    payload_text,
                    int(priority),
                    int(max_attempts),
                    _time_text(available),
                    _time_text(now),
                    _time_text(now),
                ),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_key = ?", (key,)
            ).fetchone()
            assert row is not None
            return _task_from_row(row)

    @staticmethod
    def _recover_stale_in_connection(
        connection: sqlite3.Connection, *, now: datetime
    ) -> int:
        now_text = _time_text(now)
        rows = connection.execute(
            """
            SELECT * FROM tasks
            WHERE status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
            """,
            (now_text,),
        ).fetchall()
        for row in rows:
            if row["cancel_requested"]:
                status = TaskStatus.CANCELLED.value
                completed_at = now_text
                error = row["error"]
            elif row["attempts"] >= row["max_attempts"]:
                status = TaskStatus.FAILED.value
                completed_at = now_text
                error = row["error"] or "lease expired after final attempt"
            else:
                status = TaskStatus.PENDING.value
                completed_at = None
                error = row["error"]
            connection.execute(
                """
                UPDATE tasks SET status = ?, available_at = ?, lease_owner = NULL,
                    lease_token = NULL, lease_expires_at = NULL, error = ?,
                    completed_at = ?, updated_at = ? WHERE task_id = ?
                """,
                (status, now_text, error, completed_at, now_text, row["task_id"]),
            )
        return len(rows)

    def recover_stale(self, *, now: datetime | None = None) -> int:
        current = _coerce_now(now)
        with self.storage.transaction(immediate=True) as connection:
            return self._recover_stale_in_connection(connection, now=current)

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        kinds: Sequence[str] | None = None,
        model_digest: str | None = None,
        payload_filters: Mapping[str, str] | None = None,
        now: datetime | None = None,
    ) -> TaskRecord | None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive and finite")
        current = _coerce_now(now)
        now_text = _time_text(current)
        lease_expires = _time_text(current + timedelta(seconds=lease_seconds))
        token = uuid.uuid4().hex
        kind_clause = ""
        parameters: list[Any] = [now_text]
        if kinds is not None:
            normalized_kinds = tuple(dict.fromkeys(kinds))
            if not normalized_kinds:
                return None
            placeholders = ",".join("?" for _ in normalized_kinds)
            kind_clause = f" AND kind IN ({placeholders})"
            parameters.extend(normalized_kinds)
        identity_clause = ""
        if model_digest is not None:
            if not model_digest:
                raise ValueError("model_digest filter must not be empty")
            identity_clause += " AND model_digest = ?"
            parameters.append(model_digest)
        for key, value in sorted((payload_filters or {}).items()):
            if not key or not key.replace("_", "").isalnum():
                raise ValueError("payload filter keys must be alphanumeric identifiers")
            identity_clause += f" AND json_extract(payload_json, '$.{key}') = ?"
            parameters.append(str(value))

        with self.storage.transaction(immediate=True) as connection:
            self._recover_stale_in_connection(connection, now=current)
            row = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'pending' AND cancel_requested = 0
                    AND attempts < max_attempts AND available_at <= ?
                """
                + kind_clause
                + identity_clause
                + " ORDER BY priority DESC, created_at, task_id LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE tasks SET status = 'running', attempts = attempts + 1,
                    lease_owner = ?, lease_token = ?, lease_expires_at = ?,
                    updated_at = ?, error = NULL
                WHERE task_id = ? AND status = 'pending' AND cancel_requested = 0
                """,
                (worker_id, token, lease_expires, now_text, row["task_id"]),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
            ).fetchone()
            assert claimed is not None
            return _task_from_row(claimed)

    @staticmethod
    def _require_active_lease(
        connection: sqlite3.Connection,
        task_id_or_key: str,
        worker_id: str,
        lease_token: str,
        now: datetime,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? OR task_key = ? LIMIT 1",
            (task_id_or_key, task_id_or_key),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown task: {task_id_or_key}")
        if row["status"] != TaskStatus.RUNNING.value:
            raise TaskLeaseError(f"task is not running: {task_id_or_key}")
        if row["lease_owner"] != worker_id or row["lease_token"] != lease_token:
            raise TaskLeaseError("task lease owner or token does not match")
        expires = row["lease_expires_at"]
        if expires is None or expires <= _time_text(now):
            raise TaskLeaseError("task lease has expired")
        return row

    def heartbeat(
        self,
        task_id_or_key: str,
        worker_id: str,
        lease_token: str,
        *,
        lease_seconds: float = 300.0,
        now: datetime | None = None,
    ) -> TaskRecord:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive and finite")
        current = _coerce_now(now)
        expires = _time_text(current + timedelta(seconds=lease_seconds))
        with self.storage.transaction(immediate=True) as connection:
            row = self._require_active_lease(
                connection, task_id_or_key, worker_id, lease_token, current
            )
            if row["cancel_requested"]:
                raise TaskCancelledError(f"task cancellation requested: {row['task_id']}")
            connection.execute(
                "UPDATE tasks SET lease_expires_at = ?, updated_at = ? WHERE task_id = ?",
                (expires, _time_text(current), row["task_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
            ).fetchone()
            assert updated is not None
            return _task_from_row(updated)

    def complete(
        self,
        task_id_or_key: str,
        worker_id: str,
        lease_token: str,
        *,
        result_artifact_id: str | None = None,
        now: datetime | None = None,
    ) -> TaskRecord:
        current = _coerce_now(now)
        now_text = _time_text(current)
        with self.storage.transaction(immediate=True) as connection:
            row = self._require_active_lease(
                connection, task_id_or_key, worker_id, lease_token, current
            )
            if row["cancel_requested"]:
                connection.execute(
                    """
                    UPDATE tasks SET status = 'cancelled', lease_owner = NULL,
                        lease_token = NULL, lease_expires_at = NULL,
                        completed_at = ?, updated_at = ? WHERE task_id = ?
                    """,
                    (now_text, now_text, row["task_id"]),
                )
            else:
                connection.execute(
                    """
                    UPDATE tasks SET status = 'complete', result_artifact_id = ?,
                        lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                        completed_at = ?, updated_at = ? WHERE task_id = ?
                    """,
                    (result_artifact_id, now_text, now_text, row["task_id"]),
                )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
            ).fetchone()
            assert updated is not None
            return _task_from_row(updated)

    def fail(
        self,
        task_id_or_key: str,
        worker_id: str,
        lease_token: str,
        error: str,
        *,
        retry: bool = True,
        retry_delay_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> TaskRecord:
        if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        current = _coerce_now(now)
        now_text = _time_text(current)
        with self.storage.transaction(immediate=True) as connection:
            row = self._require_active_lease(
                connection, task_id_or_key, worker_id, lease_token, current
            )
            cancelled = bool(row["cancel_requested"])
            should_retry = retry and not cancelled and row["attempts"] < row["max_attempts"]
            if cancelled:
                status = TaskStatus.CANCELLED.value
            elif should_retry:
                status = TaskStatus.PENDING.value
            else:
                status = TaskStatus.FAILED.value
            available = _time_text(current + timedelta(seconds=retry_delay_seconds))
            completed = None if should_retry else now_text
            connection.execute(
                """
                UPDATE tasks SET status = ?, available_at = ?, error = ?,
                    lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                    completed_at = ?, updated_at = ? WHERE task_id = ?
                """,
                (status, available, error[:10000], completed, now_text, row["task_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
            ).fetchone()
            assert updated is not None
            return _task_from_row(updated)

    def cancel(self, task_id_or_key: str, *, now: datetime | None = None) -> TaskRecord:
        current = _coerce_now(now)
        now_text = _time_text(current)
        with self.storage.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ? OR task_key = ? LIMIT 1",
                (task_id_or_key, task_id_or_key),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown task: {task_id_or_key}")
            if row["status"] == TaskStatus.PENDING.value:
                connection.execute(
                    """
                    UPDATE tasks SET status = 'cancelled', cancel_requested = 1,
                        completed_at = ?, updated_at = ? WHERE task_id = ?
                    """,
                    (now_text, now_text, row["task_id"]),
                )
            elif row["status"] == TaskStatus.RUNNING.value:
                connection.execute(
                    "UPDATE tasks SET cancel_requested = 1, updated_at = ? WHERE task_id = ?",
                    (now_text, row["task_id"]),
                )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
            ).fetchone()
            assert updated is not None
            return _task_from_row(updated)

    def retry(
        self,
        task_id_or_key: str,
        *,
        reset_attempts: bool = True,
        now: datetime | None = None,
    ) -> TaskRecord:
        """Explicitly return a failed/cancelled task to the pending queue."""

        current = _coerce_now(now)
        now_text = _time_text(current)
        with self.storage.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ? OR task_key = ? LIMIT 1",
                (task_id_or_key, task_id_or_key),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown task: {task_id_or_key}")
            if row["status"] not in {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}:
                raise TaskError("only failed or cancelled tasks can be explicitly retried")
            attempts = 0 if reset_attempts else int(row["attempts"])
            if attempts >= int(row["max_attempts"]):
                raise TaskError("task retry requires reset_attempts after reaching max_attempts")
            connection.execute(
                """
                UPDATE tasks SET status = 'pending', attempts = ?, available_at = ?,
                    lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                    cancel_requested = 0, error = NULL, completed_at = NULL,
                    updated_at = ? WHERE task_id = ?
                """,
                (attempts, now_text, now_text, row["task_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
            ).fetchone()
            assert updated is not None
            return _task_from_row(updated)

    def is_cancel_requested(self, task_id_or_key: str) -> bool:
        task = self.get(task_id_or_key)
        if task is None:
            raise KeyError(f"unknown task: {task_id_or_key}")
        return task.cancel_requested or task.status is TaskStatus.CANCELLED

    def get(self, task_id_or_key: str) -> TaskRecord | None:
        row = self.storage.fetch_task_row(task_id_or_key)
        return None if row is None else _task_from_row(row)

    def require(self, task_id_or_key: str) -> TaskRecord:
        task = self.get(task_id_or_key)
        if task is None:
            raise KeyError(f"unknown task: {task_id_or_key}")
        return task

    def list(
        self, *, status: TaskStatus | str | None = None, kind: str | None = None
    ) -> list[TaskRecord]:
        status_text = status.value if isinstance(status, TaskStatus) else status
        return [
            _task_from_row(row)
            for row in self.storage.fetch_task_rows(status=status_text, kind=kind)
        ]

    def counts(self) -> dict[str, int]:
        result = {status.value: 0 for status in TaskStatus}
        with self.storage.connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        for row in rows:
            result[row["status"]] = row["count"]
        return result


__all__ = [
    "TaskCancelledError",
    "TaskError",
    "TaskLeaseError",
    "TaskQueue",
    "TaskRecord",
    "make_task_key",
    "task_key",
]
