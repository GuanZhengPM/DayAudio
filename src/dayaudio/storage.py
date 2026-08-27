"""SQLite-backed durable metadata for DayAudio.

The database stores identities and provenance; large immutable payloads belong
in :mod:`dayaudio.cas`.  Every public write is transactional and idempotent
where a deterministic identifier is supplied.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .cas import canonical_json_bytes, digest_json
from .paths import filesystem_path
from .types import AsrSegment, SourceRecord

SCHEMA_VERSION = 2


class StorageError(RuntimeError):
    pass


class StorageConflictError(StorageError):
    """An immutable identifier was reused for different content."""


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    kind: str
    sha256: str
    path: str
    size_bytes: int
    source_id: str | None = None
    task_key: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_text(value: Mapping[str, Any] | Sequence[Any] | None) -> str:
    return canonical_json_bytes({} if value is None else value).decode("utf-8")


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise StorageError("stored JSON metadata is not an object")
    return decoded


def _validate_sha256(value: str, field: str = "sha256") -> str:
    digest = value.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be 64 hexadecimal characters")
    return digest


def _validate_finite(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


_MIGRATION_1 = (
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        source_sha256 TEXT NOT NULL UNIQUE,
        source_path TEXT NOT NULL,
        source_name TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
        duration_seconds REAL,
        decoded_duration_seconds REAL,
        codec TEXT,
        sample_rate INTEGER,
        channels INTEGER,
        recording_start TEXT,
        recording_time_basis TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
        source_id TEXT REFERENCES sources(source_id),
        task_key TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        task_key TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        source_id TEXT REFERENCES sources(source_id),
        source_sha256 TEXT NOT NULL,
        range_start TEXT NOT NULL,
        range_end TEXT NOT NULL,
        model_digest TEXT NOT NULL,
        config_digest TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL CHECK(status IN ('pending','running','complete','failed','cancelled')),
        priority INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
        max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts > 0),
        available_at TEXT NOT NULL,
        lease_owner TEXT,
        lease_token TEXT,
        lease_expires_at TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
        error TEXT,
        result_artifact_id TEXT REFERENCES artifacts(artifact_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS segments (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        segment_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision > 0),
        source_id TEXT NOT NULL REFERENCES sources(source_id),
        block_id TEXT,
        start REAL NOT NULL,
        end REAL NOT NULL CHECK(end > start),
        text TEXT NOT NULL CHECK(length(trim(text)) > 0),
        model_id TEXT,
        model_revision TEXT,
        confidence REAL,
        language TEXT,
        anomaly_flags_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        raw_artifact_id TEXT REFERENCES artifacts(artifact_id),
        created_at TEXT NOT NULL,
        UNIQUE(segment_id, revision)
    )
    """,
)

_MIGRATION_2 = (
    """
    CREATE TABLE IF NOT EXISTS source_locations (
        source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
        source_path TEXT NOT NULL,
        source_name TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        PRIMARY KEY(source_id, source_path)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_source_kind ON artifacts(source_id, kind, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_key, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(status, available_at, priority DESC, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_lease ON tasks(status, lease_expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_segments_source_time ON segments(source_id, start, end)",
    "CREATE INDEX IF NOT EXISTS idx_segments_block ON segments(block_id, start)",
    """
    INSERT OR IGNORE INTO source_locations(source_id, source_path, source_name, observed_at)
    SELECT source_id, source_path, source_name, created_at FROM sources
    """,
)

_MIGRATIONS: dict[int, tuple[str, ...]] = {1: _MIGRATION_1, 2: _MIGRATION_2}


class Storage:
    """Open a SQLite store and migrate it to the current schema."""

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 30.0) -> None:
        self.timeout = float(timeout)
        self._lock = threading.RLock()
        self._anchor: sqlite3.Connection | None = None
        if str(path) == ":memory:":
            self.path = Path(":memory:")
            self._database = f"file:dayaudio-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._anchor = self._new_connection()
        else:
            self.path = Path(path).expanduser().resolve()
            filesystem_database = filesystem_path(self.path)
            filesystem_database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._database = str(filesystem_database)
            self._uri = False
        self._migrate()
        self._secure_files()

    def _secure_files(self) -> None:
        if self._uri or os.name == "nt":
            return
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database,
            timeout=self.timeout,
            isolation_level=None,
            uri=self._uri,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout * 1000))}")
        if not self._uri:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            self._secure_files()
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._new_connection()
        try:
            yield connection
        finally:
            connection.close()
            self._secure_files()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a rollback-safe transaction."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _migrate(self) -> None:
        with self._lock, self.connection() as connection:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise StorageError(
                    f"database schema {current} is newer than supported schema {SCHEMA_VERSION}"
                )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            for version in range(current + 1, SCHEMA_VERSION + 1):
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _MIGRATIONS[version]:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT OR REPLACE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, utc_now()),
                    )
                    connection.execute(f"PRAGMA user_version = {version}")
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    @property
    def schema_version(self) -> int:
        with self.connection() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def journal_mode(self) -> str:
        with self.connection() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def close(self) -> None:
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None
        self._secure_files()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- Sources ---------------------------------------------------------

    @staticmethod
    def _source_from_row(row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            source_id=row["source_id"],
            source_sha256=row["source_sha256"],
            source_path=row["source_path"],
            source_name=row["source_name"],
            size_bytes=row["size_bytes"],
            duration_seconds=row["duration_seconds"],
            decoded_duration_seconds=row["decoded_duration_seconds"],
            codec=row["codec"],
            sample_rate=row["sample_rate"],
            channels=row["channels"],
            recording_start=row["recording_start"],
            recording_time_basis=row["recording_time_basis"],
        )

    def upsert_source(self, record: SourceRecord) -> SourceRecord:
        digest = _validate_sha256(record.source_sha256, "source_sha256")
        if record.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        duration = _validate_finite(record.duration_seconds, "duration_seconds")
        decoded_duration = _validate_finite(
            record.decoded_duration_seconds, "decoded_duration_seconds"
        )
        if duration is not None and duration < 0:
            raise ValueError("duration_seconds must not be negative")
        if decoded_duration is not None and decoded_duration < 0:
            raise ValueError("decoded_duration_seconds must not be negative")
        if record.sample_rate is not None and record.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if record.channels is not None and record.channels <= 0:
            raise ValueError("channels must be positive")

        now = utc_now()
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM sources WHERE source_sha256 = ?", (digest,)
            ).fetchone()
            if existing is not None:
                if existing["size_bytes"] != record.size_bytes:
                    raise StorageConflictError("identical SHA-256 has a different stored size")
                connection.execute(
                    "INSERT OR IGNORE INTO source_locations"
                    "(source_id, source_path, source_name, observed_at) VALUES (?, ?, ?, ?)",
                    (existing["source_id"], record.source_path, record.source_name, now),
                )
                return self._source_from_row(existing)

            id_collision = connection.execute(
                "SELECT source_sha256 FROM sources WHERE source_id = ?", (record.source_id,)
            ).fetchone()
            if id_collision is not None:
                raise StorageConflictError(
                    f"source_id {record.source_id!r} already names different content"
                )

            connection.execute(
                """
                INSERT INTO sources(
                    source_id, source_sha256, source_path, source_name, size_bytes,
                    duration_seconds, decoded_duration_seconds, codec, sample_rate,
                    channels, recording_start, recording_time_basis, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.source_id,
                    digest,
                    record.source_path,
                    record.source_name,
                    record.size_bytes,
                    duration,
                    decoded_duration,
                    record.codec,
                    record.sample_rate,
                    record.channels,
                    record.recording_start,
                    record.recording_time_basis,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO source_locations"
                "(source_id, source_path, source_name, observed_at) VALUES (?, ?, ?, ?)",
                (record.source_id, record.source_path, record.source_name, now),
            )
        return record

    add_source = upsert_source

    def add_source_location(
        self, source_id: str, source_path: str, source_name: str | None = None
    ) -> None:
        name = source_name or Path(source_path).name
        with self.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO source_locations"
                "(source_id, source_path, source_name, observed_at) VALUES (?, ?, ?, ?)",
                (source_id, source_path, name, utc_now()),
            )

    def source_locations(self, source_id: str) -> tuple[str, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT source_path FROM source_locations WHERE source_id = ? ORDER BY observed_at, source_path",
                (source_id,),
            ).fetchall()
        return tuple(row["source_path"] for row in rows)

    def get_source(self, source_id: str) -> SourceRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        return None if row is None else self._source_from_row(row)

    def require_source(self, source_id: str) -> SourceRecord:
        record = self.get_source(source_id)
        if record is None:
            raise KeyError(f"unknown source_id: {source_id}")
        return record

    def find_source_by_sha256(self, source_sha256: str) -> SourceRecord | None:
        digest = _validate_sha256(source_sha256, "source_sha256")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE source_sha256 = ?", (digest,)
            ).fetchone()
        return None if row is None else self._source_from_row(row)

    get_source_by_sha256 = find_source_by_sha256

    def find_source_by_path(self, source_path: str | os.PathLike[str]) -> SourceRecord | None:
        path_text = str(source_path)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT s.* FROM sources s
                JOIN source_locations l ON l.source_id = s.source_id
                WHERE l.source_path = ? ORDER BY l.observed_at DESC LIMIT 1
                """,
                (path_text,),
            ).fetchone()
        return None if row is None else self._source_from_row(row)

    def list_sources(self) -> list[SourceRecord]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM sources ORDER BY created_at, source_id").fetchall()
        return [self._source_from_row(row) for row in rows]

    def set_decoded_duration(self, source_id: str, duration_seconds: float) -> None:
        duration = _validate_finite(duration_seconds, "duration_seconds")
        if duration is None or duration < 0:
            raise ValueError("duration_seconds must not be negative")
        with self.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE sources SET decoded_duration_seconds = ?, updated_at = ? WHERE source_id = ?",
                (duration, utc_now(), source_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown source_id: {source_id}")

    # -- Artifacts -------------------------------------------------------

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            kind=row["kind"],
            sha256=row["sha256"],
            path=row["path"],
            size_bytes=row["size_bytes"],
            source_id=row["source_id"],
            task_key=row["task_key"],
            metadata=_json_object(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def add_artifact(
        self,
        *,
        kind: str,
        sha256: str,
        path: str | os.PathLike[str],
        size_bytes: int,
        source_id: str | None = None,
        task_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRecord:
        digest = _validate_sha256(sha256)
        if not kind.strip():
            raise ValueError("artifact kind must not be empty")
        if size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        identity = {
            "kind": kind,
            "sha256": digest,
            "source_id": source_id,
            "task_key": task_key,
        }
        resolved_id = artifact_id or f"artifact-{digest_json(identity)[:32]}"
        record = ArtifactRecord(
            artifact_id=resolved_id,
            kind=kind,
            sha256=digest,
            path=str(path),
            size_bytes=size_bytes,
            source_id=source_id,
            task_key=task_key,
            metadata=dict(metadata or {}),
            created_at=utc_now(),
        )
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (resolved_id,)
            ).fetchone()
            if existing is not None:
                existing_record = self._artifact_from_row(existing)
                immutable_existing = (
                    existing_record.kind,
                    existing_record.sha256,
                    existing_record.size_bytes,
                    existing_record.source_id,
                    existing_record.task_key,
                )
                immutable_new = (kind, digest, size_bytes, source_id, task_key)
                if immutable_existing != immutable_new:
                    raise StorageConflictError(
                        f"artifact_id {resolved_id!r} already names different content"
                    )
                return existing_record
            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, kind, sha256, path, size_bytes, source_id,
                    task_key, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.artifact_id,
                    record.kind,
                    record.sha256,
                    record.path,
                    record.size_bytes,
                    record.source_id,
                    record.task_key,
                    _json_text(record.metadata),
                    record.created_at,
                ),
            )
        return record

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        return None if row is None else self._artifact_from_row(row)

    def list_artifacts(
        self, *, source_id: str | None = None, kind: str | None = None, task_key: str | None = None
    ) -> list[ArtifactRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (("source_id", source_id), ("kind", kind), ("task_key", task_key)):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM artifacts{where} ORDER BY created_at, artifact_id", parameters
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def delete_artifacts(self, *, kind: str, task_key: str) -> list[ArtifactRecord]:
        """Delete metadata for one exact derived-artifact scope.

        Callers remain responsible for deleting returned local files after
        checking whether another artifact record references the same path.
        """

        if not kind or not task_key:
            raise ValueError("artifact deletion requires exact kind and task_key")
        with self.transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE kind = ? AND task_key = ?",
                (kind, task_key),
            ).fetchall()
            connection.execute(
                "DELETE FROM artifacts WHERE kind = ? AND task_key = ?",
                (kind, task_key),
            )
        return [self._artifact_from_row(row) for row in rows]

    # -- ASR segments ----------------------------------------------------

    @staticmethod
    def _segment_from_row(row: sqlite3.Row) -> AsrSegment:
        flags = json.loads(row["anomaly_flags_json"] or "[]")
        return AsrSegment(
            segment_id=row["segment_id"],
            source_id=row["source_id"],
            start=row["start"],
            end=row["end"],
            text=row["text"],
            revision=row["revision"],
            model_id=row["model_id"],
            model_revision=row["model_revision"],
            confidence=row["confidence"],
            language=row["language"],
            block_id=row["block_id"],
            anomaly_flags=tuple(flags),
            metadata=_json_object(row["metadata_json"]),
        )

    def add_segment(
        self, segment: AsrSegment, *, raw_artifact_id: str | None = None
    ) -> AsrSegment:
        start = _validate_finite(segment.start, "start")
        end = _validate_finite(segment.end, "end")
        confidence = _validate_finite(segment.confidence, "confidence")
        if start is None or end is None or start < 0 or end <= start:
            raise ValueError("segment range must satisfy 0 <= start < end")
        now = utc_now()
        flags_json = canonical_json_bytes(list(segment.anomaly_flags)).decode("utf-8")
        metadata_json = _json_text(segment.metadata)
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM segments WHERE segment_id = ? AND revision = ?",
                (segment.segment_id, segment.revision),
            ).fetchone()
            if existing is not None:
                existing_segment = self._segment_from_row(existing)
                same_content = (
                    existing["source_id"] == segment.source_id
                    and existing["block_id"] == segment.block_id
                    and existing["start"] == start
                    and existing["end"] == end
                    and existing["text"] == segment.text
                    and existing["model_id"] == segment.model_id
                    and existing["model_revision"] == segment.model_revision
                    and existing["confidence"] == confidence
                    and existing["language"] == segment.language
                    and existing["anomaly_flags_json"] == flags_json
                    and existing["metadata_json"] == metadata_json
                    and existing["raw_artifact_id"] == raw_artifact_id
                )
                if not same_content:
                    raise StorageConflictError(
                        f"segment {segment.segment_id!r} revision {segment.revision} conflicts"
                    )
                return existing_segment
            connection.execute(
                """
                INSERT INTO segments(
                    segment_id, revision, source_id, block_id, start, end, text,
                    model_id, model_revision, confidence, language,
                    anomaly_flags_json, metadata_json, raw_artifact_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.segment_id,
                    segment.revision,
                    segment.source_id,
                    segment.block_id,
                    start,
                    end,
                    segment.text,
                    segment.model_id,
                    segment.model_revision,
                    confidence,
                    segment.language,
                    flags_json,
                    metadata_json,
                    raw_artifact_id,
                    now,
                ),
            )
        return segment

    def get_segment(self, segment_id: str, revision: int | None = None) -> AsrSegment | None:
        with self.connection() as connection:
            if revision is None:
                row = connection.execute(
                    "SELECT * FROM segments WHERE segment_id = ? ORDER BY revision DESC LIMIT 1",
                    (segment_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM segments WHERE segment_id = ? AND revision = ?",
                    (segment_id, revision),
                ).fetchone()
        return None if row is None else self._segment_from_row(row)

    def list_segments(
        self,
        *,
        source_id: str | None = None,
        block_id: str | None = None,
        latest_only: bool = True,
    ) -> list[AsrSegment]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if source_id is not None:
            clauses.append("s.source_id = ?")
            parameters.append(source_id)
        if block_id is not None:
            clauses.append("s.block_id = ?")
            parameters.append(block_id)
        if latest_only:
            clauses.append(
                "s.revision = (SELECT MAX(s2.revision) FROM segments s2 WHERE s2.segment_id = s.segment_id)"
            )
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT s.* FROM segments s{where} ORDER BY s.source_id, s.start, s.end, s.segment_id, s.revision",
                parameters,
            ).fetchall()
        return [self._segment_from_row(row) for row in rows]

    # -- Low-level task reads used by TaskQueue and diagnostics ----------

    def fetch_task_row(self, task_id_or_key: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute(
                "SELECT * FROM tasks WHERE task_id = ? OR task_key = ? LIMIT 1",
                (task_id_or_key, task_id_or_key),
            ).fetchone()

    def fetch_task_rows(
        self, *, status: str | None = None, kind: str | None = None
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        parameters: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as connection:
            return connection.execute(
                f"SELECT * FROM tasks{where} ORDER BY priority DESC, created_at, task_id",
                parameters,
            ).fetchall()

    def task_queue(self) -> "Any":
        from .tasks import TaskQueue

        return TaskQueue(self)


Database = Storage


__all__ = [
    "ArtifactRecord",
    "Database",
    "SCHEMA_VERSION",
    "Storage",
    "StorageConflictError",
    "StorageError",
    "utc_now",
]
