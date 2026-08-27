"""Small, dependency-free content-addressed and atomic file utilities.

The CAS deliberately keys objects by the bytes that were written, not by a
filename or a media container's metadata.  Callers keep semantic metadata in
SQLite and use this module only for immutable bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from dayaudio.paths import filesystem_path, filesystem_tree_path

DEFAULT_CHUNK_SIZE = 1024 * 1024
CAS_FILE_MODE = 0o600 if os.name == "nt" else 0o400


def sha256_stream(stream: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return a SHA-256 digest without assuming the stream is seekable."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Hash a regular file using bounded memory."""

    source = Path(path)
    filesystem_source = filesystem_path(source)
    if not filesystem_source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    with filesystem_source.open("rb") as handle:
        return sha256_stream(handle, chunk_size=chunk_size)


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for hashes and task identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; unsupported platforms may reject it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes | bytearray | memoryview,
    *,
    mode: int = 0o600,
) -> Path:
    """Durably replace *path* with *data* using a same-directory rename."""

    target = Path(path)
    # The target itself may still be below MAX_PATH while the generated
    # same-directory temporary filename crosses it.
    filesystem_target = filesystem_path(target, force_extended=True)
    filesystem_target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".atomic-",
            suffix=".tmp",
            dir=filesystem_target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, filesystem_target)
        temporary_name = None
        _fsync_directory(filesystem_target.parent)
        return target
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding), mode=mode)


@dataclass(frozen=True, slots=True)
class CASObject:
    sha256: str
    path: Path
    size_bytes: int


class ContentAddressedStore:
    """Immutable SHA-256 object store with atomic installation."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.object_root = self.root / "sha256"
        filesystem_path(self.object_root).mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(self.root, 0o700)
            os.chmod(self.object_root, 0o700)

    @staticmethod
    def _validate_digest(sha256: str) -> str:
        digest = sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("sha256 must be 64 hexadecimal characters")
        return digest

    def path_for(self, sha256: str) -> Path:
        digest = self._validate_digest(sha256)
        return self.object_root / digest[:2] / digest[2:4] / digest

    def exists(self, sha256: str, *, verify: bool = False) -> bool:
        path = self.path_for(sha256)
        if not filesystem_path(path).is_file():
            return False
        return not verify or sha256_file(path) == sha256.lower()

    def get(self, sha256: str, *, verify: bool = False) -> CASObject:
        digest = self._validate_digest(sha256)
        path = self.path_for(digest)
        filesystem_object = filesystem_path(path)
        if not filesystem_object.is_file():
            raise FileNotFoundError(f"CAS object does not exist: {digest}")
        if verify and sha256_file(path) != digest:
            raise OSError(f"CAS integrity check failed: {digest}")
        return CASObject(digest, path, filesystem_object.stat().st_size)

    def _install_staged_file(
        self,
        temporary_name: str,
        target: Path,
        *,
        digest: str,
        size: int,
    ) -> tuple[CASObject, bool]:
        """Atomically install a completed CAS staging file.

        Windows ``os.replace`` can race with another caller that is verifying
        the same destination and surface ``PermissionError``.  ``os.rename``
        instead elects one writer without replacing an existing immutable
        object.  Losing writers verify and reuse the winner.  POSIX retains the
        original replace behavior.

        The boolean result says whether this method consumed the staging path.
        """

        filesystem_target = filesystem_path(target, force_extended=True)
        filesystem_target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(temporary_name, CAS_FILE_MODE)
        try:
            if os.name == "nt":
                os.rename(temporary_name, filesystem_target)
            else:
                os.replace(temporary_name, filesystem_target)
        except (FileExistsError, PermissionError) as error:
            if os.name != "nt":
                raise
            try:
                existing = self.get(digest, verify=True)
            except FileNotFoundError:
                raise error
            if existing.size_bytes != size:
                raise OSError(f"CAS size mismatch: {digest}")
            return existing, False
        _fsync_directory(filesystem_target.parent)
        return CASObject(digest, target, size), True

    def put_bytes(self, data: bytes | bytearray | memoryview) -> CASObject:
        raw = bytes(data)
        digest = hashlib.sha256(raw).hexdigest()
        target = self.path_for(digest)
        if filesystem_path(target).exists():
            existing = self.get(digest, verify=True)
            if existing.size_bytes != len(raw):  # defensive; SHA match already checked
                raise OSError(f"CAS size mismatch: {digest}")
            return existing

        filesystem_parent = filesystem_path(target.parent, force_extended=True)
        filesystem_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".cas-",
                suffix=".tmp",
                dir=filesystem_parent,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            result, consumed = self._install_staged_file(
                temporary_name, target, digest=digest, size=len(raw)
            )
            if consumed:
                temporary_name = None
            return result
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def put_json(self, value: Any) -> CASObject:
        return self.put_bytes(canonical_json_bytes(value))

    def put_file(
        self,
        source: str | os.PathLike[str],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> CASObject:
        """Copy *source* into the CAS while hashing it in one streaming pass."""

        source_path = Path(source)
        filesystem_source = filesystem_path(source_path)
        if not filesystem_source.is_file():
            raise FileNotFoundError(f"not a regular file: {source_path}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        staging = self.object_root / ".staging"
        filesystem_staging = filesystem_path(staging, force_extended=True)
        filesystem_staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary_name: str | None = None
        digest = hashlib.sha256()
        size = 0
        try:
            with filesystem_source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".cas-",
                suffix=".tmp",
                dir=filesystem_staging,
                delete=False,
            ) as target_handle:
                temporary_name = target_handle.name
                while True:
                    chunk = source_handle.read(chunk_size)
                    if not chunk:
                        break
                    digest.update(chunk)
                    target_handle.write(chunk)
                    size += len(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())

            hexdigest = digest.hexdigest()
            target = self.path_for(hexdigest)
            filesystem_path(target.parent, force_extended=True).mkdir(
                parents=True, exist_ok=True, mode=0o700
            )
            if filesystem_path(target).exists():
                existing = self.get(hexdigest, verify=True)
                if existing.size_bytes != size:
                    raise OSError(f"CAS size mismatch: {hexdigest}")
                return existing

            result, consumed = self._install_staged_file(
                temporary_name, target, digest=hexdigest, size=size
            )
            if consumed:
                temporary_name = None
            return result
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def verify(self, sha256: str) -> bool:
        return self.exists(sha256, verify=True)

    def iter_objects(self) -> Iterable[CASObject]:
        filesystem_root = filesystem_tree_path(self.object_root)
        if not filesystem_root.exists():
            return
        for path in filesystem_root.glob("[0-9a-f][0-9a-f]/[0-9a-f][0-9a-f]/*"):
            if not path.is_file():
                continue
            try:
                digest = self._validate_digest(path.name)
            except ValueError:
                continue
            if path.name != digest:
                continue
            if path.parent.name != digest[2:4] or path.parent.parent.name != digest[:2]:
                continue
            yield CASObject(digest, self.path_for(digest), path.stat().st_size)


CASStore = ContentAddressedStore


__all__ = [
    "CASObject",
    "CASStore",
    "ContentAddressedStore",
    "atomic_write_bytes",
    "atomic_write_text",
    "canonical_json_bytes",
    "digest_json",
    "sha256_file",
    "sha256_stream",
]
