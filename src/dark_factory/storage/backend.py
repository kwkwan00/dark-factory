"""Abstract storage backend for pipeline input/output files.

Two implementations:

- :class:`LocalStorage` — reads and writes to the local filesystem
  (the default, matching the project's current behavior).
- :class:`S3Storage` — reads and writes to an AWS S3 bucket.  Requires
  ``boto3`` (optional dependency).

Both backends expose the same low-level API so the rest of the codebase
doesn't need to know which one is active.  The higher-level
:class:`RunStorage` wraps a backend with per-run key conventions::

    s3://dark-factory-bucket/
      run-20260412-051047-9677/
        input/          ← uploaded requirements
        output/         ← generated code, reports, artifacts
      run-20260412-052423-e77f/
        input/
        output/

The :func:`get_storage` factory returns a singleton backend; callers
create :class:`RunStorage` instances as needed for each run.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import structlog

log = structlog.get_logger()


# ── Abstract interface ────────────────────────────────────────────────────────


class StorageBackend(ABC):
    """Unified interface for file storage operations.

    Keys are logical paths (e.g. ``run-123/output/main.py``).  The
    backend translates them to the appropriate physical location
    (filesystem path or S3 object key).
    """

    # ── Single file operations ────────────────────────────────────────

    @abstractmethod
    def read_text(self, key: str) -> str:
        """Read a file and return its text content.

        :raises FileNotFoundError: if the key does not exist.
        """

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read a file and return raw bytes."""

    @abstractmethod
    def write_text(self, key: str, content: str) -> None:
        """Write text content to a file, creating parent dirs/prefixes."""

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Write raw bytes to a file."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if the key exists."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a single file.  No-op if it doesn't exist."""

    # ── Prefix / directory operations ─────────────────────────────────

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """List all keys under *prefix*, returned as full logical paths."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Recursively delete everything under *prefix*."""

    # ── Bulk transfer (local ↔ storage) ───────────────────────────────

    @abstractmethod
    def download_to_local(self, key: str, local_path: Path) -> None:
        """Download a single file to a local path."""

    @abstractmethod
    def download_prefix_to_local(self, prefix: str, local_dir: Path) -> None:
        """Download all files under *prefix* into *local_dir*,
        preserving the relative directory structure."""

    @abstractmethod
    def upload_from_local(self, local_path: Path, key: str) -> None:
        """Upload a single local file to storage."""

    @abstractmethod
    def sync_local_to_storage(self, local_dir: Path, prefix: str) -> int:
        """Upload every file under *local_dir* to *prefix*, preserving
        relative paths.  Returns the number of files synced."""

    # ── Presigned URLs (S3 only, no-op for local) ─────────────────────

    def presign_url(self, key: str, expires: int = 3600) -> str | None:
        """Return a presigned download URL, or None if not applicable."""
        return None

    # ── Iteration helper ──────────────────────────────────────────────

    @abstractmethod
    def walk(self, prefix: str) -> Iterator[tuple[str, int]]:
        """Yield ``(relative_key, size_bytes)`` for every file under
        *prefix*.  Analogous to ``os.walk`` but flat."""


# ── Per-run storage wrapper ───────────────────────────────────────────────────


class RunStorage:
    """Convenience wrapper that scopes a :class:`StorageBackend` to a
    single pipeline run with four sub-folders::

        {run_id}/
          input/          ← uploaded raw files
          requirements/   ← parsed Requirement models (post-ingest)
          specs/          ← generated Spec models (post-spec-stage)
          output/         ← generated code, reports, artifacts

    Usage::

        store = RunStorage(get_storage(), "run-20260412-051047-9677")
        store.write_input("meeting.docx", data)
        store.write_requirement("req-abc123.json", json_str)
        store.write_spec("auth-login.json", spec_json)
        store.write_output("main.py", code)
        store.sync_output_from_local(scratch_dir)
        store.download_input_to_local(local_dir)
    """

    def __init__(self, backend: StorageBackend, run_id: str) -> None:
        self.backend = backend
        self.run_id = run_id

    def _input_key(self, path: str) -> str:
        return f"{self.run_id}/input/{path}"

    def _requirements_key(self, path: str) -> str:
        return f"{self.run_id}/requirements/{path}"

    def _specs_key(self, path: str) -> str:
        return f"{self.run_id}/specs/{path}"

    def _output_key(self, path: str) -> str:
        return f"{self.run_id}/output/{path}"

    @property
    def input_prefix(self) -> str:
        return f"{self.run_id}/input"

    @property
    def requirements_prefix(self) -> str:
        return f"{self.run_id}/requirements"

    @property
    def specs_prefix(self) -> str:
        return f"{self.run_id}/specs"

    @property
    def output_prefix(self) -> str:
        return f"{self.run_id}/output"

    @property
    def run_prefix(self) -> str:
        return self.run_id

    # ── Input operations ──────────────────────────────────────────────

    def write_input(self, path: str, content: str) -> None:
        """Write a text file to the run's input area."""
        self.backend.write_text(self._input_key(path), content)

    def write_input_bytes(self, path: str, data: bytes) -> None:
        """Write binary data to the run's input area."""
        self.backend.write_bytes(self._input_key(path), data)

    def upload_input_from_local(self, local_path: Path, dest_name: str) -> None:
        """Upload a local file to the run's input area."""
        self.backend.upload_from_local(local_path, self._input_key(dest_name))

    def sync_input_from_local(self, local_dir: Path) -> int:
        """Sync a local directory into the run's input area."""
        return self.backend.sync_local_to_storage(local_dir, self.input_prefix)

    def download_input_to_local(self, local_dir: Path) -> None:
        """Download all input files to a local directory for processing."""
        self.backend.download_prefix_to_local(self.input_prefix, local_dir)

    def list_input(self) -> list[str]:
        """List all keys in the run's input area."""
        return self.backend.list_keys(self.input_prefix)

    # ── Requirements operations ───────────────────────────────────────

    def write_requirement(self, path: str, content: str) -> None:
        """Write a parsed requirement to the run's requirements area."""
        self.backend.write_text(self._requirements_key(path), content)

    def read_requirement(self, path: str) -> str:
        """Read a requirement file from the run's requirements area."""
        return self.backend.read_text(self._requirements_key(path))

    def list_requirements(self) -> list[str]:
        """List all keys in the run's requirements area."""
        return self.backend.list_keys(self.requirements_prefix)

    def sync_requirements_from_local(self, local_dir: Path) -> int:
        """Sync a local directory into the run's requirements area."""
        return self.backend.sync_local_to_storage(local_dir, self.requirements_prefix)

    def download_requirements_to_local(self, local_dir: Path) -> None:
        """Download all requirements to a local directory."""
        self.backend.download_prefix_to_local(self.requirements_prefix, local_dir)

    def walk_requirements(self) -> Iterator[tuple[str, int]]:
        """Yield ``(relative_path, size_bytes)`` for requirement files."""
        return self.backend.walk(self.requirements_prefix)

    # ── Specs operations ──────────────────────────────────────────────

    def write_spec(self, path: str, content: str) -> None:
        """Write a generated spec to the run's specs area."""
        self.backend.write_text(self._specs_key(path), content)

    def read_spec(self, path: str) -> str:
        """Read a spec file from the run's specs area."""
        return self.backend.read_text(self._specs_key(path))

    def list_specs(self) -> list[str]:
        """List all keys in the run's specs area."""
        return self.backend.list_keys(self.specs_prefix)

    def sync_specs_from_local(self, local_dir: Path) -> int:
        """Sync a local directory into the run's specs area."""
        return self.backend.sync_local_to_storage(local_dir, self.specs_prefix)

    def download_specs_to_local(self, local_dir: Path) -> None:
        """Download all specs to a local directory."""
        self.backend.download_prefix_to_local(self.specs_prefix, local_dir)

    def walk_specs(self) -> Iterator[tuple[str, int]]:
        """Yield ``(relative_path, size_bytes)`` for spec files."""
        return self.backend.walk(self.specs_prefix)

    # ── Output operations ─────────────────────────────────────────────

    def write_output(self, path: str, content: str) -> None:
        """Write a text file to the run's output area."""
        self.backend.write_text(self._output_key(path), content)

    def write_output_bytes(self, path: str, data: bytes) -> None:
        """Write binary data to the run's output area."""
        self.backend.write_bytes(self._output_key(path), data)

    def read_output(self, path: str) -> str:
        """Read a text file from the run's output area."""
        return self.backend.read_text(self._output_key(path))

    def read_output_bytes(self, path: str) -> bytes:
        """Read binary data from the run's output area."""
        return self.backend.read_bytes(self._output_key(path))

    def output_exists(self, path: str) -> bool:
        """Check if a file exists in the run's output area."""
        return self.backend.exists(self._output_key(path))

    def sync_output_from_local(self, local_dir: Path) -> int:
        """Sync a local scratch directory into the run's output area."""
        return self.backend.sync_local_to_storage(local_dir, self.output_prefix)

    def download_output_to_local(self, local_dir: Path) -> None:
        """Download all output files to a local directory."""
        self.backend.download_prefix_to_local(self.output_prefix, local_dir)

    def list_output(self) -> list[str]:
        """List all keys in the run's output area."""
        return self.backend.list_keys(self.output_prefix)

    def walk_output(self) -> Iterator[tuple[str, int]]:
        """Yield ``(relative_path, size_bytes)`` for output files."""
        return self.backend.walk(self.output_prefix)

    def presign_output(self, path: str, expires: int = 3600) -> str | None:
        """Return a presigned URL for an output file."""
        return self.backend.presign_url(self._output_key(path), expires)

    # ── Run-level operations ──────────────────────────────────────────

    def delete_run(self) -> None:
        """Delete all data for this run (input, requirements, specs, output)."""
        self.backend.delete_prefix(self.run_prefix)


# ── Local filesystem implementation ───────────────────────────────────────────


class LocalStorage(StorageBackend):
    """Store files on the local filesystem under a root directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _abs(self, key: str) -> Path:
        return self.root / key

    # ── Single file ───────────────────────────────────────────────────

    def read_text(self, key: str) -> str:
        p = self._abs(key)
        if not p.is_file():
            raise FileNotFoundError(f"Key not found: {key}")
        return p.read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, key: str) -> bytes:
        p = self._abs(key)
        if not p.is_file():
            raise FileNotFoundError(f"Key not found: {key}")
        return p.read_bytes()

    def write_text(self, key: str, content: str) -> None:
        p = self._abs(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._abs(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._abs(key).exists()

    def delete(self, key: str) -> None:
        p = self._abs(key)
        if p.is_file():
            p.unlink()

    # ── Prefix ────────────────────────────────────────────────────────

    def list_keys(self, prefix: str) -> list[str]:
        base = self._abs(prefix)
        if not base.is_dir():
            return []
        root_resolved = self.root.resolve()
        return sorted(
            str(f.resolve().relative_to(root_resolved))
            for f in base.rglob("*")
            if f.is_file()
        )

    def delete_prefix(self, prefix: str) -> None:
        base = self._abs(prefix)
        if base.is_dir():
            shutil.rmtree(base, ignore_errors=True)

    # ── Bulk transfer ─────────────────────────────────────────────────

    def download_to_local(self, key: str, local_path: Path) -> None:
        src = self._abs(key)
        if not src.is_file():
            raise FileNotFoundError(f"Key not found: {key}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(local_path))

    def download_prefix_to_local(self, prefix: str, local_dir: Path) -> None:
        base = self._abs(prefix)
        if not base.is_dir():
            return
        # No-op when the target is the same directory (LocalStorage
        # scratch dir IS the storage dir).
        if base.resolve() == local_dir.resolve():
            return
        for src in base.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(base)
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))

    def upload_from_local(self, local_path: Path, key: str) -> None:
        dest = self._abs(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))

    def sync_local_to_storage(self, local_dir: Path, prefix: str) -> int:
        # No-op when the source is the same directory as the target.
        target = self._abs(prefix)
        if target.resolve() == local_dir.resolve():
            return 0
        count = 0
        for src in local_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(local_dir)
            dest = self._abs(f"{prefix}/{rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
            count += 1
        return count

    def walk(self, prefix: str) -> Iterator[tuple[str, int]]:
        base = self._abs(prefix)
        if not base.is_dir():
            return
        for f in sorted(base.rglob("*")):
            if f.is_file():
                try:
                    yield str(f.relative_to(base)), f.stat().st_size
                except (OSError, ValueError):
                    pass


# ── S3 implementation ─────────────────────────────────────────────────────────


class S3Storage(StorageBackend):
    """Store files in a dedicated AWS S3 bucket.

    Keys are stored directly in the bucket — no static prefix.  The
    per-run layout (``{run_id}/input/``, ``{run_id}/output/``) is
    enforced by :class:`RunStorage`, not by this class.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install it with: pip install boto3"
            )

        self.bucket = bucket

        session_kwargs: dict = {"region_name": region}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key

        from botocore.exceptions import ClientError

        session = boto3.Session(**session_kwargs)
        client_kwargs: dict = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        self._client = session.client("s3", **client_kwargs)
        self._botocore_error = ClientError
        log.info(
            "s3_storage_init",
            bucket=bucket,
            region=region,
            endpoint=endpoint_url or "default",
        )

    # ── Single file ───────────────────────────────────────────────────

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8", errors="replace")

    def read_bytes(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Key not found in S3: {key}")
        except self._botocore_error as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise FileNotFoundError(f"Key not found in S3: {key}")
            raise

    def write_text(self, key: str, content: str) -> None:
        self.write_bytes(key, content.encode("utf-8"))

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._botocore_error:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    # ── Prefix ────────────────────────────────────────────────────────

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = prefix if prefix.endswith("/") else prefix + "/"

        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys)

    def delete_prefix(self, prefix: str) -> None:
        full_prefix = prefix if prefix.endswith("/") else prefix + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        to_delete: list[dict] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                to_delete.append({"Key": obj["Key"]})
        if not to_delete:
            return
        # S3 delete_objects accepts up to 1000 keys per call
        for i in range(0, len(to_delete), 1000):
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": to_delete[i : i + 1000], "Quiet": True},
            )

    # ── Bulk transfer ─────────────────────────────────────────────────

    def download_to_local(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(local_path))

    def download_prefix_to_local(self, prefix: str, local_dir: Path) -> None:
        full_prefix = prefix if prefix.endswith("/") else prefix + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(full_prefix) :]
                if not rel:
                    continue
                local_path = local_dir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(self.bucket, obj["Key"], str(local_path))

    def upload_from_local(self, local_path: Path, key: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, key)

    def sync_local_to_storage(self, local_dir: Path, prefix: str) -> int:
        count = 0
        for src in local_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(local_dir)
            key = f"{prefix}/{rel}"
            self._client.upload_file(str(src), self.bucket, key)
            count += 1
        return count

    def presign_url(self, key: str, expires: int = 3600) -> str | None:
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires,
            )
        except Exception as exc:
            log.warning("s3_presign_failed", key=key, error=str(exc))
            return None

    def walk(self, prefix: str) -> Iterator[tuple[str, int]]:
        full_prefix = prefix if prefix.endswith("/") else prefix + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(full_prefix) :]
                if rel:
                    yield rel, obj.get("Size", 0)


# ── Factory ───────────────────────────────────────────────────────────────────

_singleton: StorageBackend | None = None


def get_storage(local_root: Path | None = None) -> StorageBackend:
    """Return the configured storage backend (singleton).

    Reads ``STORAGE_BACKEND`` env var:

    - ``local`` (default) → :class:`LocalStorage` rooted at
      *local_root* (defaults to ``./output``).
    - ``s3`` → :class:`S3Storage` with bucket from ``S3_BUCKET``.

    The *local_root* parameter is only used on first call (when the
    singleton is created).  Subsequent calls return the cached instance.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend == "s3":
        bucket = os.getenv("S3_BUCKET", "")
        if not bucket:
            raise ValueError(
                "S3_BUCKET environment variable is required when "
                "STORAGE_BACKEND=s3"
            )
        _singleton = S3Storage(
            bucket=bucket,
            region=os.getenv("S3_REGION", "us-east-1").strip(),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        )
    else:
        root = local_root or Path("./output")
        root.mkdir(parents=True, exist_ok=True)
        _singleton = LocalStorage(root=root.resolve())

    log.info("storage_backend_initialized", backend=backend)
    return _singleton


def reset_storage() -> None:
    """Reset the singleton — used by tests."""
    global _singleton
    _singleton = None
