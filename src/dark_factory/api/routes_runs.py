"""Per-run file tree + file preview endpoints.

Powers the "Run Detail" popup window — given a ``run_id`` the pipeline
wrote its artifacts under ``<pipeline.output_dir>/<run_id>/``, so these
endpoints let the UI browse that tree and preview individual files.

Safety rails:
- ``run_id`` must match ``RUN_ID_RE``: alphanumeric + dash/underscore.
- The resolved target file must be ``is_relative_to`` the run directory
  (prevents ``..`` traversal through symlinks or otherwise).
- Files larger than :data:`MAX_FILE_BYTES` are refused (returns 413).
- Files that don't decode as UTF-8 or that look binary (null-byte in
  the first 1 KiB) are refused (returns 415) — the UI is a text viewer,
  not an image/pdf/zip viewer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

log = structlog.get_logger()

router = APIRouter()

# Matches the pattern used by orchestrator.py (``run-YYYYMMDD-HHMMSS-xxxx``)
# but also accepts anything alphanumeric/dash/underscore for flexibility.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

# Cap the preview at 1 MiB so a runaway output doesn't OOM the browser.
MAX_FILE_BYTES = 1 * 1024 * 1024

# Entries with more than this many children are truncated in the tree
# response — protects against pathological output dirs from crashing
# the SPA with a 50k-node JSON blob.
MAX_ENTRIES_PER_DIR = 2000


def _run_dir(request: Request, run_id: str) -> Path:
    """Resolve the output directory for ``run_id`` with safety checks."""
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    settings = request.app.state.settings
    base = Path(settings.pipeline.output_dir).resolve()
    run_dir = (base / run_id).resolve()

    # Defence in depth: even though run_id is validated above, confirm
    # the resolved path is still inside base before touching it.
    if not run_dir.is_relative_to(base):
        raise HTTPException(status_code=400, detail="run_id escapes output dir")

    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"No output directory for run {run_id!r}",
        )
    return run_dir


def _build_tree(root: Path) -> dict[str, Any]:
    """Walk ``root`` and return a nested dict describing the file tree.

    Each directory node: ``{name, type: 'dir', path, children: [...]}``.
    Each file node:      ``{name, type: 'file', path, size}``.
    ``path`` is relative to ``root`` so the frontend can pass it straight
    back to :func:`get_run_file`.
    """

    def walk(dir_path: Path, rel_prefix: str) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        try:
            raw = sorted(
                dir_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError as exc:
            return {
                "name": dir_path.name,
                "type": "dir",
                "path": rel_prefix,
                "children": [],
                "error": f"permission denied: {exc}",
            }

        truncated = False
        if len(raw) > MAX_ENTRIES_PER_DIR:
            raw = raw[:MAX_ENTRIES_PER_DIR]
            truncated = True

        for entry in raw:
            # Skip symlinks that point outside the run dir — ``is_relative_to``
            # on the resolved target is the definitive check.
            try:
                resolved = entry.resolve()
                if not resolved.is_relative_to(root):
                    continue
            except (OSError, RuntimeError):
                continue

            rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            if entry.is_dir():
                entries.append(walk(entry, rel))
            elif entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                entries.append(
                    {
                        "name": entry.name,
                        "type": "file",
                        "path": rel,
                        "size": size,
                    }
                )
            # Skip sockets, devices, broken symlinks, etc.

        node: dict[str, Any] = {
            "name": dir_path.name if rel_prefix else "",
            "type": "dir",
            "path": rel_prefix,
            "children": entries,
        }
        if truncated:
            node["truncated"] = True
        return node

    return walk(root, "")


@router.get("/runs/{run_id}/files")
def get_run_files(request: Request, run_id: str):
    """Return the file tree under the run's output directory."""
    run_dir = _run_dir(request, run_id)
    try:
        tree = _build_tree(run_dir)
    except Exception as exc:
        log.warning("run_files_walk_failed", run_id=run_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read output dir: {exc}",
        ) from exc

    # Count files at the top level for the UI's summary line.
    def _count(node: dict[str, Any]) -> tuple[int, int]:
        files = 0
        total_bytes = 0
        for child in node.get("children", []):
            if child.get("type") == "file":
                files += 1
                total_bytes += int(child.get("size") or 0)
            else:
                f, b = _count(child)
                files += f
                total_bytes += b
        return files, total_bytes

    file_count, total_bytes = _count(tree)
    return {
        "run_id": run_id,
        "root": str(run_dir),
        "tree": tree,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


@router.get("/runs/{run_id}/file")
def get_run_file(
    request: Request,
    run_id: str,
    path: str = Query(..., description="Path relative to run directory"),
):
    """Return the text content of a single file inside the run directory."""
    run_dir = _run_dir(request, run_id)

    # Normalise the relative path — strip leading slashes so ``/spec.json``
    # and ``spec.json`` both work, reject absolute paths and '..' segments.
    rel = path.strip().lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")
    if ".." in Path(rel).parts:
        raise HTTPException(status_code=400, detail="path traversal rejected")

    target = (run_dir / rel).resolve()

    # Symlink defence: the resolved target must still be inside run_dir.
    if not target.is_relative_to(run_dir):
        raise HTTPException(
            status_code=400, detail="resolved path escapes run directory"
        )

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {rel}")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"stat failed: {exc}"
        ) from exc

    if size > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file too large ({size} bytes > {MAX_FILE_BYTES} max). "
                "Download via direct filesystem access instead."
            ),
        )

    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"read failed: {exc}"
        ) from exc

    # Reject obvious binary blobs — null byte in the sniff window is a
    # reliable signal. UTF-8 decode catches the rest.
    sniff = raw[:1024]
    if b"\x00" in sniff:
        raise HTTPException(
            status_code=415,
            detail="binary file: preview refused",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=415,
            detail=f"non-UTF8 file: {exc}",
        ) from exc

    return {
        "run_id": run_id,
        "path": rel,
        "size": size,
        "content": text,
    }
