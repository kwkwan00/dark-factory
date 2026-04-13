"""Per-run file tree + file preview endpoints.

Powers the "Output" tab in the Run Detail popup — given a ``run_id``
the pipeline wrote its artifacts under ``{run_id}/output/`` in the
configured storage backend (local filesystem or S3).

Safety rails:
- ``run_id`` must match ``RUN_ID_RE``: alphanumeric + dash/underscore.
- Path traversal (``..``) is rejected at the API level.
- Files larger than :data:`MAX_FILE_BYTES` are refused (returns 413).
- Files that don't decode as UTF-8 or that look binary (null-byte in
  the first 1 KiB) are refused (returns 415).
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

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


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")


def _get_storage(request: Request):
    """Return the storage backend from app state, or create one."""
    storage = getattr(request.app.state, "storage", None)
    if storage is not None:
        return storage
    from dark_factory.storage.backend import get_storage
    return get_storage()


def _build_tree_from_walk(
    entries: list[tuple[str, int]],
) -> dict[str, Any]:
    """Build a nested tree structure from a flat list of (path, size) tuples.

    Works for both local filesystem ``walk()`` output and S3 ``walk()``
    output — both return flat relative paths.
    """
    root: dict[str, Any] = {
        "name": "",
        "type": "dir",
        "path": "",
        "children": [],
    }

    # Index directories by their path for fast lookup
    dir_nodes: dict[str, dict[str, Any]] = {"": root}

    # Sort entries so directories are implicitly created in order
    sorted_entries = sorted(entries, key=lambda e: e[0].lower())

    if len(sorted_entries) > MAX_ENTRIES_PER_DIR * 10:
        sorted_entries = sorted_entries[: MAX_ENTRIES_PER_DIR * 10]
        root["truncated"] = True

    for rel_path, size in sorted_entries:
        parts = rel_path.split("/")
        filename = parts[-1]
        dir_parts = parts[:-1]

        # Ensure all parent directories exist in the tree
        current_path = ""
        parent_node = root
        for part in dir_parts:
            current_path = f"{current_path}/{part}" if current_path else part
            if current_path not in dir_nodes:
                dir_node: dict[str, Any] = {
                    "name": part,
                    "type": "dir",
                    "path": current_path,
                    "children": [],
                }
                dir_nodes[current_path] = dir_node
                if len(parent_node["children"]) < MAX_ENTRIES_PER_DIR:
                    parent_node["children"].append(dir_node)
            parent_node = dir_nodes[current_path]

        # Add the file node
        file_node: dict[str, Any] = {
            "name": filename,
            "type": "file",
            "path": rel_path,
            "size": size,
        }
        if len(parent_node["children"]) < MAX_ENTRIES_PER_DIR:
            parent_node["children"].append(file_node)

    # Sort children: directories first, then files, both alphabetical
    def _sort_children(node: dict[str, Any]) -> None:
        children = node.get("children", [])
        children.sort(
            key=lambda c: (c["type"] != "dir", c["name"].lower())
        )
        for child in children:
            if child["type"] == "dir":
                _sort_children(child)

    _sort_children(root)
    return root


def _count_tree(node: dict[str, Any]) -> tuple[int, int]:
    """Count files and total bytes in a tree node."""
    files = 0
    total_bytes = 0
    for child in node.get("children", []):
        if child.get("type") == "file":
            files += 1
            total_bytes += int(child.get("size") or 0)
        else:
            f, b = _count_tree(child)
            files += f
            total_bytes += b
    return files, total_bytes


@router.get("/runs/{run_id}/files")
def get_run_files(request: Request, run_id: str):
    """Return the file tree under the run's output directory."""
    _validate_run_id(run_id)
    storage = _get_storage(request)

    from dark_factory.storage.backend import RunStorage
    rs = RunStorage(storage, run_id)

    try:
        entries = list(rs.walk_output())
    except Exception as exc:
        log.warning("run_files_walk_failed", run_id=run_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read output dir: {exc}",
        ) from exc

    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"No output files for run {run_id!r}",
        )

    tree = _build_tree_from_walk(entries)
    file_count, total_bytes = _count_tree(tree)

    # Return JSONResponse directly to bypass FastAPI's jsonable_encoder,
    # which hits RecursionError on deeply nested file trees.
    return JSONResponse({
        "run_id": run_id,
        "root": rs.output_prefix,
        "tree": tree,
        "file_count": file_count,
        "total_bytes": total_bytes,
    })


@router.get("/runs/{run_id}/file")
def get_run_file(
    request: Request,
    run_id: str,
    path: str = Query(..., description="Path relative to run output directory"),
):
    """Return the text content of a single file inside the run's output."""
    _validate_run_id(run_id)
    storage = _get_storage(request)

    from dark_factory.storage.backend import RunStorage
    rs = RunStorage(storage, run_id)

    # Normalise the relative path — strip leading slashes so ``/spec.json``
    # and ``spec.json`` both work, reject absolute paths and '..' segments.
    rel = path.strip().lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")
    if ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="path traversal rejected")

    if not rs.output_exists(rel):
        raise HTTPException(status_code=404, detail=f"file not found: {rel}")

    try:
        raw = rs.read_output_bytes(rel)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"file not found: {rel}")
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"read failed: {exc}"
        ) from exc

    size = len(raw)
    if size > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file too large ({size} bytes > {MAX_FILE_BYTES} max). "
                "Download via direct filesystem access or presigned URL instead."
            ),
        )

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

    # Include presigned URL when available (S3 backend)
    presigned = rs.presign_output(rel)

    result: dict[str, Any] = {
        "run_id": run_id,
        "path": rel,
        "size": size,
        "content": text,
    }
    if presigned:
        result["presigned_url"] = presigned

    return result
