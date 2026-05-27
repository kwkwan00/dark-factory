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

import difflib
import re
import tempfile
import zipfile
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from dark_factory.api.validators import RUN_ID_RE

log = structlog.get_logger()

router = APIRouter()

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


# ── Zip download ──────────────────────────────────────────────────────────────

MAX_ZIP_BYTES = 500 * 1024 * 1024  # 500 MiB cap


@router.get("/runs/{run_id}/download")
def download_run_zip(request: Request, run_id: str):
    """Stream the run's output directory as a zip archive."""
    _validate_run_id(run_id)
    storage = _get_storage(request)

    from dark_factory.storage.backend import RunStorage
    rs = RunStorage(storage, run_id)

    entries = list(rs.walk_output())
    if not entries:
        raise HTTPException(status_code=404, detail=f"No output files for run {run_id!r}")

    total_bytes = sum(size for _, size in entries)
    if total_bytes > MAX_ZIP_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Output too large ({total_bytes} bytes > {MAX_ZIP_BYTES} max)",
        )

    # Build zip in a spooled temp file (spills to disk over 50 MiB)
    tmp = tempfile.SpooledTemporaryFile(max_size=50 * 1024 * 1024)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path, _ in entries:
            try:
                data = rs.read_output_bytes(rel_path)
                zf.writestr(rel_path, data)
            except Exception:
                pass  # skip unreadable files
    tmp.seek(0)

    def _iter():
        while True:
            chunk = tmp.read(64 * 1024)
            if not chunk:
                break
            yield chunk
        tmp.close()

    return StreamingResponse(
        _iter(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{run_id}-output.zip"',
        },
    )


# ── Diff between runs ────────────────────────────────────────────────────────

MAX_DIFF_FILES = 500
MAX_DIFF_FILE_BYTES = 1 * 1024 * 1024
MAX_DIFF_LINES = 2000  # cap diff output to prevent huge payloads


@router.get("/runs/diff")
def diff_runs(
    request: Request,
    run_a: str = Query(...),
    run_b: str = Query(...),
    include_unchanged: bool = Query(default=False),
):
    """Return file-level unified diffs between two runs' outputs.

    Optimized: uses pre-computed MD5 manifests (written at sync time)
    to skip content reads for identical files. Falls back to on-the-fly
    hashing if no manifest exists.
    """
    import hashlib

    _validate_run_id(run_a)
    _validate_run_id(run_b)
    storage = _get_storage(request)

    from dark_factory.storage.backend import RunStorage
    rs_a = RunStorage(storage, run_a)
    rs_b = RunStorage(storage, run_b)

    files_a = {rel: size for rel, size in rs_a.walk_output()}
    files_b = {rel: size for rel, size in rs_b.walk_output()}

    # Load pre-computed MD5 manifests (written during sync_output_from_local)
    md5_a = rs_a.get_md5_manifest()
    md5_b = rs_b.get_md5_manifest()

    all_paths = sorted(set(files_a) | set(files_b))
    if len(all_paths) > MAX_DIFF_FILES:
        all_paths = all_paths[:MAX_DIFF_FILES]

    stats = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
    results: list[dict[str, Any]] = []

    for path in all_paths:
        in_a = path in files_a
        in_b = path in files_b

        if not in_a:
            stats["added"] += 1
            results.append({"path": path, "status": "added"})
            continue
        if not in_b:
            stats["removed"] += 1
            results.append({"path": path, "status": "removed"})
            continue

        # Fast path: check pre-computed MD5 manifest first (no file reads)
        hash_a = md5_a.get(path)
        hash_b = md5_b.get(path)
        if hash_a and hash_b:
            if hash_a == hash_b:
                stats["unchanged"] += 1
                if include_unchanged:
                    results.append({"path": path, "status": "unchanged"})
                continue
            # Hashes differ — file is definitely modified. Check size
            # to decide whether to compute a text diff.
            size_a, size_b = files_a[path], files_b[path]
            if size_a > MAX_DIFF_FILE_BYTES or size_b > MAX_DIFF_FILE_BYTES:
                stats["modified"] += 1
                results.append({"path": path, "status": "modified", "diff": "(file too large to diff)"})
                continue
            # Fall through to read + diff below
        else:
            # No manifest — check sizes as a fast pre-filter
            size_a, size_b = files_a[path], files_b[path]
            if size_a > MAX_DIFF_FILE_BYTES or size_b > MAX_DIFF_FILE_BYTES:
                status = "unchanged" if size_a == size_b else "modified"
                stats[status] += 1
                if status == "modified":
                    results.append({"path": path, "status": "modified", "diff": "(file too large to diff)"})
                elif include_unchanged:
                    results.append({"path": path, "status": "unchanged"})
                continue

        try:
            raw_a = rs_a.read_output_bytes(path)
            raw_b = rs_b.read_output_bytes(path)
        except Exception:
            results.append({"path": path, "status": "error"})
            continue

        # If we didn't have manifest hashes, check content identity now
        if not hash_a or not hash_b:
            if hashlib.md5(raw_a).digest() == hashlib.md5(raw_b).digest():
                stats["unchanged"] += 1
                if include_unchanged:
                    results.append({"path": path, "status": "unchanged"})
                continue

        # Skip binary files
        if b"\x00" in raw_a[:1024] or b"\x00" in raw_b[:1024]:
            stats["modified"] += 1
            results.append({"path": path, "status": "binary"})
            continue

        try:
            text_a = raw_a.decode("utf-8").splitlines(keepends=True)
            text_b = raw_b.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            stats["modified"] += 1
            results.append({"path": path, "status": "binary"})
            continue

        diff_lines = list(difflib.unified_diff(
            text_a, text_b,
            fromfile=f"{run_a}/{path}",
            tofile=f"{run_b}/{path}",
        ))

        stats["modified"] += 1
        if len(diff_lines) > MAX_DIFF_LINES:
            diff_text = "".join(diff_lines[:MAX_DIFF_LINES])
            diff_text += f"\n... ({len(diff_lines) - MAX_DIFF_LINES} more lines truncated)"
        else:
            diff_text = "".join(diff_lines)

        results.append({
            "path": path,
            "status": "modified",
            "diff": diff_text,
        })

    return {
        "run_a": run_a,
        "run_b": run_b,
        "files": results,
        "stats": stats,
    }


# ── Run comparison ────────────────────────────────────────────────────────────


@router.get("/runs/compare")
def compare_runs(
    request: Request,
    run_a: str = Query(...),
    run_b: str = Query(...),
):
    """Return side-by-side metrics for two runs."""
    _validate_run_id(run_a)
    _validate_run_id(run_b)

    memory_repo = getattr(request.app.state, "memory_repo", None)
    metrics_client = getattr(request.app.state, "metrics_client", None)

    def _load_run(rid: str) -> dict[str, Any]:
        run_data: dict[str, Any] = {"run_id": rid}
        # Neo4j run node
        if memory_repo is not None:
            try:
                run_node = memory_repo.get_run_by_id(rid)
                if run_node is not None:
                    run_data.update(run_node)
            except Exception:
                pass
            # Feature-level episode summaries
            try:
                episodes = memory_repo.get_episodes_for_run(run_id=rid)
                run_data["episodes"] = [
                    {
                        "feature": ep.get("feature"),
                        "outcome": ep.get("outcome"),
                        "turns_used": ep.get("turns_used"),
                        "duration_seconds": ep.get("duration_seconds"),
                    }
                    for ep in episodes
                ]
            except Exception:
                run_data["episodes"] = []
        # Postgres: per-feature swarm events
        if metrics_client is not None:
            try:
                with metrics_client.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT feature, status, duration_seconds "
                            "FROM swarm_feature_events WHERE run_id = %s "
                            "ORDER BY timestamp",
                            (rid,),
                        )
                        run_data["features"] = [
                            dict(row) for row in cur.fetchall()
                        ]
                        cur.execute(
                            "SELECT sum(cost_usd) AS total_cost, "
                            "       sum(input_tokens) AS total_input_tokens, "
                            "       sum(output_tokens) AS total_output_tokens, "
                            "       count(*) AS llm_calls "
                            "FROM llm_calls WHERE run_id = %s",
                            (rid,),
                        )
                        cost_row = cur.fetchone()
                        if cost_row:
                            run_data["llm_cost"] = dict(cost_row)
            except Exception:
                pass
        return run_data

    return {
        "run_a": _load_run(run_a),
        "run_b": _load_run(run_b),
    }
