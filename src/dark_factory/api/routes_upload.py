"""File upload endpoint for drag-and-drop requirements on the Run Pipeline tab."""

from __future__ import annotations

import pathlib
import shutil
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

log = structlog.get_logger()

router = APIRouter()

# Upload configuration
UPLOAD_DIR = pathlib.Path("uploads").resolve()
# Business documents (Word, Excel, PDF, transcripts) are routinely larger
# than plain-text requirements, so the per-file cap is bumped to 25 MB and
# the total-upload cap to 150 MB. Still small enough to be impossible to
# DoS the worker with a single request.
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB per file
MAX_TOTAL_SIZE = 150 * 1024 * 1024  # 150 MB across all files in a single upload (M10)
MAX_FILES = 50

# Two tiers of allowed extensions.
#
# NATIVE: handled by the existing IngestStage fast path (JSON/YAML
# structured parse, plain-text LLM splitter). These file types are
# well-understood and don't need a deep-agent extraction pass.
#
# RICH: business documents from requirements meetings (Word, Excel,
# transcripts, HTML, XML, PDF, RTF, ...) — dispatched to a clean-context
# Claude Agent SDK subagent that reads the file with the appropriate
# python library (python-docx, openpyxl, pypdf, beautifulsoup4,
# striprtf, lxml) and extracts discrete testable requirements to a
# staging JSON file. See ``dark_factory.stages.doc_extraction``.
NATIVE_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}
RICH_EXTENSIONS = {
    # Office documents
    ".docx",
    ".xlsx",
    ".pptx",
    # PDFs
    ".pdf",
    # Rich text
    ".rtf",
    # Markup
    ".html",
    ".htm",
    ".xml",
    # Tabular
    ".csv",
    # Transcript / log files
    ".vtt",
    ".srt",
    ".log",
}
ALLOWED_EXTENSIONS = NATIVE_EXTENSIONS | RICH_EXTENSIONS
# M11: upload directories older than this are eligible for cleanup
UPLOAD_TTL_SECONDS = 24 * 60 * 60  # 24h


def _validate_filename(filename: str) -> str:
    """Strip any path components and verify the extension is allowed."""
    safe_name = pathlib.Path(filename).name  # strips directory traversal
    if not safe_name:
        raise HTTPException(status_code=400, detail="Empty filename")
    ext = pathlib.Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return safe_name


def _cleanup_old_uploads() -> None:
    """M11: best-effort removal of upload directories older than ``UPLOAD_TTL_SECONDS``.

    Called opportunistically at the start of each upload so we don't need
    a separate background task. Failures are silent (best-effort cleanup).
    """
    if not UPLOAD_DIR.exists():
        return
    import time as _time

    now = _time.time()
    for entry in UPLOAD_DIR.iterdir():
        try:
            if not entry.is_dir():
                continue
            age = now - entry.stat().st_mtime
            if age > UPLOAD_TTL_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
                log.info("upload_dir_pruned", path=str(entry), age_seconds=int(age))
        except Exception:
            pass


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Accept one or more requirement files and stage them in an upload directory.

    Returns ``{"path": "./uploads/<uuid>", "files": [...]}`` — the path can be
    used as the ``requirements_path`` in ``POST /api/agent/run``.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}); max is {MAX_FILES}",
        )

    # M11: opportunistic cleanup of old upload directories
    _cleanup_old_uploads()

    upload_id = uuid4().hex[:12]
    target_dir = UPLOAD_DIR / upload_id
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    cumulative_total = 0  # M10: track size across all files
    try:
        for upload in files:
            if not upload.filename:
                continue
            safe_name = _validate_filename(upload.filename)
            dest = (target_dir / safe_name).resolve()

            # Double-check: ensure the resolved destination is still inside target_dir
            if not dest.is_relative_to(target_dir):
                raise HTTPException(
                    status_code=400, detail=f"Invalid filename: {upload.filename}"
                )

            # Stream the file to disk, aborting if too large
            total = 0
            with dest.open("wb") as out:
                while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
                    total += len(chunk)
                    if total > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{safe_name}' exceeds {MAX_FILE_SIZE} bytes",
                        )
                    cumulative_total += len(chunk)
                    if cumulative_total > MAX_TOTAL_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Total upload size exceeds {MAX_TOTAL_SIZE} bytes",
                        )
                    out.write(chunk)

            saved.append(safe_name)
            log.info("file_uploaded", filename=safe_name, size=total, upload_id=upload_id)

        # Return a relative path so it passes the RunRequest path validator
        relative_path = f"./uploads/{upload_id}"
        return {"upload_id": upload_id, "path": relative_path, "files": saved}

    except HTTPException:
        # Clean up the partial directory on error
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        log.error("upload_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
