"""Tests for the file upload endpoint."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cleanup_uploads():
    """Remove the uploads directory after each test."""
    yield
    uploads = Path("uploads")
    if uploads.exists():
        shutil.rmtree(uploads, ignore_errors=True)


def test_upload_single_file(api_client):
    """A single .md file can be uploaded and returns a usable path."""
    content = b"# Requirement\n\nThe system shall do X."
    resp = api_client.post(
        "/api/upload",
        files={"files": ("requirement.md", io.BytesIO(content), "text/markdown")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"].startswith("./uploads/")
    assert data["files"] == ["requirement.md"]

    # The file should exist on disk
    uploaded = Path(data["path"]) / "requirement.md"
    assert uploaded.exists()
    assert uploaded.read_bytes() == content


def test_upload_multiple_files(api_client):
    """Multiple files are saved together in one upload directory."""
    resp = api_client.post(
        "/api/upload",
        files=[
            ("files", ("a.md", io.BytesIO(b"req A"), "text/markdown")),
            ("files", ("b.md", io.BytesIO(b"req B"), "text/markdown")),
            ("files", ("c.json", io.BytesIO(b"{}"), "application/json")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["files"]) == 3
    assert set(data["files"]) == {"a.md", "b.md", "c.json"}


def test_upload_rejects_disallowed_extension(api_client):
    """Files with disallowed extensions are rejected."""
    resp = api_client.post(
        "/api/upload",
        files={"files": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"].lower()


def test_upload_strips_path_traversal_in_filename(api_client):
    """A filename like '../../etc/passwd' is stripped to just the basename."""
    resp = api_client.post(
        "/api/upload",
        files={"files": ("../../etc/passwd.md", io.BytesIO(b"data"), "text/markdown")},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Should be saved as 'passwd.md' (basename only)
    assert data["files"] == ["passwd.md"]
    # Nothing should have been written outside the upload dir
    assert not Path("../../etc/passwd.md").exists()


def test_upload_rejects_too_many_files(api_client):
    """Uploads exceeding MAX_FILES are rejected."""
    files = [
        ("files", (f"file{i}.md", io.BytesIO(b"x"), "text/markdown"))
        for i in range(51)
    ]
    resp = api_client.post("/api/upload", files=files)
    assert resp.status_code == 400
    assert "too many" in resp.json()["detail"].lower()


def test_upload_rejects_oversized_file(api_client):
    """Files exceeding MAX_FILE_SIZE are rejected."""
    big_content = b"x" * (26 * 1024 * 1024)  # 26 MB > 25 MB limit
    resp = api_client.post(
        "/api/upload",
        files={"files": ("big.md", io.BytesIO(big_content), "text/markdown")},
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


def test_upload_accepts_rich_business_document_extensions(api_client):
    """Word, Excel, PDF, HTML, RTF, XML, CSV and transcript extensions
    are all accepted and stored verbatim. Ingestion-time extraction via
    the clean-context deep agent is tested separately in
    ``tests/test_doc_extraction.py``."""
    rich_samples = [
        ("meeting-notes.docx", b"PK\x03\x04fake-docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("features.xlsx", b"PK\x03\x04fake-xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("brief.pdf", b"%PDF-1.4\n", "application/pdf"),
        ("spec.html", b"<html><body>x</body></html>", "text/html"),
        ("schema.xml", b"<?xml version='1.0'?><root/>", "application/xml"),
        ("notes.rtf", b"{\\rtf1 hi}", "application/rtf"),
        ("table.csv", b"title,description\nA,B\n", "text/csv"),
        ("transcript.vtt", b"WEBVTT\n\n00:00.000 --> 00:01.000\nhi", "text/vtt"),
    ]
    resp = api_client.post(
        "/api/upload",
        files=[
            ("files", (name, io.BytesIO(data), ctype))
            for name, data, ctype in rich_samples
        ],
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert len(data["files"]) == len(rich_samples)
    assert set(data["files"]) == {name for name, _, _ in rich_samples}

    # All files should exist on disk with their original bytes intact —
    # extraction happens at ingest time, not at upload time, so the
    # upload endpoint stores raw content.
    upload_dir = Path(data["path"])
    for name, content, _ in rich_samples:
        stored = upload_dir / name
        assert stored.exists()
        assert stored.read_bytes() == content


def test_upload_path_can_be_used_as_requirements_path(api_client):
    """The path returned by /upload passes the RunRequest path validator."""
    from dark_factory.api.routes_agent import RunRequest

    resp = api_client.post(
        "/api/upload",
        files={"files": ("r.md", io.BytesIO(b"requirement"), "text/markdown")},
    )
    path = resp.json()["path"]

    # Should validate successfully (no exception raised)
    req = RunRequest(requirements_path=path)
    assert req.requirements_path == path
