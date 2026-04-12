"""Tests for the per-run detail endpoints.

Covers:
- ``GET /api/metrics/runs/{run_id}`` — aggregated per-run metrics
- ``GET /api/runs/{run_id}/files`` — file tree under the run output dir
- ``GET /api/runs/{run_id}/file?path=…`` — file content preview

The metrics endpoint is tested in two modes: postgres-disabled (returns
an ``enabled: false`` payload) and postgres-enabled with a mocked
``MetricsRepository`` that returns a canned dict.

The file endpoints are tested against a real ``tmp_path`` tree so the
safety rails (path traversal, symlinks escaping the run dir, oversize
files, binary refusal) run against the real filesystem.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── /api/metrics/runs/{run_id} ─────────────────────────────────────────────


def test_run_detail_rejects_invalid_run_id(api_client):
    """Characters outside ``[A-Za-z0-9_-]`` that survive URL normalization
    must be rejected before we even touch the metrics store — backstops
    the regex against path/SQL shenanigans. Note: ``..`` and ``/`` get
    normalized away by httpx before the request even reaches Starlette,
    so we only test characters that actually reach the handler."""
    for bad in ["run id with space", "run;drop", "run.json", "run:inject"]:
        resp = api_client.get(f"/api/metrics/runs/{bad}")
        assert resp.status_code == 400, f"bad run_id {bad!r} accepted"
        assert "invalid run_id" in resp.text.lower()


def test_run_detail_returns_disabled_when_no_metrics_client(api_client):
    """When Postgres isn't configured the endpoint responds 200 with
    ``enabled: false`` so the frontend can render a neutral empty state
    instead of surfacing a 503."""
    # The api_client fixture already mounts the app with no metrics_client
    # (lifespan returns None when POSTGRES_ENABLED is false in tests).
    resp = api_client.get("/api/metrics/runs/run-20260101-000000-abcd")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["run_id"] == "run-20260101-000000-abcd"


def test_run_detail_returns_404_when_run_missing(api_client):
    """With the metrics client wired up but the run not found in
    ``pipeline_runs``, the endpoint must 404 so the popup can show
    a helpful empty state instead of rendering an empty payload."""
    from dark_factory.api.app import app

    # Install a fake metrics_client into app.state so _repo_or_none() builds
    # a repo; patch MetricsRepository so query_run_detail returns None.
    fake_client = MagicMock()
    original = app.state.metrics_client
    app.state.metrics_client = fake_client
    try:
        with patch(
            "dark_factory.metrics.repository.MetricsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.query_run_detail.return_value = None
            mock_repo_cls.return_value = mock_repo

            resp = api_client.get("/api/metrics/runs/run-missing")
    finally:
        app.state.metrics_client = original

    assert resp.status_code == 404
    assert "not found" in resp.text.lower()


def test_run_detail_returns_repo_payload(api_client):
    """Happy path: repo returns a detail dict and the endpoint echoes it
    under ``{enabled: true, ...}``."""
    from dark_factory.api.app import app

    fake_payload = {
        "run": {
            "run_id": "run-20260101-000000-abcd",
            "status": "success",
            "spec_count": 3,
            "feature_count": 2,
            "pass_rate": 0.9,
            "duration_seconds": 42.0,
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:42+00:00",
            "error": None,
            "metadata": {},
        },
        "llm": {
            "totals": {"total_calls": 10, "total_cost_usd": 0.12},
            "per_phase": [{"phase": "spec", "calls": 5}],
        },
        "swarm_events": [],
        "agent_stats": [],
        "tool_calls": [],
        "incidents": [],
        "eval_metrics": [],
        "artifacts": {"summary": {}, "per_language": []},
        "decomposition": [],
    }

    fake_client = MagicMock()
    original = app.state.metrics_client
    app.state.metrics_client = fake_client
    try:
        with patch(
            "dark_factory.metrics.repository.MetricsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.query_run_detail.return_value = fake_payload
            mock_repo_cls.return_value = mock_repo

            resp = api_client.get(
                "/api/metrics/runs/run-20260101-000000-abcd"
            )
    finally:
        app.state.metrics_client = original

    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["run"]["run_id"] == "run-20260101-000000-abcd"
    assert body["llm"]["totals"]["total_calls"] == 10


# ── /api/runs/{run_id}/files ───────────────────────────────────────────────


def _install_output_dir(api_client, tmp_path: Path) -> Path:
    """Point the app's settings.pipeline.output_dir at tmp_path and return it."""
    from dark_factory.api.app import app

    app.state.settings.pipeline.output_dir = str(tmp_path)
    return tmp_path


def test_run_files_returns_404_for_missing_run_dir(api_client, tmp_path):
    _install_output_dir(api_client, tmp_path)
    resp = api_client.get("/api/runs/run-does-not-exist/files")
    assert resp.status_code == 404


def test_run_files_rejects_bad_run_id_format(api_client, tmp_path):
    _install_output_dir(api_client, tmp_path)
    # Path with a dot — the validator rejects anything outside [A-Za-z0-9_-].
    resp = api_client.get("/api/runs/run..etc/files")
    assert resp.status_code in (400, 404)


def test_run_files_returns_tree(api_client, tmp_path):
    """Happy path: build a realistic tree under ``<output_dir>/<run_id>``
    and assert the response contains both files, the directory, and the
    roll-up counts."""
    _install_output_dir(api_client, tmp_path)

    run_id = "run-20260101-000000-test"
    run_dir = tmp_path / run_id
    (run_dir / "src").mkdir(parents=True)
    (run_dir / "src" / "main.py").write_text("print('hi')\n")
    (run_dir / "spec.json").write_text('{"ok": true}\n')

    resp = api_client.get(f"/api/runs/{run_id}/files")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["file_count"] == 2
    assert body["total_bytes"] > 0

    # Tree is nested: root -> [spec.json, src/ -> main.py]
    tree = body["tree"]
    assert tree["type"] == "dir"
    names = {c["name"] for c in tree["children"]}
    assert names == {"src", "spec.json"}

    src_node = next(c for c in tree["children"] if c["name"] == "src")
    assert src_node["type"] == "dir"
    assert src_node["children"][0]["name"] == "main.py"
    assert src_node["children"][0]["path"] == "src/main.py"


# ── /api/runs/{run_id}/file ────────────────────────────────────────────────


def test_run_file_returns_text_content(api_client, tmp_path):
    _install_output_dir(api_client, tmp_path)

    run_id = "run-20260101-000000-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text('{"key": "value"}\n')

    resp = api_client.get(f"/api/runs/{run_id}/file?path=spec.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "spec.json"
    assert '"value"' in body["content"]
    assert body["size"] > 0


def test_run_file_rejects_path_traversal(api_client, tmp_path):
    """``..`` segments in the path query must be rejected — prevents the
    caller from reading files outside the run dir via ``../secret.txt``."""
    _install_output_dir(api_client, tmp_path)

    run_id = "run-traversal-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "inner.txt").write_text("ok")
    # Write a sensitive file OUTSIDE the run dir
    (tmp_path / "secret.txt").write_text("hunter2")

    resp = api_client.get(f"/api/runs/{run_id}/file?path=../secret.txt")
    assert resp.status_code == 400
    assert "traversal" in resp.text.lower()


def test_run_file_rejects_symlink_escape(api_client, tmp_path):
    """A symlink inside the run dir that points outside it must be
    refused — the resolved target escapes the run dir."""
    _install_output_dir(api_client, tmp_path)

    run_id = "run-symlink-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("do not read")

    link = run_dir / "escape.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    resp = api_client.get(f"/api/runs/{run_id}/file?path=escape.txt")
    # The resolved target is outside run_dir, so the safety rail fires.
    assert resp.status_code == 400
    assert "escapes" in resp.text.lower()


def test_run_file_refuses_oversized_file(api_client, tmp_path, monkeypatch):
    """Files larger than MAX_FILE_BYTES must return 413 — monkeypatch the
    limit down so we don't have to write a 1 MiB file in tests."""
    from dark_factory.api import routes_runs

    monkeypatch.setattr(routes_runs, "MAX_FILE_BYTES", 16)
    _install_output_dir(api_client, tmp_path)

    run_id = "run-oversize-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "big.txt").write_text("x" * 100)

    resp = api_client.get(f"/api/runs/{run_id}/file?path=big.txt")
    assert resp.status_code == 413
    assert "too large" in resp.text.lower()


def test_run_file_refuses_binary_content(api_client, tmp_path):
    """Files with null bytes in the first 1 KiB are treated as binary
    and refused — the UI is a text viewer."""
    _install_output_dir(api_client, tmp_path)

    run_id = "run-binary-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "binary.bin").write_bytes(b"PNG\x00\x00\x00some bytes")

    resp = api_client.get(f"/api/runs/{run_id}/file?path=binary.bin")
    assert resp.status_code == 415
    assert "binary" in resp.text.lower()


def test_run_file_returns_404_for_missing_file(api_client, tmp_path):
    _install_output_dir(api_client, tmp_path)

    run_id = "run-404-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)

    resp = api_client.get(f"/api/runs/{run_id}/file?path=missing.txt")
    assert resp.status_code == 404
