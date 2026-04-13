"""Tests for admin route helpers and the destructive admin endpoints."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Unit tests: admin helper functions ─────────────────────────────────────


def test_clear_output_dir_path_safety():
    """_clear_output_dir refuses to delete a directory outside cwd."""
    from dark_factory.api.routes_admin import _clear_output_dir

    request = MagicMock()
    settings = MagicMock()
    settings.pipeline.output_dir = "/tmp/totally-outside-cwd"
    request.app.state.settings = settings

    with pytest.raises(ValueError, match="not inside cwd"):
        _clear_output_dir(request)


def test_clear_output_dir_wipes_and_recreates(tmp_path, monkeypatch):
    """_clear_output_dir deletes files and recreates the directory."""
    from dark_factory.api.routes_admin import _clear_output_dir

    # Create a subdirectory inside tmp_path to act as output_dir
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "file1.txt").write_text("hello")
    (output_dir / "file2.txt").write_text("world")

    request = MagicMock()
    settings = MagicMock()
    settings.pipeline.output_dir = str(output_dir)
    request.app.state.settings = settings

    # Monkeypatch cwd to tmp_path so the safety check passes
    monkeypatch.chdir(tmp_path)

    result = _clear_output_dir(request)
    assert result["files_deleted"] == 2
    assert result["bytes_freed"] > 0
    assert output_dir.exists()  # recreated empty
    assert list(output_dir.iterdir()) == []


def test_clear_neo4j_counts_and_deletes():
    """_clear_neo4j runs count then DETACH DELETE on all nodes."""
    from dark_factory.api.routes_admin import _clear_neo4j

    request = MagicMock()
    session = MagicMock()
    # First call: count query
    count_result = MagicMock()
    count_result.single.return_value = {"cnt": 42}
    # Second call: delete query
    session.run.side_effect = [count_result, None]
    request.app.state.neo4j_client.session.return_value.__enter__ = MagicMock(return_value=session)
    request.app.state.neo4j_client.session.return_value.__exit__ = MagicMock(return_value=False)

    result = _clear_neo4j(request)
    assert result["nodes_deleted"] == 42
    assert session.run.call_count == 2


def test_clear_qdrant_disabled_when_no_vector_repo():
    """_clear_qdrant returns disabled status when vector_repo is None."""
    from dark_factory.api.routes_admin import _clear_qdrant

    request = MagicMock()
    request.app.state.vector_repo = None

    result = _clear_qdrant(request)
    assert result["status"] == "disabled"


def test_clear_postgres_disabled_when_no_metrics_client():
    """_clear_postgres returns disabled status when metrics_client is None."""
    from dark_factory.api.routes_admin import _clear_postgres

    request = MagicMock()
    request.app.state.metrics_client = None

    result = _clear_postgres(request)
    assert result["status"] == "disabled"


def test_clear_prometheus_in_process_only_when_disabled():
    """_clear_prometheus resets in-process collectors even when prometheus
    is disabled in settings."""
    from dark_factory.api.routes_admin import _clear_prometheus

    request = MagicMock()
    settings = MagicMock()
    settings.prometheus = None
    request.app.state.settings = settings

    result = _clear_prometheus(request)
    assert result["status"] == "in_process_only"
    assert "in_process" in result


def test_clear_prometheus_in_process_only_when_enabled_false():
    """_clear_prometheus with enabled=False still resets in-process collectors."""
    from dark_factory.api.routes_admin import _clear_prometheus

    request = MagicMock()
    settings = MagicMock()
    prom_cfg = MagicMock()
    prom_cfg.enabled = False
    settings.prometheus = prom_cfg
    request.app.state.settings = settings

    result = _clear_prometheus(request)
    assert result["status"] == "in_process_only"
    assert result["reason"] == "prometheus.enabled=false"


def test_reset_all_zeroes_counters():
    """reset_all() clears all dark_factory_ collectors and returns counts."""
    from dark_factory.metrics.prometheus import (
        observe_llm_call,
        reset_all,
    )

    # Create some state first
    observe_llm_call(client="test", model="m1", phase="p1",
                     latency_seconds=0.1, input_tokens=10, output_tokens=5)

    result = reset_all()
    assert isinstance(result, dict)
    assert "cleared_collectors" in result
    assert "reinitialised_collectors" in result
    assert "skipped_collectors" in result
    total = result["cleared_collectors"] + result["reinitialised_collectors"]
    assert total > 0


def test_clear_output_dir_nonexistent_is_safe(tmp_path, monkeypatch):
    """_clear_output_dir handles a non-existent output directory gracefully."""
    from dark_factory.api.routes_admin import _clear_output_dir

    output_dir = tmp_path / "nonexistent_output"

    request = MagicMock()
    settings = MagicMock()
    settings.pipeline.output_dir = str(output_dir)
    request.app.state.settings = settings

    monkeypatch.chdir(tmp_path)

    result = _clear_output_dir(request)
    assert result["files_deleted"] == 0
    assert result["bytes_freed"] == 0
    assert output_dir.exists()  # created empty


def test_clear_postgres_truncates_tables():
    """_clear_postgres truncates metrics tables when client is available."""
    from dark_factory.api.routes_admin import _clear_postgres

    request = MagicMock()
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    request.app.state.metrics_client.connection.return_value.__enter__ = MagicMock(return_value=conn)
    request.app.state.metrics_client.connection.return_value.__exit__ = MagicMock(return_value=False)

    result = _clear_postgres(request)
    assert "tables_truncated" in result
    assert len(result["tables_truncated"]) > 0
    assert cursor.execute.call_count > 0


# ── /api/admin/clear-all endpoint tests ────────────────────────────────────


def test_admin_clear_all_requires_confirm(api_client):
    """Without ``?confirm=yes`` the endpoint returns 400 and does not touch
    any store. This is the primary safety rail — a stray fetch or double-
    click in the UI can't wipe state by accident."""
    resp = api_client.post("/api/admin/clear-all")
    assert resp.status_code == 400
    assert "confirm=yes" in resp.text


def test_admin_clear_all_rejects_wrong_confirm_value(api_client):
    """Any confirm value other than the literal string 'yes' is rejected —
    guards against ``?confirm=true`` / ``?confirm=1`` type mistakes."""
    for value in ["true", "1", "YES", "Yes", "y", "ok"]:
        resp = api_client.post(f"/api/admin/clear-all?confirm={value}")
        assert resp.status_code == 400, f"confirm={value!r} should have been rejected"


def test_admin_clear_all_refuses_during_active_run(api_client):
    """With the ``run_lock`` held, the endpoint returns 409 — the caller
    must cancel the in-flight run before clearing state."""
    from dark_factory.api.app import app

    async def _hold_lock_then_clear():
        lock: asyncio.Lock = app.state.run_lock
        async with lock:
            return await asyncio.to_thread(
                api_client.post, "/api/admin/clear-all?confirm=yes"
            )

    resp = asyncio.run(_hold_lock_then_clear())
    assert resp.status_code == 409
    assert "in progress" in resp.text.lower()


def test_admin_clear_all_reports_per_store_results(api_client):
    """Happy path: the endpoint runs each clear step and reports a
    dict of cleared things, plus any per-store errors."""
    from dark_factory.api.app import app

    # Stub every store's clear helper to a known payload — we're
    # verifying the wiring, not the individual clear implementations.
    with (
        patch(
            "dark_factory.api.routes_admin._clear_neo4j",
            return_value={"nodes_deleted": 42},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_qdrant",
            return_value={"collections_cleared": ["dark_factory_memories"]},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_postgres",
            return_value={"tables_truncated": ["pipeline_runs"]},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_output_dir",
            return_value={"files_deleted": 5, "bytes_freed": 1024},
        ),
    ):
        resp = api_client.post("/api/admin/clear-all?confirm=yes")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["cleared"]["neo4j"]["nodes_deleted"] == 42
    assert body["cleared"]["qdrant"]["collections_cleared"] == ["dark_factory_memories"]
    assert body["cleared"]["postgres"]["tables_truncated"] == ["pipeline_runs"]
    assert body["cleared"]["output_dir"]["files_deleted"] == 5
    assert body["errors"] == {}


def test_admin_clear_all_continues_through_per_store_errors(api_client):
    """A failure in one store (e.g. Qdrant unreachable) does NOT abort
    the other stores — each failure is captured in ``errors`` and the
    remaining stores still run to completion."""
    with (
        patch(
            "dark_factory.api.routes_admin._clear_neo4j",
            return_value={"nodes_deleted": 10},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_qdrant",
            side_effect=RuntimeError("qdrant offline"),
        ),
        patch(
            "dark_factory.api.routes_admin._clear_postgres",
            return_value={"tables_truncated": ["pipeline_runs"]},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_output_dir",
            return_value={"files_deleted": 0, "bytes_freed": 0},
        ),
    ):
        resp = api_client.post("/api/admin/clear-all?confirm=yes")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "partial"
    # Healthy stores still cleared
    assert body["cleared"]["neo4j"]["nodes_deleted"] == 10
    assert body["cleared"]["postgres"]["tables_truncated"] == ["pipeline_runs"]
    # Failing store captured
    assert "qdrant" in body["errors"]
    assert "qdrant offline" in body["errors"]["qdrant"]


def test_admin_clear_all_skip_output_dir(api_client):
    """``include_output_dir=false`` preserves the output directory."""
    with (
        patch(
            "dark_factory.api.routes_admin._clear_neo4j",
            return_value={"nodes_deleted": 0},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_qdrant",
            return_value={"status": "disabled"},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_postgres",
            return_value={"status": "disabled"},
        ),
        patch(
            "dark_factory.api.routes_admin._clear_output_dir",
        ) as mock_clear_output,
    ):
        resp = api_client.post(
            "/api/admin/clear-all?confirm=yes&include_output_dir=false"
        )

    assert resp.status_code == 200
    assert resp.json()["cleared"]["output_dir"] == {"status": "skipped"}
    mock_clear_output.assert_not_called()
