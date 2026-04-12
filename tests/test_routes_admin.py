"""Tests for the destructive admin endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


# ── Per-helper unit tests ─────────────────────────────────────────────────


def test_clear_output_dir_refuses_paths_outside_cwd(tmp_path, monkeypatch):
    """Safety: if the operator has pointed output_dir outside the working
    directory, the clear function refuses to delete anything. Prevents a
    misconfigured config from wiping unrelated host files."""
    from dark_factory.api.routes_admin import _clear_output_dir
    from dark_factory.config import Settings

    # Build a settings object pointing at an absolute path OUTSIDE cwd
    outside = Path("/tmp/definitely-not-in-cwd-admin-test")
    settings = Settings()
    settings.pipeline.output_dir = str(outside)

    req = MagicMock()
    req.app.state.settings = settings

    with pytest.raises(ValueError, match="not inside cwd"):
        _clear_output_dir(req)


def test_clear_output_dir_wipes_and_recreates(tmp_path, monkeypatch):
    """Happy path: files inside the output dir are deleted and the dir
    itself is recreated empty."""
    from dark_factory.api.routes_admin import _clear_output_dir
    from dark_factory.config import Settings

    # Create a realistic output dir tree inside cwd
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "file1.py").write_text("hello")
    (output_dir / "nested").mkdir()
    (output_dir / "nested" / "file2.txt").write_text("world" * 10)

    monkeypatch.chdir(tmp_path)

    settings = Settings()
    settings.pipeline.output_dir = "./output"

    req = MagicMock()
    req.app.state.settings = settings

    result = _clear_output_dir(req)

    assert result["files_deleted"] == 2
    assert result["bytes_freed"] > 0
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_clear_neo4j_counts_and_deletes_all_nodes():
    """_clear_neo4j issues a count query and a DETACH DELETE, returning
    the pre-deletion node count."""
    from dark_factory.api.routes_admin import _clear_neo4j

    mock_session = MagicMock()
    # First session.run() is the count query — returns a record-like with
    # a ``cnt`` field; second is the DETACH DELETE.
    count_result = MagicMock()
    count_result.single.return_value = {"cnt": 1234}
    mock_session.run.side_effect = [count_result, None]

    mock_client = MagicMock()
    mock_client.session.return_value.__enter__.return_value = mock_session

    req = MagicMock()
    req.app.state.neo4j_client = mock_client

    result = _clear_neo4j(req)
    assert result == {"nodes_deleted": 1234}

    # Verify the two queries were run
    sqls = [call.args[0] for call in mock_session.run.call_args_list]
    assert any("count(n)" in sql for sql in sqls)
    assert any("DETACH DELETE" in sql for sql in sqls)


def test_clear_qdrant_disabled_when_vector_repo_is_none():
    from dark_factory.api.routes_admin import _clear_qdrant

    req = MagicMock()
    req.app.state.vector_repo = None
    assert _clear_qdrant(req) == {"status": "disabled"}


def test_clear_postgres_disabled_when_client_is_none():
    from dark_factory.api.routes_admin import _clear_postgres

    req = MagicMock()
    req.app.state.metrics_client = None
    assert _clear_postgres(req) == {"status": "disabled"}


# ── Prometheus clearing ─────────────────────────────────────────────────────


def test_clear_prometheus_in_process_only_when_disabled():
    """If ``settings.prometheus.enabled`` is false we skip the remote
    delete entirely but still reset in-process collectors."""
    from dark_factory.api.routes_admin import _clear_prometheus

    settings = MagicMock()
    settings.prometheus.enabled = False
    req = MagicMock()
    req.app.state.settings = settings

    with patch(
        "dark_factory.metrics.prometheus.reset_all",
        return_value={
            "cleared_collectors": 10,
            "reinitialised_collectors": 3,
            "skipped_collectors": 0,
        },
    ):
        result = _clear_prometheus(req)

    assert result["status"] == "in_process_only"
    assert result["in_process"]["cleared_collectors"] == 10
    assert result["reason"] == "prometheus.enabled=false"


def test_clear_prometheus_happy_path(monkeypatch):
    """Happy path: both delete_series and clean_tombstones return 204,
    and we report completed + in-process reset stats."""
    from dark_factory.api.routes_admin import _clear_prometheus

    settings = MagicMock()
    settings.prometheus.enabled = True
    settings.prometheus.url = "http://prom.test:9090"
    req = MagicMock()
    req.app.state.settings = settings

    class FakeResponse:
        def __init__(self, status_code: int = 204, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.calls: list[tuple[str, dict]] = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url: str, params: dict | None = None) -> FakeResponse:
            self.calls.append((url, params or {}))
            return FakeResponse(204)

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeClient)

    with patch(
        "dark_factory.metrics.prometheus.reset_all",
        return_value={
            "cleared_collectors": 25,
            "reinitialised_collectors": 4,
            "skipped_collectors": 0,
        },
    ):
        result = _clear_prometheus(req)

    assert result["status"] == "completed"
    assert result["series_deleted"] is True
    assert result["tombstones_cleaned"] is True
    assert result["in_process"]["cleared_collectors"] == 25
    assert result["url"] == "http://prom.test:9090"


def test_clear_prometheus_reports_admin_api_disabled(monkeypatch):
    """If Prometheus returns 404 on the delete endpoint, it means the
    admin API wasn't enabled on the server. Report that clearly so the
    operator knows they need ``--web.enable-admin-api``."""
    from dark_factory.api.routes_admin import _clear_prometheus

    settings = MagicMock()
    settings.prometheus.enabled = True
    settings.prometheus.url = "http://prom.test:9090"
    req = MagicMock()
    req.app.state.settings = settings

    class FakeResponse:
        def __init__(self, status_code: int = 404) -> None:
            self.status_code = status_code
            self.text = "admin APIs disabled"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url: str, params: dict | None = None) -> FakeResponse:
            return FakeResponse(404)

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeClient)

    with patch(
        "dark_factory.metrics.prometheus.reset_all",
        return_value={
            "cleared_collectors": 5,
            "reinitialised_collectors": 1,
            "skipped_collectors": 0,
        },
    ):
        result = _clear_prometheus(req)

    assert result["status"] == "admin_api_disabled"
    assert "--web.enable-admin-api" in result["hint"]
    # In-process reset still happened
    assert result["in_process"]["cleared_collectors"] == 5


def test_clear_prometheus_reports_unreachable(monkeypatch):
    """Network error → status=unreachable, in-process reset still reported."""
    from dark_factory.api.routes_admin import _clear_prometheus

    settings = MagicMock()
    settings.prometheus.enabled = True
    settings.prometheus.url = "http://prom.test:9090"
    req = MagicMock()
    req.app.state.settings = settings

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "Client", FakeClient)

    with patch(
        "dark_factory.metrics.prometheus.reset_all",
        return_value={
            "cleared_collectors": 7,
            "reinitialised_collectors": 1,
            "skipped_collectors": 0,
        },
    ):
        result = _clear_prometheus(req)

    assert result["status"] == "unreachable"
    assert "connection refused" in result["error"]
    assert result["in_process"]["cleared_collectors"] == 7


def test_reset_all_actually_zeroes_counters():
    """Integration-ish: bump some dark_factory counters and verify that
    ``reset_all`` brings them back to 0. Uses the real global REGISTRY
    so this catches prometheus_client API drift."""
    from dark_factory.metrics import prometheus as prom

    # Labelled counter
    prom.llm_calls_total.labels(
        client="anthropic", model="claude-opus-4-6", phase="spec"
    ).inc(5)
    # Unlabelled gauge
    prom.background_loop_active_tasks.set(42)
    # Unlabelled counter
    prom.metrics_events_dropped_total.inc(3)

    # Sanity: they're non-zero before reset
    sample_before = {
        s.name: s.value
        for s in prom.llm_calls_total.collect()[0].samples
        if s.name.endswith("_total")
    }
    assert any(v > 0 for v in sample_before.values())
    assert prom.background_loop_active_tasks._value.get() == 42
    assert prom.metrics_events_dropped_total._value.get() == 3

    report = prom.reset_all()
    assert report["cleared_collectors"] > 0

    # Labelled counter has no children after clear()
    assert len(prom.llm_calls_total._metrics) == 0
    # Unlabelled gauge is back to 0
    assert prom.background_loop_active_tasks._value.get() == 0
    # Unlabelled counter is back to 0
    assert prom.metrics_events_dropped_total._value.get() == 0
