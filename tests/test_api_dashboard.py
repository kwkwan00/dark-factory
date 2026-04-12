"""Tests for the REST dashboard endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dark_factory.api.app import app


# ── Graph Gap Finder ───────────────────────────────────────────────────────────


def _install_neo4j_side_effects(mock_session, unplanned, specs, req_count):
    """Configure a mock Neo4j session for the gaps endpoint.

    The endpoint makes four session.run() calls in order:
      1. unplanned requirements
      2. all specs with requirement_ids
      3. requirement count (on a second session; we lump it in for
         simplicity — see below)
    """
    count_result = MagicMock()
    count_result.single.return_value = {"cnt": req_count}
    mock_session.run.side_effect = [unplanned, specs, count_result]


def test_graph_gaps_postgres_disabled_returns_unplanned_only(api_client):
    """When Postgres isn't configured the endpoint still returns the
    Neo4j-derived unplanned requirement list and sets
    ``enabled_postgres=false`` so the UI can render a partial view."""
    app.state.metrics_client = None

    mock_session = MagicMock()
    app.state.neo4j_client.session.return_value.__enter__.return_value = mock_session
    app.state.neo4j_client.session.return_value.__exit__.return_value = False

    _install_neo4j_side_effects(
        mock_session,
        unplanned=[
            {
                "id": "req-1",
                "title": "Login",
                "priority": "high",
                "source_file": "reqs/auth.md",
            }
        ],
        specs=[],
        req_count=1,
    )

    resp = api_client.get("/api/graph/gaps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled_postgres"] is False
    assert data["stale_days"] == 7
    assert len(data["unplanned_requirements"]) == 1
    assert data["unplanned_requirements"][0]["id"] == "req-1"
    # Postgres-backed lists are empty when the store is off.
    assert data["specs_without_artifacts"] == []
    assert data["specs_failing_evals"] == []
    assert data["stale_requirements"] == []
    assert data["totals"]["requirements"] == 1


def test_graph_gaps_happy_path_with_postgres(api_client):
    """Full end-to-end with mocked Neo4j + Postgres. Exercises all four
    gap categories at once:
      - req-1 is unplanned (no IMPLEMENTS)
      - spec-1 has no artifacts (not in artifact_writes)
      - spec-2 failed its latest eval
      - req-3 is stale (last eval > stale_days ago)
    """
    from datetime import datetime, timedelta, timezone

    # Neo4j side: one unplanned req, three specs with IMPLEMENTS links
    mock_session = MagicMock()
    app.state.neo4j_client.session.return_value.__enter__.return_value = mock_session
    app.state.neo4j_client.session.return_value.__exit__.return_value = False

    _install_neo4j_side_effects(
        mock_session,
        unplanned=[
            {
                "id": "req-1",
                "title": "Login",
                "priority": "high",
                "source_file": "reqs/auth.md",
            }
        ],
        specs=[
            {
                "id": "spec-1",
                "title": "Login spec",
                "capability": "auth",
                "requirement_ids": ["req-2"],
            },
            {
                "id": "spec-2",
                "title": "Logout spec",
                "capability": "auth",
                "requirement_ids": ["req-2"],
            },
            {
                "id": "spec-3",
                "title": "Stale spec",
                "capability": "legacy",
                "requirement_ids": ["req-3"],
            },
        ],
        req_count=3,
    )

    # Postgres side: stub out metrics_client.connection() chain.
    # artifact_writes → only spec-2 and spec-3 have artifacts.
    # eval_metrics   → spec-2 failed its latest eval; spec-3's latest
    #                  eval is 30 days old.
    now = datetime.now(tz=timezone.utc)
    old = now - timedelta(days=30)

    class _FakeCursor:
        def __init__(self):
            self._result: list[dict] = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            if "artifact_writes" in sql:
                self._result = [
                    {"spec_id": "spec-2"},
                    {"spec_id": "spec-3"},
                ]
            elif "eval_metrics" in sql:
                self._result = [
                    {
                        "spec_id": "spec-2",
                        "metric_name": "correctness",
                        "score": 0.2,
                        "passed": False,
                        "timestamp": now,
                    },
                    {
                        "spec_id": "spec-3",
                        "metric_name": "correctness",
                        "score": 0.95,
                        "passed": True,
                        "timestamp": old,
                    },
                ]
            else:
                self._result = []

        def fetchall(self):
            return self._result

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _FakeCursor()

    fake_client = MagicMock()
    fake_client.connection.return_value.__enter__.return_value = _FakeConn()
    fake_client.connection.return_value.__exit__.return_value = False
    original = app.state.metrics_client
    app.state.metrics_client = fake_client
    try:
        resp = api_client.get("/api/graph/gaps?stale_days=7")
    finally:
        app.state.metrics_client = original

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled_postgres"] is True

    # 1. Unplanned: just req-1
    unplanned_ids = [r["id"] for r in data["unplanned_requirements"]]
    assert unplanned_ids == ["req-1"]

    # 2. Specs without artifacts: spec-1 only
    ids_without_artifacts = [s["id"] for s in data["specs_without_artifacts"]]
    assert ids_without_artifacts == ["spec-1"]

    # 3. Failing evals: spec-2 only
    failing_ids = [s["id"] for s in data["specs_failing_evals"]]
    assert failing_ids == ["spec-2"]
    # Passed through from Postgres
    assert data["specs_failing_evals"][0]["score"] == 0.2
    assert data["specs_failing_evals"][0]["metric_name"] == "correctness"

    # 4. Stale requirements: req-3 (its spec-3 was evaluated 30d ago),
    # NOT req-2 (its spec-2 was evaluated just now). req-1 has no specs
    # at all so it doesn't appear in the req_to_specs map and therefore
    # is not reported as stale (it's already in unplanned — avoids
    # double-counting).
    stale_ids = [r["id"] for r in data["stale_requirements"]]
    assert stale_ids == ["req-3"]


def test_graph_gaps_postgres_error_degrades_gracefully(api_client):
    """If the Postgres query raises, we still return the Neo4j-derived
    lists and mark ``enabled_postgres=false`` with a ``postgres_error``
    field — the UI can show a banner and render what it has."""
    mock_session = MagicMock()
    app.state.neo4j_client.session.return_value.__enter__.return_value = mock_session
    app.state.neo4j_client.session.return_value.__exit__.return_value = False

    _install_neo4j_side_effects(
        mock_session,
        unplanned=[],
        specs=[
            {
                "id": "spec-1",
                "title": "X",
                "capability": None,
                "requirement_ids": [],
            }
        ],
        req_count=0,
    )

    fake_client = MagicMock()
    fake_client.connection.side_effect = RuntimeError("connection refused")
    original = app.state.metrics_client
    app.state.metrics_client = fake_client
    try:
        resp = api_client.get("/api/graph/gaps")
    finally:
        app.state.metrics_client = original

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled_postgres"] is False
    assert "connection refused" in data["postgres_error"]
    # Neo4j-only payload is still present
    assert data["unplanned_requirements"] == []


# ── Run History ───────────────────────────────────────────────────────────────


def test_history_when_memory_disabled(api_client):
    app.state.memory_repo = None
    resp = api_client.get("/api/history")

    assert resp.status_code == 200
    assert resp.json()["runs"] == []
    assert "message" in resp.json()


def test_history_returns_runs(api_client):
    app.state.memory_repo.get_run_history.return_value = [
        {"id": "run-1", "status": "success", "pass_rate": 0.9, "duration_seconds": 10.5}
    ]
    resp = api_client.get("/api/history")

    assert resp.status_code == 200
    assert len(resp.json()["runs"]) == 1
    assert resp.json()["runs"][0]["id"] == "run-1"


def test_history_limit_parameter(api_client):
    app.state.memory_repo.get_run_history.return_value = []
    api_client.get("/api/history?limit=5")
    app.state.memory_repo.get_run_history.assert_called_once_with(limit=5)


# ── Memory Search ─────────────────────────────────────────────────────────────


def test_memory_search_requires_keywords(api_client):
    resp = api_client.get("/api/memory/search")
    assert resp.status_code == 422  # missing required query param


def test_memory_search_returns_results(api_client):
    app.state.memory_repo.search_patterns.return_value = [
        {"description": "Use dependency injection", "relevance_score": 0.95}
    ]
    app.state.memory_repo.search_mistakes.return_value = []
    app.state.memory_repo.search_solutions.return_value = []
    app.state.memory_repo.get_strategies.return_value = []

    resp = api_client.get("/api/memory/search?keywords=injection")

    assert resp.status_code == 200
    data = resp.json()
    assert data["keywords"] == "injection"
    assert len(data["results"]) == 1
    assert data["results"][0]["type"] == "pattern"


def test_memory_list_returns_all(api_client):
    """GET /api/memory/list returns memories without requiring keywords."""
    app.state.memory_repo.list_memories.return_value = [
        {"id": "p1", "type": "pattern", "description": "use DI", "relevance_score": 0.9},
        {"id": "m1", "type": "mistake", "description": "missing tests", "relevance_score": 0.7},
        {"id": "s1", "type": "solution", "description": "add fixture", "relevance_score": 0.6},
    ]
    resp = api_client.get("/api/memory/list")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["type"] == "all"
    assert len(data["results"]) == 3
    app.state.memory_repo.list_memories.assert_called_once_with(
        memory_type="all", limit=100
    )


def test_memory_list_filters_by_type(api_client):
    app.state.memory_repo.list_memories.return_value = [
        {"id": "p1", "type": "pattern", "description": "x", "relevance_score": 0.5},
    ]
    resp = api_client.get("/api/memory/list?type=pattern&limit=20")

    assert resp.status_code == 200
    assert resp.json()["type"] == "pattern"
    app.state.memory_repo.list_memories.assert_called_once_with(
        memory_type="pattern", limit=20
    )


def test_memory_list_invalid_type_rejected(api_client):
    """Invalid type values fail validation (Literal enum)."""
    resp = api_client.get("/api/memory/list?type=invalid")
    assert resp.status_code == 422


def test_memory_list_when_disabled(api_client):
    app.state.memory_repo = None
    resp = api_client.get("/api/memory/list")

    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["total"] == 0


def test_memory_search_when_disabled(api_client):
    app.state.memory_repo = None
    resp = api_client.get("/api/memory/search?keywords=auth")

    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_memory_search_type_filter(api_client):
    """The ?type= parameter filters to a single memory type."""
    app.state.memory_repo.search_patterns.return_value = [{"desc": "x"}]
    resp = api_client.get("/api/memory/search?keywords=auth&type=pattern")

    assert resp.status_code == 200
    app.state.memory_repo.search_patterns.assert_called_once_with(keywords="auth")
    # Other search methods should NOT have been called
    app.state.memory_repo.search_mistakes.assert_not_called()


def test_memory_search_invalid_type(api_client):
    """Invalid type= value is rejected by Literal validation."""
    resp = api_client.get("/api/memory/search?keywords=auth&type=invalid")
    assert resp.status_code == 422


# ── Eval Scores ───────────────────────────────────────────────────────────────


def test_eval_root_endpoint_returns_runs(api_client):
    """GET /api/eval returns the hierarchical run → spec → attempts structure."""
    fake_runs = [
        {
            "run_id": "run-1",
            "timestamp": "2026-04-10T12:00:00Z",
            "status": "success",
            "pass_rate": 0.85,
            "spec_count": 1,
            "specs": [
                {
                    "spec_id": "spec-1",
                    "feature_name": "auth",
                    "evals": [
                        {
                            "id": "eval-1",
                            "eval_type": "spec",
                            "overall_score": 0.85,
                            "all_passed": True,
                            "timestamp": "2026-04-10T12:01:00Z",
                            "metrics": [
                                {"name": "Spec Correctness", "score": 0.85, "passed": True, "reason": "ok"},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    app.state.memory_repo.list_evals_by_run.return_value = fake_runs

    resp = api_client.get("/api/eval")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_id"] == "run-1"
    assert data["runs"][0]["specs"][0]["spec_id"] == "spec-1"
    assert data["runs"][0]["specs"][0]["evals"][0]["overall_score"] == pytest.approx(0.85)


def test_eval_root_endpoint_respects_run_limit(api_client):
    app.state.memory_repo.list_evals_by_run.return_value = []
    api_client.get("/api/eval?run_limit=5")
    app.state.memory_repo.list_evals_by_run.assert_called_once_with(run_limit=5)


def test_eval_root_endpoint_when_disabled(api_client):
    app.state.memory_repo = None
    resp = api_client.get("/api/eval")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []


def test_eval_root_endpoint_filters_by_run_id(api_client):
    """``GET /api/eval?run_id=run-X`` filters the response to a single
    run. Used by the per-run popup's Evaluations screen so it doesn't
    have to fetch every run's evals just to display one."""
    fake_runs = [
        {
            "run_id": "run-1",
            "timestamp": "2026-04-10T12:00:00Z",
            "status": "success",
            "pass_rate": 0.85,
            "spec_count": 1,
            "specs": [],
        },
        {
            "run_id": "run-target",
            "timestamp": "2026-04-11T12:00:00Z",
            "status": "partial",
            "pass_rate": 0.5,
            "spec_count": 2,
            "specs": [{"spec_id": "spec-x", "feature_name": "f", "evals": []}],
        },
        {
            "run_id": "run-3",
            "timestamp": "2026-04-12T12:00:00Z",
            "status": "success",
            "pass_rate": 1.0,
            "spec_count": 1,
            "specs": [],
        },
    ]
    app.state.memory_repo.list_evals_by_run.return_value = fake_runs

    resp = api_client.get("/api/eval?run_id=run-target")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_id"] == "run-target"


def test_eval_root_endpoint_filters_by_run_id_no_match(api_client):
    """A run_id that doesn't appear in the memory_repo result returns
    an empty array, NOT an error. The popup uses this to render an
    empty state."""
    app.state.memory_repo.list_evals_by_run.return_value = [
        {
            "run_id": "run-1",
            "timestamp": "2026-04-10T12:00:00Z",
            "status": "success",
            "pass_rate": 0.85,
            "spec_count": 0,
            "specs": [],
        }
    ]

    resp = api_client.get("/api/eval?run_id=run-does-not-exist")
    assert resp.status_code == 200
    assert resp.json()["runs"] == []


def test_eval_root_endpoint_rejects_invalid_run_id(api_client):
    """``run_id`` query param is regex-validated to prevent injection
    or pathological input from reaching the filter."""
    for bad in ["run with space", "run/etc", "run.json", "run;drop"]:
        resp = api_client.get(f"/api/eval?run_id={bad}")
        # FastAPI may translate some characters via URL normalization;
        # anything that reaches the handler must be 422.
        assert resp.status_code in (422, 400, 404), (
            f"bad run_id {bad!r} accepted"
        )


def test_eval_history_empty(api_client):
    app.state.memory_repo.get_eval_history.return_value = []
    resp = api_client.get("/api/eval/spec-abc123")

    assert resp.status_code == 200
    assert resp.json()["spec_id"] == "spec-abc123"
    assert resp.json()["history"] == []


def test_eval_history_with_records(api_client):
    app.state.memory_repo.get_eval_history.return_value = [
        {"overall_score": 0.88, "all_passed": True, "eval_type": "spec", "timestamp": "2026-01-01"},
    ]
    resp = api_client.get("/api/eval/spec-abc123")

    assert resp.status_code == 200
    assert len(resp.json()["history"]) == 1
    assert resp.json()["history"][0]["overall_score"] == pytest.approx(0.88)


def test_eval_spec_id_rejects_injection(api_client):
    """Spec IDs with special characters are rejected by regex."""
    resp = api_client.get("/api/eval/spec'; DROP TABLE specs;--")
    assert resp.status_code == 422


# ── Pipeline Settings ─────────────────────────────────────────────────────────


def test_get_settings_returns_current_pipeline_config(api_client):
    """GET /api/settings returns the current PipelineConfig fields."""
    resp = api_client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "max_parallel_features" in data
    assert "max_parallel_specs" in data
    assert "max_spec_handoffs" in data
    assert "max_codegen_handoffs" in data
    assert "spec_eval_threshold" in data
    assert "output_dir" in data
    # Model fields — llm_model is the main swarm model, eval_model is
    # the DeepEval judge. Both are now editable from the Settings tab
    # instead of being read-only metadata.
    assert "llm_model" in data
    assert "eval_model" in data
    assert isinstance(data["llm_model"], str) and data["llm_model"]
    assert isinstance(data["eval_model"], str) and data["eval_model"]


def test_patch_settings_updates_llm_model(api_client):
    """PATCH /api/settings can change the main LLM model at runtime."""
    from dark_factory.api.app import app

    original = app.state.settings.llm.model
    try:
        resp = api_client.patch(
            "/api/settings", json={"llm_model": "claude-opus-4-6"}
        )
        assert resp.status_code == 200
        assert resp.json()["llm_model"] == "claude-opus-4-6"
        assert app.state.settings.llm.model == "claude-opus-4-6"
    finally:
        app.state.settings.llm.model = original


def test_patch_settings_updates_eval_model_and_propagates_to_metrics_module(
    api_client,
):
    """PATCH /api/settings with a new ``eval_model`` must both update the
    settings object AND call ``set_eval_model`` so DeepEval builders
    created on the next pipeline run pick up the new judge model."""
    from dark_factory.api.app import app
    from dark_factory.evaluation.metrics import get_eval_model

    original_settings = app.state.settings.evaluation.eval_model
    original_module = get_eval_model()
    try:
        resp = api_client.patch(
            "/api/settings", json={"eval_model": "gpt-4.1"}
        )
        assert resp.status_code == 200
        assert resp.json()["eval_model"] == "gpt-4.1"
        # Settings object updated
        assert app.state.settings.evaluation.eval_model == "gpt-4.1"
        # Metrics module global also updated — this is the key
        # invariant. Without it the next run's GEval builders would
        # still construct metrics against the old model.
        assert get_eval_model() == "gpt-4.1"
    finally:
        from dark_factory.evaluation.metrics import set_eval_model

        app.state.settings.evaluation.eval_model = original_settings
        set_eval_model(original_module)


def test_patch_settings_rejects_empty_llm_model(api_client):
    resp = api_client.patch("/api/settings", json={"llm_model": ""})
    assert resp.status_code == 422


def test_patch_settings_rejects_empty_eval_model(api_client):
    resp = api_client.patch("/api/settings", json={"eval_model": ""})
    assert resp.status_code == 422


def test_patch_settings_updates_codegen_handoffs(api_client):
    """PATCH /api/settings can mutate max_codegen_handoffs in place."""
    from dark_factory.api.app import app

    original = app.state.settings.pipeline.max_codegen_handoffs
    try:
        resp = api_client.patch(
            "/api/settings", json={"max_codegen_handoffs": 30}
        )
        assert resp.status_code == 200
        assert resp.json()["max_codegen_handoffs"] == 30
        assert app.state.settings.pipeline.max_codegen_handoffs == 30
    finally:
        app.state.settings.pipeline.max_codegen_handoffs = original


def test_patch_settings_rejects_codegen_handoffs_below_min(api_client):
    """max_codegen_handoffs must be >= 5."""
    resp = api_client.patch(
        "/api/settings", json={"max_codegen_handoffs": 1}
    )
    assert resp.status_code == 422


def test_patch_settings_rejects_codegen_handoffs_above_max(api_client):
    """max_codegen_handoffs must be <= 100."""
    resp = api_client.patch(
        "/api/settings", json={"max_codegen_handoffs": 200}
    )
    assert resp.status_code == 422


def test_patch_settings_updates_in_place(api_client):
    """PATCH /api/settings mutates the live settings object."""
    from dark_factory.api.app import app

    original = app.state.settings.pipeline.max_parallel_specs
    try:
        resp = api_client.patch(
            "/api/settings", json={"max_parallel_specs": 8}
        )
        assert resp.status_code == 200
        assert resp.json()["max_parallel_specs"] == 8
        assert app.state.settings.pipeline.max_parallel_specs == 8
    finally:
        app.state.settings.pipeline.max_parallel_specs = original


def test_patch_settings_partial_update(api_client):
    """Only fields in the body are updated; others stay unchanged."""
    from dark_factory.api.app import app

    original_features = app.state.settings.pipeline.max_parallel_features
    original_threshold = app.state.settings.pipeline.spec_eval_threshold
    try:
        resp = api_client.patch(
            "/api/settings", json={"max_spec_handoffs": 7}
        )
        assert resp.status_code == 200
        # The unchanged fields stay the same
        assert resp.json()["max_parallel_features"] == original_features
        assert resp.json()["spec_eval_threshold"] == original_threshold
        # The changed field is reflected
        assert resp.json()["max_spec_handoffs"] == 7
    finally:
        app.state.settings.pipeline.max_spec_handoffs = 5


def test_patch_settings_rejects_out_of_range(api_client):
    """Out-of-range values fail Pydantic validation with 422."""
    resp = api_client.patch(
        "/api/settings", json={"max_parallel_specs": 999}
    )
    assert resp.status_code == 422


def test_patch_settings_rejects_negative_threshold(api_client):
    resp = api_client.patch(
        "/api/settings", json={"spec_eval_threshold": -0.1}
    )
    assert resp.status_code == 422


def test_patch_settings_rejects_threshold_above_one(api_client):
    resp = api_client.patch(
        "/api/settings", json={"spec_eval_threshold": 1.5}
    )
    assert resp.status_code == 422


# ── Health ────────────────────────────────────────────────────────────────────


def test_health_all_ok(api_client):
    health = {"neo4j": (True, "Connected"), "qdrant": (True, "Connected")}

    with patch("dark_factory.ui.health.check_all", return_value=health):
        resp = api_client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["neo4j"]["ok"] is True
    assert data["qdrant"]["ok"] is True


def test_health_service_down(api_client):
    health = {"neo4j": (False, "Connection refused"), "qdrant": (True, "Connected")}

    with patch("dark_factory.ui.health.check_all", return_value=health):
        resp = api_client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["neo4j"]["ok"] is False
    assert "Connection refused" in data["neo4j"]["message"]


# ── File Watcher ──────────────────────────────────────────────────────────────


def test_watch_stop_when_not_running(api_client):
    app.state.watcher = None
    resp = api_client.post("/api/watch/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_running"


def test_watch_status_when_stopped(api_client):
    app.state.watcher = None
    resp = api_client.get("/api/watch/status")
    assert resp.status_code == 200
    assert resp.json()["running"] is False


def test_watch_start_and_stop(api_client):
    mock_watcher = MagicMock()
    mock_watcher.is_running = True
    mock_watcher.paths = ["./openspec/specs"]

    with patch("dark_factory.ui.watcher.FileWatcher", return_value=mock_watcher):
        resp = api_client.post("/api/watch/start")

    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
    mock_watcher.start.assert_called_once()

    # Now stop it
    app.state.watcher = mock_watcher
    resp = api_client.post("/api/watch/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    mock_watcher.stop.assert_called_once()
    assert app.state.watcher is None


def test_watch_start_already_running(api_client):
    mock_watcher = MagicMock()
    mock_watcher.is_running = True
    mock_watcher.paths = ["./specs"]
    app.state.watcher = mock_watcher

    resp = api_client.post("/api/watch/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_running"
