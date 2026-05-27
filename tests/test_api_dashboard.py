"""Tests for the REST dashboard endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dark_factory.api.app import app


# ── Run-scoped Gap Finder ──────────────────────────────────────────────────────


def test_run_gaps_derives_from_traceability(api_client):
    """GET /api/graph/gaps/{run_id} derives gap analysis from the
    traceability matrix. Exercises:
      - spec-1 has no artifacts → specs_without_artifacts
      - spec-2 has files but eval failed → specs_failing_evals
      - req-1 has overall_status "no_specs" → unimplemented_requirements
    """
    trace_response = {
        "run_id": "run-test-1",
        "rows": [
            {
                "requirement": {"id": "req-1", "title": "Login", "priority": "high"},
                "specs": [],
                "overall_status": "no_specs",
            },
            {
                "requirement": {"id": "req-2", "title": "Dashboard", "priority": "medium"},
                "specs": [
                    {
                        "id": "spec-1",
                        "title": "Dashboard spec",
                        "capability": "dashboard",
                        "files": [],
                        "test_files": [],
                        "eval_scores": {},
                        "all_passed": None,
                    },
                    {
                        "id": "spec-2",
                        "title": "Charts spec",
                        "capability": "dashboard",
                        "files": [{"path": "charts.py"}],
                        "test_files": [],
                        "eval_scores": {"correctness": 0.3},
                        "all_passed": False,
                    },
                ],
                "overall_status": "fail",
            },
        ],
    }

    # Mock get_traceability to return our test data, and Neo4j for
    # the structural gap queries (broken deps, cap islands, episodes).
    mock_session = MagicMock()
    app.state.neo4j_client.session.return_value.__enter__.return_value = mock_session
    app.state.neo4j_client.session.return_value.__exit__.return_value = False
    # Neo4j queries: broken_deps, cap_groups, cap_adjacency, episodes
    mock_session.run.return_value = []

    with patch(
        "dark_factory.api.routes_dashboard.get_traceability",
        return_value=trace_response,
    ):
        resp = api_client.get("/api/graph/gaps/run-test-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run-test-1"

    # spec-1 has no files → specs_without_artifacts
    no_artifacts_ids = [s["id"] for s in data["specs_without_artifacts"]]
    assert "spec-1" in no_artifacts_ids

    # spec-2 has all_passed=False → specs_failing_evals
    failing_ids = [s["id"] for s in data["specs_failing_evals"]]
    assert failing_ids == ["spec-2"]
    assert data["specs_failing_evals"][0]["eval_scores"]["correctness"] == 0.3

    # req-1 has overall_status "no_specs" → unimplemented
    unimpl_ids = [r["id"] for r in data["unimplemented_requirements"]]
    assert unimpl_ids == ["req-1"]

    # Totals
    assert data["totals"]["requirements"] == 2
    assert data["totals"]["specs"] == 2


def test_run_gaps_clean_run_returns_no_gaps(api_client):
    """A run where every spec has files and passing evals returns
    empty gap lists."""
    trace_response = {
        "run_id": "run-clean",
        "rows": [
            {
                "requirement": {"id": "req-1", "title": "Login", "priority": "high"},
                "specs": [
                    {
                        "id": "spec-1",
                        "title": "Login spec",
                        "capability": "auth",
                        "files": [{"path": "auth.py"}],
                        "test_files": [{"path": "test_auth.py"}],
                        "eval_scores": {"correctness": 0.95},
                        "all_passed": True,
                    },
                ],
                "overall_status": "pass",
            },
        ],
    }

    mock_session = MagicMock()
    app.state.neo4j_client.session.return_value.__enter__.return_value = mock_session
    app.state.neo4j_client.session.return_value.__exit__.return_value = False
    mock_session.run.return_value = []

    with patch(
        "dark_factory.api.routes_dashboard.get_traceability",
        return_value=trace_response,
    ):
        resp = api_client.get("/api/graph/gaps/run-clean")

    assert resp.status_code == 200
    data = resp.json()
    assert data["specs_without_artifacts"] == []
    assert data["specs_failing_evals"] == []
    assert data["unimplemented_requirements"] == []
    assert data["broken_dependencies"] == []
    assert data["capability_islands"] == []
    assert data["missing_episodes"] == []


def test_run_gaps_invalid_run_id_rejected(api_client):
    """Run IDs with special characters are rejected by regex."""
    resp = api_client.get("/api/graph/gaps/run;drop")
    assert resp.status_code == 422


# ── Requirements Refinery ──────────────────────────────────────────────────────


def test_patch_requirement_updates_fields(api_client):
    """PATCH /api/graph/requirements/{id} merges fields and keeps the ID."""
    from dark_factory.models.domain import Priority, Requirement

    existing = Requirement(
        id="req-abc",
        title="Old Title",
        description="Old desc",
        source_file="reqs.md",
        priority=Priority.MEDIUM,
        tags=["auth"],
    )
    mock_repo = MagicMock()
    mock_repo.get_requirement.return_value = existing
    mock_repo.upsert_requirement.return_value = None

    with patch(
        "dark_factory.graph.repository.GraphRepository",
        return_value=mock_repo,
    ):
        resp = api_client.patch(
            "/api/graph/requirements/req-abc",
            json={"title": "New Title", "priority": "high"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "req-abc"
    assert data["title"] == "New Title"
    assert data["priority"] == "high"
    # Description and source_file preserved from existing
    assert data["description"] == "Old desc"
    assert data["source_file"] == "reqs.md"


def test_patch_requirement_not_found(api_client):
    """PATCH returns 404 for a nonexistent requirement."""
    mock_repo = MagicMock()
    mock_repo.get_requirement.return_value = None

    with patch(
        "dark_factory.graph.repository.GraphRepository",
        return_value=mock_repo,
    ):
        resp = api_client.patch(
            "/api/graph/requirements/req-nope",
            json={"title": "Nope"},
        )

    assert resp.status_code == 404


def test_export_requirements_json(api_client):
    """GET /api/graph/requirements/export returns a JSON array."""
    from dark_factory.models.domain import Priority, Requirement

    mock_repo = MagicMock()
    mock_repo.get_all_requirements.return_value = [
        Requirement(
            id="req-1",
            title="Login",
            description="User can log in",
            source_file="reqs.md",
            priority=Priority.HIGH,
            tags=["auth"],
        ),
    ]

    with patch(
        "dark_factory.graph.repository.GraphRepository",
        return_value=mock_repo,
    ):
        resp = api_client.get("/api/graph/requirements/export")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "req-1"
    assert data[0]["title"] == "Login"


def test_export_refinery_zip(api_client):
    """POST /api/refinery/export returns a ZIP with report + requirement files."""
    import io
    import zipfile

    payload = {
        "summary": "Refined 2 requirements",
        "pass_summaries": ["Pass 1 done"],
        "refined_requirements": [
            {
                "id": "req-1",
                "original_title": "Login",
                "original_description": "User login",
                "title": "User Authentication",
                "description": "Detailed login with MFA",
                "priority": "high",
                "tags": ["auth"],
                "relationships": [
                    {"target_id": "req-2", "type": "depends_on", "rationale": "needs session"},
                ],
                "suggested_specs": [
                    {
                        "title": "Login Flow",
                        "capability": "auth",
                        "description": "Main login",
                        "acceptance_criteria": ["user can log in"],
                    },
                ],
                "changes": ["Expanded description"],
                "pass_context": "Pass 4",
            },
            {
                "id": "req-2",
                "original_title": "Session",
                "original_description": "Session mgmt",
                "title": "Session Management",
                "description": "Handle user sessions",
                "priority": "medium",
                "tags": ["auth"],
                "relationships": [],
                "suggested_specs": [],
                "changes": [],
                "pass_context": "",
            },
        ],
        "suggested_memories": [],
        "new_relationships_count": 1,
        "requirements_modified_count": 2,
        "requirements_unchanged_count": 0,
        "source_run_id": "run-test",
        "methodology": "5-pass analysis",
        "evidence_summary": "eval failures drove changes",
        "risk_areas": ["session handling"],
    }

    resp = api_client.post("/api/refinery/export", json=payload)
    assert resp.status_code == 200
    assert "application/zip" in resp.headers.get("content-type", "")

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()

    # Report file exists
    assert "refinery-run-test/REPORT.md" in names
    report_md = zf.read("refinery-run-test/REPORT.md").decode()
    assert "# Requirements Refinery Report" in report_md
    assert "Refined 2 requirements" in report_md
    assert "5-pass analysis" in report_md
    assert "session handling" in report_md

    # Individual requirement files (2 requirements = 2 files)
    assert "refinery-run-test/requirements/req-1.md" in names
    assert "refinery-run-test/requirements/req-2.md" in names

    req1_md = zf.read("refinery-run-test/requirements/req-1.md").decode()
    assert "# User Authentication" in req1_md
    assert "depends_on" in req1_md
    assert "Login Flow" in req1_md

    req2_md = zf.read("refinery-run-test/requirements/req-2.md").decode()
    assert "# Session Management" in req2_md


# ── Refinery Persistence ───────────────────────────────────────────────────────


def test_refinery_history_lists_results(api_client, tmp_path):
    """GET /api/refinery/history returns saved results."""
    from dark_factory.storage.backend import get_storage

    storage = get_storage()
    # Write a fake result
    metadata = {
        "id": "refinery-20260418-120000-abcd",
        "timestamp": "2026-04-18T12:00:00Z",
        "source_run_id": "run-test",
        "source_mode": "run",
        "requirements_count": 5,
        "requirements_modified": 3,
        "requirements_unchanged": 2,
        "relationships_count": 2,
        "suggested_memories_count": 1,
        "duration_seconds": 100.0,
    }
    import json as _json

    storage.write_text(
        "refinery/refinery-20260418-120000-abcd/metadata.json",
        _json.dumps(metadata),
    )

    resp = api_client.get("/api/refinery/history")
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert any(r["id"] == "refinery-20260418-120000-abcd" for r in results)

    # Cleanup
    storage.delete_prefix("refinery/refinery-20260418-120000-abcd/")


def test_refinery_load_saved_result(api_client):
    """GET /api/refinery/{result_id} loads a saved response."""
    from dark_factory.storage.backend import get_storage

    import json as _json

    storage = get_storage()
    response_data = {
        "summary": "Test refinement",
        "refined_requirements": [],
        "suggested_memories": [],
    }
    storage.write_text(
        "refinery/refinery-20260418-130000-ef01/response.json",
        _json.dumps(response_data),
    )

    resp = api_client.get("/api/refinery/refinery-20260418-130000-ef01")
    assert resp.status_code == 200
    assert resp.json()["summary"] == "Test refinement"

    # Cleanup
    storage.delete_prefix("refinery/refinery-20260418-130000-ef01/")


def test_refinery_load_nonexistent_returns_404(api_client):
    resp = api_client.get("/api/refinery/refinery-nonexistent-0000")
    assert resp.status_code == 404


def test_refinery_delete_result(api_client):
    """DELETE /api/refinery/{result_id} removes from storage."""
    from dark_factory.storage.backend import get_storage

    import json as _json

    storage = get_storage()
    storage.write_text(
        "refinery/refinery-20260418-140000-9999/metadata.json",
        _json.dumps({"id": "refinery-20260418-140000-9999"}),
    )

    resp = api_client.delete("/api/refinery/refinery-20260418-140000-9999")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "refinery-20260418-140000-9999"

    # Verify it's gone
    assert storage.list_keys("refinery/refinery-20260418-140000-9999/") == []


# ── Memory Dedup Check ────────────────────────────────────────────────────────


def test_memory_check_duplicate_returns_valid_shape(api_client):
    """POST /api/memory/check-duplicate returns the expected response shape."""
    resp = api_client.post(
        "/api/memory/check-duplicate",
        json={
            "type": "strategy",
            "description": "xyzzy_unique_test_string_42_" + str(id(api_client)),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # Should have all expected keys regardless of match/no-match
    assert "is_duplicate" in data
    assert "existing_id" in data
    assert "existing_description" in data
    assert "similarity" in data


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
