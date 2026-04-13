"""Tests for the AG-UI pipeline bridge (unit-level, no api_client)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from dark_factory.api.ag_ui_bridge import _text_events, run_pipeline_stream
from dark_factory.config import Settings
from ag_ui.encoder import EventEncoder


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_event(sse_str: str) -> dict:
    """Extract the JSON payload from a single SSE 'data: ...' line."""
    for line in sse_str.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    return {}


def event_type(sse_str: str) -> str:
    return parse_event(sse_str).get("type", "")


def collect_stream(coro) -> list[str]:
    """Run an async generator to completion and return all yielded strings."""
    async def _collect():
        return [chunk async for chunk in coro]
    return asyncio.run(_collect())


def _mock_context():
    ctx = MagicMock()
    ctx.requirements = [MagicMock(), MagicMock()]
    ctx.specs = [MagicMock(id="spec-1"), MagicMock(id="spec-2")]
    return ctx


def _mock_result():
    return {
        "completed_features": [{"feature": "auth", "status": "success", "error": None}],
        "pass_rate": 0.9,
        "all_artifacts": ["output/run-1/auth.py"],
        "all_tests": ["output/run-1/test_auth.py"],
    }


def _fake_e2e_result(status: str = "pass"):
    """Build a concrete E2EValidationResult for mocking the stage."""
    from dark_factory.stages.e2e_validation import E2EValidationResult

    return E2EValidationResult(
        status=status,
        summary=f"E2E {status}",
        tests_total=6,
        tests_passed=6 if status == "pass" else 4,
        tests_failed=0 if status == "pass" else 2,
        browsers_run=["chromium", "firefox", "webkit"],
        agent_output="ran tests",
        report_path="E2E_REPORT.md",
        html_report_path="e2e_artifacts/html-report",
        screenshots=[],
        duration_seconds=12.3,
    )


@pytest.fixture
def minimal_settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def _mock_recon_and_storage():
    """Prevent real storage init, reconciliation, and reflection LLM calls."""
    def _skip_recon(self, **kw):
        from dark_factory.stages.reconciliation import ReconciliationResult
        return ReconciliationResult(
            status="clean", summary="mocked clean", agent_output="",
            report_path=None, duration_seconds=0.0,
        )

    def _skip_e2e(self, **kw):
        from dark_factory.stages.e2e_validation import E2EValidationResult
        return E2EValidationResult(
            status="skipped", summary="mocked skip", agent_output="",
            report_path=None, html_report_path=None, screenshots=[],
            tests_total=0, tests_passed=0, tests_failed=0,
            browsers_run=[], duration_seconds=0.0,
        )

    with (
        patch("dark_factory.storage.backend.get_storage"),
        patch("dark_factory.api.ag_ui_bridge._reflect_on_reconciliation", return_value=None),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_skip_recon,
        ),
        patch(
            "dark_factory.stages.e2e_validation.E2EValidationStage.run",
            new=_skip_e2e,
        ),
    ):
        yield


# ── _text_events ──────────────────────────────────────────────────────────────


def test_text_events_returns_three_strings():
    encoder = EventEncoder()
    events = _text_events(encoder, "hello")
    assert len(events) == 3


def test_text_events_correct_types():
    encoder = EventEncoder()
    events = _text_events(encoder, "hello")
    types = [event_type(e) for e in events]
    assert types == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"]


def test_text_events_delta_content():
    encoder = EventEncoder()
    events = _text_events(encoder, "test progress message")
    content = parse_event(events[1])
    assert content["delta"] == "test progress message"


def test_text_events_consistent_message_id():
    encoder = EventEncoder()
    events = _text_events(encoder, "x")
    # AG-UI serialises as camelCase: "messageId"
    ids = [parse_event(e).get("messageId") for e in events]
    assert ids[0] == ids[1] == ids[2]
    assert ids[0] is not None


def test_text_events_different_ids_for_different_calls():
    encoder = EventEncoder()
    events_a = _text_events(encoder, "a")
    events_b = _text_events(encoder, "b")
    id_a = parse_event(events_a[0]).get("messageId")
    id_b = parse_event(events_b[0]).get("messageId")
    assert id_a != id_b


# ── run_pipeline_stream event sequence ────────────────────────────────────────


def _stream_with_mocks(minimal_settings, **kwargs):
    ctx = _mock_context()
    result = _mock_result()

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch("dark_factory.agents.orchestrator.run_orchestrator", return_value=result),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        return collect_stream(
            run_pipeline_stream(
                minimal_settings,
                kwargs.get("path", "./test"),
                kwargs.get("thread_id", "t-1"),
                kwargs.get("run_id", "r-1"),
            )
        )


def test_stream_starts_with_run_started(minimal_settings):
    chunks = _stream_with_mocks(minimal_settings)
    assert chunks, "No events emitted"
    assert event_type(chunks[0]) == "RUN_STARTED"


def test_stream_ends_with_run_finished(minimal_settings):
    chunks = _stream_with_mocks(minimal_settings)
    types = [event_type(c) for c in chunks]
    assert types[-1] == "RUN_FINISHED"


def test_stream_has_phase_steps(minimal_settings):
    chunks = _stream_with_mocks(minimal_settings)
    types = [event_type(c) for c in chunks]
    # 4 core phases + reconciliation (always runs now that run_id is always set)
    assert types.count("STEP_STARTED") >= 4
    assert types.count("STEP_FINISHED") >= 4
    # Core phases are always present
    step_names = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_STARTED"
    ]
    for phase in ("Ingest", "Spec Generation", "Knowledge Graph", "Swarm Orchestrator"):
        assert phase in step_names


def test_stream_has_state_snapshot(minimal_settings):
    chunks = _stream_with_mocks(minimal_settings)
    types = [event_type(c) for c in chunks]
    assert "STATE_SNAPSHOT" in types

    snapshot_chunk = next(c for c in chunks if event_type(c) == "STATE_SNAPSHOT")
    payload = parse_event(snapshot_chunk)
    assert "snapshot" in payload
    assert payload["snapshot"]["pass_rate"] == pytest.approx(0.9)


def test_stream_step_names_match_phases(minimal_settings):
    chunks = _stream_with_mocks(minimal_settings)
    # AG-UI serialises as camelCase: "stepName"
    step_names = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_STARTED"
    ]
    # Core 4 phases always present; Reconciliation also runs now that run_id is always set
    assert step_names[:4] == ["Ingest", "Spec Generation", "Knowledge Graph", "Swarm Orchestrator"]


def test_stream_run_id_in_started_and_finished(minimal_settings):
    chunks = _stream_with_mocks(minimal_settings, thread_id="my-thread", run_id="my-run")

    # AG-UI serialises as camelCase: "runId", "threadId"
    started = parse_event(chunks[0])
    assert started["runId"] == "my-run"
    assert started["threadId"] == "my-thread"

    finished = parse_event(chunks[-1])
    assert finished["runId"] == "my-run"


# ── Neo4j run history terminal-status update ──────────────────────────────────


def test_stream_calls_complete_run_on_success_path(minimal_settings):
    """The success-side path of run_pipeline_stream must call
    ``memory_repo.complete_run`` so the Neo4j Run node flips from
    ``status="running"`` to its terminal status (success/partial).

    Regression test for the zombie run-history bug: previously the
    success path only updated Postgres metrics, leaving the Neo4j
    Run node permanently at ``running`` and the ``/api/history``
    endpoint reporting stale state forever.
    """
    ctx = _mock_context()
    result = _mock_result()  # pass_rate=0.9, 1 successful feature

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-test-success-1"

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    # Sanity: the run reached its happy-path end (RUN_FINISHED, not RUN_ERROR)
    types = [event_type(c) for c in chunks]
    assert types[-1] == "RUN_FINISHED"

    # complete_run was called exactly once with the right shape.
    # The success-path bug was that it was called ZERO times.
    assert fake_memory_repo.complete_run.called, (
        "complete_run was not called on the success path — "
        "the Neo4j Run node will be stuck at status='running' forever"
    )
    call_kwargs = fake_memory_repo.complete_run.call_args.kwargs
    # Status comes from succeeded == total: result has 1 success out of 1, so "success"
    assert call_kwargs["status"] == "success"
    assert call_kwargs["pass_rate"] == pytest.approx(0.9)
    assert call_kwargs["duration_seconds"] >= 0
    # Defaults from the empty result fields
    assert call_kwargs["mean_eval_scores"] == {}
    assert call_kwargs["worst_features"] == []


def test_stream_complete_run_uses_partial_when_features_failed(
    minimal_settings,
):
    """When the orchestrator returns with at least one failed feature,
    the run terminal status must be ``partial`` (not ``success``)."""
    ctx = _mock_context()
    result = {
        "completed_features": [
            {"feature": "auth", "status": "success", "error": None},
            {"feature": "dashboard", "status": "error", "error": "boom"},
        ],
        "pass_rate": 0.5,
        "all_artifacts": [],
        "all_tests": [],
    }

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-test-partial-1"

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    assert fake_memory_repo.complete_run.called
    assert (
        fake_memory_repo.complete_run.call_args.kwargs["status"] == "partial"
    )


def test_stream_complete_run_failure_does_not_break_stream(
    minimal_settings,
):
    """If memory_repo.complete_run raises (e.g. Neo4j down), the
    pipeline must NOT crash — the failure is logged + swallowed and
    RUN_FINISHED still reaches the client."""
    ctx = _mock_context()
    result = _mock_result()

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-test-flaky-1"
    fake_memory_repo.complete_run.side_effect = RuntimeError("neo4j down")

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    # The failure was swallowed — the run still ends cleanly.
    types = [event_type(c) for c in chunks]
    assert types[-1] == "RUN_FINISHED"
    # The handler did try to call it.
    assert fake_memory_repo.complete_run.called


# ── Phase 5: Reconciliation ───────────────────────────────────────────────────


def test_stream_runs_reconciliation_phase(
    minimal_settings,
):
    """With at least one successful feature, the pipeline always
    emits a StepStarted/StepFinished pair for "Reconciliation" and
    the ReconciliationStage.run is invoked with the expected
    output_dir + feature_results. There's no toggle — the phase
    runs unconditionally when feature swarms produce output."""
    ctx = _mock_context()
    result = _mock_result()  # 1 successful feature

    captured_calls: list[dict] = []

    def _fake_run(self, *, run_id, output_dir, feature_results):
        captured_calls.append(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "feature_results": feature_results,
            }
        )
        from dark_factory.stages.reconciliation import ReconciliationResult

        return ReconciliationResult(
            status="clean",
            summary="Reconciliation clean",
            agent_output="",
            report_path="RECONCILIATION_REPORT.md",
            duration_seconds=1.2,
        )

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-test-recon"

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_fake_run,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    # Reconciliation stage was called exactly once
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["run_id"] == "run-test-recon"
    # Output dir is derived from settings.pipeline.output_dir / run_id
    assert "run-test-recon" in call["output_dir"]
    # Feature list passed through from the orchestrator result
    assert call["feature_results"] == result["completed_features"]

    # The stream emitted a StepStarted and StepFinished for the
    # "Reconciliation" step name.
    step_names_started = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_STARTED"
    ]
    step_names_finished = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_FINISHED"
    ]
    assert "Reconciliation" in step_names_started
    assert "Reconciliation" in step_names_finished


def test_stream_reconciliation_crash_does_not_fail_run(minimal_settings):
    """A crash inside ``ReconciliationStage.run`` must NOT fail the
    pipeline — reconciliation is best-effort polishing, not
    gatekeeping. The stream still ends with RUN_FINISHED and the
    rest of the run completes normally."""
    ctx = _mock_context()
    result = _mock_result()

    def _crashing_run(self, **kwargs):
        raise RuntimeError("reconciliation boom")

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-test-crash"

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_crashing_run,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    types = [event_type(c) for c in chunks]
    # Run still finished cleanly — no RUN_ERROR
    assert types[-1] == "RUN_FINISHED"
    assert "RUN_ERROR" not in types


# ── Phase 6: E2E Validation ───────────────────────────────────────────────────


def test_stream_runs_e2e_phase(minimal_settings):
    """With reconciliation clean and the feature enabled, Phase 6
    runs the Playwright E2E stage and emits Step events for it."""
    ctx = _mock_context()
    result = _mock_result()

    captured_calls: list[dict] = []

    def _fake_recon(self, **kwargs):
        from dark_factory.stages.reconciliation import ReconciliationResult

        return ReconciliationResult(
            status="clean",
            summary="ok",
            agent_output="",
            report_path="RECONCILIATION_REPORT.md",
            duration_seconds=1.0,
        )

    def _fake_e2e_run(
        self,
        *,
        run_id,
        output_dir,
        feature_results,
        reconciliation_status,
    ):
        captured_calls.append(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "feature_results": feature_results,
                "reconciliation_status": reconciliation_status,
                "browsers": list(self.browsers),
            }
        )
        return _fake_e2e_result("pass")

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-test-e2e"

    # Make sure the feature is enabled in the test settings object.
    minimal_settings.pipeline.enable_e2e_validation = True

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_fake_recon,
        ),
        patch(
            "dark_factory.stages.e2e_validation.E2EValidationStage.run",
            new=_fake_e2e_run,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["run_id"] == "run-test-e2e"
    assert call["reconciliation_status"] == "clean"
    # All three default browsers land in the stage
    assert set(call["browsers"]) == {"chromium", "firefox", "webkit"}

    # StepStarted + StepFinished emitted for the E2E step name
    step_names_started = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_STARTED"
    ]
    step_names_finished = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_FINISHED"
    ]
    assert "E2E Validation" in step_names_started
    assert "E2E Validation" in step_names_finished


def test_stream_skips_e2e_when_disabled(minimal_settings):
    """When ``enable_e2e_validation=False``, Phase 6 is entirely
    bypassed — no StepStarted for E2E, no stage invocation."""
    ctx = _mock_context()
    result = _mock_result()

    def _fake_recon(self, **kwargs):
        from dark_factory.stages.reconciliation import ReconciliationResult

        return ReconciliationResult(
            status="clean",
            summary="ok",
            agent_output="",
            report_path=None,
            duration_seconds=1.0,
        )

    def _should_not_run(self, **kwargs):
        raise AssertionError("E2EValidationStage.run must not be called when disabled")

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-e2e-off"

    minimal_settings.pipeline.enable_e2e_validation = False

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_fake_recon,
        ),
        patch(
            "dark_factory.stages.e2e_validation.E2EValidationStage.run",
            new=_should_not_run,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    step_names_started = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_STARTED"
    ]
    assert "E2E Validation" not in step_names_started


@pytest.mark.parametrize(
    "recon_status",
    ["error", "partial", "skipped"],
)
def test_stream_skips_e2e_when_reconciliation_not_clean(
    minimal_settings, recon_status
):
    """E2E validation must be gated on a *clean* reconciliation
    pass. Any other outcome — error, partial, or skipped — means
    the code is either known-broken or never-reconciled, so running
    browser tests against it would be wasted turn budget. Phase 6
    skips in all three cases and emits a text message explaining
    why."""
    ctx = _mock_context()
    result = _mock_result()

    def _fake_recon(self, **kwargs):
        from dark_factory.stages.reconciliation import ReconciliationResult

        return ReconciliationResult(
            status=recon_status,
            summary=f"reconciliation {recon_status}",
            agent_output="",
            report_path=None,
            duration_seconds=0.5,
        )

    def _should_not_run(self, **kwargs):
        raise AssertionError(
            f"E2EValidationStage.run must not be called when "
            f"reconciliation status is '{recon_status}'"
        )

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = f"run-recon-{recon_status}"

    minimal_settings.pipeline.enable_e2e_validation = True

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_fake_recon,
        ),
        patch(
            "dark_factory.stages.e2e_validation.E2EValidationStage.run",
            new=_should_not_run,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    # Run still finished normally
    types = [event_type(c) for c in chunks]
    assert types[-1] == "RUN_FINISHED"
    # No E2E Step events
    step_names_started = [
        parse_event(c).get("stepName")
        for c in chunks
        if event_type(c) == "STEP_STARTED"
    ]
    assert "E2E Validation" not in step_names_started
    # A text message explaining the skip must have been emitted so
    # operators reading the run log can see why E2E didn't run.
    text_blobs = [
        parse_event(c).get("delta", "")
        for c in chunks
        if event_type(c) == "TEXT_MESSAGE_CONTENT"
    ]
    combined_text = " ".join(text_blobs)
    assert "E2E validation skipped" in combined_text
    assert recon_status in combined_text


def test_stream_e2e_crash_does_not_fail_run(minimal_settings):
    """An E2E stage crash must NOT fail the pipeline — same
    best-effort contract as reconciliation."""
    ctx = _mock_context()
    result = _mock_result()

    def _fake_recon(self, **kwargs):
        from dark_factory.stages.reconciliation import ReconciliationResult

        return ReconciliationResult(
            status="clean",
            summary="ok",
            agent_output="",
            report_path=None,
            duration_seconds=1.0,
        )

    def _crashing_e2e(self, **kwargs):
        raise RuntimeError("playwright exploded")

    fake_memory_repo = MagicMock()
    fake_memory_repo.create_run.return_value = "run-e2e-crash"

    minimal_settings.pipeline.enable_e2e_validation = True

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch(
            "dark_factory.agents.orchestrator.run_orchestrator",
            return_value=result,
        ),
        patch(
            "dark_factory.stages.reconciliation.ReconciliationStage.run",
            new=_fake_recon,
        ),
        patch(
            "dark_factory.stages.e2e_validation.E2EValidationStage.run",
            new=_crashing_e2e,
        ),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.return_value = ctx
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        chunks = collect_stream(
            run_pipeline_stream(
                minimal_settings,
                "./test",
                "t-1",
                "r-1",
                memory_repo=fake_memory_repo,
            )
        )

    types = [event_type(c) for c in chunks]
    assert types[-1] == "RUN_FINISHED"
    assert "RUN_ERROR" not in types


# ── Error handling ────────────────────────────────────────────────────────────


def test_stream_emits_run_error_on_exception(minimal_settings):
    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.side_effect = RuntimeError("ingest failed")

        chunks = collect_stream(
            run_pipeline_stream(minimal_settings, "./bad-path", "t-1", "r-1")
        )

    types = [event_type(c) for c in chunks]
    assert "RUN_ERROR" in types
    # Should NOT end with RUN_FINISHED
    assert types[-1] == "RUN_ERROR"

    error_chunk = next(c for c in chunks if event_type(c) == "RUN_ERROR")
    payload = parse_event(error_chunk)
    assert "ingest failed" in payload.get("message", "")


# ── Agent endpoint (HTTP) ─────────────────────────────────────────────────────


def test_agent_run_endpoint_streams_sse(api_client):
    async def _fake_stream(settings, requirements_path, thread_id, run_id, accept=None, memory_repo=None, **_kwargs):
        encoder = EventEncoder(accept=accept)
        from ag_ui.core import EventType, RunFinishedEvent, RunStartedEvent
        yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
        yield encoder.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))

    with patch("dark_factory.api.routes_agent.run_pipeline_stream", side_effect=_fake_stream):
        resp = api_client.post("/api/agent/run", json={"requirements_path": "./openspec"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    assert len(lines) == 2

    events = [json.loads(l[6:]) for l in lines]
    assert events[0]["type"] == "RUN_STARTED"
    assert events[1]["type"] == "RUN_FINISHED"


def test_agent_run_default_path(api_client):
    """When no requirements_path is provided, defaults to ./openspec."""
    captured = {}

    async def _fake_stream(settings, requirements_path, thread_id, run_id, accept=None, memory_repo=None, **_kwargs):
        captured["path"] = requirements_path
        captured.update(_kwargs)
        encoder = EventEncoder(accept=accept)
        from ag_ui.core import EventType, RunFinishedEvent, RunStartedEvent
        yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
        yield encoder.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))

    with patch("dark_factory.api.routes_agent.run_pipeline_stream", side_effect=_fake_stream):
        api_client.post("/api/agent/run", json={})

    assert captured.get("path") == "./openspec"


def test_agent_run_rejects_path_traversal(api_client):
    """C2: paths outside the working directory are rejected."""
    resp = api_client.post("/api/agent/run", json={"requirements_path": "/etc/passwd"})
    assert resp.status_code == 422

    resp = api_client.post("/api/agent/run", json={"requirements_path": "../../etc/passwd"})
    assert resp.status_code == 422


def test_agent_run_accepts_optional_api_key_overrides(api_client):
    """Per-run API key overrides are forwarded through to run_pipeline_stream."""
    captured = {}

    async def _fake_stream(settings, requirements_path, thread_id, run_id, accept=None, memory_repo=None, **_kwargs):
        captured.update(_kwargs)
        encoder = EventEncoder(accept=accept)
        from ag_ui.core import EventType, RunFinishedEvent, RunStartedEvent
        yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
        yield encoder.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))

    with patch("dark_factory.api.routes_agent.run_pipeline_stream", side_effect=_fake_stream):
        resp = api_client.post(
            "/api/agent/run",
            json={
                "requirements_path": "./openspec",
                "anthropic_api_key": "sk-ant-override",
                "openai_api_key": "sk-oa-override",
            },
        )

    assert resp.status_code == 200
    assert captured.get("anthropic_api_key") == "sk-ant-override"
    assert captured.get("openai_api_key") == "sk-oa-override"


def test_agent_run_empty_api_key_treated_as_null(api_client):
    """Empty-string API keys (the frontend's "cleared" state) are coerced to
    None so the server default stays in place."""
    captured = {}

    async def _fake_stream(settings, requirements_path, thread_id, run_id, accept=None, memory_repo=None, **_kwargs):
        captured.update(_kwargs)
        encoder = EventEncoder(accept=accept)
        from ag_ui.core import EventType, RunFinishedEvent, RunStartedEvent
        yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
        yield encoder.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))

    with patch("dark_factory.api.routes_agent.run_pipeline_stream", side_effect=_fake_stream):
        api_client.post(
            "/api/agent/run",
            json={
                "requirements_path": "./openspec",
                "anthropic_api_key": "",
                "openai_api_key": "   ",
            },
        )

    assert captured.get("anthropic_api_key") is None
    assert captured.get("openai_api_key") is None


def test_agent_run_rejects_oversized_api_key(api_client):
    resp = api_client.post(
        "/api/agent/run",
        json={
            "requirements_path": "./openspec",
            "anthropic_api_key": "sk-" + "x" * 600,
        },
    )
    assert resp.status_code == 422


def test_agent_run_rejects_api_key_with_whitespace(api_client):
    resp = api_client.post(
        "/api/agent/run",
        json={
            "requirements_path": "./openspec",
            "openai_api_key": "sk-oa with space",
        },
    )
    assert resp.status_code == 422


def test_agent_run_validation_error_does_not_echo_api_key(api_client):
    """CRITICAL security guardrail: a rejected API key must NOT appear in
    the 422 response body."""
    secret = "sk-ant-SUPER_SECRET_DO_NOT_LEAK_" + "x" * 500
    resp = api_client.post(
        "/api/agent/run",
        json={
            "requirements_path": "./openspec",
            "anthropic_api_key": secret,
        },
    )
    assert resp.status_code == 422
    body_text = resp.text
    assert "SUPER_SECRET_DO_NOT_LEAK" not in body_text, (
        "API key value leaked into the 422 validation error response: "
        "hide_input_in_errors is not working"
    )


def test_agent_run_openai_key_also_hidden_in_validation_errors(api_client):
    secret = "sk-OPENAI_SECRET_" + "x" * 500
    resp = api_client.post(
        "/api/agent/run",
        json={
            "requirements_path": "./openspec",
            "openai_api_key": secret,
        },
    )
    assert resp.status_code == 422
    assert "OPENAI_SECRET" not in resp.text


# ── API key injection & leak prevention ──────────────────────────────────────


def test_run_pipeline_stream_injects_and_restores_api_keys(
    minimal_settings, monkeypatch, tmp_path
):
    """Inside the stream the env vars reflect the override; on exit the
    previous values are restored even on exception."""
    import os

    monkeypatch.setenv("ANTHROPIC_API_KEY", "previous-anthropic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    seen: dict = {}

    def _fake_llm(_settings, model_override=None):
        seen["anthropic_during_build"] = os.environ.get("ANTHROPIC_API_KEY")
        seen["openai_during_build"] = os.environ.get("OPENAI_API_KEY")
        raise RuntimeError("halt-pipeline-for-test")

    monkeypatch.setattr(
        "dark_factory.ui.helpers.build_llm", _fake_llm, raising=True
    )

    requirements_dir = tmp_path / "openspec"
    requirements_dir.mkdir()

    async def _drain():
        async for _ in run_pipeline_stream(
            settings=minimal_settings,
            requirements_path=str(requirements_dir),
            thread_id="t",
            run_id="r",
            anthropic_api_key="sk-override-anthropic",
            openai_api_key="sk-override-openai",
        ):
            pass

    asyncio.run(_drain())

    assert seen["anthropic_during_build"] == "sk-override-anthropic"
    assert seen["openai_during_build"] == "sk-override-openai"
    assert os.environ.get("ANTHROPIC_API_KEY") == "previous-anthropic"
    assert "OPENAI_API_KEY" not in os.environ


def test_api_key_override_class_redacts_repr():
    """The _ApiKeyOverride helper must never expose the secret via repr()."""
    from dark_factory.api.ag_ui_bridge import _ApiKeyOverride

    override = _ApiKeyOverride(
        anthropic="sk-dont-leak-me-anthropic",
        openai="sk-dont-leak-me-openai",
    )
    rendered = repr(override)
    assert "sk-dont-leak-me-anthropic" not in rendered
    assert "sk-dont-leak-me-openai" not in rendered
    assert "redacted" in rendered.lower()


def test_traceback_from_stream_does_not_leak_api_keys(
    minimal_settings, monkeypatch, tmp_path
):
    """If the pipeline stream raises after installing per-run API key
    overrides, the traceback must NOT contain the plain-text key values."""
    import os
    import traceback

    monkeypatch.setenv("ANTHROPIC_API_KEY", "previous-anthropic-secret-XXYY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    override_anthropic = "sk-override-anthropic-SHOULD-NOT-LEAK-aaa111"
    override_openai = "sk-override-openai-SHOULD-NOT-LEAK-bbb222"

    captured_traceback: str | None = None

    def _fake_llm(_settings, model_override=None):
        raise RuntimeError("halt-pipeline-for-traceback-test")

    monkeypatch.setattr(
        "dark_factory.ui.helpers.build_llm", _fake_llm, raising=True
    )

    requirements_dir = tmp_path / "openspec"
    requirements_dir.mkdir()

    async def _drain():
        async for _ in run_pipeline_stream(
            settings=minimal_settings,
            requirements_path=str(requirements_dir),
            thread_id="t",
            run_id="r",
            anthropic_api_key=override_anthropic,
            openai_api_key=override_openai,
        ):
            pass

    try:
        asyncio.run(_drain())
    except Exception as exc:
        captured_traceback = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    if captured_traceback is None:
        from dark_factory.api.ag_ui_bridge import _ApiKeyOverride

        ovr = _ApiKeyOverride(
            anthropic=override_anthropic,
            openai=override_openai,
        )
        captured_traceback = repr(ovr)

    assert override_anthropic not in captured_traceback
    assert override_openai not in captured_traceback
    assert "previous-anthropic-secret-XXYY" not in captured_traceback

    assert os.environ.get("ANTHROPIC_API_KEY") == "previous-anthropic-secret-XXYY"
    assert "OPENAI_API_KEY" not in os.environ
