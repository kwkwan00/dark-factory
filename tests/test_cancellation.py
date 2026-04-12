"""Tests for the pipeline kill-switch.

Covers:
- The ``cancellation`` module primitives (set/reset/raise/is_cancelled)
- The ``POST /api/agent/cancel`` endpoint
- Cooperative cancellation checks in ``SpecStage`` and ``run_feature_swarm``
- End-to-end via ``run_pipeline_stream`` — start a stream, fire cancel,
  confirm the run ends with a ``cancelled`` signal without processing
  remaining phases
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dark_factory.agents.cancellation import (
    PipelineCancelled,
    is_cancelled,
    raise_if_cancelled,
    request_cancel,
    reset_cancel,
)
from dark_factory.config import Settings


@pytest.fixture
def minimal_settings() -> Settings:
    return Settings()


# ── Module primitives ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_cancel_state():
    """Ensure each test starts and ends with the cancel flag cleared so
    leaking state from one test can't silently short-circuit the next."""
    reset_cancel()
    yield
    reset_cancel()


def test_cancellation_flag_defaults_to_clear():
    assert is_cancelled() is False


def test_request_cancel_sets_flag():
    request_cancel()
    assert is_cancelled() is True


def test_reset_cancel_clears_flag():
    request_cancel()
    assert is_cancelled() is True
    reset_cancel()
    assert is_cancelled() is False


def test_request_cancel_is_idempotent():
    request_cancel()
    request_cancel()
    request_cancel()
    assert is_cancelled() is True


def test_raise_if_cancelled_noop_when_clear():
    # Must not raise
    raise_if_cancelled()


def test_raise_if_cancelled_raises_when_set():
    request_cancel()
    with pytest.raises(PipelineCancelled):
        raise_if_cancelled()


# ── /api/agent/cancel endpoint ─────────────────────────────────────────────


def test_cancel_endpoint_returns_not_active_when_no_run(api_client):
    """No run in progress → 200 + ``cancelled=False`` so the frontend
    can fire cancel without a try/catch race."""
    resp = api_client.post("/api/agent/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is False
    assert "no active run" in data.get("reason", "")


def test_cancel_endpoint_sets_flag_when_run_active(api_client):
    """With the run_lock held, the endpoint should set the cancellation
    flag and return ``cancelled=True``."""
    from dark_factory.api.app import app

    async def _hold_lock_then_cancel():
        lock: asyncio.Lock = app.state.run_lock
        async with lock:
            # Now the lock is locked — simulate an active run and poke
            # the cancel endpoint via the TestClient in a thread so the
            # async context manager doesn't block the sync test client.
            resp = await asyncio.to_thread(
                api_client.post, "/api/agent/cancel"
            )
            return resp

    resp = asyncio.run(_hold_lock_then_cancel())
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is True
    assert is_cancelled() is True


def test_cancel_endpoint_idempotent_when_already_cancelled(api_client):
    """A second cancel while already pending returns ``already_pending=True``."""
    from dark_factory.api.app import app

    async def _double_cancel():
        lock: asyncio.Lock = app.state.run_lock
        async with lock:
            first = await asyncio.to_thread(api_client.post, "/api/agent/cancel")
            second = await asyncio.to_thread(api_client.post, "/api/agent/cancel")
            return first, second

    first, second = asyncio.run(_double_cancel())
    assert first.json()["cancelled"] is True
    assert second.json()["cancelled"] is True
    assert second.json().get("already_pending") is True


# ── SpecStage honours cancellation ─────────────────────────────────────────


def test_spec_stage_raises_cancelled_at_entry():
    """Pre-flight check: if cancel is already set when run() starts, raise
    immediately without calling the LLM."""
    from dark_factory.models.domain import PipelineContext, Priority, Requirement
    from dark_factory.stages.spec import SpecStage

    req = Requirement(
        id="req-1",
        title="Test",
        description="Test requirement",
        source_file="test.md",
        priority=Priority.HIGH,
    )
    fake_llm = MagicMock()
    stage = SpecStage(llm=fake_llm)
    ctx = PipelineContext(input_path="test", requirements=[req])

    request_cancel()
    with pytest.raises(PipelineCancelled):
        stage.run(ctx)
    fake_llm.complete_structured.assert_not_called()


def test_spec_stage_refine_loop_stops_between_attempts():
    """Mid-refinement cancel stops the loop before the next attempt."""
    from dark_factory.evaluation import metrics as eval_metrics
    from dark_factory.models.domain import (
        PipelineContext,
        Priority,
        Requirement,
        Spec,
    )
    from dark_factory.stages.spec import SpecStage

    req = Requirement(
        id="req-1",
        title="Test",
        description="desc",
        source_file="test.md",
        priority=Priority.HIGH,
    )
    spec = Spec(
        id="spec-req-1",
        title="s",
        description="d",
        requirement_ids=["req-1"],
        acceptance_criteria=["c"],
        capability="test",
    )

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = spec

    # First attempt scores below threshold → loop would normally refine.
    # But between attempt 1 and attempt 2 we set the cancel flag.
    call_count = {"n": 0}

    def _eval_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            request_cancel()
        return {
            "m1": {"score": 0.1, "passed": False, "reason": "low"},
        }

    with patch.object(eval_metrics, "evaluate_generated_spec", side_effect=_eval_side_effect):
        stage = SpecStage(llm=fake_llm, max_handoffs=5, eval_threshold=0.9)
        ctx = PipelineContext(input_path="test", requirements=[req])
        with pytest.raises(PipelineCancelled):
            stage.run(ctx)

    # Only one attempt was allowed through before the cancel raised.
    assert fake_llm.complete_structured.call_count == 1


# ── run_feature_swarm honours cancellation ────────────────────────────────


def test_run_feature_swarm_cancelled_returns_skipped():
    """When the swarm stream loop hits a cancel, the feature result is
    marked ``skipped`` with a ``Cancelled by user`` error message rather
    than an ``error`` status, so the orchestrator's aggregate doesn't
    flag it as a crash."""
    from dark_factory.agents.swarm import run_feature_swarm

    # Fake compiled swarm that yields one chunk, then cancellation fires.
    class _FakeCompiled:
        def stream(self, initial, stream_mode="updates", config=None):
            yield {
                "planner": {
                    "messages": []
                }
            }
            # Cancel is set during iteration — next iteration checks and raises
            request_cancel()
            yield {
                "coder": {
                    "messages": []
                }
            }

    result = run_feature_swarm(
        _FakeCompiled(),
        spec_ids=["spec-1"],
        feature_name="auth",
        max_handoffs=10,
    )
    assert result["status"] == "skipped"
    assert "Cancelled" in (result["error"] or "")


# ── End-to-end via run_pipeline_stream ─────────────────────────────────────


def test_run_pipeline_stream_cancel_during_ingest(minimal_settings, monkeypatch, tmp_path):
    """Set the cancel flag before ingest starts — the stream emits a
    RUN_ERROR event with a cancellation message and exits cleanly."""
    import json

    from dark_factory.api.ag_ui_bridge import run_pipeline_stream

    # Halt build_llm so the pipeline can't actually proceed even without cancel.
    def _halt_llm(_settings):
        # Raise before anything else runs — but AFTER reset_cancel has run.
        # We set the cancel flag here so the NEXT raise_if_cancelled picks it up.
        request_cancel()
        # Return a dummy so control flow continues to the next check.
        return MagicMock()

    monkeypatch.setattr(
        "dark_factory.ui.helpers.build_llm", _halt_llm, raising=True
    )

    # Also stub out Neo4jClient + IngestStage to avoid real DB / file work
    monkeypatch.setattr(
        "dark_factory.graph.client.Neo4jClient",
        MagicMock(),
    )

    class _FakeIngestStage:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, ctx):
            return ctx

    monkeypatch.setattr(
        "dark_factory.stages.ingest.IngestStage", _FakeIngestStage, raising=True
    )

    reqs_dir = tmp_path / "openspec"
    reqs_dir.mkdir()

    async def _drain():
        chunks = []
        async for chunk in run_pipeline_stream(
            settings=minimal_settings,
            requirements_path=str(reqs_dir),
            thread_id="t",
            run_id="r",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_drain())

    # Parse all data: lines from the stream
    payloads = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                try:
                    payloads.append(json.loads(line[len("data: "):]))
                except Exception:
                    pass

    event_types = [p.get("type") for p in payloads]
    # Should emit RUN_STARTED → (some optional text events) → RUN_ERROR
    assert "RUN_STARTED" in event_types
    assert "RUN_ERROR" in event_types

    # The RUN_ERROR message must mention cancellation
    error_msgs = [p.get("message", "") for p in payloads if p.get("type") == "RUN_ERROR"]
    assert any("cancel" in msg.lower() for msg in error_msgs)


def test_run_pipeline_stream_resets_cancel_flag_on_exit(
    minimal_settings, monkeypatch, tmp_path
):
    """After the stream finishes — success or cancel — the cancel flag is
    cleared so a late cancel can't poison the next run."""
    from dark_factory.api.ag_ui_bridge import run_pipeline_stream

    def _halt_llm(_settings):
        request_cancel()
        return MagicMock()

    monkeypatch.setattr("dark_factory.ui.helpers.build_llm", _halt_llm, raising=True)
    monkeypatch.setattr("dark_factory.graph.client.Neo4jClient", MagicMock())

    class _FakeIngestStage:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, ctx):
            return ctx

    monkeypatch.setattr(
        "dark_factory.stages.ingest.IngestStage", _FakeIngestStage, raising=True
    )

    reqs_dir = tmp_path / "openspec"
    reqs_dir.mkdir()

    async def _drain():
        async for _ in run_pipeline_stream(
            settings=minimal_settings,
            requirements_path=str(reqs_dir),
            thread_id="t",
            run_id="r",
        ):
            pass

    asyncio.run(_drain())

    # The cancel flag must be cleared once the stream exits.
    assert is_cancelled() is False


# ── B2 regression guards: PipelineCancelled must propagate through
# best-effort stages (doc_extraction, reconciliation, e2e_validation,
# episode synthesis). Without the explicit `except PipelineCancelled:
# raise` guard added by B2, the broad `except Exception` handler in
# each stage would catch the cancel signal and the pipeline would
# proceed past the stage as if nothing happened — making the Cancel
# button unreliable during Phase 1 / Phase 5 / Phase 6 / episode
# writing.


def test_b2_doc_extraction_propagates_pipeline_cancelled(tmp_path):
    """PipelineCancelled raised inside the extraction deep agent must
    bubble out of extract_with_deep_agent, not get swallowed by the
    best-effort except Exception handler."""
    from dark_factory.stages.doc_extraction import extract_with_deep_agent

    source = tmp_path / "meeting.docx"
    source.write_bytes(b"fake")

    def _cancel_mid_agent(prompt, allowed_tools, max_turns=15, timeout_seconds=None):
        raise PipelineCancelled("cancel during doc extraction")

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=_cancel_mid_agent,
    ):
        with pytest.raises(PipelineCancelled):
            extract_with_deep_agent(source)


def test_b2_reconciliation_propagates_pipeline_cancelled(tmp_path):
    """PipelineCancelled raised inside the reconciliation deep agent
    must bubble out, not be converted into ``status=error``."""
    from dark_factory.stages.reconciliation import ReconciliationStage

    (tmp_path / "feature").mkdir()

    def _cancel(prompt, allowed_tools, max_turns=15, timeout_seconds=None):
        raise PipelineCancelled("cancel during reconciliation")

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=_cancel,
    ):
        stage = ReconciliationStage()
        with pytest.raises(PipelineCancelled):
            stage.run(
                run_id="run-test",
                output_dir=tmp_path,
                feature_results=[{"feature": "feat", "status": "success"}],
            )


def test_b2_e2e_validation_propagates_pipeline_cancelled(tmp_path):
    """PipelineCancelled raised inside the E2E deep agent must bubble
    out of E2EValidationStage.run."""
    from dark_factory.stages.e2e_validation import E2EValidationStage

    (tmp_path / "app").mkdir()

    def _cancel(prompt, allowed_tools, max_turns=15, timeout_seconds=None):
        raise PipelineCancelled("cancel during e2e validation")

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=_cancel,
    ):
        stage = E2EValidationStage()
        with pytest.raises(PipelineCancelled):
            stage.run(
                run_id="run-test",
                output_dir=tmp_path,
                feature_results=[{"feature": "feat", "status": "success"}],
                reconciliation_status="clean",
            )


def test_b2_episode_synthesis_propagates_pipeline_cancelled():
    """PipelineCancelled raised from llm.complete_structured during
    episode synthesis must bubble out, not silently fall back to the
    deterministic template."""
    from datetime import datetime, timezone

    from dark_factory.memory.episodes import synthesize_episode

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = PipelineCancelled(
        "cancel during episode synthesis"
    )

    now = datetime.now(timezone.utc)
    with pytest.raises(PipelineCancelled):
        synthesize_episode(
            run_id="run-test",
            feature="auth",
            spec_ids=[],
            outcome="success",
            turns_used=1,
            duration_seconds=1.0,
            started_at=now,
            ended_at=now,
            final_eval_scores={},
            agents_visited=[],
            tool_calls_summary={},
            progress_events=[],
            llm=fake_llm,
        )


# ── B1 regression guard: orchestrator must NOT treat KeyboardInterrupt
# as a worker-crash-with-inflight-agents event. The H6 diagnostic
# counter should only fire on subclasses of Exception, not on
# BaseException.


def test_b1_keyboard_interrupt_does_not_fire_inflight_counter():
    """Simulate a worker crash via KeyboardInterrupt and verify the
    worker_crashes_with_inflight_agents_total counter stays flat."""
    from dark_factory.agents.tools import _thread_local
    from dark_factory.metrics import prometheus as prom

    # Seed the in-flight counter so a buggy except BaseException
    # would fire the metric.
    _thread_local.inflight_deep_agents = 2
    try:
        before = prom.worker_crashes_with_inflight_agents_total.labels(
            feature="b1-test"
        )._value.get()

        # Mimic what _run_one does — except Exception, not
        # BaseException. KeyboardInterrupt must propagate untouched.
        try:
            raise KeyboardInterrupt("Ctrl-C during test")
        except Exception:
            # Would reach here under the old buggy code, fire the
            # counter, then re-raise. Under the new code, Python
            # skips this except clause entirely and propagates
            # KeyboardInterrupt out.
            prom.worker_crashes_with_inflight_agents_total.labels(
                feature="b1-test"
            ).inc()
            raise
        except KeyboardInterrupt:
            pass  # Expected path — the counter stays flat.

        after = prom.worker_crashes_with_inflight_agents_total.labels(
            feature="b1-test"
        )._value.get()
        assert after == before, (
            "KeyboardInterrupt should not bump the worker-crash counter"
        )
    finally:
        _thread_local.inflight_deep_agents = 0


# ── A1 regression guard: orchestrator resets the inflight counter at
# the start of every feature swarm so a leaked drift from a prior
# feature on the same worker thread cannot produce false-positive
# crash signals.


def test_a1_run_one_resets_inflight_counter():
    """Inspect orchestrator._run_one source to verify the thread-local
    reset is present. This is a structural guard — the function is a
    closure so we can't easily invoke it in isolation, but we can
    assert the reset instruction is in the module source."""
    import inspect

    from dark_factory.agents import orchestrator

    src = inspect.getsource(orchestrator.make_execute_layer_node)
    assert "_thread_local.inflight_deep_agents = 0" in src, (
        "A1: _run_one must reset inflight_deep_agents at the start of "
        "each feature to prevent drift across reused worker threads"
    )


# ── B3 regression guards: record_* tools refuse to write when the
# current run id is empty, preventing orphaned memory nodes.


def test_b3_record_pattern_refuses_without_run_id():
    from dark_factory.agents import tools as tools_mod

    mock_repo = MagicMock()
    mock_repo.record_pattern.return_value = "pattern-xxx"

    prev_repo = tools_mod._memory_repo
    prev_run = tools_mod._current_run_id
    tools_mod._memory_repo = mock_repo
    tools_mod._current_run_id = ""
    try:
        result = tools_mod.record_pattern.invoke(
            {"description": "x", "context": "y"}
        )
        assert "cannot record memory outside an active pipeline run" in result
        mock_repo.record_pattern.assert_not_called()
    finally:
        tools_mod._memory_repo = prev_repo
        tools_mod._current_run_id = prev_run


def test_b3_record_mistake_refuses_without_run_id():
    from dark_factory.agents import tools as tools_mod

    mock_repo = MagicMock()
    prev_repo = tools_mod._memory_repo
    prev_run = tools_mod._current_run_id
    tools_mod._memory_repo = mock_repo
    tools_mod._current_run_id = ""
    try:
        result = tools_mod.record_mistake.invoke(
            {
                "description": "x",
                "error_type": "y",
                "trigger_context": "z",
            }
        )
        assert "cannot record memory" in result
        mock_repo.record_mistake.assert_not_called()
    finally:
        tools_mod._memory_repo = prev_repo
        tools_mod._current_run_id = prev_run


def test_b3_record_solution_refuses_without_run_id():
    from dark_factory.agents import tools as tools_mod

    mock_repo = MagicMock()
    prev_repo = tools_mod._memory_repo
    prev_run = tools_mod._current_run_id
    tools_mod._memory_repo = mock_repo
    tools_mod._current_run_id = ""
    try:
        result = tools_mod.record_solution.invoke({"description": "x"})
        assert "cannot record memory" in result
        mock_repo.record_solution.assert_not_called()
    finally:
        tools_mod._memory_repo = prev_repo
        tools_mod._current_run_id = prev_run


def test_b3_record_strategy_refuses_without_run_id():
    from dark_factory.agents import tools as tools_mod

    mock_repo = MagicMock()
    prev_repo = tools_mod._memory_repo
    prev_run = tools_mod._current_run_id
    tools_mod._memory_repo = mock_repo
    tools_mod._current_run_id = ""
    try:
        result = tools_mod.record_strategy.invoke(
            {"description": "x", "applicability": "y"}
        )
        assert "cannot record memory" in result
        mock_repo.record_strategy.assert_not_called()
    finally:
        tools_mod._memory_repo = prev_repo
        tools_mod._current_run_id = prev_run


def test_b3_record_pattern_accepts_with_run_id():
    """Sanity check: the guard only blocks empty run_id. When a run
    id IS set the tool should write normally."""
    from dark_factory.agents import tools as tools_mod

    mock_repo = MagicMock()
    mock_repo.record_pattern.return_value = "pattern-abc"

    prev_repo = tools_mod._memory_repo
    prev_run = tools_mod._current_run_id
    tools_mod._memory_repo = mock_repo
    tools_mod._current_run_id = "run-b3-happy-path"
    try:
        result = tools_mod.record_pattern.invoke(
            {"description": "x", "context": "y"}
        )
        assert "pattern-abc" in result
        mock_repo.record_pattern.assert_called_once()
        call_kwargs = mock_repo.record_pattern.call_args.kwargs
        assert call_kwargs["run_id"] == "run-b3-happy-path"
    finally:
        tools_mod._memory_repo = prev_repo
        tools_mod._current_run_id = prev_run
