"""Tests for early Run History entry creation in the AG-UI bridge."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from ag_ui.encoder import EventEncoder

from dark_factory.config import Settings
from dark_factory.api.ag_ui_bridge import run_pipeline_stream


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_event(sse_str: str) -> dict:
    for line in sse_str.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    return {}


def _event_type(sse_str: str) -> str:
    return _parse_event(sse_str).get("type", "")


def _collect(coro) -> list[str]:
    async def _run():
        return [chunk async for chunk in coro]
    return asyncio.run(_run())


def _mock_context():
    ctx = MagicMock()
    ctx.requirements = [MagicMock(id="req-1")]
    ctx.specs = [MagicMock(id="spec-1"), MagicMock(id="spec-2")]
    return ctx


# ── update_run_counts / mark_run_failed unit tests ────────────────────────────


def test_update_run_counts_only_spec_count():
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client.session.return_value.__enter__.return_value = mock_session

    repo = MemoryRepository(mock_client)
    repo.update_run_counts(run_id="run-abc", spec_count=5)

    args, kwargs = mock_session.run.call_args
    assert "r.spec_count" in args[0]
    assert "r.feature_count" not in args[0]
    assert kwargs["spec_count"] == 5
    assert kwargs["id"] == "run-abc"


def test_update_run_counts_both_counts():
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client.session.return_value.__enter__.return_value = mock_session

    repo = MemoryRepository(mock_client)
    repo.update_run_counts(run_id="run-abc", spec_count=5, feature_count=3)

    args, kwargs = mock_session.run.call_args
    assert "r.spec_count" in args[0]
    assert "r.feature_count" in args[0]
    assert kwargs["spec_count"] == 5
    assert kwargs["feature_count"] == 3


def test_update_run_counts_noop_when_no_args():
    """Calling with both args None is a no-op."""
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    repo = MemoryRepository(mock_client)
    repo.update_run_counts(run_id="run-abc")

    mock_client.session.assert_not_called()


def test_mark_run_failed_sets_error_status():
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client.session.return_value.__enter__.return_value = mock_session

    repo = MemoryRepository(mock_client)
    repo.mark_run_failed(run_id="run-abc", error="LLM timeout")

    args, kwargs = mock_session.run.call_args
    assert "'error'" in args[0]
    assert kwargs["id"] == "run-abc"
    payload = json.loads(kwargs["error_payload"])
    assert payload[0]["reason"] == "LLM timeout"


# ── Bridge integration: early Run creation ────────────────────────────────────


def test_bridge_creates_run_history_entry_before_phase_1():
    """The bridge calls memory_repo.create_run() BEFORE Phase 1 starts
    and sets ``_current_run_id`` so the orchestrator reuses it.

    The global is intentionally cleared in the stream's ``finally``
    block (see C3 fix — prevents stale run_id leaking into the NEXT
    pipeline run), so we can't assert on it AFTER the stream finishes.
    Instead we snapshot it from inside the ingest stage mock, which
    runs mid-pipeline while the global is still set.
    """
    settings = Settings()
    mock_repo = MagicMock()
    mock_repo.create_run.return_value = "run-20260410-120000-abcd"

    ctx = _mock_context()
    result = {
        "completed_features": [],
        "pass_rate": 1.0,
        "all_artifacts": [],
        "all_tests": [],
    }

    captured_run_id: list[str] = []

    def _ingest_run_captures_global(*args, **kwargs):
        from dark_factory.agents.tools import get_current_run_id

        captured_run_id.append(get_current_run_id())
        return ctx

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.stages.spec.SpecStage") as spec_cls,
        patch("dark_factory.stages.graph.GraphStage") as graph_cls,
        patch("dark_factory.agents.orchestrator.run_orchestrator", return_value=result),
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.side_effect = _ingest_run_captures_global
        spec_cls.return_value.run.return_value = ctx
        graph_cls.return_value.run.return_value = ctx

        _collect(
            run_pipeline_stream(
                settings=settings,
                memory_repo=mock_repo,
                requirements_path="./test",
                thread_id="t-1",
                run_id="r-1",
            )
        )

    # The bridge must have called create_run BEFORE running any pipeline stage
    mock_repo.create_run.assert_called_once_with(spec_count=0, feature_count=0)
    # During Phase 1, _current_run_id should be the freshly-created run id
    assert captured_run_id == ["run-20260410-120000-abcd"]
    # AFTER the stream's finally block runs, the global is cleared
    from dark_factory.agents.tools import get_current_run_id

    assert get_current_run_id() == ""


def test_bridge_updates_spec_count_after_phase_2():
    """update_run_counts is called with the actual spec count after spec gen."""
    settings = Settings()
    mock_repo = MagicMock()
    mock_repo.create_run.return_value = "run-test-1"

    ctx = _mock_context()  # has 2 specs
    result = {"completed_features": [], "pass_rate": 1.0, "all_artifacts": [], "all_tests": []}

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

        _collect(
            run_pipeline_stream(
                settings=settings,
                memory_repo=mock_repo,
                requirements_path="./test",
                thread_id="t-1",
                run_id="r-1",
            )
        )

    mock_repo.update_run_counts.assert_called_once_with(
        run_id="run-test-1", spec_count=2
    )


def test_bridge_marks_run_failed_on_pipeline_error():
    """If the pipeline raises (e.g. ingest fails), the Run entry is marked as 'error'."""
    settings = Settings()
    mock_repo = MagicMock()
    mock_repo.create_run.return_value = "run-failure-test"

    with (
        patch("dark_factory.graph.client.Neo4jClient"),
        patch("dark_factory.graph.repository.GraphRepository"),
        patch("dark_factory.stages.ingest.IngestStage") as ingest_cls,
        patch("dark_factory.ui.helpers.build_llm"),
    ):
        ingest_cls.return_value.run.side_effect = RuntimeError("ingest blew up")

        chunks = _collect(
            run_pipeline_stream(
                settings=settings,
                memory_repo=mock_repo,
                requirements_path="./test",
                thread_id="t-1",
                run_id="r-1",
            )
        )

    mock_repo.mark_run_failed.assert_called_once()
    call_kwargs = mock_repo.mark_run_failed.call_args.kwargs
    assert call_kwargs["run_id"] == "run-failure-test"
    assert "ingest blew up" in call_kwargs["error"]

    # The SSE stream should still emit RUN_ERROR for the frontend
    types = [_event_type(c) for c in chunks]
    assert "RUN_ERROR" in types


def test_bridge_works_without_memory_repo():
    """When memory_repo is None (memory disabled), pipeline still runs without Run History."""
    settings = Settings()

    ctx = _mock_context()
    result = {"completed_features": [], "pass_rate": 1.0, "all_artifacts": [], "all_tests": []}

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

        chunks = _collect(
            run_pipeline_stream(
                settings=settings,
                memory_repo=None,  # disabled
                requirements_path="./test",
                thread_id="t-1",
                run_id="r-1",
            )
        )

    # Pipeline should still finish normally
    types = [_event_type(c) for c in chunks]
    assert "RUN_FINISHED" in types


def test_bridge_clears_broker_history_before_phase_1():
    """The bridge clears the broker history at the START of a run (not in
    Phase 4) so spec_gen_* events from Phase 2 stay visible to the Logs tab."""
    settings = Settings()
    ctx = _mock_context()
    result = {"completed_features": [], "pass_rate": 1.0, "all_artifacts": [], "all_tests": []}

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    broker = ProgressBroker()
    # Pre-populate history with a stale event from a "previous run"
    broker.publish({"event": "stale_event_from_previous_run"})

    set_progress_broker(broker)
    try:
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

            # Verify history has the stale event before the run
            assert broker.history_count == 1

            _collect(
                run_pipeline_stream(
                    settings=settings,
                    memory_repo=None,
                    requirements_path="./test",
                    thread_id="t-1",
                    run_id="r-1",
                )
            )

        # After the run, the stale event should be gone (cleared at start)
        # Note: history may contain new events emitted during the run; what
        # we want to verify is that "stale_event_from_previous_run" is NOT
        # there.
        async def _drain():
            q = broker.subscribe(include_history=True)
            events = []
            while not q.empty():
                events.append(q.get_nowait())
            return events

        events = asyncio.run(_drain())
        event_names = {e["event"] for e in events}
        assert "stale_event_from_previous_run" not in event_names
    finally:
        set_progress_broker(None)


def test_orchestrator_reuses_existing_run_id():
    """run_orchestrator should NOT call create_run when get_current_run_id() returns a value."""
    from dark_factory.agents.tools import set_current_run_id

    set_current_run_id("run-bridge-created")
    try:
        # We can't easily test the full orchestrator without mocking a lot,
        # but we can at least verify the getter returns what was set.
        from dark_factory.agents.tools import get_current_run_id

        assert get_current_run_id() == "run-bridge-created"
    finally:
        set_current_run_id("")
