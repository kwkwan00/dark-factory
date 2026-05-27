"""Tests for the requirement-level resume feature.

Covers three layers:

1. ``ResumeRegistry`` is a no-op when Postgres is disabled.
2. ``ResumeRegistry`` calls through to ``RefineryMetricsRepository``
   when wired.
3. ``run_refinery_stream(resume_run_id=...)`` replays cached completions
   from the registry and only re-runs pending requirements.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dark_factory.api.refinery.models import (
    RefinedRequirement,
    SuggestedMemory,
)
from dark_factory.api.refinery.resume import ResumeRegistry


# ─────────────────────────────────────────────────────────────────────
# 1. ResumeRegistry: no-op semantics when client is None
# ─────────────────────────────────────────────────────────────────────


def test_resume_registry_disabled_when_client_is_none():
    reg = ResumeRegistry(None)

    assert reg.enabled is False
    # All write methods return None silently.
    reg.start(
        refinery_run_id="refinery-test",
        source_mode="run",
        source_run_id="run-1",
        input_snapshot={"requirements": []},
    )
    reg.mark_status("refinery-test", "completed")
    reg.cache_debate(
        refinery_run_id="refinery-test",
        requirement_id="req-1",
        convergence_status="converged",
        refined_payload={},
    )
    # Reads return None / empty when disabled.
    assert reg.load_run("refinery-test") is None
    assert reg.load_completed("refinery-test") == {}


# ─────────────────────────────────────────────────────────────────────
# 2. ResumeRegistry: delegation to RefineryMetricsRepository
# ─────────────────────────────────────────────────────────────────────


class _StubRepo:
    """Records every method invocation so tests can assert."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._run: dict[str, Any] | None = {
            "status": "in_progress",
            "source_mode": "run",
            "source_run_id": "run-A",
            "input_snapshot": {"requirements": [{"id": "r-1"}, {"id": "r-2"}]},
            "requirements_count": 2,
            "started_at": None,
        }
        self._completed: dict[str, dict] = {}

    def record_run_start(self, **kwargs):
        self.calls.append(("record_run_start", kwargs))

    def record_input_snapshot(self, **kwargs):
        self.calls.append(("record_input_snapshot", kwargs))

    def record_run_status(self, **kwargs):
        self.calls.append(("record_run_status", kwargs))

    def record_debate_completion(self, **kwargs):
        self.calls.append(("record_debate_completion", kwargs))
        self._completed[kwargs["requirement_id"]] = {
            "convergence_status": kwargs["convergence_status"],
            "refined": kwargs["refined_payload"],
            "suggested_memories": kwargs["suggested_memories"],
            "completed_at": None,
        }

    def load_run_for_resume(self, refinery_run_id):
        return self._run

    def load_completed_debates(self, refinery_run_id):
        return self._completed


def test_resume_registry_delegates_to_repo():
    repo = _StubRepo()
    # Bypass the constructor's repo build — inject the stub directly.
    reg = ResumeRegistry(metrics_client=None)
    reg._repo = repo  # type: ignore[attr-defined]

    reg.start(
        refinery_run_id="refinery-A",
        source_mode="run",
        source_run_id="run-A",
        input_snapshot={"requirements": [{"id": "r-1"}, {"id": "r-2"}]},
    )
    # start() makes three calls: record_run_start, record_input_snapshot,
    # record_run_status('in_progress').
    names = [c[0] for c in repo.calls]
    assert names == [
        "record_run_start",
        "record_input_snapshot",
        "record_run_status",
    ]
    assert repo.calls[0][1]["requirements_count"] == 2
    assert repo.calls[2][1]["status"] == "in_progress"

    repo.calls.clear()
    reg.mark_status("refinery-A", "completed")
    assert repo.calls == [(
        "record_run_status",
        {"refinery_run_id": "refinery-A", "status": "completed"},
    )]

    repo.calls.clear()
    reg.cache_debate(
        refinery_run_id="refinery-A",
        requirement_id="r-1",
        convergence_status="converged",
        refined_payload={"id": "r-1"},
        suggested_memories=[],
    )
    assert repo.calls[0][0] == "record_debate_completion"
    assert reg.load_completed("refinery-A")["r-1"]["convergence_status"] == "converged"


def test_mark_status_rejects_invalid_value():
    repo = _StubRepo()
    reg = ResumeRegistry(metrics_client=None)
    reg._repo = repo  # type: ignore[attr-defined]

    reg.mark_status("refinery-A", "not_a_real_status")
    # Invalid status is silently dropped — no repo call.
    assert repo.calls == []


# ─────────────────────────────────────────────────────────────────────
# 3. run_refinery_stream resume integration
# ─────────────────────────────────────────────────────────────────────


from tests.refinery.conftest import drive_async_gen, make_stub_request


def _fake_refined(req_id: str, title: str = "Refined") -> RefinedRequirement:
    return RefinedRequirement(
        id=req_id,
        original_title="Original",
        original_description="Original description",
        title=title,
        description="Refined description",
        priority="medium",
        tags=[],
    )


@pytest.fixture
def stub_run_phase2():
    """Patch run_phase2_generator so the test never invokes the real
    LangGraph debate. Returns a list of requirement IDs the stub was
    asked to refine."""

    seen: list[str] = []

    def _fake(*, req, **kwargs):
        seen.append(req.get("id", "?"))
        return _fake_refined(req.get("id", "?")), [], ""

    with patch(
        "dark_factory.api.refinery.debate.runner.run_phase2_generator",
        side_effect=_fake,
    ):
        yield seen




def test_resume_replays_cached_and_runs_only_pending(stub_run_phase2):
    """Two requirements; r-1 is cached as completed, r-2 is pending.
    Resume should replay r-1 from cache and only invoke the panel
    for r-2."""

    cached_refined = _fake_refined("r-1", title="Cached r-1")

    class _RepoWithOneCached(_StubRepo):
        def __init__(self):
            super().__init__()
            self._completed = {
                "r-1": {
                    "convergence_status": "converged",
                    "refined": cached_refined.model_dump(mode="json"),
                    "suggested_memories": [],
                    "completed_at": None,
                },
            }

    stub_repo = _RepoWithOneCached()

    # Patch ResumeRegistry to inject our stub repo regardless of the
    # metrics_client argument the orchestrator passes in.
    def _fake_registry_init(self, metrics_client):
        self._repo = stub_repo

    from dark_factory.api.refinery import resume as resume_mod
    from dark_factory.api.refinery import stream as stream_mod

    with patch.object(resume_mod.ResumeRegistry, "__init__", _fake_registry_init), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(metrics_client="anything-truthy"),
            run_id=None,
            input_path=None,
            direct=None,
            resume_run_id="refinery-A",
        )
        events = drive_async_gen(gen)

    # The stub seen list reflects which requirements were sent through
    # the panel. With r-1 cached, only r-2 should run.
    assert stub_run_phase2 == ["r-2"]

    # The done event should carry both refined requirements (r-1 from
    # cache, r-2 freshly produced).
    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None, events
    refined_ids = {r["id"] for r in done["data"]["refined_requirements"]}
    assert refined_ids == {"r-1", "r-2"}


def test_resume_unknown_run_yields_error(stub_run_phase2):
    """Resume against a run id Postgres doesn't know about → error
    event, no panel invocation."""

    class _RepoNoRun(_StubRepo):
        def load_run_for_resume(self, refinery_run_id):
            return None

    stub_repo = _RepoNoRun()

    def _fake_registry_init(self, metrics_client):
        self._repo = stub_repo

    from dark_factory.api.refinery import resume as resume_mod
    from dark_factory.api.refinery import stream as stream_mod

    with patch.object(resume_mod.ResumeRegistry, "__init__", _fake_registry_init), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(metrics_client="anything-truthy"),
            run_id=None,
            input_path=None,
            direct=None,
            resume_run_id="refinery-MISSING",
        )
        events = drive_async_gen(gen)

    # Panel was never asked to refine anything.
    assert stub_run_phase2 == []
    err = next((e for e in events if e.get("phase") == "error"), None)
    assert err is not None
    assert "unknown to" in err["message"]


def test_resume_drops_malformed_cached_payload(stub_run_phase2):
    """Documents current behavior: a corrupt cached ``refined`` payload
    is caught at model_validate time, the requirement is silently
    dropped from the refined set, and it does NOT get re-run via the
    panel — so the final response is missing it.

    Captures a real risk surface (silent data loss on cache corruption)
    so any future regression of the recovery semantics shows up here.
    """

    class _RepoCorrupted(_StubRepo):
        def __init__(self):
            super().__init__()
            self._completed = {
                "r-1": {
                    "convergence_status": "converged",
                    "refined": {"id": "r-1"},  # missing required fields
                    "suggested_memories": [],
                    "completed_at": None,
                },
            }

    stub_repo = _RepoCorrupted()

    def _fake_registry_init(self, metrics_client):
        self._repo = stub_repo

    from dark_factory.api.refinery import resume as resume_mod
    from dark_factory.api.refinery import stream as stream_mod

    with patch.object(resume_mod.ResumeRegistry, "__init__", _fake_registry_init), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(metrics_client="anything-truthy"),
            run_id=None,
            input_path=None,
            direct=None,
            resume_run_id="refinery-A",
        )
        events = drive_async_gen(gen)

    # Pending list excludes r-1 (it's "in cache"), but replay drops it
    # silently — r-1 is neither cached-replayed nor panel-refined.
    assert stub_run_phase2 == ["r-2"]
    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    refined_ids = {r["id"] for r in done["data"]["refined_requirements"]}
    assert refined_ids == {"r-2"}, refined_ids


def test_resume_skips_malformed_memory_but_keeps_requirement(stub_run_phase2):
    """A cached completion with one malformed ``suggested_memory`` and
    one valid one → the requirement still appears in the refined set;
    the bad memory is silently skipped, the good one survives."""

    cached_refined = _fake_refined("r-1", title="Cached r-1")

    class _RepoMemoryCorrupted(_StubRepo):
        def __init__(self):
            super().__init__()
            self._completed = {
                "r-1": {
                    "convergence_status": "converged",
                    "refined": cached_refined.model_dump(mode="json"),
                    # First entry is malformed, second is well-formed.
                    "suggested_memories": [
                        {"not": "a valid memory"},
                        SuggestedMemory(
                            type="pattern",
                            description="reusable timeout pattern",
                            context="auth flows",
                            applicability="all",
                            source_feature="judge",
                            rationale="cited by panel",
                        ).model_dump(mode="json"),
                    ],
                    "completed_at": None,
                },
            }

    stub_repo = _RepoMemoryCorrupted()

    def _fake_registry_init(self, metrics_client):
        self._repo = stub_repo

    from dark_factory.api.refinery import resume as resume_mod
    from dark_factory.api.refinery import stream as stream_mod

    with patch.object(resume_mod.ResumeRegistry, "__init__", _fake_registry_init), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(metrics_client="anything-truthy"),
            run_id=None,
            input_path=None,
            direct=None,
            resume_run_id="refinery-A",
        )
        events = drive_async_gen(gen)

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    refined_ids = {r["id"] for r in done["data"]["refined_requirements"]}
    assert refined_ids == {"r-1", "r-2"}
    # Exactly one memory survives; the malformed one was dropped.
    mem_descriptions = [m["description"] for m in done["data"].get("suggested_memories") or []]
    assert "reusable timeout pattern" in mem_descriptions


def test_resume_preserves_run_context_to_panel(stub_run_phase2):
    """The saved input_snapshot's run_context survives the resume path
    and reaches run_phase2_generator — proves Phase 1's outputs are
    not silently dropped on resume."""

    captured_kwargs: list[dict] = []

    # Re-patch run_phase2_generator with a richer spy.
    def _capture(*, req, **kwargs):
        captured_kwargs.append({"req_id": req.get("id"), **kwargs})
        return _fake_refined(req.get("id", "?")), [], ""

    class _RepoWithCtx(_StubRepo):
        def __init__(self):
            super().__init__()
            self._run = {
                **self._run,
                "input_snapshot": {
                    "requirements": [{"id": "r-1"}],
                    "run_context": {"run_id": "run-X", "marker": "preserved"},
                },
            }

    stub_repo = _RepoWithCtx()

    def _fake_registry_init(self, metrics_client):
        self._repo = stub_repo

    from dark_factory.api.refinery import resume as resume_mod
    from dark_factory.api.refinery import stream as stream_mod

    with patch(
        "dark_factory.api.refinery.debate.runner.run_phase2_generator",
        side_effect=_capture,
    ), patch.object(resume_mod.ResumeRegistry, "__init__", _fake_registry_init), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(metrics_client="anything-truthy"),
            run_id=None,
            input_path=None,
            direct=None,
            resume_run_id="refinery-A",
        )
        drive_async_gen(gen)

    assert len(captured_kwargs) == 1
    ctx = captured_kwargs[0].get("run_context")
    assert ctx is not None
    assert ctx.get("marker") == "preserved"
    assert ctx.get("run_id") == "run-X"


def test_resume_already_completed_yields_error(stub_run_phase2):
    """Resume against a completed run → error event pointing the
    operator at GET /api/refinery/{id}."""

    class _RepoCompleted(_StubRepo):
        def __init__(self):
            super().__init__()
            self._run = {**self._run, "status": "completed"}  # type: ignore[dict-item]

    stub_repo = _RepoCompleted()

    def _fake_registry_init(self, metrics_client):
        self._repo = stub_repo

    from dark_factory.api.refinery import resume as resume_mod
    from dark_factory.api.refinery import stream as stream_mod

    with patch.object(resume_mod.ResumeRegistry, "__init__", _fake_registry_init), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(metrics_client="anything-truthy"),
            run_id=None,
            input_path=None,
            direct=None,
            resume_run_id="refinery-DONE",
        )
        events = drive_async_gen(gen)

    assert stub_run_phase2 == []
    err = next((e for e in events if e.get("phase") == "error"), None)
    assert err is not None
    assert "already" in err["message"]
