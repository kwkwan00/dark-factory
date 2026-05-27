"""Tests for the Phase 3 cross-set review (Judge.review_set + the
structural patcher) wired into the SSE stream."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dark_factory.api.refinery.contracts import CrossReviewReport
from dark_factory.api.refinery.models import RefinedRequirement
from dark_factory.api.refinery.roles.judge import JudgeRole


# ─────────────────────────────────────────────────────────────────────
# Fixtures
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
    """Patch run_phase2_generator so the SSE flow never invokes a real
    panel — each requirement comes back with a deterministic
    RefinedRequirement keyed on its id."""

    def _fake(*, req, **kwargs):
        return _fake_refined(req.get("id", "?"), title=f"Refined {req.get('id')}"), [], ""

    with patch(
        "dark_factory.api.refinery.debate.runner.run_phase2_generator",
        side_effect=_fake,
    ):
        yield




def _two_requirements_direct() -> list[dict]:
    """Helper: spin up the SSE flow on TWO direct-mode requirements
    (forcing Phase 3 to run since n>1 — direct mode normally has n=1
    so we patch in two via the gather path)."""

    return [
        {"id": "r-1", "title": "A", "description": "A desc", "priority": "medium", "tags": []},
        {"id": "r-2", "title": "B", "description": "B desc", "priority": "medium", "tags": []},
    ]


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_phase3_skipped_when_single_requirement(stub_run_phase2):
    """Single-requirement runs short-circuit Phase 3 — no review_set
    call and no reconciling SSE events."""

    review_calls: list[dict] = []

    def _spy(self, refined_set, traces, run_context):
        review_calls.append({"len": len(refined_set)})
        return CrossReviewReport()

    from dark_factory.api.refinery import stream as stream_mod

    with patch.object(JudgeRole, "review_set", _spy), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(),
            run_id=None,
            input_path=None,
            direct={"title": "only one", "description": "only one req"},
        )
        events = drive_async_gen(gen)

    # review_set was never called.
    assert review_calls == []
    # No reconciling events.
    assert not any(e.get("phase") == "reconciling" for e in events)
    # Done event still emits.
    assert any(e.get("phase") == "done" for e in events)


def test_phase3_runs_review_set_with_two_requirements(stub_run_phase2):
    """When n>1, Phase 3 runs and JudgeRole.review_set is called once
    with the full refined set."""

    review_calls: list[dict] = []

    def _spy(self, refined_set, traces, run_context):
        review_calls.append({
            "len": len(refined_set),
            "ids": [d.requirement_id for d in refined_set],
        })
        return CrossReviewReport(summary="reviewed cleanly")

    from dark_factory.api.refinery import gather as gather_mod
    from dark_factory.api.refinery import stream as stream_mod

    def _fake_gather(request, run_id):
        return {
            "run_id": run_id,
            "source_mode": "run",
            "traceability": {
                "rows": [
                    {"requirement": r} for r in _two_requirements_direct()
                ],
            },
        }

    with patch.object(JudgeRole, "review_set", _spy), \
         patch.object(gather_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(),
            run_id="run-1",
            input_path=None,
            direct=None,
        )
        events = drive_async_gen(gen)

    # review_set was called once with both requirements.
    assert len(review_calls) == 1
    assert sorted(review_calls[0]["ids"]) == ["r-1", "r-2"]
    # SSE saw the reconciling phase.
    recon_events = [e for e in events if e.get("phase") == "reconciling"]
    assert len(recon_events) >= 1, events
    # Done event still emits.
    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None


def test_phase3_applies_relationship_fix(stub_run_phase2):
    """A relationship_fix returned by the Judge gets patched onto the
    refined set in place — the Done event's response carries the new
    edge."""

    def _spy(self, refined_set, traces, run_context):
        return CrossReviewReport(
            summary="added one dep",
            relationship_fixes=[{
                "source_id": "r-1",
                "target_id": "r-2",
                "action": "add",
                "type": "depends_on",
                "rationale": "r-1 needs r-2's session manager",
            }],
        )

    from dark_factory.api.refinery import gather as gather_mod
    from dark_factory.api.refinery import stream as stream_mod

    def _fake_gather(request, run_id):
        return {
            "run_id": run_id,
            "source_mode": "run",
            "traceability": {
                "rows": [
                    {"requirement": r} for r in _two_requirements_direct()
                ],
            },
        }

    with patch.object(JudgeRole, "review_set", _spy), \
         patch.object(gather_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(),
            run_id="run-1",
            input_path=None,
            direct=None,
        )
        events = drive_async_gen(gen)

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    # r-1 should now depend_on r-2 in the final refined set.
    refined = {r["id"]: r for r in done["data"]["refined_requirements"]}
    r1_rels = refined["r-1"]["relationships"]
    assert any(
        rel["target_id"] == "r-2" and rel["type"] == "depends_on"
        for rel in r1_rels
    ), r1_rels


def test_phase3_surfaces_risk_areas_and_unresolved_debates(stub_run_phase2):
    """The Judge's risk_areas and unresolved_debates flow into
    RefineryResponse.risk_areas via _collect_risk_areas."""

    def _spy(self, refined_set, traces, run_context):
        return CrossReviewReport(
            summary="reviewed",
            risk_areas=["set-level: missing auth requirement"],
            unresolved_debates=[
                {"requirement_id": "r-1", "rounds_run": 3, "reason": "Security blocker"},
            ],
        )

    from dark_factory.api.refinery import gather as gather_mod
    from dark_factory.api.refinery import stream as stream_mod

    def _fake_gather(request, run_id):
        return {
            "run_id": run_id,
            "source_mode": "run",
            "traceability": {
                "rows": [
                    {"requirement": r} for r in _two_requirements_direct()
                ],
            },
        }

    with patch.object(JudgeRole, "review_set", _spy), \
         patch.object(gather_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(),
            run_id="run-1",
            input_path=None,
            direct=None,
        )
        events = drive_async_gen(gen)

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    risks = done["data"]["risk_areas"]
    # The Judge's set-level risk made it through.
    assert any("set-level: missing auth requirement" in r for r in risks)
    # The unresolved debate surfaced as an "Unresolved debate" entry.
    assert any("Unresolved debate (r-1)" in r for r in risks)


def test_phase3_review_set_exception_doesnt_crash_the_run(stub_run_phase2):
    """When Judge.review_set raises, Phase 3 swallows the failure with
    a 'failed' summary and the orchestrator still emits the Done
    event with the (un-patched) refined set."""

    def _boom(self, refined_set, traces, run_context):
        raise RuntimeError("review provider 503")

    from dark_factory.api.refinery import gather as gather_mod
    from dark_factory.api.refinery import stream as stream_mod

    def _fake_gather(request, run_id):
        return {
            "run_id": run_id,
            "source_mode": "run",
            "traceability": {
                "rows": [
                    {"requirement": r} for r in _two_requirements_direct()
                ],
            },
        }

    with patch.object(JudgeRole, "review_set", _boom), \
         patch.object(gather_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "gather_run_context", _fake_gather), \
         patch.object(stream_mod, "save_refinery_result"), \
         patch.object(stream_mod, "_get_refinery_config",
                      return_value=("gpt-5.4", "high", 10, 60)):
        gen = stream_mod.run_refinery_stream(
            make_stub_request(),
            run_id="run-1",
            input_path=None,
            direct=None,
        )
        events = drive_async_gen(gen)

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    # The Refinery still produced a response with both requirements.
    refined_ids = {r["id"] for r in done["data"]["refined_requirements"]}
    assert refined_ids == {"r-1", "r-2"}
