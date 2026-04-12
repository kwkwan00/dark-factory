"""Tests for the parallel SpecStage."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from dark_factory.models.domain import (
    PipelineContext,
    Priority,
    Requirement,
    Spec,
)
from dark_factory.stages.spec import SpecStage


@pytest.fixture(autouse=True)
def _mock_evaluate_spec():
    """Stub out the DeepEval call so tests don't hit the real OpenAI API.

    Default returns score=1.0 (above any threshold) so the refinement loop
    exits after one attempt unless a test overrides ``side_effect``.
    """
    with patch(
        "dark_factory.evaluation.metrics.evaluate_generated_spec",
        return_value={"correctness": {"score": 1.0, "passed": True, "reason": None}},
    ) as mock:
        yield mock


def _make_req(idx: int) -> Requirement:
    return Requirement(
        id=f"req-{idx}",
        title=f"Requirement {idx}",
        description=f"Do thing {idx}",
        source_file="test.md",
        priority=Priority.HIGH,
    )


def _make_spec(req: Requirement) -> Spec:
    return Spec(
        id=f"spec-{req.id}",
        title=f"Spec for {req.title}",
        description="generated",
        requirement_ids=[req.id],
        acceptance_criteria=["test passes"],
        capability="test-feature",
    )


def test_spec_stage_no_requirements_returns_empty():
    fake_llm = MagicMock()
    stage = SpecStage(llm=fake_llm)
    ctx = PipelineContext(input_path="test", requirements=[])
    result = stage.run(ctx)
    assert result.specs == []
    fake_llm.complete_structured.assert_not_called()


def test_spec_stage_single_requirement_succeeds():
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    stage = SpecStage(llm=fake_llm)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 1
    assert result.specs[0].id == "spec-req-1"


def test_spec_stage_processes_in_parallel():
    """Multiple requirements are processed concurrently — verified by
    timing. Each fake LLM call sleeps for 0.5s. With max_parallel=4,
    4 requirements should finish in ~0.5s, not ~2s."""
    requirements = [_make_req(i) for i in range(4)]

    def slow_complete(*, prompt, response_model, system):
        time.sleep(0.5)
        # Extract the requirement number from the prompt
        # The prompt template includes "Requirement ID: req-N"
        for line in prompt.split("\n"):
            if "Requirement ID:" in line:
                req_id = line.split(":", 1)[1].strip()
                return Spec(
                    id=f"spec-{req_id}",
                    title=f"Spec for {req_id}",
                    description="generated",
                    requirement_ids=[req_id],
                    acceptance_criteria=["c"],
                    capability="cap",
                )
        raise ValueError("could not parse req id")

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = slow_complete

    stage = SpecStage(llm=fake_llm, max_parallel=4)
    ctx = PipelineContext(input_path="test", requirements=requirements)

    start = time.monotonic()
    result = stage.run(ctx)
    duration = time.monotonic() - start

    assert len(result.specs) == 4
    # Sequential would take ~2.0s; parallel with 4 workers should be ~0.5-1.0s
    assert duration < 1.5, f"expected parallel execution, took {duration:.2f}s"


def test_spec_stage_preserves_requirement_order():
    """Output specs match the input requirement order even when futures
    complete out of order (small reqs finish first)."""
    requirements = [_make_req(i) for i in range(5)]

    # Make later requirements finish faster (reverse order)
    def staggered_complete(*, prompt, response_model, system):
        for line in prompt.split("\n"):
            if "Requirement ID:" in line:
                req_id = line.split(":", 1)[1].strip()
                idx = int(req_id.split("-")[1])
                # Reqs at the front sleep longer; back reqs return immediately
                time.sleep((5 - idx) * 0.05)
                return Spec(
                    id=f"spec-{req_id}",
                    title=f"Spec {idx}",
                    description="x",
                    requirement_ids=[req_id],
                    acceptance_criteria=["c"],
                    capability="c",
                )
        raise ValueError

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = staggered_complete

    stage = SpecStage(llm=fake_llm, max_parallel=5)
    ctx = PipelineContext(input_path="test", requirements=requirements)
    result = stage.run(ctx)

    # Should match the input order: req-0, req-1, ..., req-4
    assert [s.id for s in result.specs] == [
        "spec-req-0",
        "spec-req-1",
        "spec-req-2",
        "spec-req-3",
        "spec-req-4",
    ]


def test_spec_stage_one_failure_does_not_stop_others():
    """A failed requirement is logged but other requirements still complete."""
    requirements = [_make_req(i) for i in range(3)]
    call_count = {"n": 0}
    lock = threading.Lock()

    def complete_with_one_failure(*, prompt, response_model, system):
        with lock:
            call_count["n"] += 1
            current = call_count["n"]
        for line in prompt.split("\n"):
            if "Requirement ID:" in line:
                req_id = line.split(":", 1)[1].strip()
                if "req-1" in req_id:
                    raise RuntimeError("simulated LLM error")
                return Spec(
                    id=f"spec-{req_id}",
                    title=f"Spec {current}",
                    description="x",
                    requirement_ids=[req_id],
                    acceptance_criteria=["c"],
                    capability="c",
                )
        raise ValueError

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = complete_with_one_failure

    stage = SpecStage(llm=fake_llm, max_parallel=3)
    ctx = PipelineContext(input_path="test", requirements=requirements)
    result = stage.run(ctx)

    # 2 succeeded, 1 failed — order preserved among the survivors
    assert len(result.specs) == 2
    assert result.specs[0].id == "spec-req-0"
    assert result.specs[1].id == "spec-req-2"


def test_spec_stage_all_failures_raise():
    """If every requirement fails, the first error is raised."""
    import pytest

    requirements = [_make_req(i) for i in range(2)]

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = RuntimeError("LLM down")

    stage = SpecStage(llm=fake_llm, max_parallel=2)
    ctx = PipelineContext(input_path="test", requirements=requirements)
    with pytest.raises(RuntimeError, match="LLM down"):
        stage.run(ctx)


def test_spec_stage_max_parallel_clamped_to_min_1():
    """max_parallel=0 or negative is clamped to 1 (still works, just sequentially)."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    stage = SpecStage(llm=fake_llm, max_parallel=0)
    assert stage.max_parallel == 1

    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)
    assert len(result.specs) == 1


def test_spec_stage_early_exits_when_threshold_met(_mock_evaluate_spec):
    """Default mock returns score=1.0 — first attempt should hit the threshold."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    stage = SpecStage(llm=fake_llm, max_handoffs=5, eval_threshold=0.8)
    ctx = PipelineContext(input_path="test", requirements=[req])
    stage.run(ctx)

    # Only one LLM call because the first score (1.0) >= threshold (0.8)
    assert fake_llm.complete_structured.call_count == 1


def test_spec_stage_iterates_until_threshold(_mock_evaluate_spec):
    """When eval scores below threshold, the stage refines on subsequent attempts."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    # First two attempts score below 0.8, third attempt hits 0.9 → early exit
    _mock_evaluate_spec.side_effect = [
        {"correctness": {"score": 0.5, "passed": False, "reason": "weak"}},
        {"correctness": {"score": 0.6, "passed": False, "reason": "still weak"}},
        {"correctness": {"score": 0.9, "passed": True, "reason": "good"}},
    ]

    stage = SpecStage(llm=fake_llm, max_handoffs=5, eval_threshold=0.8)
    ctx = PipelineContext(input_path="test", requirements=[req])
    stage.run(ctx)

    assert fake_llm.complete_structured.call_count == 3


def test_spec_stage_caps_at_max_handoffs(_mock_evaluate_spec):
    """If the score never reaches the threshold, stop after max_handoffs."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    # Always return a low score
    _mock_evaluate_spec.return_value = {
        "correctness": {"score": 0.3, "passed": False, "reason": "bad"},
    }

    stage = SpecStage(llm=fake_llm, max_handoffs=4, eval_threshold=0.8)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert fake_llm.complete_structured.call_count == 4
    # Still returns the best (only) spec
    assert len(result.specs) == 1


def test_spec_stage_returns_best_score_across_attempts(_mock_evaluate_spec):
    """The spec with the highest eval score is returned, not the latest."""
    req = _make_req(1)

    # Three different specs returned across three attempts
    spec_a = Spec(id="spec-a", title="A", description="x", requirement_ids=["req-1"], acceptance_criteria=["c"], capability="c")
    spec_b = Spec(id="spec-b", title="B", description="x", requirement_ids=["req-1"], acceptance_criteria=["c"], capability="c")
    spec_c = Spec(id="spec-c", title="C", description="x", requirement_ids=["req-1"], acceptance_criteria=["c"], capability="c")

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = [spec_a, spec_b, spec_c]

    # B is the best (0.7); A is 0.5; C is 0.6. None hit the 0.8 threshold.
    _mock_evaluate_spec.side_effect = [
        {"m": {"score": 0.5, "passed": False, "reason": "x"}},
        {"m": {"score": 0.7, "passed": False, "reason": "x"}},
        {"m": {"score": 0.6, "passed": False, "reason": "x"}},
    ]

    stage = SpecStage(llm=fake_llm, max_handoffs=3, eval_threshold=0.8)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 1
    assert result.specs[0].id == "spec-b"  # the best-scoring attempt


def test_spec_stage_refine_prompt_includes_previous_feedback(_mock_evaluate_spec):
    """The refine prompt feeds the previous spec + critic feedback back to the LLM."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    _mock_evaluate_spec.side_effect = [
        {"correctness": {"score": 0.4, "passed": False, "reason": "missing edge cases"}},
        {"correctness": {"score": 0.95, "passed": True, "reason": "good"}},
    ]

    stage = SpecStage(llm=fake_llm, max_handoffs=3, eval_threshold=0.8)
    ctx = PipelineContext(input_path="test", requirements=[req])
    stage.run(ctx)

    # Two LLM calls: initial + one refinement
    assert fake_llm.complete_structured.call_count == 2

    # The second call should be a refinement prompt that includes the feedback
    second_call_prompt = fake_llm.complete_structured.call_args_list[1].kwargs["prompt"]
    assert "previously generated" in second_call_prompt.lower()
    assert "missing edge cases" in second_call_prompt
    assert "0.40" in second_call_prompt  # the previous score


def test_spec_stage_events_visible_to_logs_subscriber(_mock_evaluate_spec):
    """Spec gen events flow through the broker so a Logs tab subscriber sees them.

    This is the behavior the Agent Logs tab depends on: it subscribes to the
    broker and expects spec_gen_* events from Phase 2 to arrive alongside
    swarm events from Phase 4.
    """
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            # Logs-tab-style subscription: get history on connect
            queue = broker.subscribe(include_history=True)
            # Empty initially
            assert queue.qsize() == 0

            requirements = [_make_req(i) for i in range(2)]
            fake_llm = MagicMock()
            fake_llm.complete_structured.side_effect = [
                _make_spec(requirements[0]),
                _make_spec(requirements[1]),
            ]

            stage = SpecStage(llm=fake_llm, max_parallel=2)
            ctx = PipelineContext(input_path="test", requirements=requirements)
            stage.run(ctx)

            await asyncio.sleep(0.05)

            # Drain everything the Logs tab would have received
            received = []
            while not queue.empty():
                received.append(queue.get_nowait())

            event_types = {e["event"] for e in received}
            # All 5 spec event types must reach the subscriber
            assert "spec_gen_layer_started" in event_types
            assert "spec_gen_started" in event_types
            assert "spec_handoff" in event_types  # default mock returns 1.0 → 1 handoff
            assert "spec_gen_completed" in event_types
            assert "spec_gen_layer_completed" in event_types
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_spec_stage_emits_eval_rubric_event(_mock_evaluate_spec):
    """SpecStage emits an eval_rubric event with per-metric breakdown for each attempt."""
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)
            req = _make_req(1)
            fake_llm = MagicMock()
            fake_llm.complete_structured.return_value = _make_spec(req)

            # Simulate multi-metric DeepEval results
            _mock_evaluate_spec.return_value = {
                "Spec Correctness": {
                    "score": 0.85,
                    "passed": True,
                    "reason": "Logic is correct",
                },
                "Spec Coherence": {
                    "score": 0.78,
                    "passed": True,
                    "reason": "Mostly clear",
                },
                "Spec Instruction Following": {
                    "score": 0.92,
                    "passed": True,
                    "reason": "Follows the brief",
                },
                "Spec Safety": {
                    "score": 0.95,
                    "passed": True,
                    "reason": "No issues",
                },
            }

            stage = SpecStage(llm=fake_llm, max_handoffs=3, eval_threshold=0.8)
            ctx = PipelineContext(input_path="test", requirements=[req])
            stage.run(ctx)

            await asyncio.sleep(0.05)

            events = []
            while not queue.empty():
                events.append(queue.get_nowait())

            rubric_events = [e for e in events if e["event"] == "eval_rubric"]
            assert len(rubric_events) >= 1
            rubric = rubric_events[0]
            assert rubric["requirement_id"] == "req-1"
            assert rubric["attempt"] == 1
            assert rubric["max_handoffs"] == 3
            assert rubric["avg_score"] == pytest.approx(0.875, abs=0.01)
            assert rubric["threshold"] == 0.8

            # All four metrics should be in the payload
            metrics = rubric["metrics"]
            assert len(metrics) == 4
            metric_names = {m["name"] for m in metrics}
            assert "Spec Correctness" in metric_names
            assert "Spec Coherence" in metric_names
            assert "Spec Instruction Following" in metric_names
            assert "Spec Safety" in metric_names

            # Each metric has score, passed, reason
            correctness = next(m for m in metrics if m["name"] == "Spec Correctness")
            assert correctness["score"] == 0.85
            assert correctness["passed"] is True
            assert "Logic is correct" in correctness["reason"]
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_spec_stage_emits_one_rubric_per_attempt(_mock_evaluate_spec):
    """When refinement loops, one rubric event is emitted per attempt."""
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)
            req = _make_req(1)
            fake_llm = MagicMock()
            fake_llm.complete_structured.return_value = _make_spec(req)

            # Three attempts: low → medium → high (early-exit on the third)
            _mock_evaluate_spec.side_effect = [
                {"m1": {"score": 0.3, "passed": False, "reason": "weak"}},
                {"m1": {"score": 0.6, "passed": False, "reason": "better"}},
                {"m1": {"score": 0.95, "passed": True, "reason": "good"}},
            ]

            stage = SpecStage(llm=fake_llm, max_handoffs=5, eval_threshold=0.8)
            ctx = PipelineContext(input_path="test", requirements=[req])
            stage.run(ctx)

            await asyncio.sleep(0.05)

            events = []
            while not queue.empty():
                events.append(queue.get_nowait())

            rubric_events = [e for e in events if e["event"] == "eval_rubric"]
            assert len(rubric_events) == 3
            assert [r["attempt"] for r in rubric_events] == [1, 2, 3]
            assert [r["avg_score"] for r in rubric_events] == [0.3, 0.6, 0.95]
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_spec_stage_handoff_progress_events(_mock_evaluate_spec):
    """Handoff iterations emit spec_handoff progress events."""
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)
            req = _make_req(1)
            fake_llm = MagicMock()
            fake_llm.complete_structured.return_value = _make_spec(req)

            _mock_evaluate_spec.side_effect = [
                {"m": {"score": 0.3, "passed": False, "reason": "x"}},
                {"m": {"score": 0.6, "passed": False, "reason": "x"}},
                {"m": {"score": 0.95, "passed": True, "reason": "x"}},
            ]

            stage = SpecStage(llm=fake_llm, max_handoffs=5, eval_threshold=0.8)
            ctx = PipelineContext(input_path="test", requirements=[req])
            stage.run(ctx)

            await asyncio.sleep(0.05)

            events = []
            while not queue.empty():
                events.append(queue.get_nowait())

            handoff_events = [e for e in events if e["event"] == "spec_handoff"]
            assert len(handoff_events) == 3  # 3 attempts before threshold met
            # Final event should have the final_score
            completed = [e for e in events if e["event"] == "spec_gen_completed"]
            assert len(completed) == 1
            assert completed[0]["final_score"] >= 0.8
            assert completed[0]["attempts"] == 3
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_spec_stage_emits_progress_events():
    """SpecStage emits progress events through the global broker."""
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)
            requirements = [_make_req(i) for i in range(2)]

            fake_llm = MagicMock()
            fake_llm.complete_structured.side_effect = [
                _make_spec(requirements[0]),
                _make_spec(requirements[1]),
            ]
            stage = SpecStage(llm=fake_llm, max_parallel=2)
            ctx = PipelineContext(input_path="test", requirements=requirements)
            stage.run(ctx)

            # Allow callbacks to drain
            await asyncio.sleep(0.05)

            events = []
            while not queue.empty():
                events.append(queue.get_nowait())

            event_types = [e["event"] for e in events]
            assert "spec_gen_layer_started" in event_types
            assert event_types.count("spec_gen_started") == 2
            assert event_types.count("spec_gen_completed") == 2
            assert "spec_gen_layer_completed" in event_types
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


# ── Decomposition tests ─────────────────────────────────────────────────────


def _make_plan(req_id: str, titles: list[str], deps: dict[str, list[str]] | None = None):
    """Build a `_SpecPlan` with the given sibling titles and title-based deps."""
    from dark_factory.stages.spec import _PlannedSpec, _SpecPlan

    deps = deps or {}
    return _SpecPlan(
        parent_requirement_id=req_id,
        specs=[
            _PlannedSpec(
                title=t,
                description=f"purpose of {t}",
                capability=t.lower().replace(" ", "-"),
                depends_on=deps.get(t, []),
                rationale="test",
            )
            for t in titles
        ],
    )


def _decomposition_side_effect(plans_by_req: dict[str, object]):
    """Return a side_effect for ``llm.complete_structured`` that returns a
    planned ``_SpecPlan`` for planner calls and a generated ``Spec`` for
    refinement calls. Uses ``response_model`` to distinguish.
    """
    from dark_factory.stages.spec import _SpecPlan

    def side_effect(*, prompt, response_model, system):
        if response_model is _SpecPlan:
            # Extract the parent requirement id from the planner prompt
            for line in prompt.split("\n"):
                if line.strip().startswith("ID:"):
                    rid = line.split(":", 1)[1].strip()
                    plan = plans_by_req.get(rid)
                    if plan is not None:
                        return plan
                    break
            raise RuntimeError(f"no plan configured for prompt: {prompt[:120]}…")
        # Refinement: look for the target spec id hint in the prompt
        target_id = None
        for marker in ('id (MUST equal "', '"{target_spec_id}"'):
            pass
        for line in prompt.split("\n"):
            if 'id: MUST equal "' in line or 'id (MUST equal "' in line:
                try:
                    target_id = line.split('"', 2)[1]
                except IndexError:
                    pass
                break
        parent_req_id = None
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("ID:") and parent_req_id is None:
                parent_req_id = stripped.split(":", 1)[1].strip()
        return Spec(
            id=target_id or f"spec-{parent_req_id or 'unknown'}",
            title="generated",
            description="refined description",
            requirement_ids=[parent_req_id or "unknown"],
            acceptance_criteria=["must work"],
            capability="test-cap",
        )

    return side_effect


def test_decomposition_default_is_off():
    """SpecStage defaults to enable_decomposition=False so existing callers
    that only mock `complete_structured` → Spec keep working."""
    stage = SpecStage(llm=MagicMock())
    assert stage.enable_decomposition is False
    assert stage.max_specs_per_requirement == 12


def test_decomposition_enabled_plans_then_refines():
    """With decomposition on, one requirement → multiple sub-specs, each
    with the parent requirement id in requirement_ids."""
    req = _make_req(1)
    plan = _make_plan("req-1", ["Auth", "Session", "API"])
    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = _decomposition_side_effect(
        {"req-1": plan}
    )

    stage = SpecStage(llm=fake_llm, enable_decomposition=True)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 3
    assert [s.id for s in result.specs] == [
        "spec-req-1-00",
        "spec-req-1-01",
        "spec-req-1-02",
    ]
    for s in result.specs:
        assert "req-1" in s.requirement_ids


def test_decomposition_resolves_title_dependencies():
    """Planner title-based deps become generated spec-id deps."""
    req = _make_req(1)
    plan = _make_plan(
        "req-1",
        ["Auth", "Session", "API"],
        deps={"Session": ["Auth"], "API": ["Auth", "Session"]},
    )
    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = _decomposition_side_effect(
        {"req-1": plan}
    )

    stage = SpecStage(llm=fake_llm, enable_decomposition=True)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    by_index = {s.id: s for s in result.specs}
    auth = by_index["spec-req-1-00"]
    session = by_index["spec-req-1-01"]
    api = by_index["spec-req-1-02"]

    assert auth.dependencies == []
    assert session.dependencies == ["spec-req-1-00"]
    assert sorted(api.dependencies) == ["spec-req-1-00", "spec-req-1-01"]


def test_decomposition_unknown_title_in_depends_on_is_ignored():
    """Planner-fabricated dep titles that don't match any sibling are dropped
    silently (with a debug log); the pipeline does not crash."""
    req = _make_req(1)
    plan = _make_plan(
        "req-1",
        ["Auth", "Session"],
        deps={"Session": ["Auth", "Ghost"]},
    )
    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = _decomposition_side_effect(
        {"req-1": plan}
    )

    stage = SpecStage(llm=fake_llm, enable_decomposition=True)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    by_index = {s.id: s for s in result.specs}
    assert by_index["spec-req-1-01"].dependencies == ["spec-req-1-00"]


def test_decomposition_planner_failure_falls_back_to_single_spec():
    """When the planner raises, SpecStage falls back to a single-spec plan
    that still uses the decomposed refine path, so one sub-spec comes out
    with id `spec-<req>-00` and a `spec_plan_failed` event is emitted."""
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker
    from dark_factory.stages.spec import _SpecPlan

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)

            req = _make_req(1)

            def side_effect(*, prompt, response_model, system):
                if response_model is _SpecPlan:
                    raise RuntimeError("planner LLM exploded")
                return Spec(
                    id="spec-req-1-00",
                    title=req.title,
                    description="x",
                    requirement_ids=["req-1"],
                    acceptance_criteria=["c"],
                    capability="c",
                )

            fake_llm = MagicMock()
            fake_llm.complete_structured.side_effect = side_effect

            stage = SpecStage(llm=fake_llm, enable_decomposition=True)
            ctx = PipelineContext(input_path="test", requirements=[req])
            result = stage.run(ctx)

            # Single spec generated via the fallback plan
            assert len(result.specs) == 1
            assert result.specs[0].id == "spec-req-1-00"

            # And a spec_plan_failed event was emitted
            await asyncio.sleep(0.05)
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
            event_types = [e["event"] for e in events]
            assert "spec_plan_failed" in event_types
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_decomposition_respects_max_specs_per_requirement():
    """When the planner returns more sub-specs than the cap, the cap wins."""
    req = _make_req(1)
    plan = _make_plan("req-1", [f"Slice{i}" for i in range(10)])
    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = _decomposition_side_effect(
        {"req-1": plan}
    )

    stage = SpecStage(
        llm=fake_llm,
        enable_decomposition=True,
        max_specs_per_requirement=4,
    )
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 4
    assert [s.id for s in result.specs] == [
        "spec-req-1-00",
        "spec-req-1-01",
        "spec-req-1-02",
        "spec-req-1-03",
    ]


def test_decomposition_forces_target_id_even_when_llm_ignores_it():
    """If the architect returns the wrong spec id, the stage overwrites it
    so dependency resolution stays consistent."""
    req = _make_req(1)
    plan = _make_plan("req-1", ["Auth"])
    fake_llm = MagicMock()

    from dark_factory.stages.spec import _SpecPlan

    def side_effect(*, prompt, response_model, system):
        if response_model is _SpecPlan:
            return plan
        return Spec(
            id="wrong-id",  # architect ignored the instruction
            title="Auth",
            description="x",
            requirement_ids=[],  # also missing the parent req id
            acceptance_criteria=["c"],
            capability="c",
        )

    fake_llm.complete_structured.side_effect = side_effect
    stage = SpecStage(llm=fake_llm, enable_decomposition=True)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 1
    assert result.specs[0].id == "spec-req-1-00"
    assert "req-1" in result.specs[0].requirement_ids


def test_decomposition_emits_plan_events():
    """Planning phase emits spec_plan_started / spec_plan_completed /
    spec_plan_resolved events via the broker."""
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)
            req = _make_req(1)
            plan = _make_plan("req-1", ["Auth", "Session"], deps={"Session": ["Auth"]})
            fake_llm = MagicMock()
            fake_llm.complete_structured.side_effect = _decomposition_side_effect(
                {"req-1": plan}
            )

            stage = SpecStage(llm=fake_llm, enable_decomposition=True)
            ctx = PipelineContext(input_path="test", requirements=[req])
            stage.run(ctx)

            await asyncio.sleep(0.05)
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())

            event_types = [e["event"] for e in events]
            assert "spec_plan_started" in event_types
            assert "spec_plan_completed" in event_types
            assert "spec_plan_resolved" in event_types

            plan_completed = next(e for e in events if e["event"] == "spec_plan_completed")
            assert plan_completed["sub_spec_count"] == 2
            assert plan_completed["titles"] == ["Auth", "Session"]

            resolved = next(e for e in events if e["event"] == "spec_plan_resolved")
            assert resolved["resolved"] == 1
            assert resolved["unresolved"] == 0
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


# ── Preflight skip (reuse_existing_specs) ──────────────────────────────────


def test_spec_stage_preflight_skips_existing_specs_when_reuse_on():
    """When ``reuse_existing_specs=True`` and the graph repo reports a
    target spec id already exists, the stage must NOT invoke the LLM
    for that work unit. It loads the existing Spec from the repo and
    passes it through into ``context.specs`` unchanged.

    Regression test for the "spec swarm duplicates work on the same
    requirements across re-runs" complaint. Decomposition is off here
    so work units are 1:1 with requirements and target ids are simply
    ``spec-<req.id>``.
    """
    req_reused = _make_req(1)
    req_new = _make_req(2)
    reused_target_id = f"spec-{req_reused.id}"
    new_target_id = f"spec-{req_new.id}"

    # The reused spec as it would have been persisted on a previous run.
    reused_spec = Spec(
        id=reused_target_id,
        title="Previously generated spec",
        description="from last run",
        requirement_ids=[req_reused.id],
        acceptance_criteria=["already validated"],
        capability="auth",
    )

    fake_llm = MagicMock()
    # Only req_new should hit the LLM.
    fake_llm.complete_structured.return_value = _make_spec(req_new)

    # Mock repo: reports reused_target_id exists, returns the full spec
    # from get_specs. new_target_id is NOT reported so refinement runs.
    fake_repo = MagicMock()
    fake_repo.existing_spec_ids.return_value = {reused_target_id}
    fake_repo.get_specs.return_value = [reused_spec]

    stage = SpecStage(
        llm=fake_llm,
        graph_repo=fake_repo,
        reuse_existing_specs=True,
        enable_decomposition=False,
    )
    ctx = PipelineContext(input_path="test", requirements=[req_reused, req_new])
    result = stage.run(ctx)

    # Both specs end up in context.specs — the reused one passed through,
    # the new one freshly generated.
    assert len(result.specs) == 2
    result_ids = {s.id for s in result.specs}
    assert reused_target_id in result_ids
    assert new_target_id in result_ids

    # The LLM was called exactly once — for the non-reused requirement.
    # Anything higher means we didn't actually skip.
    assert fake_llm.complete_structured.call_count == 1

    # The repo was consulted for the full list of target ids, then
    # asked for the existing spec's full record.
    fake_repo.existing_spec_ids.assert_called_once()
    called_with = fake_repo.existing_spec_ids.call_args[0][0]
    assert set(called_with) == {reused_target_id, new_target_id}
    fake_repo.get_specs.assert_called_once_with([reused_target_id])


def test_spec_stage_preflight_skip_disabled_by_default():
    """Without ``reuse_existing_specs=True`` the preflight path is
    inert — the repo is never consulted and every requirement runs
    through the refinement loop. Protects unit tests and callers that
    don't provide a repo."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    fake_repo = MagicMock()

    # Default: reuse_existing_specs=False
    stage = SpecStage(llm=fake_llm, graph_repo=fake_repo)
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 1
    fake_repo.existing_spec_ids.assert_not_called()
    fake_repo.get_specs.assert_not_called()
    assert fake_llm.complete_structured.call_count == 1


def test_spec_stage_preflight_handles_neo4j_errors_gracefully():
    """If the bulk existing_spec_ids query raises (e.g. Neo4j flaking),
    the stage must NOT fail the run — it just skips the optimisation
    and does the full work. Reliability > performance."""
    req = _make_req(1)
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _make_spec(req)

    fake_repo = MagicMock()
    fake_repo.existing_spec_ids.side_effect = RuntimeError("neo4j down")

    stage = SpecStage(
        llm=fake_llm,
        graph_repo=fake_repo,
        reuse_existing_specs=True,
    )
    ctx = PipelineContext(input_path="test", requirements=[req])
    result = stage.run(ctx)

    assert len(result.specs) == 1
    # Full refinement ran despite the preflight failure.
    assert fake_llm.complete_structured.call_count == 1


def test_spec_stage_log_events_identify_sub_specs_when_decomposition_on():
    """With decomposition on, a single parent requirement fans out into
    multiple sub-specs, all of which would otherwise share the same
    ``requirement_id`` / ``requirement_title`` in their log events and
    become indistinguishable. This regression test captures the
    progress broker stream and asserts that every per-attempt event
    (``spec_handoff``, ``eval_rubric``, ``spec_gen_completed``) carries
    both a ``target_spec_id`` AND a ``sub_spec_title`` — the two fields
    the Agent Logs tab uses to tell sub-specs apart.

    Investigation of the running docker container showed 108
    ``spec_handoff`` log lines across 9 requirements and 69 sub-specs
    that all *looked* like duplication of the same work — because the
    ``target_spec_id`` field was missing from every log event. This
    test pins the fix so the fields can't regress silently.
    """
    import asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker
    from dark_factory.stages.spec import _PlannedSpec, _SpecPlan

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)

            req = _make_req(1)
            # Mock the planner to return 3 distinct sub-specs so the
            # refine loop runs 3 times for the same parent requirement.
            planned_specs = [
                _PlannedSpec(
                    title="Auth flow",
                    description="Login + logout",
                    capability="auth",
                    depends_on=[],
                    rationale="",
                ),
                _PlannedSpec(
                    title="Session storage",
                    description="JWT token persistence",
                    capability="auth",
                    depends_on=[],
                    rationale="",
                ),
                _PlannedSpec(
                    title="Password reset",
                    description="Email reset flow",
                    capability="auth",
                    depends_on=[],
                    rationale="",
                ),
            ]

            fake_llm = MagicMock()
            # One spec per sub-spec refinement. The architect prompt
            # forces spec.id to equal target_spec_id inside _refine_spec
            # so we can return a simple stub here.
            fake_llm.complete_structured.side_effect = [
                _SpecPlan(parent_requirement_id=req.id, specs=planned_specs),
                _make_spec(req),
                _make_spec(req),
                _make_spec(req),
            ]

            stage = SpecStage(
                llm=fake_llm,
                max_parallel=1,  # serialise so event ordering is deterministic
                enable_decomposition=True,
            )
            ctx = PipelineContext(input_path="test", requirements=[req])
            stage.run(ctx)

            await asyncio.sleep(0.05)

            events = []
            while not queue.empty():
                events.append(queue.get_nowait())

            # Every spec_handoff event MUST identify its sub-spec.
            # Without this fix, they'd all share the parent req id and
            # be indistinguishable in the Agent Logs tab.
            handoff_events = [e for e in events if e["event"] == "spec_handoff"]
            assert len(handoff_events) == 3, (
                f"expected 1 handoff per sub-spec (3 total), got {len(handoff_events)}"
            )
            target_ids = {e.get("target_spec_id") for e in handoff_events}
            assert target_ids == {
                "spec-req-1-00",
                "spec-req-1-01",
                "spec-req-1-02",
            }, f"target_spec_id missing/wrong in handoff events: {target_ids}"

            sub_titles = {e.get("sub_spec_title") for e in handoff_events}
            assert sub_titles == {
                "Auth flow",
                "Session storage",
                "Password reset",
            }, f"sub_spec_title missing/wrong in handoff events: {sub_titles}"

            # eval_rubric events carry the same fields for the Logs
            # tab's per-metric breakdown.
            rubric_events = [e for e in events if e["event"] == "eval_rubric"]
            assert len(rubric_events) == 3
            for ev in rubric_events:
                assert ev.get("target_spec_id", "").startswith("spec-req-1-"), (
                    f"eval_rubric missing target_spec_id: {ev}"
                )
                assert ev.get("sub_spec_title") in {
                    "Auth flow",
                    "Session storage",
                    "Password reset",
                }

            # spec_gen_completed already had the field, verify it still
            # flows correctly alongside the new ones.
            completed_events = [
                e for e in events if e["event"] == "spec_gen_completed"
            ]
            assert len(completed_events) == 3
            for ev in completed_events:
                assert ev.get("sub_spec_title") in {
                    "Auth flow",
                    "Session storage",
                    "Password reset",
                }
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_spec_stage_preflight_skips_all_means_no_executor_spinup():
    """When every target spec id is already present, the stage must
    return the loaded specs without submitting anything to the thread
    pool — no LLM calls at all."""
    req_a = _make_req(1)
    req_b = _make_req(2)
    spec_a = Spec(
        id=f"spec-{req_a.id}",
        title="a",
        description="a",
        requirement_ids=[req_a.id],
    )
    spec_b = Spec(
        id=f"spec-{req_b.id}",
        title="b",
        description="b",
        requirement_ids=[req_b.id],
    )

    fake_llm = MagicMock()
    fake_repo = MagicMock()
    fake_repo.existing_spec_ids.return_value = {spec_a.id, spec_b.id}
    fake_repo.get_specs.return_value = [spec_a, spec_b]

    stage = SpecStage(
        llm=fake_llm,
        graph_repo=fake_repo,
        reuse_existing_specs=True,
    )
    ctx = PipelineContext(input_path="test", requirements=[req_a, req_b])
    result = stage.run(ctx)

    assert {s.id for s in result.specs} == {spec_a.id, spec_b.id}
    fake_llm.complete_structured.assert_not_called()
