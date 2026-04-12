"""Unit tests for the Prometheus metrics collectors.

prometheus_client collectors are module-level singletons on the default
registry, so these tests read ``collector._value.get()`` (for simple
Counters/Gauges) or sample values via ``generate_latest`` and parse the
exposition output. To keep tests order-independent, we capture the
current value before the assertion and assert a **delta**, never an
absolute value.
"""

from __future__ import annotations

import pytest
from prometheus_client import generate_latest


@pytest.fixture(autouse=True)
def _reset_prometheus_collectors():
    """Clear every ``dark_factory_*`` collector's internal state before
    each test.

    prometheus_client collectors are module-level singletons on the
    default registry. Without this fixture, test order affects absolute
    counter values and a failed test can leave half-observed histograms
    that silently skew the next test's delta reads. We call ``clear()``
    on every Counter / Histogram / Gauge we declared so each test starts
    from a known-zero baseline.
    """
    from dark_factory.metrics import prometheus as _prom

    collectors = [
        attr
        for name, attr in vars(_prom).items()
        if hasattr(attr, "clear") and hasattr(attr, "_metrics")
    ]
    for c in collectors:
        try:
            c.clear()
        except Exception:
            # Labelled histograms without any labels set can refuse clear()
            # — safe to ignore because they're zero-state anyway.
            pass
    yield
    for c in collectors:
        try:
            c.clear()
        except Exception:
            pass


def _counter_value(counter, **labels) -> float:
    """Read the current value of a labelled Counter."""
    try:
        return counter.labels(**labels)._value.get() if labels else counter._value.get()
    except Exception:
        return 0.0


def _histogram_count(histogram, **labels) -> int:
    """Read the current observation count of a labelled Histogram.

    prometheus_client stores per-bucket (non-cumulative) counts internally;
    the total observation count is the sum across all buckets.
    """
    try:
        target = histogram.labels(**labels) if labels else histogram
        total = 0
        for b in target._buckets:
            value = b.get() if hasattr(b, "get") else b
            total += int(value)
        return total
    except Exception:
        return 0


def _gauge_value(gauge, **labels) -> float:
    try:
        return gauge.labels(**labels)._value.get() if labels else gauge._value.get()
    except Exception:
        return 0.0


# ── observe_llm_call ─────────────────────────────────────────────────────────


def test_observe_llm_call_increments_counters_and_histograms():
    from dark_factory.metrics.prometheus import (
        llm_calls_total,
        llm_cost_usd_total,
        llm_latency_seconds,
        llm_tokens_total,
        observe_llm_call,
    )

    before_calls = _counter_value(
        llm_calls_total,
        client="anthropic", model="claude-sonnet-4-6", phase="spec_refine",
    )
    before_input = _counter_value(
        llm_tokens_total,
        client="anthropic", model="claude-sonnet-4-6", kind="input",
    )
    before_cost = _counter_value(
        llm_cost_usd_total, client="anthropic", model="claude-sonnet-4-6",
    )
    before_latency_count = _histogram_count(
        llm_latency_seconds, client="anthropic", model="claude-sonnet-4-6",
    )

    observe_llm_call(
        client="anthropic",
        model="claude-sonnet-4-6",
        phase="spec_refine",
        latency_seconds=1.5,
        time_to_first_token_seconds=0.3,
        input_tokens=500,
        output_tokens=1200,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cost_usd=0.02,
    )

    assert _counter_value(
        llm_calls_total,
        client="anthropic", model="claude-sonnet-4-6", phase="spec_refine",
    ) == before_calls + 1
    assert _counter_value(
        llm_tokens_total,
        client="anthropic", model="claude-sonnet-4-6", kind="input",
    ) == before_input + 500
    assert _counter_value(
        llm_cost_usd_total, client="anthropic", model="claude-sonnet-4-6",
    ) == pytest.approx(before_cost + 0.02)
    assert _histogram_count(
        llm_latency_seconds, client="anthropic", model="claude-sonnet-4-6",
    ) == before_latency_count + 1


def test_observe_llm_call_error_increments_errors_counter():
    from dark_factory.metrics.prometheus import llm_errors_total, observe_llm_call

    before = _counter_value(
        llm_errors_total,
        client="anthropic", model="claude-sonnet-4-6", reason="rate_limited",
    )

    observe_llm_call(
        client="anthropic",
        model="claude-sonnet-4-6",
        error="rate limited",
        rate_limited=True,
        http_status=429,
    )

    assert _counter_value(
        llm_errors_total,
        client="anthropic", model="claude-sonnet-4-6", reason="rate_limited",
    ) == before + 1


def test_observe_llm_call_http_5xx_error_classified_correctly():
    from dark_factory.metrics.prometheus import llm_errors_total, observe_llm_call

    before = _counter_value(
        llm_errors_total, client="anthropic", model="claude-3-5-haiku", reason="http_5xx",
    )

    observe_llm_call(
        client="anthropic",
        model="claude-3-5-haiku",
        error="Internal server error",
        http_status=503,
    )

    assert _counter_value(
        llm_errors_total, client="anthropic", model="claude-3-5-haiku", reason="http_5xx",
    ) == before + 1


# ── observe_pipeline_run_* ──────────────────────────────────────────────────


def test_observe_pipeline_run_start_and_end_gauges():
    from dark_factory.metrics.prometheus import (
        observe_pipeline_run_end,
        observe_pipeline_run_start,
        pipeline_duration_seconds,
        pipeline_runs_total,
        running_pipelines,
    )

    before_running = _gauge_value(running_pipelines)
    before_success = _counter_value(pipeline_runs_total, status="success")

    observe_pipeline_run_start(run_id="run-test")
    assert _gauge_value(running_pipelines) == before_running + 1

    observe_pipeline_run_end(status="success", duration_seconds=42.0)
    assert _gauge_value(running_pipelines) == before_running
    assert _counter_value(pipeline_runs_total, status="success") == before_success + 1
    assert _histogram_count(pipeline_duration_seconds, status="success") >= 1


# ── observe_feature_event ───────────────────────────────────────────────────


def test_observe_feature_event_completed_records_duration():
    from dark_factory.metrics.prometheus import (
        feature_duration_seconds,
        feature_events_total,
        observe_feature_event,
    )

    before_count = _counter_value(
        feature_events_total, event="completed", status="success",
    )
    before_hist = feature_duration_seconds._sum.get()

    observe_feature_event(
        event="completed", status="success", duration_seconds=12.5,
    )

    assert _counter_value(
        feature_events_total, event="completed", status="success",
    ) == before_count + 1
    assert feature_duration_seconds._sum.get() == pytest.approx(before_hist + 12.5)


# ── observe_tool_call ───────────────────────────────────────────────────────


def test_observe_tool_call_success_vs_failure():
    from dark_factory.metrics.prometheus import (
        observe_tool_call,
        tool_calls_total,
    )

    before_ok = _counter_value(
        tool_calls_total, tool="write_file", agent="coder", status="success",
    )
    before_fail = _counter_value(
        tool_calls_total, tool="write_file", agent="coder", status="failure",
    )

    observe_tool_call(
        tool="write_file", agent="coder", success=True, latency_seconds=0.1,
    )
    observe_tool_call(
        tool="write_file", agent="coder", success=False,
        error="disk full", latency_seconds=0.05,
    )

    assert _counter_value(
        tool_calls_total, tool="write_file", agent="coder", status="success",
    ) == before_ok + 1
    assert _counter_value(
        tool_calls_total, tool="write_file", agent="coder", status="failure",
    ) == before_fail + 1


# ── observe_agent_activation + handoff ─────────────────────────────────────


def test_observe_agent_activation_and_handoff():
    from dark_factory.metrics.prometheus import (
        agent_activations_total,
        agent_handoffs_total,
        observe_agent_activation,
        observe_agent_handoff,
    )

    before_act = _counter_value(agent_activations_total, agent="planner")
    before_ho = _counter_value(
        agent_handoffs_total, from_agent="planner", to_agent="coder",
    )

    observe_agent_activation("planner")
    observe_agent_handoff(from_agent="planner", to_agent="coder")

    assert _counter_value(agent_activations_total, agent="planner") == before_act + 1
    assert _counter_value(
        agent_handoffs_total, from_agent="planner", to_agent="coder",
    ) == before_ho + 1


# ── observe_memory_op ───────────────────────────────────────────────────────


def test_observe_memory_op_recall_hit_vs_miss():
    from dark_factory.metrics.prometheus import (
        memory_ops_total,
        memory_recall_total,
        observe_memory_op,
    )

    before_hit = _counter_value(memory_recall_total, outcome="hit")
    before_miss = _counter_value(memory_recall_total, outcome="miss")
    before_ops = _counter_value(
        memory_ops_total, operation="recall", memory_type="none",
    )

    observe_memory_op(operation="recall", count=5, latency_seconds=0.08)
    observe_memory_op(operation="recall", count=0, latency_seconds=0.02)

    assert _counter_value(memory_recall_total, outcome="hit") == before_hit + 1
    assert _counter_value(memory_recall_total, outcome="miss") == before_miss + 1
    assert _counter_value(
        memory_ops_total, operation="recall", memory_type="none",
    ) == before_ops + 2


def test_observe_memory_op_create_pattern():
    from dark_factory.metrics.prometheus import (
        memory_ops_total,
        observe_memory_op,
    )

    before = _counter_value(
        memory_ops_total, operation="create", memory_type="pattern",
    )

    observe_memory_op(
        operation="create", memory_type="pattern", memory_id="pattern-abc",
    )

    assert _counter_value(
        memory_ops_total, operation="create", memory_type="pattern",
    ) == before + 1


# ── observe_spec_plan + eval rubric ─────────────────────────────────────────


def test_observe_spec_plan_outcomes():
    from dark_factory.metrics.prometheus import (
        decomposition_sub_specs,
        observe_spec_plan,
        spec_plan_outcomes_total,
    )

    before_success = _counter_value(spec_plan_outcomes_total, outcome="success")
    before_fallback = _counter_value(spec_plan_outcomes_total, outcome="fallback")
    before_hist = decomposition_sub_specs._sum.get()

    observe_spec_plan(outcome="success", sub_spec_count=5)
    observe_spec_plan(outcome="fallback", sub_spec_count=1)

    assert _counter_value(spec_plan_outcomes_total, outcome="success") == before_success + 1
    assert _counter_value(spec_plan_outcomes_total, outcome="fallback") == before_fallback + 1
    # Histogram only observes on 'success'
    assert decomposition_sub_specs._sum.get() == pytest.approx(before_hist + 5)


def test_observe_eval_rubric_records_counter_and_score_histogram():
    from dark_factory.metrics.prometheus import (
        eval_rubric_total,
        eval_score,
        observe_eval_rubric,
    )

    before_count = _counter_value(
        eval_rubric_total, metric_name="Spec Correctness", passed="true",
    )
    before_hist_sum = eval_score.labels(metric_name="Spec Correctness")._sum.get()

    observe_eval_rubric(
        metric_name="Spec Correctness", score=0.87, passed=True,
    )

    assert _counter_value(
        eval_rubric_total, metric_name="Spec Correctness", passed="true",
    ) == before_count + 1
    assert eval_score.labels(metric_name="Spec Correctness")._sum.get() == pytest.approx(
        before_hist_sum + 0.87
    )


# ── observe_incident + artifact + deep agent ───────────────────────────────


def test_observe_incident_counter():
    from dark_factory.metrics.prometheus import incidents_total, observe_incident

    before = _counter_value(
        incidents_total, category="llm", severity="error",
    )

    observe_incident(category="llm", severity="error")

    assert _counter_value(
        incidents_total, category="llm", severity="error",
    ) == before + 1


def test_observe_artifact_write_increments_all():
    from dark_factory.metrics.prometheus import (
        artifact_bytes_written,
        artifacts_written_total,
        bytes_written_total,
        observe_artifact_write,
    )

    before_files = _counter_value(
        artifacts_written_total, language="python", is_test="false",
    )
    before_bytes = _counter_value(bytes_written_total, language="python")
    before_hist = artifact_bytes_written._sum.get()

    observe_artifact_write(language="python", bytes_written=2048, is_test=False)

    assert _counter_value(
        artifacts_written_total, language="python", is_test="false",
    ) == before_files + 1
    assert _counter_value(bytes_written_total, language="python") == before_bytes + 2048
    assert artifact_bytes_written._sum.get() == pytest.approx(before_hist + 2048)


def test_observe_deep_agent_invocation_increments_subprocess_counter():
    from dark_factory.metrics.prometheus import (
        deep_agent_invocations_total,
        observe_deep_agent_invocation,
        subprocess_spawns_total,
    )

    before_deep = _counter_value(
        deep_agent_invocations_total, tool="claude_agent_codegen",
    )
    before_spawn = _counter_value(subprocess_spawns_total)

    observe_deep_agent_invocation("claude_agent_codegen")

    assert _counter_value(
        deep_agent_invocations_total, tool="claude_agent_codegen",
    ) == before_deep + 1
    assert _counter_value(subprocess_spawns_total) == before_spawn + 1


# ── observe_bg_loop_sample ──────────────────────────────────────────────────


def test_observe_bg_loop_sample_sets_gauges():
    from dark_factory.metrics.prometheus import (
        background_loop_active_tasks,
        background_loop_completed_tasks,
        background_loop_restarts,
        observe_bg_loop_sample,
    )

    observe_bg_loop_sample(
        active_task_count=3,
        completed_task_count=42,
        loop_restarts=1,
    )

    assert _gauge_value(background_loop_active_tasks) == 3
    assert _gauge_value(background_loop_completed_tasks) == 42
    assert _gauge_value(background_loop_restarts) == 1


# ── /metrics endpoint ───────────────────────────────────────────────────────


def test_metrics_endpoint_returns_prometheus_exposition(api_client):
    """The /metrics endpoint returns text/plain Prometheus exposition."""
    # Make sure at least one known collector has a non-zero entry so we can
    # grep for it. ``observe_llm_call`` registers a sample for the
    # ``dark_factory_llm_calls_total`` counter.
    from dark_factory.metrics.prometheus import observe_llm_call

    observe_llm_call(
        client="anthropic",
        model="claude-sonnet-4-6",
        phase="metrics_endpoint_test",
        latency_seconds=0.5,
        input_tokens=10,
        output_tokens=20,
    )

    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text

    # Key metrics should appear in the exposition
    assert "dark_factory_llm_calls_total" in body
    assert "dark_factory_pipeline_runs_total" in body
    assert "dark_factory_tool_calls_total" in body
    assert "dark_factory_incidents_total" in body
    assert "dark_factory_background_loop_active_tasks" in body
    # And the observation we just made should be present
    assert 'phase="metrics_endpoint_test"' in body


def test_generate_latest_includes_all_dark_factory_metrics():
    """Smoke check: generate_latest() exposition contains every headline
    dark_factory_* counter name."""
    body = generate_latest().decode()
    for name in [
        "dark_factory_llm_calls_total",
        "dark_factory_llm_tokens_total",
        "dark_factory_llm_cost_usd_total",
        "dark_factory_pipeline_runs_total",
        "dark_factory_pipeline_duration_seconds",
        "dark_factory_feature_events_total",
        "dark_factory_agent_activations_total",
        "dark_factory_agent_handoffs_total",
        "dark_factory_tool_calls_total",
        "dark_factory_tool_latency_seconds",
        "dark_factory_memory_ops_total",
        "dark_factory_memory_recall_total",
        "dark_factory_spec_plan_outcomes_total",
        "dark_factory_eval_rubric_total",
        "dark_factory_eval_score",
        "dark_factory_incidents_total",
        "dark_factory_artifacts_written_total",
        "dark_factory_background_loop_active_tasks",
        "dark_factory_worker_crashes_total",
        "dark_factory_deep_agent_invocations_total",
    ]:
        assert name in body, f"missing {name} in /metrics exposition"
