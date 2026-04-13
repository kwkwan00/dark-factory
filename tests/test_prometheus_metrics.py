"""Tests for Prometheus metrics observers and the /metrics HTTP endpoint.

Unit tests for the observer helpers come first (no api_client needed),
followed by the endpoint integration test that requires the ``api_client``
fixture.
"""

from __future__ import annotations

import pytest

from dark_factory.metrics import prometheus as prom


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


# ── Helpers ────────────────────────────────────────────────────────────────


def _counter_value(counter, **labels) -> float:
    """Read the current value of a labelled Counter."""
    return counter.labels(**labels)._value.get()


def _histogram_count(histogram, **labels) -> int:
    """Read the sample count of a labelled Histogram."""
    if labels:
        h = histogram.labels(**labels)
    else:
        h = histogram
    return int(h._sum._count.get()) if hasattr(h._sum, '_count') else sum(
        b.get() for b in h._buckets
    )


def _gauge_value(gauge, **labels) -> float:
    """Read the current value of a Gauge."""
    if labels:
        return gauge.labels(**labels)._value.get()
    return gauge._value.get()


# ── Unit tests: observer helpers ───────────────────────────────────────────


def test_observe_llm_call_increments_counter():
    """observe_llm_call increments the llm_calls_total counter."""
    baseline = _counter_value(prom.llm_calls_total, client="test", model="m1", phase="p1")
    prom.observe_llm_call(client="test", model="m1", phase="p1", latency_seconds=0.5,
                          input_tokens=100, output_tokens=50)
    after = _counter_value(prom.llm_calls_total, client="test", model="m1", phase="p1")
    assert after - baseline == 1.0


def test_observe_llm_call_records_tokens():
    """observe_llm_call records input and output token counters."""
    baseline_in = _counter_value(prom.llm_tokens_total, client="test", model="m1", kind="input")
    baseline_out = _counter_value(prom.llm_tokens_total, client="test", model="m1", kind="output")
    prom.observe_llm_call(client="test", model="m1", phase="p1",
                          input_tokens=200, output_tokens=80)
    assert _counter_value(prom.llm_tokens_total, client="test", model="m1", kind="input") - baseline_in == 200
    assert _counter_value(prom.llm_tokens_total, client="test", model="m1", kind="output") - baseline_out == 80


def test_observe_pipeline_run_start_and_end():
    """observe_pipeline_run_start increments running counter; end decrements gauge."""
    baseline = _counter_value(prom.pipeline_runs_total, status="running")
    prom.observe_pipeline_run_start()
    assert _counter_value(prom.pipeline_runs_total, status="running") - baseline == 1.0

    baseline_success = _counter_value(prom.pipeline_runs_total, status="success")
    prom.observe_pipeline_run_end(status="success", duration_seconds=120.0)
    assert _counter_value(prom.pipeline_runs_total, status="success") - baseline_success == 1.0


def test_observe_feature_event():
    """observe_feature_event increments the feature_events_total counter."""
    baseline = _counter_value(prom.feature_events_total, event="started", status="unknown")
    prom.observe_feature_event(event="started")
    assert _counter_value(prom.feature_events_total, event="started", status="unknown") - baseline == 1.0


def test_observe_feature_event_completed_records_duration():
    """observe_feature_event with event=completed records duration histogram."""
    prom.observe_feature_event(event="completed", status="success", duration_seconds=45.0)
    # Verify the counter was incremented
    assert _counter_value(prom.feature_events_total, event="completed", status="success") >= 1.0


def test_observe_tool_call_success():
    """observe_tool_call increments tool_calls_total with success status."""
    baseline = _counter_value(prom.tool_calls_total, tool="write_file", agent="Coder", status="success")
    prom.observe_tool_call(tool="write_file", agent="Coder", success=True, latency_seconds=0.1)
    assert _counter_value(prom.tool_calls_total, tool="write_file", agent="Coder", status="success") - baseline == 1.0


def test_observe_tool_call_failure():
    """observe_tool_call with success=False increments failure counter."""
    baseline = _counter_value(prom.tool_calls_total, tool="run_tests", agent="Tester", status="failure")
    prom.observe_tool_call(tool="run_tests", agent="Tester", success=False)
    assert _counter_value(prom.tool_calls_total, tool="run_tests", agent="Tester", status="failure") - baseline == 1.0


def test_observe_agent_activation():
    """observe_agent_activation increments agent_activations_total."""
    baseline = _counter_value(prom.agent_activations_total, agent="Coder")
    prom.observe_agent_activation(agent="Coder")
    assert _counter_value(prom.agent_activations_total, agent="Coder") - baseline == 1.0


def test_observe_memory_op():
    """observe_memory_op increments memory_ops_total."""
    baseline = _counter_value(prom.memory_ops_total, operation="create", memory_type="pattern")
    prom.observe_memory_op(operation="create", memory_type="pattern")
    assert _counter_value(prom.memory_ops_total, operation="create", memory_type="pattern") - baseline == 1.0


def test_observe_memory_op_recall_hit():
    """observe_memory_op with operation=recall and count>0 records a hit."""
    baseline = _counter_value(prom.memory_recall_total, outcome="hit")
    prom.observe_memory_op(operation="recall", memory_type="pattern", count=3, latency_seconds=0.05)
    assert _counter_value(prom.memory_recall_total, outcome="hit") - baseline == 1.0


def test_observe_memory_op_recall_miss():
    """observe_memory_op with operation=recall and count=0 records a miss."""
    baseline = _counter_value(prom.memory_recall_total, outcome="miss")
    prom.observe_memory_op(operation="recall", memory_type="pattern", count=0)
    assert _counter_value(prom.memory_recall_total, outcome="miss") - baseline == 1.0


def test_observe_spec_plan_success():
    """observe_spec_plan increments spec_plan_outcomes_total."""
    baseline = _counter_value(prom.spec_plan_outcomes_total, outcome="success")
    prom.observe_spec_plan(outcome="success", sub_spec_count=3)
    assert _counter_value(prom.spec_plan_outcomes_total, outcome="success") - baseline == 1.0


def test_observe_eval_rubric():
    """observe_eval_rubric increments eval_rubric_total."""
    baseline = _counter_value(prom.eval_rubric_total, metric_name="correctness", passed="true")
    prom.observe_eval_rubric(metric_name="correctness", score=0.85, passed=True)
    assert _counter_value(prom.eval_rubric_total, metric_name="correctness", passed="true") - baseline == 1.0


def test_observe_incident():
    """observe_incident increments incidents_total."""
    baseline = _counter_value(prom.incidents_total, category="tool_crash", severity="high")
    prom.observe_incident(category="tool_crash", severity="high")
    assert _counter_value(prom.incidents_total, category="tool_crash", severity="high") - baseline == 1.0


def test_observe_artifact_write():
    """observe_artifact_write increments artifacts_written_total."""
    baseline = _counter_value(prom.artifacts_written_total, language="python", is_test="false")
    prom.observe_artifact_write(language="python", bytes_written=1024, is_test=False)
    assert _counter_value(prom.artifacts_written_total, language="python", is_test="false") - baseline == 1.0


def test_observe_deep_agent_invocation():
    """observe_deep_agent_invocation increments the deep agent counter."""
    baseline = _counter_value(prom.deep_agent_invocations_total, tool="code_review")
    prom.observe_deep_agent_invocation(tool="code_review")
    assert _counter_value(prom.deep_agent_invocations_total, tool="code_review") - baseline == 1.0


def test_observe_bg_loop_sample_sets_gauges():
    """observe_bg_loop_sample sets gauge values directly."""
    prom.observe_bg_loop_sample(active_task_count=5, completed_task_count=100, loop_restarts=2)
    assert _gauge_value(prom.background_loop_active_tasks) == 5.0
    assert _gauge_value(prom.background_loop_completed_tasks) == 100.0
    assert _gauge_value(prom.background_loop_restarts) == 2.0


def test_generate_latest_returns_bytes():
    """generate_latest() returns bytes containing exposition text."""
    from prometheus_client import generate_latest

    prom.observe_llm_call(client="test", model="m1", phase="gen", latency_seconds=0.1,
                          input_tokens=10, output_tokens=5)
    output = generate_latest()
    assert isinstance(output, bytes)
    assert b"dark_factory_llm_calls_total" in output


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
