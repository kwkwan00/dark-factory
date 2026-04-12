"""Prometheus collectors for aggregate metrics.

This module is the single source of truth for every Prometheus metric
the app exposes. All collectors live on the default global registry so
``prometheus_client.generate_latest()`` picks them up automatically when
the FastAPI ``/metrics`` endpoint is hit.

Design principles:

- **Bounded label cardinality.** Labels here are limited to values drawn
  from small enums (status, event, client, model, phase, agent, tool,
  language, memory_type, category, severity, metric_name). High-cardinality
  fields like ``run_id``, ``requirement_id``, ``feature`` stay in Postgres.
- **One observer helper per logical event.** Instrumentation sites in the
  rest of the codebase call a single ``observe_*`` function which updates
  every related counter / histogram / gauge at once. This keeps hot paths
  uncluttered and confines all label normalisation to this module.
- **Always-on.** prometheus_client collectors are in-memory ints/floats
  with no I/O, so there's no feature flag. The ``/metrics`` endpoint is
  always exposed — the Postgres store remains optional independently.
- **Defensive.** Every helper wraps its work in a broad try/except. A
  Prometheus error must never take the pipeline down.

Naming convention: ``dark_factory_<subject>_<type>``
  - counters: ``*_total``
  - histograms: ``*_seconds`` / ``*_bytes`` / ``*_tokens``
  - gauges: no suffix
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import structlog
from prometheus_client import CONTENT_TYPE_LATEST  # noqa: F401 — re-export
from prometheus_client import REGISTRY
from prometheus_client import Counter, Gauge, Histogram, generate_latest  # noqa: F401 — re-export

log = structlog.get_logger()


# ── Buckets ──────────────────────────────────────────────────────────────────


# Latency buckets tuned for LLM calls: most < 5s, tails up to 5 minutes.
_LATENCY_BUCKETS_LLM = (
    0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0,
)

# Tool-call latency: most sub-second, tails up to 30s.
_LATENCY_BUCKETS_TOOL = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0,
)

# Feature / pipeline durations: seconds to tens of minutes.
_DURATION_BUCKETS_FEATURE = (
    1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0,
)
_DURATION_BUCKETS_PIPELINE = (
    10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, 7200.0,
)

# Token histograms — exponential so we can see both tiny and huge calls.
_TOKEN_BUCKETS = (
    50, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
)

# Eval scores always 0..1 — linear buckets.
_EVAL_SCORE_BUCKETS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
)

# Spec refinement attempt count.
_ATTEMPTS_BUCKETS = (1, 2, 3, 4, 5, 7, 10)

# Planner decomposition fanout.
_SUB_SPECS_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)

# Artifact bytes written.
_BYTES_BUCKETS = (
    256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216,
)


# ── LLM metrics ─────────────────────────────────────────────────────────────


llm_calls_total = Counter(
    "dark_factory_llm_calls_total",
    "Total LLM API calls.",
    ["client", "model", "phase"],
)

llm_tokens_total = Counter(
    "dark_factory_llm_tokens_total",
    "Total tokens consumed by LLM calls, split by kind.",
    ["client", "model", "kind"],  # kind = input | output | cache_read | cache_create
)

llm_cost_usd_total = Counter(
    "dark_factory_llm_cost_usd_total",
    "Cumulative USD cost of LLM calls (approximate, from rate table).",
    ["client", "model"],
)

llm_errors_total = Counter(
    "dark_factory_llm_errors_total",
    "LLM call failures split by reason.",
    ["client", "model", "reason"],  # reason = rate_limited | http_error | other
)

llm_retries_total = Counter(
    "dark_factory_llm_retries_total",
    "Total LLM retries (retry_count > 0 on a successful call).",
    ["client", "model"],
)

llm_latency_seconds = Histogram(
    "dark_factory_llm_latency_seconds",
    "End-to-end LLM call latency in seconds.",
    ["client", "model"],
    buckets=_LATENCY_BUCKETS_LLM,
)

llm_ttft_seconds = Histogram(
    "dark_factory_llm_time_to_first_token_seconds",
    "LLM time to first token in seconds.",
    ["client", "model"],
    buckets=_LATENCY_BUCKETS_LLM,
)

llm_input_tokens = Histogram(
    "dark_factory_llm_input_tokens",
    "LLM input token count distribution.",
    ["client", "model"],
    buckets=_TOKEN_BUCKETS,
)

llm_output_tokens = Histogram(
    "dark_factory_llm_output_tokens",
    "LLM output token count distribution.",
    ["client", "model"],
    buckets=_TOKEN_BUCKETS,
)


# ── Pipeline metrics ────────────────────────────────────────────────────────


pipeline_runs_total = Counter(
    "dark_factory_pipeline_runs_total",
    "Total pipeline runs, counted at run-start time (status=running) and "
    "re-counted at run-end time (status=success|partial|error).",
    ["status"],
)

pipeline_duration_seconds = Histogram(
    "dark_factory_pipeline_duration_seconds",
    "End-to-end pipeline run duration in seconds.",
    ["status"],
    buckets=_DURATION_BUCKETS_PIPELINE,
)

running_pipelines = Gauge(
    "dark_factory_running_pipelines",
    "Number of pipeline runs currently in the running state.",
)

# Phase 5 reconciliation — best-effort cross-feature polishing pass
# after all feature swarms complete. Outcome is ``clean`` (everything
# builds + tests pass), ``partial`` (some validation failing but no
# blocker), ``error`` (stage crashed), or ``skipped`` (disabled or
# nothing to reconcile).
reconciliation_runs_total = Counter(
    "dark_factory_reconciliation_runs_total",
    "Reconciliation phase outcomes, counted at phase completion.",
    ["status"],  # clean | partial | error | skipped
)

reconciliation_duration_seconds = Histogram(
    "dark_factory_reconciliation_duration_seconds",
    "End-to-end reconciliation phase duration in seconds.",
    ["status"],
    buckets=_DURATION_BUCKETS_PIPELINE,
)


# Phase 6 end-to-end validation (Playwright across chromium / firefox /
# webkit). Outcome is ``pass`` (every browser passed every test),
# ``partial`` (some failures but the app booted and tests ran),
# ``error`` (stage crashed / agent failed), or ``skipped`` (not a web
# app, reconciliation errored, or the feature is disabled).
e2e_validation_runs_total = Counter(
    "dark_factory_e2e_validation_runs_total",
    "E2E validation phase outcomes, counted at phase completion.",
    ["status"],  # pass | partial | error | skipped
)

e2e_validation_duration_seconds = Histogram(
    "dark_factory_e2e_validation_duration_seconds",
    "End-to-end E2E validation phase duration in seconds.",
    ["status"],
    buckets=_DURATION_BUCKETS_PIPELINE,
)

# Per-browser smoke-test counts — the sum across the labels in a
# single run should equal (tests_passed + tests_failed) from the
# stage result. Lets dashboards break "flaky on webkit" out from
# "flaky on chromium".
e2e_tests_total = Counter(
    "dark_factory_e2e_tests_total",
    "Playwright test results by browser and outcome.",
    ["browser", "status"],  # browser in {chromium, firefox, webkit}; status in {passed, failed}
)


# ── Memory (procedural memory graph) ────────────────────────────────────────
#
# Tier A introduces write-time dedup + recall effectiveness tracking.
# The counters below let dashboards answer:
#
# - "Is dedup actually catching duplicates?" → memory_writes_total
#   with outcome=deduped vs outcome=created.
# - "Are memories getting recalled at all, or is the graph dead
#   weight?" → memory_recalls_total.
# - "Is the feedback loop moving relevance scores in the right
#   direction?" → memory_relevance_adjustments_total.
memory_writes_total = Counter(
    "dark_factory_memory_writes_total",
    "Memory write operations. outcome=created (new node) or "
    "deduped (boosted an existing near-duplicate instead).",
    ["type", "outcome"],  # type in {pattern, mistake, solution, strategy}; outcome in {created, deduped}
)

memory_recalls_total = Counter(
    "dark_factory_memory_recalls_total",
    "Memory recall operations. hit=yes if at least one memory was "
    "returned, no if the recall returned an empty list.",
    ["type", "hit"],  # type in the 4 + episode; hit in {yes, no}
)

memory_relevance_adjustments_total = Counter(
    "dark_factory_memory_relevance_adjustments_total",
    "Memory relevance boost/demote operations driven by the "
    "eval-pass feedback loop. Direction=boost on eval pass, "
    "demote on eval fail.",
    ["type", "direction"],  # direction in {boost, demote, decay}
)


# ── Feature / swarm metrics ─────────────────────────────────────────────────


feature_events_total = Counter(
    "dark_factory_feature_events_total",
    "Feature lifecycle events from the swarm phase.",
    ["event", "status"],  # event = started | completed | skipped
)

feature_duration_seconds = Histogram(
    "dark_factory_feature_duration_seconds",
    "Per-feature swarm duration in seconds.",
    buckets=_DURATION_BUCKETS_FEATURE,
)

agent_activations_total = Counter(
    "dark_factory_agent_activations_total",
    "Agent activation events (each time a swarm agent takes a turn).",
    ["agent"],
)

agent_handoffs_total = Counter(
    "dark_factory_agent_handoffs_total",
    "Agent handoffs via transfer_to_* tools.",
    ["from_agent", "to_agent"],
)

worker_crashes_total = Counter(
    "dark_factory_worker_crashes_total",
    "Swarm worker thread crashes caught by the orchestrator's exception handler.",
)


# ── Tool call metrics ───────────────────────────────────────────────────────


tool_calls_total = Counter(
    "dark_factory_tool_calls_total",
    "Swarm agent tool invocations.",
    ["tool", "agent", "status"],  # status = success | failure
)

tool_latency_seconds = Histogram(
    "dark_factory_tool_latency_seconds",
    "Per-tool invocation latency in seconds.",
    ["tool"],
    buckets=_LATENCY_BUCKETS_TOOL,
)


# ── Deep agent metrics ──────────────────────────────────────────────────────


deep_agent_invocations_total = Counter(
    "dark_factory_deep_agent_invocations_total",
    "Invocations of Claude Agent SDK deep sub-agents.",
    ["tool"],
)

deep_agent_timeouts_total = Counter(
    "dark_factory_deep_agent_timeouts_total",
    "Deep sub-agent timeouts.",
)

subprocess_spawns_total = Counter(
    "dark_factory_subprocess_spawns_total",
    "Approximate count of subprocesses spawned by the Claude Agent SDK.",
)


# ── Memory metrics ──────────────────────────────────────────────────────────


memory_ops_total = Counter(
    "dark_factory_memory_ops_total",
    "Procedural memory operations.",
    ["operation", "memory_type"],  # operation = create|recall|boost|demote|decay
)

memory_recall_total = Counter(
    "dark_factory_memory_recall_total",
    "Memory recall calls split by outcome.",
    ["outcome"],  # outcome = hit | miss
)

memory_recall_latency_seconds = Histogram(
    "dark_factory_memory_recall_latency_seconds",
    "Memory recall latency in seconds.",
    buckets=_LATENCY_BUCKETS_TOOL,
)


# ── Spec decomposition + eval metrics ──────────────────────────────────────


spec_plan_outcomes_total = Counter(
    "dark_factory_spec_plan_outcomes_total",
    "Planner phase outcomes per requirement.",
    ["outcome"],  # outcome = success | fallback | empty | truncated
)

decomposition_sub_specs = Histogram(
    "dark_factory_decomposition_sub_specs_per_requirement",
    "Number of planned sub-specs per requirement.",
    buckets=_SUB_SPECS_BUCKETS,
)

eval_rubric_total = Counter(
    "dark_factory_eval_rubric_total",
    "DeepEval rubric results per metric name.",
    ["metric_name", "passed"],  # passed = true | false
)

eval_score = Histogram(
    "dark_factory_eval_score",
    "DeepEval rubric scores (0.0–1.0).",
    ["metric_name"],
    buckets=_EVAL_SCORE_BUCKETS,
)

spec_attempts_to_pass = Histogram(
    "dark_factory_spec_attempts_to_pass",
    "Number of refinement attempts before a spec hits the eval threshold.",
    buckets=_ATTEMPTS_BUCKETS,
)


# ── Incident metrics ────────────────────────────────────────────────────────


incidents_total = Counter(
    "dark_factory_incidents_total",
    "Structured incidents recorded by the error-path hooks.",
    ["category", "severity"],
)


# ── Artifact metrics ────────────────────────────────────────────────────────


artifacts_written_total = Counter(
    "dark_factory_artifacts_written_total",
    "Code artifact files written by swarm tools.",
    ["language", "is_test"],  # is_test = true | false
)

bytes_written_total = Counter(
    "dark_factory_bytes_written_total",
    "Total bytes of artifact content written.",
    ["language"],
)

artifact_bytes_written = Histogram(
    "dark_factory_artifact_bytes_written",
    "Per-file artifact size distribution.",
    buckets=_BYTES_BUCKETS,
)


# ── System / infrastructure gauges ──────────────────────────────────────────


background_loop_active_tasks = Gauge(
    "dark_factory_background_loop_active_tasks",
    "Tasks currently running on the shared BackgroundLoop event loop.",
)

background_loop_completed_tasks = Gauge(
    "dark_factory_background_loop_completed_tasks",
    "Cumulative coroutines completed via BackgroundLoop.run().",
)

background_loop_restarts = Gauge(
    "dark_factory_background_loop_restarts",
    "Number of times the BackgroundLoop singleton has been (re)created.",
)

progress_broker_subscribers = Gauge(
    "dark_factory_progress_broker_subscribers",
    "Active subscribers on the in-process progress broker.",
)

progress_broker_history_size = Gauge(
    "dark_factory_progress_broker_history_size",
    "Progress broker replay-history buffer depth.",
)

metrics_queue_depth = Gauge(
    "dark_factory_metrics_queue_depth",
    "Current depth of the MetricsRecorder background writer queue.",
)

metrics_events_dropped_total = Counter(
    "dark_factory_metrics_events_dropped_total",
    "MetricsRecorder events dropped because the writer queue was full.",
)

# H5 fix: per-kind breakdown of dropped events so dashboards can
# answer "which kind of metric is being lost?" — e.g. swarm events
# vs LLM calls vs eval results. Helps operators prioritise recovery.
metrics_events_dropped_by_kind_total = Counter(
    "dark_factory_metrics_events_dropped_by_kind_total",
    "MetricsRecorder events dropped, broken down by event kind.",
    ["kind"],
)

# H6 fix: when a feature swarm worker thread crashes with deep agent
# calls still in-flight, the orchestrator's ThreadPoolExecutor
# exception handler bumps this counter. The SDK's own async context
# managers are still responsible for the actual subprocess cleanup
# (via BackgroundLoop's event loop) — this counter is the
# diagnostic signal that tells operators whether the cleanup path
# is being exercised under fault conditions.
worker_crashes_with_inflight_agents_total = Counter(
    "dark_factory_worker_crashes_with_inflight_agents_total",
    "Feature worker crashes where deep-agent calls were still in-flight.",
    ["feature"],
)

# L5 fix: Postgres pool depth gauges. Sampled every 10s by the
# BackgroundLoop sampler so dashboards can catch pool exhaustion
# trends before they stall the pipeline. ``waiting`` is the most
# actionable field — non-zero values indicate that requesters are
# blocked on the pool being full.
postgres_pool_size = Gauge(
    "dark_factory_postgres_pool_size",
    "Current Postgres connection pool size (configured max).",
)
postgres_pool_idle = Gauge(
    "dark_factory_postgres_pool_idle",
    "Postgres connections idle in the pool.",
)
postgres_pool_active = Gauge(
    "dark_factory_postgres_pool_active",
    "Postgres connections currently checked out (in use).",
)
postgres_pool_waiting = Gauge(
    "dark_factory_postgres_pool_waiting",
    "Requesters blocked waiting for a free Postgres connection.",
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _safe(func):
    """Wrap a helper so Prometheus errors can never take the pipeline down."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:  # pragma: no cover — defensive
            log.debug("prometheus_observer_failed", helper=func.__name__, exc_info=True)
            return None
    return wrapper


def _label(value: Any, *, default: str = "unknown") -> str:
    """Coerce a label value to a bounded-cardinality string."""
    if value is None or value == "":
        return default
    return str(value)


def _zero_unlabelled(collector) -> None:
    """Reset an unlabelled Counter/Gauge/Histogram back to zero.

    prometheus_client stores the unlabelled backing value directly on
    the wrapper for Counter/Gauge (``_value``) and on ``_sum`` + each
    ``_buckets[i]`` for Histogram. We call ``.set(0)`` on each one,
    which is the same public API the metrics themselves use internally.

    Private attribute access is unavoidable — the library exposes no
    public reset API and the unlabelled wrapper IS the underlying child.
    """
    # Counter / Gauge: single ``_value`` ValueClass
    value = getattr(collector, "_value", None)
    if value is not None and hasattr(value, "set"):
        value.set(0)

    # Histogram: _sum + _buckets list of ValueClass
    sum_val = getattr(collector, "_sum", None)
    if sum_val is not None and hasattr(sum_val, "set"):
        sum_val.set(0)

    buckets = getattr(collector, "_buckets", None)
    if buckets is not None:
        for b in buckets:
            if hasattr(b, "set"):
                b.set(0)


def reset_all() -> dict[str, int]:
    """Reset every dark-factory Prometheus collector to its initial state.

    Used by the admin clear-all flow to wipe in-process counter/gauge/
    histogram values so Prometheus doesn't immediately re-scrape the old
    numbers after its TSDB is cleared.

    Only collectors whose names start with ``dark_factory_`` are touched —
    we never reset prometheus_client's own ``python_*`` / ``process_*``
    collectors, which would break liveness reporting.

    Two-pass strategy (because prometheus_client exposes no public reset):
    1. For labelled collectors, ``.clear()`` wipes every child metric.
    2. For unlabelled collectors, the wrapper IS the child — we set the
       underlying ``_value`` / ``_sum`` / ``_buckets`` back to 0 directly.

    Every step is wrapped in try/except so a single failing collector
    can't take the endpoint down.
    """
    cleared = 0
    reinitialised = 0
    skipped = 0

    # REGISTRY._names_to_collectors maps every fully-qualified metric name
    # (including per-sample children like ``*_total`` / ``*_bucket``) back
    # to its owning collector. Deduplicate by object identity.
    collectors: set = set()
    try:
        for collector in REGISTRY._names_to_collectors.values():  # type: ignore[attr-defined]
            collectors.add(collector)
    except Exception as exc:
        log.warning("prometheus_reset_enum_failed", error=str(exc))
        return {
            "cleared_collectors": 0,
            "reinitialised_collectors": 0,
            "skipped_collectors": 0,
        }

    for collector in collectors:
        # Only reset our own collectors. The registry also holds
        # prometheus_client's built-in process/gc collectors which we
        # must leave alone.
        name = getattr(collector, "_name", "") or ""
        if not name.startswith("dark_factory_"):
            continue

        try:
            labelnames = getattr(collector, "_labelnames", ()) or ()
            if labelnames:
                # Labelled: wipe all children in one shot.
                if hasattr(collector, "clear"):
                    collector.clear()
                    cleared += 1
            else:
                # Unlabelled: the wrapper backs the single child directly.
                _zero_unlabelled(collector)
                reinitialised += 1
        except Exception as exc:
            skipped += 1
            log.warning(
                "prometheus_reset_collector_failed",
                collector=name,
                error=str(exc),
            )

    log.info(
        "prometheus_collectors_reset",
        cleared=cleared,
        reinitialised=reinitialised,
        skipped=skipped,
    )
    return {
        "cleared_collectors": cleared,
        "reinitialised_collectors": reinitialised,
        "skipped_collectors": skipped,
    }


def _bool_label(value: Any) -> str:
    return "true" if value else "false"


# ── Observer helpers ────────────────────────────────────────────────────────


@_safe
def observe_llm_call(
    *,
    client: str,
    model: str,
    phase: str | None = None,
    latency_seconds: float | None = None,
    time_to_first_token_seconds: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    cost_usd: float | None = None,
    retry_count: int = 0,
    error: str | None = None,
    rate_limited: bool = False,
    http_status: int | None = None,
    **_ignored: Any,
) -> None:
    """One-stop LLM call observer — updates every LLM prometheus metric."""
    client_label = _label(client)
    model_label = _label(model)
    phase_label = _label(phase)

    llm_calls_total.labels(
        client=client_label, model=model_label, phase=phase_label
    ).inc()

    if input_tokens:
        llm_tokens_total.labels(
            client=client_label, model=model_label, kind="input"
        ).inc(input_tokens)
        llm_input_tokens.labels(
            client=client_label, model=model_label
        ).observe(input_tokens)
    if output_tokens:
        llm_tokens_total.labels(
            client=client_label, model=model_label, kind="output"
        ).inc(output_tokens)
        llm_output_tokens.labels(
            client=client_label, model=model_label
        ).observe(output_tokens)
    if cache_read_input_tokens:
        llm_tokens_total.labels(
            client=client_label, model=model_label, kind="cache_read"
        ).inc(cache_read_input_tokens)
    if cache_creation_input_tokens:
        llm_tokens_total.labels(
            client=client_label, model=model_label, kind="cache_create"
        ).inc(cache_creation_input_tokens)

    if cost_usd:
        llm_cost_usd_total.labels(
            client=client_label, model=model_label
        ).inc(cost_usd)

    if latency_seconds is not None and latency_seconds >= 0:
        llm_latency_seconds.labels(
            client=client_label, model=model_label
        ).observe(latency_seconds)
    if time_to_first_token_seconds is not None and time_to_first_token_seconds >= 0:
        llm_ttft_seconds.labels(
            client=client_label, model=model_label
        ).observe(time_to_first_token_seconds)

    if retry_count and retry_count > 0:
        llm_retries_total.labels(
            client=client_label, model=model_label
        ).inc(retry_count)

    if error:
        if rate_limited or http_status == 429:
            reason = "rate_limited"
        elif isinstance(http_status, int) and http_status >= 500:
            reason = "http_5xx"
        elif isinstance(http_status, int) and http_status >= 400:
            reason = "http_4xx"
        else:
            reason = "other"
        llm_errors_total.labels(
            client=client_label, model=model_label, reason=reason
        ).inc()


@_safe
def observe_pipeline_run_start(run_id: str | None = None, **_ignored: Any) -> None:
    pipeline_runs_total.labels(status="running").inc()
    running_pipelines.inc()


@_safe
def observe_pipeline_run_end(
    *,
    status: str,
    duration_seconds: float | None = None,
    **_ignored: Any,
) -> None:
    status_label = _label(status)
    pipeline_runs_total.labels(status=status_label).inc()
    # Floor guard: only decrement if >0 to prevent the gauge from going
    # negative on unpaired end calls (e.g. run_end called twice on the
    # same run, or a pre-start exception path that never bumped the
    # counter). A negative "running pipelines" gauge would silently lie
    # on the Metrics dashboard.
    try:
        if running_pipelines._value.get() > 0:
            running_pipelines.dec()
    except Exception:  # pragma: no cover — defensive
        pass
    if duration_seconds is not None and duration_seconds >= 0:
        pipeline_duration_seconds.labels(status=status_label).observe(duration_seconds)


@_safe
def observe_reconciliation_run(
    *,
    status: str,
    duration_seconds: float | None = None,
    **_ignored: Any,
) -> None:
    """Record the outcome of a Phase 5 reconciliation pass."""
    status_label = _label(status)
    reconciliation_runs_total.labels(status=status_label).inc()
    if duration_seconds is not None and duration_seconds >= 0:
        reconciliation_duration_seconds.labels(
            status=status_label
        ).observe(duration_seconds)


@_safe
def observe_e2e_validation_run(
    *,
    status: str,
    duration_seconds: float | None = None,
    tests_passed: int = 0,
    tests_failed: int = 0,
    browsers_run: list[str] | None = None,
    **_ignored: Any,
) -> None:
    """Record the outcome of a Phase 6 E2E validation pass.

    In addition to the phase-level counter/histogram, this also
    distributes the per-test counts across the browsers that
    actually ran — we use a flat average rather than exact
    per-browser parsing because the stage result only carries the
    aggregate counts. For a more granular view of "which browser
    is flaky" the E2E_REPORT.md table is the authoritative source.
    """
    status_label = _label(status)
    e2e_validation_runs_total.labels(status=status_label).inc()
    if duration_seconds is not None and duration_seconds >= 0:
        e2e_validation_duration_seconds.labels(
            status=status_label
        ).observe(duration_seconds)
    browsers = browsers_run or []
    if browsers and (tests_passed or tests_failed):
        # Spread the counts evenly across browsers. This is an
        # approximation — a browser that failed every test would
        # ideally show 100% failure, but we don't have per-browser
        # counts here. Dashboards that need precision should read
        # E2E_REPORT.md directly.
        per_browser_passed = tests_passed // max(1, len(browsers))
        per_browser_failed = tests_failed // max(1, len(browsers))
        for b in browsers:
            browser_label = _label(b)
            if per_browser_passed:
                e2e_tests_total.labels(
                    browser=browser_label, status="passed"
                ).inc(per_browser_passed)
            if per_browser_failed:
                e2e_tests_total.labels(
                    browser=browser_label, status="failed"
                ).inc(per_browser_failed)


@_safe
def observe_memory_write(
    *,
    memory_type: str,
    outcome: str,
    **_ignored: Any,
) -> None:
    """Record a memory write. ``outcome`` is ``created`` for new
    nodes or ``deduped`` when the dedup helper boosted an existing
    near-duplicate instead. Powers the "dedup catch rate" chart in
    the Memory metrics dashboard."""
    memory_writes_total.labels(
        type=_label(memory_type),
        outcome=_label(outcome),
    ).inc()


@_safe
def observe_memory_recall(
    *,
    memory_type: str,
    hit: bool,
    **_ignored: Any,
) -> None:
    """Record a memory recall operation. ``hit=True`` if the recall
    returned at least one result. Lets the dashboard answer "is the
    graph dead weight?" by comparing recall hit rate over time."""
    memory_recalls_total.labels(
        type=_label(memory_type),
        hit="yes" if hit else "no",
    ).inc()


@_safe
def observe_memory_relevance_adjustment(
    *,
    memory_type: str,
    direction: str,
    count: int = 1,
    **_ignored: Any,
) -> None:
    """Record relevance boost/demote/decay operations. ``direction``
    in ``{boost, demote, decay}``. Boost fires when an eval pass
    attributes success to a recalled memory; demote on eval fail;
    decay on the 5%-per-run background decay pass."""
    if count <= 0:
        return
    memory_relevance_adjustments_total.labels(
        type=_label(memory_type),
        direction=_label(direction),
    ).inc(count)


@_safe
def observe_feature_event(
    *,
    event: str,
    status: str | None = None,
    duration_seconds: float | None = None,
    **_ignored: Any,
) -> None:
    event_label = _label(event)
    status_label = _label(status)
    feature_events_total.labels(event=event_label, status=status_label).inc()
    if event_label == "completed" and duration_seconds is not None and duration_seconds >= 0:
        feature_duration_seconds.observe(duration_seconds)


@_safe
def observe_agent_activation(agent: str) -> None:
    agent_activations_total.labels(agent=_label(agent)).inc()


@_safe
def observe_agent_handoff(*, from_agent: str, to_agent: str) -> None:
    agent_handoffs_total.labels(
        from_agent=_label(from_agent),
        to_agent=_label(to_agent),
    ).inc()


@_safe
def observe_worker_crash() -> None:
    worker_crashes_total.inc()


@_safe
def observe_tool_call(
    *,
    tool: str,
    agent: str | None = None,
    success: bool | None = None,
    latency_seconds: float | None = None,
    error: str | None = None,
    **_ignored: Any,
) -> None:
    tool_label = _label(tool)
    agent_label = _label(agent)
    # Treat success=None as "success" to match the existing ToolMessage
    # convention where missing ``status`` implies a successful result.
    if success is None:
        status_label = "failure" if error else "success"
    else:
        status_label = "success" if success else "failure"
    tool_calls_total.labels(
        tool=tool_label, agent=agent_label, status=status_label
    ).inc()
    if latency_seconds is not None and latency_seconds >= 0:
        tool_latency_seconds.labels(tool=tool_label).observe(latency_seconds)


@_safe
def observe_deep_agent_invocation(tool: str) -> None:
    deep_agent_invocations_total.labels(tool=_label(tool)).inc()
    subprocess_spawns_total.inc()


@_safe
def observe_deep_agent_timeout() -> None:
    deep_agent_timeouts_total.inc()


@_safe
def observe_memory_op(
    *,
    operation: str,
    memory_type: str | None = None,
    count: int | None = None,
    latency_seconds: float | None = None,
    **_ignored: Any,
) -> None:
    op_label = _label(operation)
    type_label = _label(memory_type, default="none")
    memory_ops_total.labels(operation=op_label, memory_type=type_label).inc()

    if op_label == "recall":
        outcome = "hit" if (count or 0) > 0 else "miss"
        memory_recall_total.labels(outcome=outcome).inc()
        if latency_seconds is not None and latency_seconds >= 0:
            memory_recall_latency_seconds.observe(latency_seconds)


@_safe
def observe_spec_plan(
    *,
    outcome: str,
    sub_spec_count: int | None = None,
    **_ignored: Any,
) -> None:
    spec_plan_outcomes_total.labels(outcome=_label(outcome)).inc()
    if outcome == "success" and sub_spec_count is not None and sub_spec_count >= 0:
        decomposition_sub_specs.observe(sub_spec_count)


@_safe
def observe_eval_rubric(
    *,
    metric_name: str,
    score: float,
    passed: bool,
    **_ignored: Any,
) -> None:
    name_label = _label(metric_name)
    eval_rubric_total.labels(
        metric_name=name_label, passed=_bool_label(passed)
    ).inc()
    if score is not None:
        eval_score.labels(metric_name=name_label).observe(score)


@_safe
def observe_spec_attempts_to_pass(attempts: int) -> None:
    if attempts and attempts > 0:
        spec_attempts_to_pass.observe(attempts)


@_safe
def observe_incident(*, category: str, severity: str, **_ignored: Any) -> None:
    incidents_total.labels(
        category=_label(category),
        severity=_label(severity),
    ).inc()


@_safe
def observe_artifact_write(
    *,
    language: str | None = None,
    bytes_written: int = 0,
    is_test: bool = False,
    **_ignored: Any,
) -> None:
    language_label = _label(language, default="unknown")
    artifacts_written_total.labels(
        language=language_label,
        is_test=_bool_label(is_test),
    ).inc()
    if bytes_written and bytes_written > 0:
        bytes_written_total.labels(language=language_label).inc(bytes_written)
        artifact_bytes_written.observe(bytes_written)


@_safe
def observe_bg_loop_sample(
    *,
    active_task_count: int = 0,
    completed_task_count: int = 0,
    loop_restarts: int = 0,
    **_ignored: Any,
) -> None:
    background_loop_active_tasks.set(active_task_count)
    background_loop_completed_tasks.set(completed_task_count)
    background_loop_restarts.set(loop_restarts)


@_safe
def observe_metrics_recorder(
    *,
    queue_depth: int = 0,
    dropped_delta: int = 0,
) -> None:
    metrics_queue_depth.set(queue_depth)
    if dropped_delta > 0:
        metrics_events_dropped_total.inc(dropped_delta)


@_safe
def observe_progress_broker(
    *,
    subscribers: int = 0,
    history_size: int = 0,
) -> None:
    progress_broker_subscribers.set(subscribers)
    progress_broker_history_size.set(history_size)


@_safe
def observe_postgres_pool(
    *,
    size: int = 0,
    idle: int = 0,
    active: int = 0,
    waiting: int = 0,
) -> None:
    """L5: record Postgres pool depth gauges.

    Called from the BackgroundLoop sampler every 10s. ``waiting``
    going non-zero is the actionable signal — it means requesters
    are blocked on pool exhaustion and the pipeline is about to
    stall.
    """
    postgres_pool_size.set(size)
    postgres_pool_idle.set(idle)
    postgres_pool_active.set(active)
    postgres_pool_waiting.set(waiting)
