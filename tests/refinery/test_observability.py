"""Phase 3 tests — Prometheus catalog + ObservabilityHub routing.

Postgres writes aren't covered here because the existing suite already
gates Postgres tests on ``POSTGRES_ENABLED`` and docker-compose; tests
under ``tests/test_metrics.py`` already exercise the RepoS/conn path.
The Phase 3 tests here cover:

- Every ``observe_refinery_*`` helper is callable and wraps in ``_safe``.
- ObservabilityHub fans out to Prometheus when Postgres is unavailable.
- ObservabilityHub calls the repo when Postgres is configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from prometheus_client import REGISTRY

from dark_factory.api.refinery.contracts import MemoryKind, SourceTier, ValidationStatus
from dark_factory.api.refinery.observability import ObservabilityHub
from dark_factory.metrics import prometheus as prom


# ─────────────────────────────────────────────────────────────────────
# Prometheus catalog — every helper exists + is _safe
# ─────────────────────────────────────────────────────────────────────


def test_all_refinery_observe_helpers_are_defined():
    """The plan lists 21 metric series and a matching set of helpers.
    This test ensures the module exports match so a rename breakage
    surfaces immediately."""

    required = {
        "observe_refinery_role_call",
        "observe_refinery_rounds",
        "observe_refinery_convergence",
        "observe_refinery_escalation",
        "observe_refinery_disagreement",
        "observe_refinery_judge_score",
        "observe_refinery_rule_violation",
        "observe_refinery_rule_penalty_applied",
        "observe_refinery_rules_engine_failure",
        "observe_refinery_eval_short_circuit",
        "observe_refinery_fallback_judge_used",
        "observe_refinery_research_call",
        "observe_refinery_research_tier",
        "observe_refinery_research_internal_sufficient",
        "observe_refinery_research_budget_exhausted",
        "observe_refinery_validated_insight",
        "observe_refinery_t5_propagation_rejected",
        "observe_refinery_memory_suggested",
        "observe_refinery_memory_audit",
        "observe_refinery_memory_write_back_failure",
        "observe_refinery_memory_dedup_blocked",
        "observe_refinery_cost",
        "observe_refinery_tokens",
        "observe_refinery_dual_write_failure",
        "observe_refinery_postgres_write",
    }
    for name in required:
        assert hasattr(prom, name), f"prometheus.py missing helper: {name}"


def test_observe_helpers_are_safe_against_bad_inputs():
    """Every helper must swallow errors — a bad call cannot take the
    debate down. We exercise a variety of absurd inputs."""

    # These should all be no-ops without raising.
    prom.observe_refinery_role_call(role=None)  # type: ignore[arg-type]
    prom.observe_refinery_rounds(rounds=-5)     # negative → filtered out
    prom.observe_refinery_convergence(outcome="")
    prom.observe_refinery_escalation(reason=None)  # type: ignore[arg-type]
    prom.observe_refinery_disagreement(score=-1.0)
    prom.observe_refinery_judge_score(dimension="nonsense", score=999.0)
    prom.observe_refinery_cost(role="x", cost_usd=-1.0)
    prom.observe_refinery_tokens(role="x", tokens_in=0, tokens_out=0)


def test_rounds_histogram_increments_on_valid_input():
    before = _sample_sum("dark_factory_refinery_rounds_per_requirement_sum")
    prom.observe_refinery_rounds(rounds=3)
    after = _sample_sum("dark_factory_refinery_rounds_per_requirement_sum")
    assert after - before == 3


def test_convergence_counter_increments_per_outcome():
    before = _label_counter(
        "dark_factory_refinery_convergence_outcome_total", outcome="converged",
    )
    prom.observe_refinery_convergence(outcome="converged")
    after = _label_counter(
        "dark_factory_refinery_convergence_outcome_total", outcome="converged",
    )
    assert after - before == 1


def test_dual_write_failure_counter_has_table_label():
    before = _label_counter(
        "dark_factory_refinery_dual_write_failures_total", table="llm_calls",
    )
    prom.observe_refinery_dual_write_failure(table="llm_calls")
    after = _label_counter(
        "dark_factory_refinery_dual_write_failures_total", table="llm_calls",
    )
    assert after - before == 1


# ─────────────────────────────────────────────────────────────────────
# ObservabilityHub routing
# ─────────────────────────────────────────────────────────────────────


def test_hub_routes_to_prometheus_only_when_no_repo():
    hub = ObservabilityHub(refinery_repo=None, postgres_enabled=False)
    hub.on_run_start(
        refinery_run_id="refinery-test-1",
        source_mode="direct",
        source_run_id=None,
        requirements_count=1,
        settings_snapshot={},
    )
    hub.on_debate_end(
        refinery_run_id="refinery-test-1",
        requirement_id="req-1",
        convergence_status="converged",
        rounds_executed=2,
    )
    # No assertion on Prometheus values (other tests cover that). This
    # test just proves the hub doesn't crash without a repo.


def test_hub_calls_repo_methods_when_enabled():
    repo = MagicMock()
    repo.record_debate.return_value = 42
    hub = ObservabilityHub(refinery_repo=repo, postgres_enabled=True)

    hub.on_run_start(
        refinery_run_id="refinery-test-2",
        source_mode="run",
        source_run_id="run-abc",
        requirements_count=3,
        settings_snapshot={"key": "val"},
    )
    repo.record_run_start.assert_called_once()
    kw = repo.record_run_start.call_args.kwargs
    assert kw["refinery_run_id"] == "refinery-test-2"
    assert kw["source_mode"] == "run"
    assert kw["source_run_id"] == "run-abc"

    debate_id = hub.on_debate_end(
        refinery_run_id="refinery-test-2",
        requirement_id="req-1",
        convergence_status="short_circuited",
        rounds_executed=3,
        final_dimension_scores={"clarity": 0.5},
    )
    assert debate_id == 42
    repo.record_debate.assert_called_once()

    hub.on_round_end(
        debate_id=42,
        round_number=1,
        critic_count=4,
        critic_blockers_count=1,
        critic_warnings_count=2,
        rule_violations_count=1,
        rule_warnings_count=0,
        judge_overall=0.6,
        judge_dimensions={"clarity": 0.7, "risk_coverage": 0.5},
        disagreement_score=0.3,
        router_decision="continue",
    )
    repo.record_round.assert_called_once()


def test_hub_llm_call_fans_to_prometheus_and_repo():
    repo = MagicMock()
    hub = ObservabilityHub(refinery_repo=repo, postgres_enabled=True)
    hub.on_llm_call(
        refinery_run_id="refinery-test-3",
        role="security",
        kind="critique",
        model="claude-opus-4-7",
        requirement_id="req-1",
        round_number=1,
        tokens_in=1000,
        tokens_out=400,
        latency_ms=2500,
        cost_usd=0.05,
    )
    repo.record_llm_call.assert_called_once()
    kw = repo.record_llm_call.call_args.kwargs
    assert kw["role"] == "security"
    assert kw["kind"] == "critique"
    assert kw["model"] == "claude-opus-4-7"
    assert kw["tokens_in"] == 1000
    assert kw["tokens_out"] == 400


def test_hub_t5_propagation_rejection_fires_prometheus_only_on_non_propagated_t5():
    """Guardrail telemetry: T5 sources that fail the Editor's corroboration
    check bump the counter. Any other tier's non-propagated row does not."""

    repo = MagicMock()
    hub = ObservabilityHub(refinery_repo=repo, postgres_enabled=True)

    before = _counter("dark_factory_refinery_research_t5_propagation_rejected_total")
    # T0 source not propagated — no rejection counter bump.
    hub.on_research_source(
        debate_id=1, round_number=1, tier=SourceTier.T0_STRUCTURED,
        provider="qdrant", url=None, title="memory:pattern-42",
        propagated=False,
    )
    middle = _counter("dark_factory_refinery_research_t5_propagation_rejected_total")
    assert middle == before

    # T5 source not propagated — one rejection counter bump.
    hub.on_research_source(
        debate_id=1, round_number=1, tier=SourceTier.T5_WEB,
        provider="openai_web_search", url="https://example.com/a",
        title="example", propagated=False,
    )
    after = _counter("dark_factory_refinery_research_t5_propagation_rejected_total")
    assert after - middle == 1


def test_hub_memory_audit_routes_both_stores():
    repo = MagicMock()
    hub = ObservabilityHub(refinery_repo=repo, postgres_enabled=True)
    before = _label_counter(
        "dark_factory_refinery_memory_audit_total", outcome="saved",
    )
    hub.on_memory_audit(
        refinery_run_id="refinery-test-4",
        suggested_memory_id="mem-1",
        kind=MemoryKind.DECISION,
        source_role="judge",
        validation_status=ValidationStatus.VALIDATED,
        outcome="saved",
    )
    after = _label_counter(
        "dark_factory_refinery_memory_audit_total", outcome="saved",
    )
    assert after - before == 1
    repo.record_memory_audit.assert_called_once()
    kw = repo.record_memory_audit.call_args.kwargs
    # Kind + validation_status get coerced from enums to strings for SQL.
    assert kw["kind"] == "decision"
    assert kw["validation_status"] == "validated"


def test_hub_without_postgres_still_fires_prometheus():
    hub = ObservabilityHub(refinery_repo=None, postgres_enabled=False)
    before = _label_counter(
        "dark_factory_refinery_memory_audit_total", outcome="dismissed",
    )
    hub.on_memory_audit(
        refinery_run_id="refinery-test-5",
        suggested_memory_id="mem-2",
        kind="conflict",
        source_role="operations",
        validation_status="unvalidated",
        outcome="dismissed",
    )
    after = _label_counter(
        "dark_factory_refinery_memory_audit_total", outcome="dismissed",
    )
    assert after - before == 1


# ─────────────────────────────────────────────────────────────────────
# Grafana dashboard JSON is valid + references correct metrics
# ─────────────────────────────────────────────────────────────────────


def test_grafana_dashboard_is_valid_json_and_references_refinery_metrics():
    path = (
        Path(__file__).parent.parent.parent
        / "deploy"
        / "grafana-dashboards"
        / "refinery.json"
    )
    data = json.loads(path.read_text())
    assert data["title"].startswith("Dark Factory")
    panel_titles = {p["title"] for p in data["panels"]}
    assert "Convergence rate" in panel_titles
    assert "Top rule violations (top 10)" in panel_titles
    # At least one target must reference a refinery metric.
    exprs = " ".join(
        t.get("expr", "")
        for p in data["panels"]
        for t in (p.get("targets") or [])
    )
    assert "dark_factory_refinery_" in exprs


# ─────────────────────────────────────────────────────────────────────
# Helpers for probing Prometheus internals
# ─────────────────────────────────────────────────────────────────────


def _counter(name: str) -> float:
    """Return the sum of an unlabelled counter, or 0 if absent."""

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name:
                return sample.value
    return 0.0


def _sample_sum(name: str) -> float:
    """Alias for _counter — histogram_sum samples use the same pattern."""

    return _counter(name)


def _label_counter(name: str, **labels: str) -> float:
    """Return the sample value for a counter filtered to given labels."""

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return sample.value
    return 0.0
