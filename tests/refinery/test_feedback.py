"""Tests for the active-learning feedback loop.

Covers ``record_memory_feedback`` orchestration: telemetry write,
relevance adjustment in Neo4j, and Conflict emission on reasoned
dismissal. All sinks are stubs — no real DB or graph driver.
"""

from __future__ import annotations

from typing import Any

from dark_factory.api.refinery.feedback import record_memory_feedback


class _StubMetricsRepo:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def record_memory_feedback(self, **kwargs):
        self.calls.append(kwargs)


class _StubMemoryRepo:
    def __init__(self):
        self.boosts: list[tuple[str, str, float]] = []
        self.demotes: list[tuple[str, str, float]] = []
        self.conflicts: list[dict[str, Any]] = []

    def boost_relevance(self, node_id: str, label: str, delta: float = 0.1):
        self.boosts.append((node_id, label, delta))

    def demote_relevance(self, node_id: str, label: str, delta: float = 0.05):
        self.demotes.append((node_id, label, delta))

    def record_conflict(self, **kwargs):
        self.conflicts.append(kwargs)
        return f"conflict-{len(self.conflicts)}"


def test_feedback_invalid_decision_returns_error():
    """``decision`` must be one of accepted/dismissed/edited; anything
    else short-circuits with a clear error."""

    result = record_memory_feedback(
        memory_id="m-1",
        memory_kind="pattern",
        decision="approve",  # not allowed
    )
    assert result["ok"] is False
    assert "invalid decision" in result["reason"]


def test_feedback_unknown_kind_returns_error():
    """Unknown memory_kind → no label → no-op with reason."""

    result = record_memory_feedback(
        memory_id="m-1",
        memory_kind="unicorn",
        decision="accepted",
    )
    assert result["ok"] is False
    assert "unknown memory_kind" in result["reason"]


def test_feedback_acceptance_boosts_and_records_telemetry():
    metrics = _StubMetricsRepo()
    memory = _StubMemoryRepo()

    result = record_memory_feedback(
        memory_id="pat-1",
        memory_kind="pattern",
        decision="accepted",
        metrics_repo=metrics,
        memory_repo=memory,
        refinery_run_id="refinery-A",
        source_role="security",
    )
    assert result["ok"] is True
    assert result["telemetry_recorded"] is True
    assert result["relevance_adjusted"] is True
    assert result["conflict_emitted"] is False
    assert metrics.calls and metrics.calls[0]["decision"] == "accepted"
    assert memory.boosts == [("pat-1", "Pattern", 0.10)]
    assert memory.demotes == []
    assert memory.conflicts == []


def test_feedback_dismissal_demotes_and_emits_conflict_when_reasoned():
    metrics = _StubMetricsRepo()
    memory = _StubMemoryRepo()

    result = record_memory_feedback(
        memory_id="dec-1",
        memory_kind="decision",
        decision="dismissed",
        metrics_repo=metrics,
        memory_repo=memory,
        source_role="judge",
        reason="this decision contradicts our hard SLA — operator override",
    )
    assert result["ok"] is True
    assert result["relevance_adjusted"] is True
    assert result["conflict_emitted"] is True
    assert memory.demotes == [("dec-1", "Decision", 0.05)]
    assert memory.boosts == []
    # Conflict carries the user/role parties + cause=user_override.
    conflict = memory.conflicts[0]
    assert conflict["cause"] == "user_override"
    assert "user" in conflict["conflict_parties"]
    assert "judge" in conflict["conflict_parties"]


def test_feedback_dismissal_without_reason_skips_conflict_emission():
    """Bare dismissal (no reason text) demotes the memory but does NOT
    emit a Conflict — we don't want to flood the store with empty
    overrides."""

    memory = _StubMemoryRepo()
    result = record_memory_feedback(
        memory_id="con-1",
        memory_kind="constraint",
        decision="dismissed",
        memory_repo=memory,
        reason="   ",  # whitespace-only — counts as empty
    )
    assert result["conflict_emitted"] is False
    assert memory.demotes == [("con-1", "Constraint", 0.05)]
    assert memory.conflicts == []


def test_feedback_edit_records_telemetry_but_no_score_change():
    """``edited`` is a notable signal but neither boosts nor demotes —
    the edit content itself is the operator's correction."""

    metrics = _StubMetricsRepo()
    memory = _StubMemoryRepo()
    result = record_memory_feedback(
        memory_id="m-1",
        memory_kind="pattern",
        decision="edited",
        metrics_repo=metrics,
        memory_repo=memory,
    )
    assert result["telemetry_recorded"] is True
    assert result["relevance_adjusted"] is False
    assert memory.boosts == []
    assert memory.demotes == []


def test_feedback_no_repos_still_succeeds():
    """When neither sink is wired (test or degraded environment), the
    call still returns ok=True and just reports nothing was done."""

    result = record_memory_feedback(
        memory_id="m-1",
        memory_kind="pattern",
        decision="accepted",
    )
    assert result["ok"] is True
    assert result["telemetry_recorded"] is False
    assert result["relevance_adjusted"] is False
    assert result["conflict_emitted"] is False


def test_feedback_kind_alias_incident_routes_to_mistake_label():
    """``incident`` is an alias for the legacy ``Mistake`` label; a
    boost/demote must hit the Mistake node, not a non-existent
    Incident label."""

    memory = _StubMemoryRepo()
    record_memory_feedback(
        memory_id="inc-1",
        memory_kind="incident",
        decision="accepted",
        memory_repo=memory,
    )
    assert memory.boosts == [("inc-1", "Mistake", 0.10)]
