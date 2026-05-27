"""Tests for the adaptive role-weighting aggregator + Judge wiring.

Three layers:

1. ``adjust_role_weights`` — the bounded multiplier math.
2. ``load_role_weights`` — the precomputed-stats path (skips Postgres).
3. ``JudgeRole.defend`` — reads ``role_weights`` from evidence and
   passes them through to ``format_defend_prompt``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dark_factory.api.refinery.contracts import (
    Critique,
    CritiqueDimension,
    Draft,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.role_weighting import (
    RoleWeight,
    adjust_role_weights,
    load_role_weights,
    weight_for,
)
from dark_factory.api.refinery.roles.judge import JudgeRole


# ─────────────────────────────────────────────────────────────────────
# 1. adjust_role_weights — bounded multiplier math
# ─────────────────────────────────────────────────────────────────────


def test_below_min_total_yields_identity_multiplier():
    """A role with too few disposition rows in the window must NOT be
    re-weighted — early signal isn't trustworthy."""

    stats = [{
        "role": "security", "severity": "blocker",
        "accepted": 2, "rejected": 1, "deferred": 0, "total": 3,
    }]
    weights = adjust_role_weights(role_stats=stats)
    assert weights["security"].multiplier == pytest.approx(1.0)
    assert weights["security"].reason == "below_min_total"


def test_high_acceptance_rate_amplifies_within_bounds():
    """Acceptance rate ≥ ceiling → multiplier nudged up by one step,
    bounded by MAX_MULTIPLIER (1.40)."""

    stats = [{
        "role": "security", "severity": "blocker",
        "accepted": 9, "rejected": 1, "deferred": 0, "total": 10,
    }]
    weights = adjust_role_weights(role_stats=stats)
    w = weights["security"]
    assert w.multiplier == pytest.approx(1.10)
    assert w.reason == "high_acceptance"
    # Hard upper bound — no matter how high the rate, capped at 1.40.
    assert w.multiplier <= 1.40


def test_low_acceptance_rate_attenuates_within_bounds():
    """Acceptance rate ≤ floor → multiplier nudged down, bounded
    by MIN_MULTIPLIER (0.60). Roles never silenced."""

    stats = [{
        "role": "cost", "severity": "blocker",
        "accepted": 1, "rejected": 9, "deferred": 0, "total": 10,
    }]
    weights = adjust_role_weights(role_stats=stats)
    w = weights["cost"]
    assert w.multiplier == pytest.approx(0.90)
    assert w.reason == "low_acceptance"
    assert w.multiplier >= 0.60


def test_neutral_band_yields_identity():
    """An acceptance rate in the floor-to-ceiling band leaves the
    multiplier at 1.0 — no spurious adjustments."""

    stats = [{
        "role": "engineering", "severity": "blocker",
        "accepted": 5, "rejected": 5, "deferred": 0, "total": 10,
    }]
    weights = adjust_role_weights(role_stats=stats)
    w = weights["engineering"]
    assert w.multiplier == pytest.approx(1.0)
    assert w.reason == "neutral"


def test_severity_filter_excludes_warnings():
    """Default severity filter is 'blocker' — warnings/info are
    excluded so accepting a WARNING doesn't inflate a role's signal."""

    stats = [
        {"role": "security", "severity": "warning",
         "accepted": 10, "rejected": 0, "deferred": 0, "total": 10},
        {"role": "security", "severity": "blocker",
         "accepted": 9, "rejected": 1, "deferred": 0, "total": 10},
    ]
    weights = adjust_role_weights(role_stats=stats)
    # Only the blocker row contributes — yielding the high-acceptance
    # boost; the warning row is ignored.
    assert weights["security"].multiplier == pytest.approx(1.10)


def test_malformed_row_is_skipped_not_crashing():
    """Rows with non-int totals don't crash the aggregator."""

    stats = [
        {"role": "security", "severity": "blocker",
         "total": "ten", "accepted": "nine"},
        {"role": "engineering", "severity": "blocker",
         "accepted": 9, "rejected": 1, "deferred": 0, "total": 10},
    ]
    weights = adjust_role_weights(role_stats=stats)
    assert "security" not in weights
    assert weights["engineering"].multiplier == pytest.approx(1.10)


def test_weight_for_unknown_role_returns_default():
    weights = {"security": RoleWeight(
        role="security", multiplier=1.10,
        accepted=9, rejected=1, deferred=0, total=10,
        acceptance_rate=0.9, reason="high_acceptance",
    )}
    assert weight_for("security", weights) == pytest.approx(1.10)
    assert weight_for("ghost", weights) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# 2. load_role_weights — pre-computed path
# ─────────────────────────────────────────────────────────────────────


def test_load_role_weights_with_precomputed_stats_skips_postgres():
    """Passing ``role_stats`` lets the dashboard endpoint reuse a
    single query — no second hit on Postgres."""

    stats = [{
        "role": "security", "severity": "blocker",
        "accepted": 9, "rejected": 1, "deferred": 0, "total": 10,
    }]
    weights = load_role_weights(metrics_client=None, role_stats=stats)
    assert "security" in weights
    assert weights["security"].multiplier == pytest.approx(1.10)


def test_load_role_weights_returns_empty_when_metrics_client_missing():
    weights = load_role_weights(metrics_client=None)
    assert weights == {}


# ─────────────────────────────────────────────────────────────────────
# 3. JudgeRole.defend wiring — role weights flow through to the prompt
# ─────────────────────────────────────────────────────────────────────


def _draft() -> Draft:
    return Draft(
        requirement_id="r-1",
        title="x", description="y",
        priority="medium", tags=[],
        suggested_specs=[], relationships=[],
        produced_by="product", iteration=0,
    )


def _critique(role: str = "security") -> Critique:
    return Critique(
        author_role=role,
        severity=Severity.BLOCKER,
        dimension=CritiqueDimension.RISK_COVERAGE,
        finding="Ambiguous threat model lets stolen tokens replay forever.",
        proposed_fix="Add a 30-min token TTL and rotation on logout.",
    )


def test_defend_passes_role_weights_into_prompt_when_present():
    """When evidence carries ``role_weights``, the formatter's
    weighting block must surface in the prompt the LLM sees. We assert
    on the prompt text by spying on _call_llm."""

    role = JudgeRole()
    captured: list[str] = []

    def _spy_llm(prompt: str) -> str:
        captured.append(prompt)
        # Returning malformed JSON forces the deterministic fallback,
        # which is fine — the prompt was still rendered before the call.
        raise NotImplementedError

    with patch.object(role, "_call_llm", side_effect=_spy_llm):
        role.defend(
            _draft(), [_critique()],
            RoleContext(
                role="judge",
                requirement_id="r-1",
                round_number=1,
                evidence={"role_weights": {"security": 1.10, "cost": 0.90}},
            ),
        )

    assert captured, "Judge.defend should call _call_llm at least once"
    prompt = captured[0]
    assert "Role calibration" in prompt
    assert "security: ×1.10" in prompt
    assert "cost: ×0.90" in prompt
    # Identity entries must NOT appear (signal-to-noise discipline).
    assert "engineering" not in prompt


def test_defend_omits_calibration_block_when_no_weights():
    """No weights → the prompt is identical to pre-feature behaviour."""

    role = JudgeRole()
    captured: list[str] = []
    with patch.object(role, "_call_llm",
                      side_effect=lambda p: (captured.append(p), "")[1]):
        role.defend(
            _draft(), [_critique()],
            RoleContext(
                role="judge", requirement_id="r-1", round_number=1,
                evidence={},
            ),
        )
    assert captured
    assert "Role calibration" not in captured[0]


def test_disposition_sink_fires_from_synthesize_node():
    """End-to-end check: when a disposition_sink is wired into the
    debate graph, every rebuttal entry produces a sink call carrying
    role/severity/dimension/action — the row shape the metrics repo
    expects."""

    from dark_factory.api.refinery.contracts import ConvergenceStatus
    from dark_factory.api.refinery.debate.graph import run_debate

    from tests.refinery.conftest import (
        debate_kwargs,
        forced_judge_score,
        patched_product_propose,
    )
    from dark_factory.api.refinery.models import RefinedRequirement

    # We don't use the conftest fake_refined fixture here — build a
    # minimal one inline so this test stays self-contained. Use the
    # same shape the existing debate-graph smoke test uses.
    refined = RefinedRequirement(
        id="req-1", original_title="t", original_description="d",
        title="t", description="d", priority="medium", tags=[],
    )

    rows: list[dict] = []

    def sink(batch: list[dict]) -> None:
        rows.extend(batch)

    # Use the test conftest's stub_critic_registry helper which lands
    # at most one BLOCKER critique per round per critic. Force the
    # judge to converge on round 1 so we only see one synthesis pass.
    from tests.refinery.conftest import make_stub_critic_registry
    from dark_factory.config import PipelineConfig

    cfg = PipelineConfig()
    registry = make_stub_critic_registry(cfg)

    with patched_product_propose(refined), forced_judge_score(passing=True):
        terminal = run_debate(
            **debate_kwargs(
                title="t", description="d",
                refinery_run_id="refinery-disp-test",
            ),
            config=cfg,
            registry=registry,
            disposition_sink=sink,
        )

    assert terminal["final_trace"]["convergence_status"] == \
        ConvergenceStatus.CONVERGED.value
    # At least one disposition must have been recorded; each row must
    # carry the schema the metrics repo expects.
    assert rows, "synthesize node should have fired the disposition sink"
    expected_keys = {
        "refinery_run_id", "requirement_id", "round_number",
        "role", "severity", "dimension", "action",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys
        assert row["refinery_run_id"] == "refinery-disp-test"
        assert row["requirement_id"] == "req-1"
        assert row["round_number"] >= 1
        assert row["action"] in {"accepted", "rejected", "deferred"}


def test_defend_accepts_roleweight_objects_too():
    """Callers that hand in ``dict[str, RoleWeight]`` (the richer audit
    shape) get unpacked transparently — Judge doesn't have to know
    which form was used."""

    role = JudgeRole()
    captured: list[str] = []
    rw = {"security": RoleWeight(
        role="security", multiplier=1.10,
        accepted=9, rejected=1, deferred=0, total=10,
        acceptance_rate=0.9, reason="high_acceptance",
    )}
    with patch.object(role, "_call_llm",
                      side_effect=lambda p: (captured.append(p), "")[1]):
        role.defend(
            _draft(), [_critique()],
            RoleContext(
                role="judge", requirement_id="r-1", round_number=1,
                evidence={"role_weights": rw},
            ),
        )
    assert "security: ×1.10" in captured[0]
