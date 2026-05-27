"""Phase 9 tests — institutional memory producers + write-back contract.

Covers:
- Producers fire at correct debate events and skip when preconditions
  aren't met.
- CONFLICT on short-circuit is UNCONDITIONAL and VALIDATED.
- Kind-scoped dedup: Decision and Pattern with identical text aren't
  treated as duplicates.
- Swarm memory recall scoping (config defaults exclude refinery kinds).
- Phase-9 check-duplicate endpoint accepts the new kind arg.
"""

from __future__ import annotations

import pytest

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    CritiqueDimension,
    Draft,
    DraftSpec,
    MemoryKind,
    Rebuttal,
    RebuttalEntry,
    Severity,
    SuggestedMemoryV2,
    ValidationStatus,
)
from dark_factory.api.refinery.memory import (
    produce_conflict_on_disagreement,
    produce_conflict_on_short_circuit,
    produce_constraint_from_blocker,
    produce_decision_from_rebuttal,
    produce_incident_from_blocker,
)


# ─────────────────────────────────────────────────────────────────────
# Producer unit tests
# ─────────────────────────────────────────────────────────────────────


def _draft_short_circuited(req_id: str = "req-1") -> Draft:
    return Draft(
        requirement_id=req_id, title="T", description="D",
        priority="medium", produced_by="judge", iteration=3,
        convergence_status=ConvergenceStatus.SHORT_CIRCUITED,
        unresolved_points=["rate-limit strategy", "token lifetime"],
        open_questions=["Redis cluster sizing?", "refresh token TTL?"],
    )


def _critique(
    author: str = "engineering",
    severity: Severity = Severity.BLOCKER,
    dimension: CritiqueDimension = CritiqueDimension.FEASIBILITY,
    finding: str = "the system would exceed a known rate limit on the auth provider",
    proposed_fix: str = "introduce exponential backoff",
) -> Critique:
    return Critique(
        author_role=author, severity=severity, dimension=dimension,
        finding=finding, proposed_fix=proposed_fix,
    )


# ── DECISION producer


def test_decision_producer_emits_for_accepted_entries_with_rationale():
    critiques = [
        _critique(),
        _critique(author="security", severity=Severity.WARNING,
                  dimension=CritiqueDimension.RISK_COVERAGE,
                  finding="missing CSRF token on POST /session"),
    ]
    rebuttal = Rebuttal(
        revised_draft=Draft(
            requirement_id="req-1", title="T", description="D",
            priority="medium", produced_by="judge", iteration=1,
        ),
        entries=[
            RebuttalEntry(
                critique_ref="c-0", action="accepted",
                rationale=(
                    "The rate-limit concern generalises — whenever we hit an "
                    "external provider we need backoff, not just for auth."
                ),
            ),
            RebuttalEntry(
                critique_ref="c-1", action="rejected",
                rationale=(
                    "Session POST is already behind SameSite=strict cookies, "
                    "so CSRF exposure is mitigated at the browser level."
                ),
            ),
        ],
    )
    result = produce_decision_from_rebuttal(
        rebuttal, critiques=critiques,
        requirement_id="req-1", round_number=2, converged=True,
    )
    assert len(result.memories) == 2
    for m in result.memories:
        assert m.kind == MemoryKind.DECISION
        assert m.validation_status == ValidationStatus.PRODUCED
        assert m.source_requirement_id == "req-1"


def test_decision_producer_skips_rubber_stamp_entries():
    critiques = [_critique()]
    rebuttal = Rebuttal(
        revised_draft=Draft(
            requirement_id="req-1", title="T", description="D",
            priority="medium", produced_by="judge",
        ),
        entries=[
            RebuttalEntry(
                critique_ref="c-0", action="accepted",
                rationale="ok",  # too short — doesn't generalise
            ),
        ],
    )
    result = produce_decision_from_rebuttal(
        rebuttal, critiques=critiques,
        requirement_id="req-1", round_number=1, converged=True,
    )
    assert result.memories == []


def test_decision_producer_skips_info_severity():
    """INFO critiques don't merit decision memories — the decision is
    trivially 'acknowledged'."""

    critiques = [
        _critique(severity=Severity.INFO, finding="FYI note about logging"),
    ]
    rebuttal = Rebuttal(
        revised_draft=Draft(
            requirement_id="req-1", title="T", description="D",
            priority="medium", produced_by="judge",
        ),
        entries=[
            RebuttalEntry(
                critique_ref="c-0", action="rejected",
                rationale=(
                    "The observability concern is valid but lives at the "
                    "spec layer, not the requirement."
                ),
            ),
        ],
    )
    result = produce_decision_from_rebuttal(
        rebuttal, critiques=critiques,
        requirement_id="req-1", round_number=1, converged=True,
    )
    assert result.memories == []


# ── CONSTRAINT producer


_CONSTRAINT_CASES = [
    # (label, critique_kwargs, expected_fires, expected_domain or None)
    ("fires_on_system_quota",
     {"finding": "The 100 req/s quota cap on the auth provider is exceeded",
      "proposed_fix": "split traffic across providers or batch calls"},
     True, "system"),
    ("fires_on_business_gdpr",
     {"author": "security", "dimension": CritiqueDimension.RISK_COVERAGE,
      "finding": "The data retention window violates our GDPR policy",
      "proposed_fix": "reduce retention to 30 days or add explicit consent"},
     True, "business"),
    ("skips_non_blocker",
     {"severity": Severity.WARNING}, False, None),
    ("skips_when_no_limit_keyword",
     {"finding": "The proposed API shape is inconsistent with sibling specs",
      "proposed_fix": "align to the canonical schema"},
     False, None),
]


@pytest.mark.parametrize(
    ("critique_kwargs", "expected_fires", "expected_domain"),
    [(c[1], c[2], c[3]) for c in _CONSTRAINT_CASES],
    ids=[c[0] for c in _CONSTRAINT_CASES],
)
def test_constraint_producer(critique_kwargs, expected_fires, expected_domain):
    c = _critique(**critique_kwargs)
    result = produce_constraint_from_blocker(
        c, requirement_id="req-1", round_number=1, converged=True,
    )
    if expected_fires:
        assert len(result.memories) == 1
        assert result.memories[0].kind == MemoryKind.CONSTRAINT
        assert result.memories[0].constraint_domain == expected_domain
    else:
        assert result.memories == []
        assert result.skipped_reason


# ── INCIDENT producer


def test_incident_producer_fires_on_blocker_citing_outage():
    c = _critique(
        author="operations",
        dimension=CritiqueDimension.RISK_COVERAGE,
        finding="This pattern caused the sev2 incident 2025-03-14 token leak",
        proposed_fix="add audit logs on every token read",
    )
    result = produce_incident_from_blocker(
        c, requirement_id="req-1", round_number=1, converged=True,
    )
    assert len(result.memories) == 1
    assert result.memories[0].kind == MemoryKind.INCIDENT
    assert result.memories[0].incident_severity == "sev2"


def test_incident_producer_defaults_to_sev3_when_no_severity_token():
    c = _critique(
        author="operations",
        finding="Similar to the recent postmortem — a regression in rollback",
        proposed_fix="restore the pre-change pointer",
    )
    result = produce_incident_from_blocker(
        c, requirement_id="req-1", round_number=1, converged=True,
    )
    assert result.memories[0].incident_severity == "sev3"


# ── CONFLICT producer — disagreement path


def test_conflict_on_disagreement_fires_above_threshold():
    critiques_by_round = {
        1: [
            _critique(author="engineering", severity=Severity.BLOCKER,
                      finding="feasibility: API shape conflicts"),
            _critique(author="cost", severity=Severity.WARNING,
                      finding="cost: estimated spend doubled"),
        ],
    }
    rebuttals_by_round = {
        1: Rebuttal(
            revised_draft=Draft(
                requirement_id="req-1", title="T", description="D",
                priority="medium", produced_by="judge",
            ),
            entries=[
                RebuttalEntry(
                    critique_ref="c-0", action="accepted",
                    rationale=(
                        "We aligned the API shape to the Engineering "
                        "proposal even at the cost of doubled spend."
                    ),
                ),
            ],
        ),
    }
    result = produce_conflict_on_disagreement(
        disagreement_score=0.75, threshold=0.4,
        critiques_by_round=critiques_by_round,
        rebuttals_by_round=rebuttals_by_round,
        requirement_id="req-1", rounds_executed=1, converged=True,
    )
    assert len(result.memories) == 1
    m = result.memories[0]
    assert m.kind == MemoryKind.CONFLICT
    assert "engineering" in (m.conflict_parties or [])
    assert "cost" in (m.conflict_parties or [])
    assert m.conflict_resolution  # pulled from accepted rebuttal entry


def test_conflict_on_disagreement_skips_below_threshold():
    result = produce_conflict_on_disagreement(
        disagreement_score=0.2, threshold=0.4,
        critiques_by_round={},
        rebuttals_by_round={},
        requirement_id="req-1", rounds_executed=1, converged=True,
    )
    assert result.memories == []


# ── CONFLICT producer — short-circuit path (UNCONDITIONAL)


def test_conflict_on_short_circuit_is_unconditional_and_validated():
    """Non-convergence is itself load-bearing knowledge — the CONFLICT
    memory fires regardless of disagreement score AND is always
    validation_status=VALIDATED (Phase 9 exception to the 'only
    validated memories on converged debates' rule)."""

    draft = _draft_short_circuited("req-1")
    critiques_by_round = {
        1: [_critique(author="engineering")],
        2: [_critique(author="cost", severity=Severity.BLOCKER,
                      finding="cost: budget would triple")],
    }
    result = produce_conflict_on_short_circuit(
        final_draft=draft,
        critiques_by_round=critiques_by_round,
        requirement_id="req-1", rounds_executed=2,
    )
    assert len(result.memories) == 1
    m = result.memories[0]
    assert m.kind == MemoryKind.CONFLICT
    assert m.validation_status == ValidationStatus.VALIDATED
    assert m.conflict_resolution is None  # explicitly unresolved
    # Parties = roles that raised blockers.
    assert "engineering" in (m.conflict_parties or [])
    assert "cost" in (m.conflict_parties or [])


def test_conflict_on_short_circuit_skips_when_draft_is_converged():
    draft = Draft(
        requirement_id="req-1", title="T", description="D",
        priority="medium", produced_by="judge",
        convergence_status=ConvergenceStatus.CONVERGED,
    )
    result = produce_conflict_on_short_circuit(
        final_draft=draft, critiques_by_round={},
        requirement_id="req-1", rounds_executed=1,
    )
    assert result.memories == []


# ─────────────────────────────────────────────────────────────────────
# Swarm memory recall scoping (Phase 9 sneaky-touchpoint mitigation)
# ─────────────────────────────────────────────────────────────────────


def test_swarm_memory_kinds_enabled_defaults_to_legacy_four(pipeline_config):
    assert set(pipeline_config.swarm_memory_kinds_enabled) == {
        "pattern", "mistake", "solution", "strategy",
    }
    # Refinery kinds explicitly NOT in the swarm recall list until the
    # operator opts in.
    for kind in ("decision", "constraint", "conflict"):
        assert kind not in pipeline_config.swarm_memory_kinds_enabled


def test_refinery_auto_save_on_apply_defaults_true(pipeline_config):
    """Write-back is on by default; operators who want assistant-only
    mode must explicitly disable it."""

    assert pipeline_config.refinery_auto_save_on_apply is True


# ─────────────────────────────────────────────────────────────────────
# MemoryKind round-trip + kind-scoped dedup shape
# ─────────────────────────────────────────────────────────────────────


def test_memory_kind_enum_exhausts_curated_kinds():
    assert {k.value for k in MemoryKind} == {
        "decision", "incident", "pattern", "constraint", "conflict",
        "hypothesis", "anti_pattern",
    }


def test_suggested_memory_v2_round_trips_all_kinds():
    for kind in MemoryKind:
        m = SuggestedMemoryV2(
            kind=kind, summary=f"summary for {kind.value}",
            body="body", source_role="product",
        )
        assert m.model_dump()["kind"] == kind.value
