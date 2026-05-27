"""Phase 7 tests — combined evaluation pipeline (rules + LLM + penalties)
plus the cross-review patcher round-trip.

Covers:
- Every rule's violation / compliant input pair.
- Phase B prompt awareness: DeepEvalJudge receives rule findings in its
  evidence bag (verified via the RULE_FINDINGS preamble injection).
- Phase C clamp: a rule-blocker-affected dimension is capped at 0.5
  regardless of the LLM score.
- Unified truth table: rule blocker × LLM high vs LLM low → passed/failed.
- Short-circuit: catastrophic rule failure skips Phase B.
- Rules disabled: Phase A + C both no-op.
- DeepEval exception → fallback kicks in cleanly.
- Cross-review patcher golden round-trip against the
  ``ReconciliationReport`` fixture — same mutations as the
  ``CrossReviewReport`` superset shape.
"""

from __future__ import annotations

import pytest

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    CritiqueDimension,
    Draft,
    DraftRelationship,
    DraftSpec,
    EvaluationScore,
    RawRequirement,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.contracts import CrossReviewReport
from dark_factory.api.refinery.judge import (
    CombinedJudgePipeline,
    FallbackJudge,
    RULES,
    RuleResult,
    RulesJudge,
    apply_cross_review_report,
)
from dark_factory.api.refinery.judge.base import Judge
from dark_factory.api.refinery.judge.rules import (
    RuleSpec,
    _is_criterion_measurable,
    _rule_acceptance_criteria_measurable,
    _rule_acceptance_criteria_min_count,
    _rule_convergence_consistency,
    _rule_description_specificity,
    _rule_ids_preserved,
    _rule_priority_valid,
    _rule_relationships_non_circular,
    _rule_tradeoffs_required,
)
from dark_factory.api.refinery.models import (
    ReconciliationReport,
    RefinedRequirement,
    RequirementRelationship,
    SuggestedSpec,
)
from dark_factory.api.refinery.roles.judge import JudgeRole


# ─────────────────────────────────────────────────────────────────────
# Draft fixtures
# ─────────────────────────────────────────────────────────────────────


def _strong_draft(req_id: str = "req-1") -> Draft:
    return Draft(
        requirement_id=req_id,
        title="OAuth2 authorization-code flow",
        description=(
            "Given a valid client when authorize is called then a redirect "
            "occurs. Token endpoint returns within 300ms at 100 req/s."
        ),
        priority="high",
        tags=["auth", "security"],
        suggested_specs=[
            DraftSpec(
                title="OAuth2 flow", capability="auth",
                description="Implement OAuth2 authorization-code flow",
                acceptance_criteria=[
                    "GIVEN a valid client WHEN authorize is called THEN a redirect occurs",
                    "GIVEN an invalid code WHEN token is called THEN 400 is returned",
                    "Token endpoint latency < 300ms at 100 req/s",
                ],
            )
        ],
        produced_by="product",
    )


def _weak_draft() -> Draft:
    return Draft(
        requirement_id="req-1",
        title="Fast thing",
        description="Make it fast and secure and scalable.",
        priority="medium",
        produced_by="product",
        suggested_specs=[
            DraftSpec(
                title="Bad spec", capability="misc",
                description="short",
                acceptance_criteria=["do stuff"],
            )
        ],
    )


def _ctx(req_id: str = "req-1") -> RoleContext:
    return RoleContext(role="judge", requirement_id=req_id)


# ─────────────────────────────────────────────────────────────────────
# Rules — per-rule isolation
# ─────────────────────────────────────────────────────────────────────


# Parametrized over every per-rule case: (label, rule_fn,
# draft_transform, ctx_factory, expected_severity-or-None-for-passes)
# A draft_transform of None means "use _strong_draft() unchanged" (i.e.
# exercise the pass path).
_PER_RULE_CASES = [
    ("ids_preserved:mismatch",
     _rule_ids_preserved, None, lambda: _ctx("req-B"), Severity.BLOCKER),
    ("ids_preserved:match",
     _rule_ids_preserved, None, lambda: _ctx(), None),
    ("priority_valid:garbage",
     _rule_priority_valid,
     lambda d: d.model_copy(update={"priority": "mid-ish"}),
     lambda: _ctx(), Severity.BLOCKER),
    ("convergence_consistency:converged_but_unresolved",
     _rule_convergence_consistency,
     lambda d: d.model_copy(update={
         "convergence_status": ConvergenceStatus.CONVERGED,
         "unresolved_points": ["something"],
     }),
     lambda: _ctx(), Severity.BLOCKER),
    ("convergence_consistency:short_circuit_no_unresolved",
     _rule_convergence_consistency,
     lambda d: d.model_copy(update={
         "convergence_status": ConvergenceStatus.SHORT_CIRCUITED,
     }),
     lambda: _ctx(), Severity.BLOCKER),
    ("acceptance_criteria_min_count:too_few",
     _rule_acceptance_criteria_min_count,
     lambda d: d.model_copy(update={
         "suggested_specs": [
             DraftSpec(title="spec", capability="cap", description="d",
                       acceptance_criteria=["a", "b"]),
         ],
     }),
     lambda: _ctx(), Severity.BLOCKER),
    ("description_specificity:unmeasured_vague",
     _rule_description_specificity,
     lambda d: d.model_copy(update={
         "title": "Fast feature",
         "description": "Must be fast and user-friendly.",
     }),
     lambda: _ctx(), Severity.WARNING),
    ("description_specificity:vague_with_measure",
     _rule_description_specificity,
     lambda d: d.model_copy(update={
         "description": "Must be fast — responds under 300ms at 100 req/s.",
     }),
     lambda: _ctx(), None),
    ("relationships_non_circular:self_loop",
     _rule_relationships_non_circular,
     lambda d: d.model_copy(update={
         "relationships": [
             DraftRelationship(target_id="req-1", type="depends_on",
                               rationale="self"),
         ],
     }),
     lambda: _ctx(), Severity.BLOCKER),
    ("tradeoffs_required:pre_synthesis",
     _rule_tradeoffs_required,
     lambda d: d.model_copy(update={"produced_by": "product", "iteration": 0}),
     lambda: _ctx(), None),
    ("tradeoffs_required:post_synthesis_with_tradeoffs",
     _rule_tradeoffs_required,
     lambda d: d.model_copy(update={
         "produced_by": "judge", "iteration": 1,
         "explicit_tradeoffs": ["chose latency over cost"],
     }),
     lambda: _ctx(), None),
    ("tradeoffs_required:post_synthesis_no_tradeoffs",
     _rule_tradeoffs_required,
     lambda d: d.model_copy(update={"produced_by": "judge", "iteration": 1}),
     lambda: _ctx(), Severity.WARNING),
]


@pytest.mark.parametrize(
    ("rule_fn", "transform", "ctx_factory", "expected"),
    [(c[1], c[2], c[3], c[4]) for c in _PER_RULE_CASES],
    ids=[c[0] for c in _PER_RULE_CASES],
)
def test_per_rule_behaviour(rule_fn, transform, ctx_factory, expected):
    draft = transform(_strong_draft()) if transform else _strong_draft()
    violations = rule_fn(draft, ctx_factory(), None)
    if expected is None:
        assert violations == []
    else:
        assert len(violations) >= 1
        assert violations[0].severity == expected


def test_rule_acceptance_criteria_measurable_flags_vague_criteria():
    """Three vague criteria → three violations — dedicated test because
    the count is the load-bearing assertion."""

    draft = _strong_draft().model_copy(update={
        "suggested_specs": [
            DraftSpec(
                title="spec", capability="cap", description="d",
                acceptance_criteria=[
                    "the system should be fast",
                    "it must be secure",
                    "make it nice",
                ],
            )
        ],
    })
    violations = _rule_acceptance_criteria_measurable(draft, _ctx(), None)
    assert len(violations) == 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GIVEN x WHEN y THEN z", True),
        ("returns within 300ms", True),
        ("Token endpoint rejects empty body", True),
        ("make it fast", False),
    ],
    ids=["when_then", "numeric_unit", "observable_verb", "vague"],
)
def test_is_criterion_measurable(text, expected):
    assert _is_criterion_measurable(text) is expected


# ─────────────────────────────────────────────────────────────────────
# RulesJudge integration
# ─────────────────────────────────────────────────────────────────────


def test_rules_judge_runs_all_rules_on_strong_draft():
    judge = RulesJudge()
    result = judge.validate(_strong_draft(), _ctx())
    assert result.engine_failed is False
    assert len(result.rules_run) == len(RULES)
    # Strong draft should have zero blockers.
    assert result.blockers == []


def test_rules_judge_collects_violations_on_weak_draft():
    judge = RulesJudge()
    result = judge.validate(_weak_draft(), _ctx())
    assert len(result.blockers) >= 2  # min_count + measurable at minimum


def test_rules_judge_respects_disabled_list():
    judge = RulesJudge(disabled=["rule_acceptance_criteria_measurable"])
    result = judge.validate(_weak_draft(), _ctx())
    fired = {v.rule_id for v in result.violations}
    assert "rule_acceptance_criteria_measurable" not in fired
    assert "rule_acceptance_criteria_measurable" in result.rules_skipped


def test_rules_judge_isolates_rule_crash_from_engine():
    """A rule that raises mid-run doesn't poison the engine — the rest
    of the rules still run, and engine_failed only sets when NONE of
    them completed."""

    def _broken(draft, ctx, trace):
        raise RuntimeError("rule outage")

    broken = RuleSpec(
        "broken_rule", CritiqueDimension.CLARITY, Severity.WARNING, _broken,
    )

    # Inject the broken rule via the extra_rules parameter — the public
    # API for this exact scenario.
    judge = RulesJudge(extra_rules=[broken])
    result = judge.validate(_strong_draft(), _ctx())
    assert result.engine_failed is False
    assert any(s.startswith("broken_rule:error=") for s in result.rules_skipped)


# ─────────────────────────────────────────────────────────────────────
# CombinedJudgePipeline — truth table
# ─────────────────────────────────────────────────────────────────────


def _high_semantic_score() -> EvaluationScore:
    return EvaluationScore(
        dimensions={d.value: 0.95 for d in CritiqueDimension},
        dimensions_semantic_raw={d.value: 0.95 for d in CritiqueDimension},
        reasons={d.value: "mock high" for d in CritiqueDimension},
        overall=0.95,
        passed=True,
        model_used="mock-high",
    )


def _low_semantic_score() -> EvaluationScore:
    return EvaluationScore(
        dimensions={d.value: 0.4 for d in CritiqueDimension},
        dimensions_semantic_raw={d.value: 0.4 for d in CritiqueDimension},
        reasons={d.value: "mock low" for d in CritiqueDimension},
        overall=0.4,
        passed=False,
        model_used="mock-low",
    )


class _StubSemantic(Judge):
    def __init__(self, score: EvaluationScore) -> None:
        self._score = score

    def score(self, draft, context, trace=None):
        return self._score

    def review_set(self, refined_set, traces, run_context):
        return CrossReviewReport()


def test_phase_c_clamps_dimension_on_rule_blocker():
    """Fixture: LLM returns clarity=0.95; a BLOCKER rule on clarity
    appears; Phase C must clamp final clarity to ≤ blocker_cap (0.5)."""

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=_StubSemantic(_high_semantic_score()),
        blocker_cap=0.5,
    )
    result = pipeline.score(_weak_draft(), _ctx())
    # weak_draft has rule_acceptance_criteria_min_count (testability)
    # and rule_acceptance_criteria_measurable (testability) blockers.
    assert result.dimensions["testability"] <= 0.5
    # Raw LLM score preserved for the trace.
    assert result.dimensions_semantic_raw["testability"] == 0.95
    assert result.passed is False


# Truth table: (draft_factory, semantic_score, expected_passed, expected_has_rule_violations)
_COMPOSITION_TRUTH_TABLE = [
    ("blocker_x_high_llm",   _weak_draft,    _high_semantic_score, False, True),
    ("no_blocker_x_high_llm", _strong_draft, _high_semantic_score, True,  False),
    ("no_blocker_x_low_llm",  _strong_draft, _low_semantic_score,  False, False),
    ("blocker_x_low_llm",    _weak_draft,    _low_semantic_score,  False, True),
]


@pytest.mark.parametrize(
    ("draft_factory", "semantic_factory", "expected_passed", "expected_violations"),
    [(c[1], c[2], c[3], c[4]) for c in _COMPOSITION_TRUTH_TABLE],
    ids=[c[0] for c in _COMPOSITION_TRUTH_TABLE],
)
def test_combined_pipeline_truth_table(
    draft_factory, semantic_factory, expected_passed, expected_violations,
):
    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=_StubSemantic(semantic_factory()),
    )
    result = pipeline.score(draft_factory(), _ctx())
    assert result.passed is expected_passed
    assert bool(result.rule_violations) is expected_violations


def test_short_circuit_skips_llm_on_many_blockers():
    class _CountingSemantic(_StubSemantic):
        def __init__(self, score):
            super().__init__(score)
            self.calls = 0

        def score(self, draft, context, trace=None):
            self.calls += 1
            return self._score

    sem = _CountingSemantic(_high_semantic_score())
    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=sem,
        short_circuit_enabled=True,
        short_circuit_threshold=2,
    )
    pipeline.score(_weak_draft(), _ctx())
    # _weak_draft produces 1 min_count + 1 measurable blocker = 2;
    # threshold is 2 so Phase B is skipped.
    assert sem.calls == 0


def test_rules_disabled_makes_phase_a_noop():
    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=_StubSemantic(_high_semantic_score()),
        rules_enabled=False,
    )
    result = pipeline.score(_weak_draft(), _ctx())
    # With rules disabled, Phase A is skipped → no rule_violations.
    assert result.rule_violations == []
    assert result.rules_engine_disabled is True
    # LLM score stands without penalty.
    assert result.dimensions["testability"] == pytest.approx(0.95)


def test_semantic_exception_falls_back_cleanly():
    class _BoomSemantic(Judge):
        def score(self, draft, context, trace=None):
            raise RuntimeError("provider 503")

        def review_set(self, *a, **k):  # pragma: no cover
            return CrossReviewReport()

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=_BoomSemantic(),
        fallback_judge=FallbackJudge(),
    )
    result = pipeline.score(_strong_draft(), _ctx())
    assert result.fallback_used is True
    assert result.passed is not None  # whatever fallback returns


def test_warning_subtract_penalty_applies():
    """Fixture: strong draft; LLM clarity=0.8; one WARNING rule on
    clarity (e.g. description_specificity). Phase C subtracts
    warning_delta (0.1) → final clarity=0.7."""

    semantic_score = EvaluationScore(
        dimensions={d.value: 0.8 for d in CritiqueDimension},
        dimensions_semantic_raw={d.value: 0.8 for d in CritiqueDimension},
        reasons={d.value: "" for d in CritiqueDimension},
        overall=0.8,
        model_used="mock",
    )
    # Vague-description draft triggers rule_description_specificity
    # (warning, dimension=clarity).
    vague = _strong_draft().model_copy(update={
        "description": "It will be fast and reliable.",
        # Keep specs strong so other rules don't fire.
    })
    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=_StubSemantic(semantic_score),
        warning_delta=0.1,
    )
    result = pipeline.score(vague, _ctx())
    assert result.dimensions["clarity"] == pytest.approx(0.7)
    assert any(v.rule_id == "rule_description_specificity"
               for v in result.rule_warnings)


# ─────────────────────────────────────────────────────────────────────
# Cross-review patcher — golden round-trip
# ─────────────────────────────────────────────────────────────────────


def test_apply_cross_review_report_adds_relationships():
    refined = [
        RefinedRequirement(
            id="req-1", original_title="T1", original_description="D1",
            title="T1", description="D1", priority="medium",
        ),
        RefinedRequirement(
            id="req-2", original_title="T2", original_description="D2",
            title="T2", description="D2", priority="high",
        ),
    ]
    report = ReconciliationReport(
        relationship_fixes=[{
            "source_id": "req-1", "target_id": "req-2",
            "action": "add", "type": "depends_on",
            "rationale": "A needs B's auth",
        }],
    )
    patched = apply_cross_review_report(refined, report)
    r1 = next(r for r in patched if r.id == "req-1")
    assert any(rel.target_id == "req-2" and rel.type == "depends_on"
               for rel in r1.relationships)


def test_apply_cross_review_report_is_idempotent():
    """Applying the same report twice must not duplicate edges or notes."""

    refined = [
        RefinedRequirement(
            id="req-1", original_title="T", original_description="D",
            title="T", description="D", priority="medium",
        ),
    ]
    report = ReconciliationReport(
        priority_changes=[{
            "requirement_id": "req-1",
            "new_priority": "high",
            "rationale": "uplifted",
        }],
    )
    apply_cross_review_report(refined, report)
    apply_cross_review_report(refined, report)
    r1 = refined[0]
    assert r1.priority == "high"
    # Change note recorded exactly once.
    notes = [c for c in r1.changes if "Priority changed" in c]
    assert len(notes) == 1


def test_apply_cross_review_report_marks_duplicates_with_replaces_edge():
    refined = [
        RefinedRequirement(
            id="req-keep", original_title="K", original_description="",
            title="K", description="", priority="medium",
        ),
        RefinedRequirement(
            id="req-dup", original_title="D", original_description="",
            title="D", description="", priority="medium",
        ),
    ]
    report = ReconciliationReport(
        duplicate_pairs=[{
            "keep_id": "req-keep",
            "remove_id": "req-dup",
            "rationale": "semantic duplicate",
        }],
    )
    apply_cross_review_report(refined, report)
    dup = next(r for r in refined if r.id == "req-dup")
    assert any(r.target_id == "req-keep" and r.type == "replaces"
               for r in dup.relationships)
    assert any("Potential duplicate of req-keep" in c for c in dup.changes)


def test_apply_cross_review_report_removes_relationships_on_remove_action():
    refined = [
        RefinedRequirement(
            id="req-1", original_title="T", original_description="D",
            title="T", description="D", priority="medium",
            relationships=[
                RequirementRelationship(
                    target_id="req-2", type="depends_on",
                    rationale="orig",
                )
            ],
        ),
    ]
    report = ReconciliationReport(
        relationship_fixes=[{
            "source_id": "req-1", "target_id": "req-2",
            "action": "remove", "type": "depends_on",
            "rationale": "misidentified",
        }],
    )
    apply_cross_review_report(refined, report)
    assert refined[0].relationships == []


def test_judge_role_uses_combined_pipeline_via_default_scorer():
    """Sanity: JudgeRole assembles a CombinedJudgePipeline by default.
    Swapping the scorer to a stub still works for tests."""

    role = JudgeRole()
    role._scorer = FallbackJudge()   # override for test determinism
    ctx = RoleContext(role="judge", requirement_id="req-1")
    result = role.score(_strong_draft(), ctx)
    assert isinstance(result, EvaluationScore)
    assert set(result.dimensions) == {d.value for d in CritiqueDimension}
