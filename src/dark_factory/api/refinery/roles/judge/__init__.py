"""Judge role — synthesis + scoring + cross-req review.

The Judge owns four verbs: ``defend`` (synthesis),
``reconcile_unresolved`` (short-circuit synthesis), ``score`` (combined
rules-plus-DeepEval evaluation), and ``review_set`` (cross-requirement
adjudication).

The three LLM-driven verbs (``defend``, ``reconcile_unresolved``,
``review_set``) format a prompt from ``judge/prompt.py`` and hand it to
an injectable ``_call_llm`` hook. When no LLM is wired they fall
through to deterministic behaviour — the debate graph stays runnable
without external dependencies. ``score`` is built on the
``CombinedJudgePipeline`` and degrades to ``FallbackJudge`` when
DeepEval raises.
"""

from __future__ import annotations

from typing import Any, ClassVar

import structlog

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    CrossReviewReport,
    Critique,
    Draft,
    EvaluationScore,
    Rebuttal,
    RebuttalEntry,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
from dark_factory.api.refinery.roles.base import RoleAgent
from dark_factory.log import trace_methods

log = structlog.get_logger()


def _coerce_weight_map(raw: dict) -> dict[str, float]:
    """Normalize the evidence-bag ``role_weights`` payload to a flat
    ``{role: float}`` dict. Accepts ``RoleWeight`` objects (in-process
    callers) and plain floats (orchestrator-stamped JSON shape)."""

    out: dict[str, float] = {}
    for role, val in raw.items():
        if isinstance(val, (int, float)):
            out[role] = float(val)
        elif hasattr(val, "multiplier"):
            out[role] = float(val.multiplier)
    return out


def _coerce_dim_weight_map(raw: dict) -> dict[str, float]:
    """Normalize ``role_dim_weights`` to a flat ``{"role:dim": float}``
    dict. Accepts both the JSON-friendly string-key shape and the
    legacy tuple-keyed shape (used by in-process tests)."""

    out: dict[str, float] = {}
    for key, val in raw.items():
        if isinstance(key, tuple) and len(key) == 2:
            flat_key = f"{key[0]}:{key[1]}"
        elif isinstance(key, str):
            flat_key = key
        else:
            continue
        if isinstance(val, (int, float)):
            out[flat_key] = float(val)
        elif hasattr(val, "multiplier"):
            out[flat_key] = float(val.multiplier)
    return out


@trace_methods
class JudgeRole(RoleAgent):
    """Synthesis + verdict + cross-req review.

    Verbs:
    - ``defend`` — pass-through synth (LLM synth is the eventual replacement)
    - ``reconcile_unresolved`` — short-circuit synthesis on non-convergence
    - ``score`` — combined rules + DeepEval gate (falls back to heuristic)
    - ``review_set`` — cross-requirement review
    """

    role_name: ClassVar[str] = "judge"
    default_model: ClassVar[str] = "claude-opus-4-6"
    default_reasoning_effort: ClassVar[str] = "xhigh"

    def __init__(self) -> None:
        super().__init__()
        # Scorer is lazily built on first access — a CombinedJudgePipeline
        # (rules + DeepEval + FallbackJudge backstop). Tests can inject a
        # stub directly via ``self._scorer`` to skip the default assembly.
        self._scorer: Any | None = None

    def _call_llm(self, prompt: str) -> str:
        """LLM hook used by synthesis verbs (``defend`` /
        ``reconcile_unresolved`` / ``review_set``).

        Delegates to the shared refinery LLM helper so the synthesis
        call appears in the Agent Log under the ``judge`` badge with
        tokens + latency. Tests monkey-patch this method directly and
        never reach the helper.
        """

        from dark_factory.api.refinery.roles._shared.llm import call_refinery_llm

        return call_refinery_llm(
            prompt,
            model=self.model,
            reasoning_effort=self._reasoning_effort or self.default_reasoning_effort,
            agent="judge",
        )

    @property
    def model(self) -> str:
        return self._model or self.default_model

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            self._scorer = _build_default_scorer()
        return self._scorer

    def defend(
        self,
        draft: Draft,
        critiques: list[Critique],
        context: RoleContext,
    ) -> Rebuttal:
        """Synthesize a revised draft by folding in critiques.

        When an LLM is wired via ``_call_llm``, uses the real synthesis
        prompt from ``roles/judge/prompt.py`` and parses the returned
        JSON Rebuttal. Falls through to a deterministic pass-through
        when no LLM binding is available — keeps the debate graph
        runnable without external dependencies.
        """

        from dark_factory.api.refinery.roles.judge.prompt import (
            format_defend_prompt,
        )

        new_iteration = draft.iteration + 1
        # Calibration: prefer the orchestrator's pre-rendered block (stamped
        # once at run start — role×dim is invariant across rounds of the
        # same debate). Fall through to building from raw weight maps when
        # the orchestrator didn't pre-render (in-process tests, legacy
        # callers).
        evidence = context.evidence or {}
        cal_block = evidence.get("calibration_block")
        flat_weights: dict[str, float] = _coerce_weight_map(
            evidence.get("role_weights") or {},
        )
        flat_dim_weights: dict[str, float] = _coerce_dim_weight_map(
            evidence.get("role_dim_weights") or {},
        )
        try:
            prompt = format_defend_prompt(
                draft, critiques, new_iteration,
                role_weights=flat_weights or None,
                role_dim_weights=flat_dim_weights or None,
                weights_block_override=cal_block,
                rule_findings=evidence.get("rule_findings"),
            )
            raw = self._call_llm(prompt)
            parsed = _parse_rebuttal_json(raw, default_mode="synthesize")
            if parsed is not None:
                return parsed
        except NotImplementedError:
            pass  # no LLM wired — fall through
        except Exception as exc:
            log.warning("judge_defend_llm_failed", error=str(exc))

        # Deterministic pass-through: record each critique's disposition
        # without mutating the draft body. Useful for tests and as the
        # permanent backstop when an LLM call fails.
        revised = draft.model_copy(update={
            "produced_by": self.role_name,
            "iteration": new_iteration,
            "convergence_status": ConvergenceStatus.CONVERGED,
        })
        entries = [
            RebuttalEntry(
                critique_ref=f"c-{i}",
                action="accepted" if c.severity == Severity.BLOCKER else "deferred",
                rationale=(
                    f"pass-through disposition for {c.severity.value} "
                    f"critique on {c.dimension.value}"
                ),
            )
            for i, c in enumerate(critiques)
        ]
        return Rebuttal(revised_draft=revised, entries=entries, mode="synthesize")

    def score(
        self,
        draft: Draft,
        context: RoleContext,
        trace: Any = None,
    ) -> EvaluationScore:
        """Delegate to the configured scorer (``CombinedJudgePipeline``
        by default)."""

        return self.scorer.score(draft, context, trace)

    def prepare_calibration(
        self,
        *,
        role_weights: dict[str, float] | None = None,
        role_dim_weights: dict[str, float] | None = None,
    ) -> str:
        from dark_factory.api.refinery.roles.judge.prompt import (
            format_calibration_block,
        )

        return format_calibration_block(
            role_weights=role_weights,
            role_dim_weights=role_dim_weights,
        )

    def reconcile_unresolved(
        self,
        draft: Draft,
        critiques: list[Critique],
        prior_rebuttals: list[Rebuttal],
        scores: list[EvaluationScore],
        context: RoleContext,
    ) -> Rebuttal:
        """Short-circuit synthesis emitted when the debate hits
        max_rounds without reaching threshold. The returned Rebuttal
        marks the draft as short-circuited and surfaces the unresolved
        points + open questions so operators see what the panel
        couldn't agree on.

        Uses the real LLM prompt when ``_call_llm`` is wired; falls
        through to the deterministic short-circuit when not.
        """

        from dark_factory.api.refinery.roles.judge.prompt import (
            format_reconcile_prompt,
        )

        try:
            prompt = format_reconcile_prompt(
                draft, critiques, prior_rebuttals, scores,
            )
            raw = self._call_llm(prompt)
            parsed = _parse_rebuttal_json(raw, default_mode="reconcile_unresolved")
            if parsed is not None:
                return parsed
        except NotImplementedError:
            pass
        except Exception as exc:
            log.warning("judge_reconcile_llm_failed", error=str(exc))

        unresolved_blockers = [
            c for c in critiques if c.severity == Severity.BLOCKER
        ]
        revised = draft.model_copy(update={
            "produced_by": self.role_name,
            "iteration": draft.iteration + 1,
            "convergence_status": ConvergenceStatus.SHORT_CIRCUITED,
            "unresolved_points": [c.finding for c in unresolved_blockers],
            "open_questions": [c.proposed_fix for c in unresolved_blockers],
        })
        entries = [
            RebuttalEntry(
                critique_ref=f"c-{i}",
                action="deferred",
                rationale="Panel did not converge within max_rounds; blocker preserved",
            )
            for i, _ in enumerate(unresolved_blockers)
        ]
        return Rebuttal(
            revised_draft=revised, entries=entries, mode="reconcile_unresolved",
        )

    def review_set(
        self,
        refined_set: list[Draft],
        traces: list,
        run_context: dict,
    ) -> CrossReviewReport:
        """Cross-req review — the LLM-driven path runs through
        ``format_review_set_prompt`` + ``_call_llm`` + JSON parse.

        ``traces`` carries critique dicts from the prior round of a
        multi-pass set review (see ``set_review.py``). When non-empty
        and dict-shaped, they're rendered into a "Critics challenged
        your prior draft" block so the Judge revises against them
        instead of re-running blind. When no LLM is wired, delegates
        to the scorer's ``review_set``.
        """

        from dark_factory.api.refinery.roles.judge.prompt import (
            format_review_set_prompt,
        )

        prior_critiques = [
            t for t in (traces or [])
            if isinstance(t, dict) and "finding" in t
        ]

        try:
            prompt = format_review_set_prompt(
                refined_set, run_context,
                prior_critiques=prior_critiques or None,
            )
            raw = self._call_llm(prompt)
            parsed = _parse_cross_review_json(raw)
            if parsed is not None:
                return parsed
        except NotImplementedError:
            pass
        except Exception as exc:
            log.warning("judge_review_set_llm_failed", error=str(exc))

        # Fall through to the scorer's review_set stub.
        scorer = self.scorer
        if hasattr(scorer, "_semantic"):
            return scorer._semantic.review_set(refined_set, traces, run_context)
        return scorer.review_set(refined_set, traces, run_context)


# ─────────────────────────────────────────────────────────────────────
# JSON parsers for LLM-returned Rebuttal / CrossReviewReport
# ─────────────────────────────────────────────────────────────────────


from dark_factory.api.refinery.roles._shared.json_utils import parse_pydantic_json


def _parse_rebuttal_json(
    raw: str, *, default_mode: str = "synthesize",
) -> Rebuttal | None:
    """Parse a raw LLM response into a validated Rebuttal.

    Defaults ``mode`` to ``default_mode`` when absent. Returns ``None``
    on malformed payloads so the caller falls through to its
    deterministic branch.
    """

    return parse_pydantic_json(raw, Rebuttal, defaults={"mode": default_mode})


def _parse_cross_review_json(raw: str) -> CrossReviewReport | None:
    """Parse a raw LLM response into a validated CrossReviewReport."""

    return parse_pydantic_json(raw, CrossReviewReport)


# ─────────────────────────────────────────────────────────────────────
# Default scorer assembly
# ─────────────────────────────────────────────────────────────────────


def _build_default_scorer():
    """Build the combined-pipeline scorer from the current
    PipelineConfig. Tests that want a deterministic scorer assign
    ``judge._scorer = FallbackJudge()`` directly and skip this path."""

    from dark_factory.api.refinery.judge.composition import CombinedJudgePipeline
    from dark_factory.api.refinery.judge.rules_judge import RulesJudge
    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
    from dark_factory.config import load_settings

    try:
        cfg = load_settings().pipeline
    except Exception:  # pragma: no cover — defensive during bootstrap
        from dark_factory.config import PipelineConfig
        cfg = PipelineConfig()

    rules_judge = RulesJudge(
        disabled=list(cfg.refinery_rules_disabled),
        dimension_overrides=dict(cfg.refinery_rules_dimension_overrides),
    )

    # Prefer DeepEvalJudge when available; fall through to FallbackJudge
    # on import failure (optional dep path).
    try:
        from dark_factory.api.refinery.judge.deepeval_judge import DeepEvalJudge
        semantic: Judge = DeepEvalJudge(
            per_dim_thresholds=dict(cfg.refinery_judge_thresholds),
        )
    except Exception:  # pragma: no cover — optional dep
        semantic = FallbackJudge(
            thresholds=dict(cfg.refinery_judge_thresholds),
            overall_threshold=cfg.refinery_judge_overall_threshold,
        )

    fallback = FallbackJudge(
        thresholds=dict(cfg.refinery_judge_thresholds),
        overall_threshold=cfg.refinery_judge_overall_threshold,
    )

    return CombinedJudgePipeline(
        rules_judge=rules_judge,
        semantic_judge=semantic,
        fallback_judge=fallback,
        short_circuit_enabled=cfg.refinery_rules_short_circuit_llm,
        short_circuit_threshold=cfg.refinery_rules_short_circuit_threshold,
        blocker_cap=cfg.refinery_rules_penalty_blocker_cap,
        warning_delta=cfg.refinery_rules_penalty_warning_delta,
        overall_threshold=cfg.refinery_judge_overall_threshold,
        per_dim_thresholds=dict(cfg.refinery_judge_thresholds),
        aggregation=cfg.refinery_judge_aggregation,
        rules_enabled=cfg.refinery_rules_enabled,
    )
