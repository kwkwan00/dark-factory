"""RoleAgent ABC — the Parnas secret-hiding boundary.

Every concrete role keeps its prompt, model choice, retrieval strategy,
and decoding parameters private behind the six verbs below. The
orchestrator:

- MAY see ``role_name``, the ABC methods, and typed return values
- MAY NOT import a concrete role class directly
- MAY NOT read a role's prompt text
- MAY NOT know which model a role uses at a class level

Each verb's default raises ``NotImplementedError`` so that a misdirected
orchestrator call (e.g. asking an Engineering critic to synthesize) fails
loudly instead of silently producing nonsense.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from dark_factory.api.refinery.contracts import (
    CrossReviewReport,
    Critique,
    Draft,
    EvaluationScore,
    RawRequirement,
    Rebuttal,
    RoleContext,
)


class RoleAgent(ABC):
    """Stable contract every refinery role implements.

    Class metadata exposed to the orchestrator (safe to read):

    - ``role_name`` — unique registry key (e.g. ``"security"``)
    - ``default_model`` — model id used when no override is configured
    - ``default_reasoning_effort`` — decoding effort when no override

    Runtime configuration is applied via ``configure()`` — the
    orchestrator / registry override model + reasoning effort per-
    instance without touching the role's private state.

    Hidden implementation (orchestrator MUST NOT reach for these):

    - prompt text — owned by ``roles/<name>/prompt.py``
    - retrieval slice — each role's critique/propose decides what to pull
    - tool allowlist — roles configure their own
    - output post-processing — private to the role module
    """

    role_name: ClassVar[str] = ""
    default_model: ClassVar[str] = ""
    default_reasoning_effort: ClassVar[str] = "high"

    def __init__(self) -> None:
        # Runtime overrides default to None so ``model`` / ``reasoning_effort``
        # properties (on concrete roles) fall through to class-level defaults.
        self._model: str | None = None
        self._reasoning_effort: str | None = None

    def configure(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> "RoleAgent":
        """Apply per-instance overrides to this role.

        Used by ``RoleRegistry`` (for operator config overrides) and
        by the debate graph (to switch to the strong model on
        escalation). Returns self for chaining.
        """

        if model is not None:
            self._model = model
        if reasoning_effort is not None:
            self._reasoning_effort = reasoning_effort
        return self

    # ── Generator verb (only Product overrides by default) ────────

    def propose(
        self,
        requirement: RawRequirement,
        context: RoleContext,
    ) -> Draft:
        """Produce an initial Draft from the raw requirement."""

        raise NotImplementedError(
            f"role '{self.role_name}' is not a generator; cannot propose()"
        )

    # ── Adversarial critic verb (Eng / Sec / Ops / Cost override) ──

    def critique(
        self,
        draft: Draft,
        context: RoleContext,
    ) -> Critique:
        """Review *draft* from this role's perspective and return findings."""

        raise NotImplementedError(
            f"role '{self.role_name}' is not a critic; cannot critique()"
        )

    # ── Synthesis verb (only Judge overrides) ─────────────────────

    def defend(
        self,
        draft: Draft,
        critiques: list[Critique],
        context: RoleContext,
    ) -> Rebuttal:
        """Fold critiques into a revised draft with explicit tradeoff declarations.

        The synthesis output preserves disagreement — it does not smooth it.
        Every accepted/rejected critique must be cited in ``Rebuttal.entries``
        with a value-judgment rationale (rule_no_rubber_stamp_rebuttal
        enforces this downstream).
        """

        raise NotImplementedError(
            f"role '{self.role_name}' is not a synthesizer; cannot defend()"
        )

    # ── Short-circuit synthesis verb (only Judge overrides) ───────

    def reconcile_unresolved(
        self,
        draft: Draft,
        critiques: list[Critique],
        prior_rebuttals: list[Rebuttal],
        scores: list[EvaluationScore],
        context: RoleContext,
    ) -> Rebuttal:
        """Short-circuit synthesis invoked when max_rounds is hit without
        convergence. Its distinct prompt documents what couldn't be
        resolved rather than forcing a synthesis that does not exist.

        Returns a Rebuttal whose ``revised_draft`` has
        ``convergence_status=SHORT_CIRCUITED`` and populated
        ``unresolved_points`` / ``open_questions`` fields.
        """

        raise NotImplementedError(
            f"role '{self.role_name}' does not reconcile; not a judge"
        )

    # ── Evaluation verb (only Judge overrides) ────────────────────

    def score(
        self,
        draft: Draft,
        context: RoleContext,
        trace: "DebateTrace | None" = None,  # type: ignore[name-defined]
    ) -> EvaluationScore:
        """Gate the debate — the combined-pipeline evaluation.

        Kept distinct from ``defend`` so the scoring call does not share
        a reasoning trace with the synthesis call (reduces confirmation
        bias on the Judge's own output).
        """

        raise NotImplementedError(
            f"role '{self.role_name}' does not score; not a judge"
        )

    # ── Calibration-block render verb (only Judge overrides) ─────

    def prepare_calibration(
        self,
        *,
        role_weights: dict[str, float] | None = None,
        role_dim_weights: dict[str, float] | None = None,
    ) -> str:
        """Pre-render the per-run calibration block the role will inline
        into its prompts. The orchestrator calls this once at run start
        and stamps the result on the evidence bag so per-round prompts
        skip re-rendering an invariant table.

        Returning ``""`` means "no signal worth rendering" — the role's
        prompt formatter must accept that as a no-op.
        """

        raise NotImplementedError(
            f"role '{self.role_name}' does not prepare calibration; not a judge"
        )

    # ── Cross-requirement review verb (only Judge overrides) ─────

    def review_set(
        self,
        refined_set: list[Draft],
        traces: list["DebateTrace"],  # type: ignore[name-defined]
        run_context: dict,
    ) -> CrossReviewReport:
        """Phase-3 cross-requirement adjudication. Reads the full
        refined set + every per-requirement ``DebateTrace`` and emits
        a ``CrossReviewReport`` whose mutations are applied by
        ``apply_cross_review_report``."""

        raise NotImplementedError(
            f"role '{self.role_name}' does not review sets; not a judge"
        )
