"""Judge role's synthesis + cross-review prompt templates.

Two distinct prompts:

- ``format_defend_prompt`` — fed to ``Judge.defend`` to revise a draft
  by folding critiques in. The Judge must preserve disagreement in the
  revised draft (explicit tradeoffs), not smooth it.
- ``format_reconcile_prompt`` — fed to ``Judge.reconcile_unresolved``
  on short-circuit. Its job is to DOCUMENT what couldn't be resolved
  rather than force a synthesis that doesn't exist.
- ``format_review_set_prompt`` — fed to ``Judge.review_set`` for the
  Phase-3 cross-requirement adjudication that replaces the legacy
  reconciliation agent.

These prompts are consumed by the JudgeRole's injectable ``_call_llm``
hook. When no LLM is wired, ``JudgeRole.defend`` falls through to a
deterministic pass-through; see ``JudgeRole`` for the fallback logic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from dark_factory.api.refinery.role_weighting import combined_weight_for
from dark_factory.api.refinery.roles._shared.rule_findings import (
    render_rule_findings_block,
)

if TYPE_CHECKING:
    from dark_factory.api.refinery.contracts import (
        Critique,
        Draft,
        EvaluationScore,
        Rebuttal,
    )


_DEFEND_PREAMBLE = """\
You are the Judge on the Requirements Refinery's adversarial panel.
Your job is to SYNTHESIZE a revised requirement draft by folding in
the critics' findings — without smoothing over disagreement.

**The adversarial contract:** every accepted critique must cite a
concrete value-judgment (why this critique's fix won over the status
quo). Every rejected critique must cite a value-judgment (why the
draft is correct as-is). If two critiques propose mutually exclusive
fixes, the revised draft must name the tradeoff chosen in
``explicit_tradeoffs``.

**Phase A rule findings** (when present in the prompt) are
deterministic, mechanically-verified flaws — you cannot overrule
them. If your revised draft does not address every BLOCKER rule
finding, the next round will re-fire the same finding and refuse to
converge. Treat them as critiques with author_role="rules" already
accepted.

**Output format:** a single JSON object matching the Rebuttal schema:

```json
{
  "revised_draft": {
    "requirement_id": "...",
    "title": "...",
    "description": "...",
    "priority": "low|medium|high|critical",
    "tags": [...],
    "suggested_specs": [ {"title", "capability", "description", "acceptance_criteria"} ],
    "relationships": [ {"target_id", "type", "rationale"} ],
    "produced_by": "judge",
    "iteration": N,
    "convergence_status": "converged",
    "explicit_tradeoffs": ["we chose X over Y because ..."],
    "unresolved_points": [],
    "open_questions": []
  },
  "entries": [
    {"critique_ref": "c-0", "action": "accepted", "rationale": "..."},
    {"critique_ref": "c-1", "action": "rejected", "rationale": "..."},
    {"critique_ref": "c-2", "action": "partial",  "rationale": "accepted the spec change but deferred the migration plan to a follow-up requirement"},
    {"critique_ref": "c-3", "action": "deferred", "rationale": "needs cost data the panel doesn't have this round"}
  ],
  "mode": "synthesize"
}
```

When the debate does not fully converge, populate ``unresolved_points``
(blockers that outlasted the round) and ``open_questions`` (decisions
the panel can't make without more information) instead of leaving them
empty. Use ``deferred`` / ``partial`` actions whenever the synthesis
side-steps a critique rather than committing to it.

Rationale strings must be ≥ 25 chars and name the value-judgment,
not just restate the critique.
"""


_RECONCILE_PREAMBLE = """\
You are the Judge on the Requirements Refinery's adversarial panel.
The panel did not converge within ``max_rounds``. DO NOT force a
synthesis that doesn't exist — your job here is to DOCUMENT what
couldn't be resolved so operators see the panel's honest output.

**Your revised draft must:**
- set ``convergence_status`` to ``"short_circuited"``
- populate ``unresolved_points`` with the concrete blockers that
  outlasted every round (reference which critic raised each)
- populate ``open_questions`` with the decisions the panel couldn't
  make (framed as questions the operator must answer)
- leave ``explicit_tradeoffs`` empty (we're not choosing here —
  we're flagging)
- carry the best-available draft body so the operator has something
  to act on

**Output format:** a single JSON object matching the Rebuttal schema
with ``mode: "reconcile_unresolved"``.
"""


_REVIEW_SET_PREAMBLE = """\
You are the Judge on the Requirements Refinery's adversarial panel,
running the Phase-3 cross-requirement review across a batch of
refined requirements. Your job: identify cross-requirement issues
that no single per-requirement debate could see.

**Checks to run:**

1. **Duplicate detection** — semantically identical requirements
   produced by different per-req debates. Report in
   ``duplicate_pairs`` with ``{keep_id, remove_id, rationale}``.
2. **Coherence** — contradictory constraints across requirements,
   inconsistent terminology, priority mismatches relative to
   dependency chains. Report in ``coherence_issues``.
3. **Relationship fixes** — missing ``depends_on`` edges, invalid
   references, circular dependencies, priority inversions, and
   lineage gaps where a newer requirement evolved from an older one
   without a ``supersedes`` edge. Report each as
   ``{source_id, target_id, action, type, rationale}`` with action
   in ``{"add", "remove"}`` and ``type`` in
   ``{"depends_on", "conflicts_with", "extends", "replaces",
   "related_to", "supersedes"}``. Prefer ``supersedes`` over
   ``replaces`` whenever you want to preserve lineage; ``replaces``
   is reserved for the duplicate-marker path.
4. **Priority changes** — escalate when a requirement is on the
   dependency path to multiple high-priority items. Report in
   ``priority_changes``.
5. **Spec overlaps** — suggested specs that duplicate each other
   across requirements. Report in ``spec_overlaps``.
6. **Set-level dimension scores** — rate the SET (not individual
   requirements) on clarity / testability / feasibility /
   completeness / risk_coverage / reversibility. Report in
   ``set_scores`` and ``set_reasons``.
7. **Unresolved debates** — requirements where the per-req debate
   short-circuited without convergence. Report in
   ``unresolved_debates`` with ``{requirement_id, rounds_run, reason}``.
8. **Risk areas** — summary string list of the highest-leverage
   risks across the set.

**Output format:** a single JSON object matching the
CrossReviewReport schema. No prose outside the JSON.
"""


def format_defend_prompt(
    draft: "Draft",
    critiques: list["Critique"],
    iteration: int,
    *,
    role_weights: dict[str, float] | None = None,
    role_dim_weights: dict[str, float] | None = None,
    weights_block_override: str | None = None,
    rule_findings: list | None = None,
) -> str:
    """Build the Judge.defend synthesis prompt.

    ``role_weights`` and ``role_dim_weights`` are the flat-shape
    calibration signals the orchestrator stamped on the evidence bag.
    ``weights_block_override`` accepts a pre-rendered calibration block
    (typically built once at run start by the orchestrator and reused
    across rounds — the role×dim Cartesian doesn't change between
    rounds of the same debate). When supplied, the per-round prompt
    skips the render and inlines the string directly.

    ``rule_findings`` is Phase A's deterministic-rule output stamped on
    the evidence bag at scoring time. Surfacing it here ensures the
    Judge's synthesis addresses every BLOCKER finding instead of
    revisiting the same violation on the next round.
    """

    crit_block = "\n\n".join(
        f"**Critique c-{i}** (role={c.author_role}, severity={c.severity.value}, "
        f"dimension={c.dimension.value}):\n"
        f"  Finding: {c.finding}\n"
        f"  Proposed fix: {c.proposed_fix}"
        for i, c in enumerate(critiques)
    ) or "(no critiques this round)"

    if weights_block_override is not None:
        weights_block = weights_block_override
    else:
        weights_block = format_calibration_block(
            role_weights=role_weights, role_dim_weights=role_dim_weights,
        )

    rule_findings_block = ""
    if rule_findings:
        rule_findings_block = (
            "\n\n# Phase A rule findings (deterministic; treat as accepted critiques)\n\n"
            + render_rule_findings_block(rule_findings)
        )

    return (
        _DEFEND_PREAMBLE
        + "\n\n# Draft under review (iteration "
        + str(iteration) + ")\n\n"
        + json.dumps(draft.model_dump(), indent=2, default=str)
        + "\n\n# Critiques\n\n"
        + crit_block
        + rule_findings_block
        + weights_block
        + "\n\nReturn ONLY the Rebuttal JSON object.\n"
    )


def format_calibration_block(
    *,
    role_weights: dict[str, float] | None,
    role_dim_weights: dict[str, float] | None,
) -> str:
    """Render the calibration block. Both signals → 2D table with
    combined multipliers; only flat → legacy 1D bullet list; neither
    → empty string.

    Pure function over flat-shape inputs so the orchestrator can call
    it once at run start and stamp the result on the evidence bag.
    """

    if not role_weights and not role_dim_weights:
        return ""

    if role_dim_weights:
        return _format_2d_weights_block(
            role_weights=role_weights or {},
            role_dim_weights=role_dim_weights,
        )

    non_default = {
        role: m for role, m in (role_weights or {}).items()
        if abs(m - 1.0) > 1e-6
    }
    if not non_default:
        return ""
    lines = "\n".join(
        f"- {role}: ×{m:.2f} ("
        f"{'amplify' if m > 1.0 else 'attenuate'})"
        for role, m in sorted(non_default.items())
    )
    return (
        "\n\n# Role calibration (historical acceptance signal)\n\n"
        "Apply these multipliers when deciding which BLOCKERS to "
        "accept versus reject. Roles marked **amplify** have a "
        "strong track record — give their blockers more weight. "
        "Roles marked **attenuate** historically raise blockers "
        "the panel rejects — be skeptical, but never ignore. "
        "Multipliers are bounded; use them as a tie-breaker, not "
        "as license to silence a role.\n\n"
        + lines
    )


def _format_2d_weights_block(
    *,
    role_weights: dict[str, float],
    role_dim_weights: dict[str, float],
) -> str:
    """Render the 2D (role × dimension) calibration table. Cells whose
    combined multiplier is within 1% of identity are dropped."""

    roles: set[str] = set(role_weights.keys())
    dims: set[str] = set()
    for key in role_dim_weights.keys():
        if ":" in key:
            r, d = key.split(":", 1)
            roles.add(r)
            dims.add(d)

    if not roles or not dims:
        return ""

    rows: list[tuple[str, str, float]] = []
    for role in sorted(roles):
        for dim in sorted(dims):
            combined = combined_weight_for(
                role, dim,
                role_weights=role_weights or None,
                role_dim_weights=role_dim_weights or None,
            )
            if abs(combined - 1.0) > 0.01:
                rows.append((role, dim, combined))

    if not rows:
        return ""

    lines = "\n".join(
        f"- {role} on **{dim}**: ×{m:.2f} "
        f"({'amplify' if m > 1.0 else 'attenuate'})"
        for role, dim, m in rows
    )
    return (
        "\n\n# Role × dimension calibration (combined historical signal)\n\n"
        "Each line is the COMBINED multiplier (per-role × per-dimension, "
        "product-clamped) you should apply when deciding which BLOCKERS "
        "to accept. A role's credibility on its native dimension is "
        "weighed separately from its credibility outside its wheelhouse. "
        "**Amplify** = strong historical track record — give those "
        "blockers more sway. **Attenuate** = historically rejected — be "
        "skeptical, but never ignore. Cells not listed are identity "
        "(×1.0) — no signal yet or balanced track record.\n\n"
        + lines
    )


def format_reconcile_prompt(
    draft: "Draft",
    critiques: list["Critique"],
    prior_rebuttals: list["Rebuttal"],
    scores: list["EvaluationScore"],
) -> str:
    """Build the short-circuit reconcile prompt."""

    all_crits = "\n".join(
        f"- [{c.severity.value}/{c.dimension.value}] {c.author_role}: {c.finding}"
        for c in critiques
    ) or "(no critiques recorded)"
    scores_block = "\n".join(
        f"Round {i}: overall={s.overall:.2f} passed={s.passed} "
        f"disagreement={s.disagreement_score:.2f}"
        for i, s in enumerate(scores)
    ) or "(no scores)"

    return (
        _RECONCILE_PREAMBLE
        + "\n\n# Last draft\n\n"
        + json.dumps(draft.model_dump(), indent=2, default=str)
        + "\n\n# All critiques across rounds\n\n"
        + all_crits
        + "\n\n# Score history\n\n"
        + scores_block
        + "\n\nReturn ONLY the Rebuttal JSON (mode=reconcile_unresolved).\n"
    )


def format_review_set_prompt(
    refined_set: list["Draft"],
    run_context: dict,
    *,
    prior_critiques: list[dict] | None = None,
) -> str:
    """Build the cross-requirement review prompt.

    ``prior_critiques`` is an optional list of critique dicts
    (``{role, severity, dimension, finding, proposed_fix}``) from the
    previous critic round of a multi-pass set review. When supplied,
    they are rendered in their own block so the Judge revises the
    report against the panel's challenges instead of re-running blind.
    """

    set_block = json.dumps(
        [r.model_dump() for r in refined_set], indent=2, default=str,
    )
    run_summary = json.dumps(
        {
            "run_id": run_context.get("run_id"),
            "source_mode": run_context.get("source_mode"),
            "requirements_count": len(refined_set),
        },
        indent=2,
        default=str,
    )

    critiques_block = ""
    if prior_critiques:
        rendered = "\n\n".join(
            f"**{c.get('role','?')}** "
            f"(severity={c.get('severity','?')}, "
            f"dimension={c.get('dimension','?')}):\n"
            f"  Finding: {c.get('finding','')}\n"
            f"  Proposed fix: {c.get('proposed_fix','')}"
            for c in prior_critiques
        )
        critiques_block = (
            "\n\n# Critics challenged your prior draft — address each "
            "before returning the revised report\n\n"
            + rendered
        )

    return (
        _REVIEW_SET_PREAMBLE
        + "\n\n# Refined requirement set\n\n"
        + set_block
        + "\n\n# Run summary\n\n"
        + run_summary
        + critiques_block
        + "\n\nReturn ONLY the CrossReviewReport JSON object.\n"
    )
