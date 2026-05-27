"""Shared prompt fragments for the four adversarial critics.

Each critic's prompt = adversarial preamble + role-specific focus +
draft block + (optional) retry follow-up. Keeping the preamble and the
draft-block formatter here ensures the adversarial contract is stated
identically for every critic.
"""

from __future__ import annotations

from dark_factory.api.refinery.contracts import Draft, RoleContext

# ── Adversarial preamble (IDENTICAL text for every critic) ───────────

CRITIC_SEVERITY_BANDS = """\
**Severity bands (apply consistently):**

- ``blocker``: the draft, if shipped as-is, would cause a concrete
  domain-specific failure (security breach, runtime outage, irreversible
  data loss, ≥ 10x cost blowout, fundamentally infeasible scope).
- ``warning``: a real but non-catastrophic flaw — a missing edge case,
  a soft cost overrun, an under-specified failure mode, a smell that
  will compound across rounds if left unaddressed.
- ``info``: no genuine concern in your domain, OR a domain-relevant
  observation the panel should record but that doesn't merit revision.
  Option (b) above always lands as ``info``.

Escalate carefully: a flood of ``blocker`` severities drowns out genuine
blockers. When in doubt, pick the lower band and explain in ``finding``.
"""


CRITIC_ADVERSARIAL_PREAMBLE = """\
You are the {role_name} critic on the Requirements Refinery's
adversarial panel. Your primary dimension is **{primary_dimension}**.

**Your job is to challenge this draft — not to validate it.**

You MUST either:

(a) Identify a concrete flaw — an ambiguity, gap, contradiction, risk,
    cost, or limit — from your domain's perspective, AND return it as
    a Critique JSON object with:

      - ``author_role``: "{role_name}"
      - ``dimension``:   one of "clarity" | "testability" | "feasibility"
                         | "completeness" | "risk_coverage" | "reversibility"
                         (default to "{primary_dimension}" when in doubt)
      - ``severity``:    "info" | "warning" | "blocker"
      - ``finding``:     ≥ 20 chars, concrete description of the flaw
        — MUST name one specific draft sentence, spec, or relationship
        you stress-tested before raising the critique
      - ``proposed_fix``: ≥ 10 chars, a specific actionable remedy
      - ``cited_evidence``: optional list of quotes or references
      - ``confidence``:  0.0 - 1.0

OR (b) Return severity="info" with a finding ≥ 15 chars that explicitly
    states *what you searched for* and *why you found no genuine concern*
    in your domain. The finding MUST quote at least one specific draft
    sentence you read closely. Empty or generic "looks good" style
    findings are rejected at the schema validator.

Emit at most 3 critiques per round. If you have more findings, fold the
weaker ones into your highest-severity critique's ``cited_evidence``.

""" + CRITIC_SEVERITY_BANDS + """

Your output MUST be a single JSON object (not an array) matching the
Critique schema above. No prose, no markdown fences, just the JSON.

"""


def render_draft_block(draft: Draft) -> str:
    """Format the draft for inclusion in every critic prompt."""

    specs_block = ""
    if draft.suggested_specs:
        specs_block = "\n".join(
            f"- **{s.title}** ({s.capability}): {s.description}\n"
            + "\n".join(f"    - {c}" for c in s.acceptance_criteria)
            for s in draft.suggested_specs
        )

    rel_block = ""
    if draft.relationships:
        rel_block = "\n".join(
            f"- {r.type} → {r.target_id}: {r.rationale}"
            for r in draft.relationships
        )

    return f"""\

# Draft under review

- **ID**: {draft.requirement_id}
- **Title**: {draft.title}
- **Priority**: {draft.priority}
- **Tags**: {', '.join(draft.tags) if draft.tags else '(none)'}
- **Iteration**: {draft.iteration}

## Description

{draft.description}

## Suggested specs

{specs_block or '(no specs drafted)'}

## Relationships

{rel_block or '(no relationships declared)'}

"""


def render_followup(followup: str) -> str:
    """Append a retry follow-up message when the first attempt was
    rejected as a rubber-stamp or failed validation. Empty on first
    attempt."""

    if not followup:
        return ""
    return f"""\

---

**Important follow-up (the retry loop added this):**

{followup}

Return ONLY a single JSON object matching the Critique schema above.
"""
