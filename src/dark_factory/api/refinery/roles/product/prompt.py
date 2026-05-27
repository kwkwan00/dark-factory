"""Product role prompt — single-shot generator.

Consumed by ``ProductRole.propose``. The LLM receives the full
context in one turn (target requirement + sibling set + run evidence)
and must reply with a JSON object matching the
``RefinedRequirement`` schema plus an optional ``suggested_memories``
list. No tools, no iteration — same shared context every other panel
seat sees.
"""

from __future__ import annotations

import json

from dark_factory.api.refinery.roles._shared.formatters import (
    fmt_episodes,
    fmt_evals,
    fmt_gaps,
    fmt_learnings,
    fmt_requirements,
)
from dark_factory.api.refinery.roles._shared.memory_vocab import render_memory_kinds


def format_product_prompt(
    req: dict,
    all_requirements: list[dict],
    run_context: dict | None,
) -> str:
    """Build the Product role's single-shot prompt.

    Returns inline JSON (no tool-use side-effects) so the panel
    orchestrator can validate the draft synchronously.
    """

    has_run = run_context is not None
    req_id = req.get("id", "unknown")

    other_reqs = [r for r in all_requirements if r.get("id") != req_id]
    other_text = fmt_requirements(other_reqs) if other_reqs else "No other requirements."

    evidence_block = ""
    if has_run:
        trace = run_context.get("traceability")
        req_trace = None
        if trace and isinstance(trace, dict):
            for row in trace.get("rows", []):
                if row.get("requirement", {}).get("id") == req_id:
                    req_trace = row
                    break
        if req_trace:
            status = req_trace.get("overall_status", "?")
            specs = req_trace.get("specs", [])
            spec_lines = [
                f"  - {s.get('id', '?')} ({s.get('capability', '?')}): "
                f"passed={s.get('all_passed', '?')}, scores={s.get('eval_scores', {})}"
                for s in specs
            ]
            trace_text = (
                f"Status: {status}\nSpecs:\n" + "\n".join(spec_lines)
                if spec_lines else f"Status: {status}"
            )
        else:
            trace_text = "No traceability data for this requirement."

        run_stats = json.dumps(run_context.get("run") or {}, default=str, indent=2)[:500]
        evidence_block = f"""
## Run Evidence (from run {run_context.get("run_id", "?")})

### This Requirement's Traceability
{trace_text}

### Gap Analysis
{fmt_gaps(run_context.get("gaps"))}

### Relevant Episodes
{fmt_episodes(run_context.get("episodes", []))}

### Procedural Memories
{fmt_learnings(run_context.get("learnings", []))}

### Eval Critiques
{fmt_evals(run_context.get("evals", []))}

### Run Stats
{run_stats}
"""

    if has_run:
        analysis_instruction = (
            "Analyze this requirement against the run evidence:\n"
            "- What specs implemented it? Did they pass or fail?\n"
            "- What do the eval critiques say about why specs failed?\n"
            "- Are there episodic memories showing agent confusion?"
        )
    else:
        analysis_instruction = (
            "Analyze this requirement purely on its structure:\n"
            "- Is the language ambiguous?\n"
            "- Are constraints, edge cases, and error conditions covered?\n"
            "- Is this one feature or several bundled together?"
        )

    return f"""You are the Product agent on the Requirements Refinery panel.
Your job is to draft a refined version of ONE requirement that the
adversarial critics (Engineering, Security, Operations, Cost) will
challenge. You see the full shared context in this prompt; do not
ask for more — produce your best first draft now.

# Target Requirement

- [{req.get("id", "?")}] {req.get("title", "?")}
  (priority={req.get("priority", "?")}, tags={req.get("tags", [])})
  Description: {req.get("description", "?")}

# Other Requirements (for relationship discovery)

{other_text}
{evidence_block}

# Instructions

1. **Assess** this requirement for clarity, completeness, testability,
   scope, AND reversibility (how hard would this be to undo if it ships
   and turns out wrong — destructive migrations, public API contracts,
   regulatory commitments raise the reversibility cost).

2. **Identify relationships** with other requirements (depends_on,
   conflicts_with, extends, replaces, related_to, supersedes). Use
   ``supersedes`` (not ``replaces``) when this requirement evolves
   from a prior one and the lineage is worth keeping; reserve
   ``replaces`` for semantic duplicates the cross-review patcher
   marked. Decompose into 2-3 smaller requirements only if the bundle
   is genuinely separate features.

3. **Rewrite** the requirement:
   - Improve title for clarity and specificity
   - Expand description with constraints, edge cases, acceptance context
   - Adjust priority if evidence suggests it
   - Add tags for capability area, complexity tier, dependency depth

4. **Draft suggested_specs** (at most 4): how this should decompose
   into specs (title, capability group, 3-5 concrete acceptance
   criteria each). Acceptance criteria should follow WHEN/THEN or
   GIVEN/WHEN/THEN shape, or carry a numeric measure with units.

5. **Identify memories** worth recording. ``kind`` MUST be one of
   ``{render_memory_kinds()}``. Pick the most specific kind:
   - ``pattern`` — recurrent positive guidance (e.g. "always emit
     audit logs on permission grants").
   - ``anti_pattern`` — recurrent negative guidance (e.g. "do NOT
     store session tokens in localStorage"); pair with the
     recommended alternative in ``anti_pattern_alternative``.
   - ``hypothesis`` — an open question the panel surfaced that a
     future Research call should verify; include
     ``hypothesis_verification_query``.
   - ``decision`` / ``constraint`` / ``conflict`` / ``incident`` —
     when the rationale clearly matches that taxonomy.
   Memory descriptions must be GENERAL and REUSABLE — never reference
   a specific spec ID or requirement ID. Emit at most 5 memories.

{analysis_instruction}

# Response format

Return a SINGLE JSON object (no Markdown fences, no commentary). The
object must conform to this schema:

{{
  "id": {json.dumps(req_id)},
  "original_title": {json.dumps(req.get("title", ""))},
  "original_description": {json.dumps(req.get("description", ""))},
  "title": "refined title",
  "description": "refined description with more detail",
  "priority": "low|medium|high|critical",
  "tags": ["tag1", "tag2"],
  "relationships": [
    {{"target_id": "other-req-id", "type": "depends_on", "rationale": "..."}}
  ],
  "suggested_specs": [
    {{
      "title": "spec title",
      "capability": "kebab-case-group",
      "description": "what this spec covers",
      "acceptance_criteria": ["criterion 1", "criterion 2"]
    }}
  ],
  "changes": ["What you changed and why"],
  "pass_context": "Your reasoning for the changes",
  "suggested_memories": [
    {{
      "type": "pattern",
      "kind": "pattern",
      "description": "General reusable insight",
      "context": "",
      "applicability": "When this applies",
      "rationale": "Why this should be remembered"
    }},
    {{
      "type": "pattern",
      "kind": "anti_pattern",
      "description": "Approach to avoid and why it harms",
      "anti_pattern_alternative": "Use X instead",
      "anti_pattern_harm": "What breaks if we pick the rejected approach",
      "applicability": "Where this anti-pattern shows up",
      "rationale": "Why this should be remembered"
    }},
    {{
      "type": "pattern",
      "kind": "hypothesis",
      "description": "Open question the panel couldn't settle this round",
      "hypothesis_verification_query": "Concrete query the Research agent can run",
      "hypothesis_status": "open",
      "applicability": "When this hypothesis matters",
      "rationale": "What decision unblocks once we verify"
    }}
  ]
}}

The ``suggested_memories`` field is optional — omit or use an empty
array when there's nothing memory-worthy. ``type`` is the legacy
shape field (set to "pattern" for forward-compat); ``kind`` is the
authoritative taxonomy field and MUST be one of
``{render_memory_kinds()}``.
"""
