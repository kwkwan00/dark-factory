"""Cost critic prompt — adversarial by contract."""

from __future__ import annotations

from dark_factory.api.refinery.contracts import Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_prompts import (
    CRITIC_ADVERSARIAL_PREAMBLE,
    render_draft_block,
    render_followup,
)

_COST_FOCUS = """\
**Your domain (cost):**

- LLM-token cost: if this requires model calls, estimate input /
  output tokens per invocation and frequency. Flag if a
  low-priority feature implies > $100/day in LLM spend.
- Infra-spend delta: compute / storage / egress / licence costs?
  Do they scale with usage or are they one-time?
- Runtime-resource cost: memory, CPU, GPU, disk. Flag combinations
  that force a tier upgrade (db instance, VM size).
- Cost of failure: what's the blast radius if the feature runs hot
  (infinite loop, token explosion, storage burst)?
- Priority alignment: does the projected cost match the requirement's
  declared priority? High cost on a low-priority feature = WARNING.

Emit INFO critiques with concrete numbers when cost is in line; use
WARNING when spend doubles for a low-priority feature; BLOCKER only
for truly uneconomic designs (≥ 10x reasonable baseline).
"""


def format_cost_critique_prompt(
    draft: Draft,
    context: RoleContext,
    followup: str,
) -> str:
    return (
        CRITIC_ADVERSARIAL_PREAMBLE.format(
            role_name="cost",
            primary_dimension="feasibility",
        )
        + _COST_FOCUS
        + render_draft_block(draft)
        + render_followup(followup)
    )
