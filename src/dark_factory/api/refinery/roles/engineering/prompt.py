"""Engineering critic prompt — adversarial by contract.

The adversarial preamble is identical across the four critic roles; the
role-specific middle block declares Engineering's domain concerns.
"""

from __future__ import annotations

from dark_factory.api.refinery.contracts import Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_prompts import (
    CRITIC_ADVERSARIAL_PREAMBLE,
    render_draft_block,
    render_followup,
)
from dark_factory.api.refinery.roles._shared.stack import STACK_DESCRIPTION

_ENGINEERING_FOCUS = f"""\
**Your domain (engineering):**

- Feasibility: can this be implemented with the declared stack
  ({STACK_DESCRIPTION})? Flag unstated migrations or dependencies.
- Decomposition fit: is the requirement one cohesive capability or
  several bundled together? Over-bundling should be a BLOCKER.
- API shape + coupling: does the requirement imply an API surface
  that conflicts with sibling requirements or existing spec contracts?
- Integration points: what does this requirement assume about other
  systems' outputs or inputs? Are those assumptions justified?
- Scalability: does the requirement imply any load / throughput /
  concurrency bounds? Are they explicit?
"""


def format_engineering_critique_prompt(
    draft: Draft,
    context: RoleContext,
    followup: str,
) -> str:
    return (
        CRITIC_ADVERSARIAL_PREAMBLE.format(
            role_name="engineering",
            primary_dimension="feasibility",
        )
        + _ENGINEERING_FOCUS
        + render_draft_block(draft)
        + render_followup(followup)
    )
