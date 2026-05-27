"""Operations critic prompt — adversarial by contract."""

from __future__ import annotations

from dark_factory.api.refinery.contracts import Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_prompts import (
    CRITIC_ADVERSARIAL_PREAMBLE,
    render_draft_block,
    render_followup,
)

_OPERATIONS_FOCUS = """\
**Your domain (operations):**

- Rollback + migrations: irreversible operations (DROP, DELETE,
  destructive schema changes, data migrations without a reverse path,
  public-API contracts without versioning) are BLOCKER. Tag these
  critiques with ``dimension="reversibility"``.
- Observability: what metrics / logs / traces does this require? If
  the requirement doesn't specify them, the operator can't tell when
  it's broken. Missing observability is at least WARNING. Tag with
  ``dimension="completeness"``.
- Failure modes: what can go wrong at runtime? Timeouts, partial
  writes, exhausted budgets, upstream outages. Are they handled?
- Capacity: what's the expected load? What happens at 2x, 10x?
- SLO impact: does this requirement affect an existing SLO? Is the
  impact accounted for?
- Incident / runbook memory: if an outage of this shape has been
  recorded before, surface the reference.
"""


def format_operations_critique_prompt(
    draft: Draft,
    context: RoleContext,
    followup: str,
) -> str:
    return (
        CRITIC_ADVERSARIAL_PREAMBLE.format(
            role_name="operations",
            primary_dimension="reversibility",
        )
        + _OPERATIONS_FOCUS
        + render_draft_block(draft)
        + render_followup(followup)
    )
