"""Security critic prompt — adversarial by contract."""

from __future__ import annotations

from dark_factory.api.refinery.contracts import Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_prompts import (
    CRITIC_ADVERSARIAL_PREAMBLE,
    render_draft_block,
    render_followup,
)

_SECURITY_FOCUS = """\
**Your domain (security):**

- Authentication + authorization: does the requirement specify who
  can perform each action and how identity is verified? Missing
  authN / coarse authZ is a BLOCKER.
- Data exposure: what sensitive data does this touch? PII, tokens,
  credentials, audit logs? Are exposure surfaces enumerated?
- Injection + input handling: where does user input flow? Are the
  boundaries validated? Consider SQL, command, prompt, XSS vectors.
- Supply chain: does the requirement add external dependencies or
  runtime inputs (LLM prompts, remote APIs, user-uploaded code)?
- Compliance: GDPR / SOC2 / PCI-DSS / HIPAA implications unstated?
- Incident memory: if the requirement resembles a pattern that has
  already caused a production incident, surface the reference.
"""


def format_security_critique_prompt(
    draft: Draft,
    context: RoleContext,
    followup: str,
) -> str:
    return (
        CRITIC_ADVERSARIAL_PREAMBLE.format(
            role_name="security",
            primary_dimension="risk_coverage",
        )
        + _SECURITY_FOCUS
        + render_draft_block(draft)
        + render_followup(followup)
    )
