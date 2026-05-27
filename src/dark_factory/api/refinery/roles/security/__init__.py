"""Security critic — identifies abuse cases, vulnerabilities, compliance."""

from __future__ import annotations

from typing import ClassVar

from dark_factory.api.refinery.contracts import CritiqueDimension, Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_role import _CriticBaseRole
from dark_factory.api.refinery.roles.security.prompt import (
    format_security_critique_prompt,
)
from dark_factory.log import trace_methods


@trace_methods
class SecurityRole(_CriticBaseRole):
    """Identifies authN/authZ gaps, data-exposure surfaces, injection
    vectors, supply-chain risks, compliance misses."""

    role_name: ClassVar[str] = "security"
    default_model: ClassVar[str] = "claude-opus-4-6"
    default_reasoning_effort: ClassVar[str] = "xhigh"
    default_dimension: ClassVar[CritiqueDimension] = CritiqueDimension.RISK_COVERAGE

    def _build_prompt(
        self,
        draft: Draft,
        context: RoleContext,
        followup: str,
    ) -> str:
        return format_security_critique_prompt(draft, context, followup)
