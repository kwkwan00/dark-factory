"""Engineering critic — challenges feasibility, scalability, architecture."""

from __future__ import annotations

from typing import ClassVar

from dark_factory.api.refinery.contracts import CritiqueDimension, Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_role import _CriticBaseRole
from dark_factory.api.refinery.roles.engineering.prompt import (
    format_engineering_critique_prompt,
)
from dark_factory.log import trace_methods


@trace_methods
class EngineeringRole(_CriticBaseRole):
    """Challenges feasibility, decomposition fit, API shape, coupling,
    integration with the existing stack."""

    role_name: ClassVar[str] = "engineering"
    default_model: ClassVar[str] = "claude-sonnet-4-6"
    default_reasoning_effort: ClassVar[str] = "high"
    default_dimension: ClassVar[CritiqueDimension] = CritiqueDimension.FEASIBILITY

    def _build_prompt(
        self,
        draft: Draft,
        context: RoleContext,
        followup: str,
    ) -> str:
        return format_engineering_critique_prompt(draft, context, followup)
