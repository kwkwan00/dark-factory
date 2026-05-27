"""Operations critic — evaluates reliability, observability, failure modes."""

from __future__ import annotations

from typing import ClassVar

from dark_factory.api.refinery.contracts import CritiqueDimension, Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_role import _CriticBaseRole
from dark_factory.api.refinery.roles.operations.prompt import (
    format_operations_critique_prompt,
)
from dark_factory.log import trace_methods


@trace_methods
class OperationsRole(_CriticBaseRole):
    """Challenges runtime behaviour: observability, failure modes,
    rollback paths, capacity, SLO impact."""

    role_name: ClassVar[str] = "operations"
    default_model: ClassVar[str] = "claude-sonnet-4-6"
    default_reasoning_effort: ClassVar[str] = "medium"
    default_dimension: ClassVar[CritiqueDimension] = CritiqueDimension.COMPLETENESS

    def _build_prompt(
        self,
        draft: Draft,
        context: RoleContext,
        followup: str,
    ) -> str:
        return format_operations_critique_prompt(draft, context, followup)
