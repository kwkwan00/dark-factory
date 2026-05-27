"""Cost critic — assesses economic feasibility and resource efficiency."""

from __future__ import annotations

from typing import ClassVar

from dark_factory.api.refinery.contracts import CritiqueDimension, Draft, RoleContext
from dark_factory.api.refinery.roles._shared.critic_role import _CriticBaseRole
from dark_factory.api.refinery.roles.cost.prompt import (
    format_cost_critique_prompt,
)
from dark_factory.log import trace_methods


@trace_methods
class CostRole(_CriticBaseRole):
    """Estimates LLM-token cost, infra-spend deltas, runtime-resource
    cost of the draft's suggested_specs. Returns INFO critiques with
    numbers; WARNING when spend doubles for a low-priority feature."""

    role_name: ClassVar[str] = "cost"
    default_model: ClassVar[str] = "gpt-5.4"
    default_reasoning_effort: ClassVar[str] = "medium"
    default_dimension: ClassVar[CritiqueDimension] = CritiqueDimension.FEASIBILITY

    def _build_prompt(
        self,
        draft: Draft,
        context: RoleContext,
        followup: str,
    ) -> str:
        return format_cost_critique_prompt(draft, context, followup)
