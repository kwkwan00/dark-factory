"""Layered-sourcing Research agent — 6 tiers, 4-role pipeline, 3 guardrails."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dark_factory.api.refinery.research.agent import ResearchAgent, ResearchNote
from dark_factory.api.refinery.research.base import (
    ResearchProvider,
    TierBudgetExhausted,
)
from dark_factory.api.refinery.research.providers import (
    AcademicProvider,
    InternalDocsProvider,
    InternalKnowledgeProvider,
    ObservabilityProvider,
    OfficialDocsProvider,
    WebSearchProvider,
)

if TYPE_CHECKING:
    from dark_factory.config import PipelineConfig


def build_research_agent_from_config(
    config: "PipelineConfig",
    *,
    trust_weights: dict[int, float] | None = None,
) -> ResearchAgent:
    """Construct a ``ResearchAgent`` with the standard 6-tier provider
    set wired from ``PipelineConfig``.

    ``trust_weights`` overrides the per-tier trust map — pass the
    learning-adjusted weights from
    ``research.learning.load_adjusted_trust_weights`` when available.
    When omitted, the config's base weights are used (degraded mode —
    no provider-trust learning).

    Both the orchestrator (``stream.py``) and the debate graph's
    fallback factory (``debate/graph.py``) call this so the provider
    set + bootstrapping stay in one place.
    """

    from dark_factory.api.refinery.contracts import SourceTier

    providers = {
        SourceTier.T0_STRUCTURED: InternalKnowledgeProvider(),
        SourceTier.T1_INTERNAL: InternalDocsProvider(),
        SourceTier.T2_OBSERVABILITY: ObservabilityProvider(),
        SourceTier.T3_OFFICIAL: OfficialDocsProvider(
            domain_allowlist=list(config.refinery_research_official_url_allowlist),
        ),
        SourceTier.T4_ACADEMIC: AcademicProvider(),
        SourceTier.T5_WEB: WebSearchProvider(),
    }
    return ResearchAgent(
        providers=providers,
        tier_budgets=dict(config.refinery_research_tier_budgets),
        enabled_tiers=list(config.refinery_research_enabled_tiers),
        trust_weights=(
            dict(trust_weights)
            if trust_weights is not None
            else dict(config.refinery_research_tier_trust_weights)
        ),
        internal_sufficient_threshold=config.refinery_research_internal_sufficient_threshold,
        editor_min_confidence=config.refinery_research_editor_min_confidence,
        max_sources_per_call=config.refinery_research_max_sources_per_call,
    )


__all__ = [
    "AcademicProvider",
    "InternalDocsProvider",
    "InternalKnowledgeProvider",
    "ObservabilityProvider",
    "OfficialDocsProvider",
    "ResearchAgent",
    "ResearchNote",
    "ResearchProvider",
    "TierBudgetExhausted",
    "WebSearchProvider",
    "build_research_agent_from_config",
]
