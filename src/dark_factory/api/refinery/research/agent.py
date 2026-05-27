"""ResearchAgent — the 4-role pipeline over the 6 tier providers.

Librarian → Analyst → Editor → Historian.

- **Librarian** queries internal tiers FIRST; short-circuits if T0/T1
  are sufficient. External tiers only fire on a genuine gap.
- **Analyst** (tool-less by design) extracts atomic claims from the
  already-fetched source set. No fresh browsing mid-extraction.
- **Editor** clusters semantically-equivalent claims, trust-weights
  across tiers, and emits ``ValidatedInsight`` records. Guardrail 1
  enforced here: a T5-only cluster can't cross the boundary.
- **Historian** converts ValidatedInsights into ``SuggestedMemoryV2``
  records with provenance (tier mix, confidence, citations), ready
  for the memory write-back path.

Output schema: ``ResearchNote`` wraps everything the debate needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import structlog
from pydantic import BaseModel, Field

from dark_factory.api.refinery.contracts import (
    ExtractedClaim,
    MemoryKind,
    Source,
    SourceTier,
    SuggestedMemoryV2,
    ValidatedInsight,
    ValidationStatus,
)
from dark_factory.api.refinery.research.base import (
    ResearchProvider,
    TierBudgetExhausted,
)
from dark_factory.log import trace_methods
from dark_factory.metrics.prometheus import (
    observe_refinery_research_budget_exhausted,
    observe_refinery_research_internal_sufficient,
    observe_refinery_research_tier,
    observe_refinery_t5_propagation_rejected,
    observe_refinery_validated_insight,
)

log = structlog.get_logger()


# Default trust weight per tier, used by the Editor when confidence is
# computed. Plan's "Source Prioritization Model" table; mutable at
# operator config time via ``PipelineConfig.refinery_research_tier_trust_weights``.
_DEFAULT_TRUST_WEIGHTS: dict[SourceTier, float] = {
    SourceTier.T0_STRUCTURED: 1.00,
    SourceTier.T1_INTERNAL: 0.95,
    SourceTier.T2_OBSERVABILITY: 0.90,
    SourceTier.T3_OFFICIAL: 0.80,
    SourceTier.T4_ACADEMIC: 0.60,
    SourceTier.T5_WEB: 0.30,
}


class ResearchNote(BaseModel):
    """ResearchAgent output — what the debate graph's ``research_node``
    stores in state and what the Judge reads in its next synthesis."""

    query: str
    validated_insights: list[ValidatedInsight] = Field(default_factory=list)
    suggested_memories: list[SuggestedMemoryV2] = Field(default_factory=list)
    tier_mix_summary: dict[int, int] = Field(default_factory=dict)
    raw_sources_not_propagated: int = 0
    tier_budgets_remaining: dict[int, int] = Field(default_factory=dict)
    round_emitted: int = 0


AnalystFn = Callable[[list[Source], str], list[ExtractedClaim]]
"""``(sources, query) → claims`` — signals injection for tests."""

ClusterFn = Callable[[list[ExtractedClaim]], list[list[ExtractedClaim]]]
"""``(claims) → clusters`` — default is naive by ``claim`` string prefix."""


@dataclass
class _TierSlot:
    """Accounting slot for one tier's provider + budget."""

    provider: ResearchProvider
    budget: int
    exhausted: bool = False


@trace_methods
class ResearchAgent:
    """Orchestrates the 4-role pipeline with budget + guardrail tracking."""

    def __init__(
        self,
        *,
        providers: dict[SourceTier, ResearchProvider],
        tier_budgets: dict[int, int],
        enabled_tiers: list[int],
        trust_weights: dict[int, float] | None = None,
        internal_sufficient_threshold: float = 0.75,
        editor_min_confidence: float = 0.55,
        max_sources_per_call: int = 20,
        analyst_fn: AnalystFn | None = None,
        cluster_fn: ClusterFn | None = None,
    ) -> None:
        self._providers = providers
        self._slots: dict[SourceTier, _TierSlot] = {}
        for tier_val in enabled_tiers:
            tier = SourceTier(tier_val)
            if tier in providers:
                budget = tier_budgets.get(tier_val, 4)
                self._slots[tier] = _TierSlot(
                    provider=providers[tier], budget=budget,
                )
                # Reset per-invocation budget on each agent instance.
                if hasattr(providers[tier], "reset_budget"):
                    providers[tier].reset_budget(budget)
        self._trust = (
            {SourceTier(k): v for k, v in (trust_weights or {}).items()}
            if trust_weights else dict(_DEFAULT_TRUST_WEIGHTS)
        )
        self._internal_threshold = internal_sufficient_threshold
        self._editor_min = editor_min_confidence
        self._max_sources = max_sources_per_call
        self._analyst = analyst_fn or _default_analyst
        self._cluster = cluster_fn or _default_cluster

    # ── Public entry point ────────────────────────────────────────────

    def run(
        self,
        *,
        query: str,
        round_number: int = 0,
    ) -> ResearchNote:
        """Execute the 4-role pipeline for *query*."""

        sources = self._librarian(query)
        claims = self._analyst(sources[: self._max_sources], query)
        insights = self._editor(claims, sources)
        memories = self._historian(insights)
        return ResearchNote(
            query=query,
            validated_insights=insights,
            suggested_memories=memories,
            tier_mix_summary=_tier_mix(sources),
            raw_sources_not_propagated=_count_not_propagated(sources, insights),
            tier_budgets_remaining={
                int(t.value): max(0, slot.budget - slot.provider._calls)  # type: ignore[attr-defined]
                for t, slot in self._slots.items()
            },
            round_emitted=round_number,
        )

    # ── Librarian ─────────────────────────────────────────────────────

    def _librarian(self, query: str) -> list[Source]:
        """Internal-first triage. Hit T0/T1 first; skip external tiers
        when the internal signal is sufficient."""

        internal_sources: list[Source] = []
        for tier in (SourceTier.T0_STRUCTURED, SourceTier.T1_INTERNAL):
            internal_sources.extend(self._tier_search(tier, query))

        if self._internal_sufficient(internal_sources):
            observe_refinery_research_internal_sufficient()
            log.info(
                "research_internal_first_short_circuit",
                internal_count=len(internal_sources),
            )
            return internal_sources

        # T2 observability only when the query is measurable-shaped.
        external_sources: list[Source] = []
        if _query_looks_measurable(query):
            external_sources.extend(
                self._tier_search(SourceTier.T2_OBSERVABILITY, query)
            )

        # T3, T4, T5 in descending trust order.
        for tier in (
            SourceTier.T3_OFFICIAL,
            SourceTier.T4_ACADEMIC,
            SourceTier.T5_WEB,
        ):
            external_sources.extend(self._tier_search(tier, query))

        return internal_sources + external_sources

    def _tier_search(
        self, tier: SourceTier, query: str,
    ) -> list[Source]:
        slot = self._slots.get(tier)
        if slot is None or slot.exhausted:
            return []
        observe_refinery_research_tier(tier=int(tier.value))
        try:
            return slot.provider.search(query, slot.budget)
        except TierBudgetExhausted:
            slot.exhausted = True
            observe_refinery_research_budget_exhausted(tier=int(tier.value))
            return []
        except Exception as exc:
            log.warning(
                "research_tier_search_failed",
                tier=int(tier.value), error=str(exc),
            )
            return []

    def _internal_sufficient(self, sources: list[Source]) -> bool:
        """Internal-first triage is 'sufficient' only when we have at
        least two T0/T1 hits — a single internal hit still needs
        external corroboration before the editor will propagate it,
        so the librarian descends through external tiers for the
        cross-reference check."""

        internal = [
            s for s in sources
            if s.tier in (SourceTier.T0_STRUCTURED, SourceTier.T1_INTERNAL)
            and (s.chunk or s.title).strip()
        ]
        return len(internal) >= 2

    # ── Editor ────────────────────────────────────────────────────────

    def _editor(
        self,
        claims: list[ExtractedClaim],
        sources: list[Source],
    ) -> list[ValidatedInsight]:
        """Cluster + trust-weight + emit ValidatedInsights. Guardrail 1:
        a cluster whose ONLY tier is T5 cannot propagate."""

        if not claims:
            return []
        source_by_id = {s.id: s for s in sources}
        clusters = self._cluster(claims)
        insights: list[ValidatedInsight] = []

        for cluster in clusters:
            tier_mix = {
                source_by_id[c.source_id].tier
                for c in cluster
                if c.source_id in source_by_id
            }
            # Guardrail 1 — T5-only clusters are filtered.
            if tier_mix == {SourceTier.T5_WEB}:
                observe_refinery_t5_propagation_rejected()
                continue

            # Cross-reference requirement: ≥ 2 distinct tiers OR ≥ 1
            # T0/T1 internal hit.
            internal_hit = any(
                t in (SourceTier.T0_STRUCTURED, SourceTier.T1_INTERNAL)
                for t in tier_mix
            )
            if not (len(tier_mix) >= 2 or internal_hit):
                continue

            confidence = self._confidence(tier_mix, cluster)
            if confidence < self._editor_min:
                continue

            kind = _classify_kind(cluster)
            summary = _synthesize_summary(cluster)
            citations = [c.source_id for c in cluster]
            insight = ValidatedInsight(
                kind=kind,
                summary=summary,
                detail=" | ".join(c.claim for c in cluster[:5]),
                confidence=confidence,
                source_tier_mix=sorted(tier_mix, key=lambda t: t.value),
                citations=citations,
                contradicting_claims=_find_contradictions(cluster),
                is_incident=_is_incident_cluster(cluster),
            )
            insights.append(insight)
            observe_refinery_validated_insight(kind=kind)
        return insights

    def _confidence(
        self,
        tier_mix: set[SourceTier],
        cluster: list[ExtractedClaim],
    ) -> float:
        """Trust-weighted confidence.

        The score rewards the PRESENCE of a high-trust tier rather than
        averaging high + low trust together — a T0+T5 cluster should
        outscore a T5-only cluster by a wide margin (which the straight
        mean doesn't deliver when trust weights span 0.3 to 1.0).

        Formula: ``(max_trust + mean_trust) / 2 * agreement``. A pure-
        T5 cluster still caps at 0.3 × agreement; a mixed cluster with
        any T0 internal hit clears the min-confidence bar.
        """

        if not tier_mix:
            return 0.0
        weights = [self._trust.get(t, 0.0) for t in tier_mix]
        mean_trust = sum(weights) / len(weights)
        max_trust = max(weights)
        blended = (max_trust + mean_trust) / 2.0
        # Agreement rises with cluster size (more corroborating claims).
        agreement = min(1.0, 0.6 + 0.1 * len(cluster))
        return min(max_trust, blended * agreement)

    # ── Historian ─────────────────────────────────────────────────────

    def _historian(
        self,
        insights: list[ValidatedInsight],
    ) -> list[SuggestedMemoryV2]:
        """Convert validated insights into SuggestedMemory records with
        provenance. Memory kind comes from the insight's ``kind``:

        - pattern      → MemoryKind.PATTERN
        - risk         → MemoryKind.INCIDENT (if ``is_incident=True``) else CONSTRAINT
        - constraint   → MemoryKind.CONSTRAINT
        - tradeoff     → MemoryKind.CONFLICT
        - decision     → MemoryKind.DECISION
        - hypothesis   → MemoryKind.HYPOTHESIS
        - anti_pattern → MemoryKind.ANTI_PATTERN
        """

        kind_map: dict[str, MemoryKind] = {
            "pattern": MemoryKind.PATTERN,
            "constraint": MemoryKind.CONSTRAINT,
            "tradeoff": MemoryKind.CONFLICT,
            "decision": MemoryKind.DECISION,
            "hypothesis": MemoryKind.HYPOTHESIS,
            "anti_pattern": MemoryKind.ANTI_PATTERN,
        }
        out: list[SuggestedMemoryV2] = []
        for ins in insights:
            if ins.kind == "risk":
                kind = MemoryKind.INCIDENT if ins.is_incident else MemoryKind.CONSTRAINT
            else:
                kind = kind_map.get(ins.kind, MemoryKind.PATTERN)

            out.append(SuggestedMemoryV2(
                kind=kind,
                summary=ins.summary[:120],
                body=ins.detail or ins.summary,
                source_role="research",
                rationale=f"external research with confidence={ins.confidence:.2f}",
                validation_status=(
                    ValidationStatus.VALIDATED
                    if ins.confidence >= 0.75 else ValidationStatus.UNVALIDATED
                ),
                provenance_source_tier_mix=list(ins.source_tier_mix),
                provenance_confidence=ins.confidence,
                provenance_citations=list(ins.citations),
            ))
        return out


# ─────────────────────────────────────────────────────────────────────
# Default helpers (overridable via constructor hooks)
# ─────────────────────────────────────────────────────────────────────


def _default_analyst(
    sources: list[Source], query: str,
) -> list[ExtractedClaim]:
    """Pure-Python default: one claim per source, pulled from the first
    sentence of the chunk. Real LLM-backed analysts inject via
    ``ResearchAgent(analyst_fn=...)``."""

    claims: list[ExtractedClaim] = []
    for src in sources:
        text = (src.chunk or src.title or "").strip()
        if not text:
            continue
        # First sentence, bounded to 280 chars.
        first = text.split(".")[0].strip()[:280]
        if not first:
            continue
        claims.append(ExtractedClaim(
            source_id=src.id,
            claim=first,
            confidence_from_source=0.7,
        ))
    return claims


def _default_cluster(
    claims: list[ExtractedClaim],
) -> list[list[ExtractedClaim]]:
    """Group claims that share a leading trigram. Deterministic fallback
    for tests — real implementations use embedding clustering."""

    buckets: dict[str, list[ExtractedClaim]] = {}
    for c in claims:
        tokens = c.claim.lower().split()
        key = " ".join(tokens[:3]) if tokens else ""
        buckets.setdefault(key, []).append(c)
    return [v for v in buckets.values() if v]


def _classify_kind(
    cluster: list[ExtractedClaim],
) -> Literal["pattern", "risk", "constraint", "tradeoff", "decision"]:
    """Heuristic classifier driven by claim keywords."""

    text = " ".join(c.claim.lower() for c in cluster)
    if any(kw in text for kw in ("outage", "incident", "vulnerab", "cve")):
        return "risk"
    if any(kw in text for kw in ("rate limit", "cap", "quota", "budget", "gdpr")):
        return "constraint"
    if any(kw in text for kw in ("tradeoff", "versus", "instead of")):
        return "tradeoff"
    if any(kw in text for kw in ("decision", "chose", "adopted")):
        return "decision"
    return "pattern"


def _synthesize_summary(cluster: list[ExtractedClaim]) -> str:
    """Pick the shortest claim as the summary (terse wins in summaries)."""

    return min((c.claim for c in cluster), key=len)


def _find_contradictions(cluster: list[ExtractedClaim]) -> list[str]:
    """Surface claims that declare opposite outcomes for the same topic.
    Minimal heuristic: if both 'not' and the non-negated verb appear in
    the cluster, record the one with 'not'."""

    out: list[str] = []
    for c in cluster:
        text = c.claim.lower()
        if " not " in text or text.startswith("not "):
            out.append(c.claim)
    return out


def _is_incident_cluster(cluster: list[ExtractedClaim]) -> bool:
    text = " ".join(c.claim.lower() for c in cluster)
    return any(kw in text for kw in ("incident", "postmortem", "sev1", "sev2"))


def _tier_mix(sources: list[Source]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for s in sources:
        counts[int(s.tier.value)] = counts.get(int(s.tier.value), 0) + 1
    return counts


def _count_not_propagated(
    sources: list[Source], insights: list[ValidatedInsight],
) -> int:
    cited = set()
    for ins in insights:
        cited.update(ins.citations)
    return sum(1 for s in sources if s.id not in cited)


def _query_looks_measurable(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in (
        "latency", "throughput", "load", "rate", "p95", "p99",
        "error rate", "uptime", "req/s",
    ))
