"""T2 — Observability (Prometheus + Postgres metrics).

Reads time-series + forensic data to validate assumptions against
reality ("what's the actual P95 latency of this endpoint?"). Phase 8
ships the abstract wiring; concrete backends inject via constructor
so tests don't require a live Prometheus.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import structlog

from dark_factory.api.refinery.contracts import Source, SourceTier
from dark_factory.api.refinery.research.base import (
    ResearchProvider,
    TierBudgetExhausted,
)

log = structlog.get_logger()


MetricsProbe = Callable[[str], list[dict[str, Any]]]
"""``(query) → list[{metric, value, unit, context}]``"""


class ObservabilityProvider(ResearchProvider):
    tier: ClassVar[SourceTier] = SourceTier.T2_OBSERVABILITY
    provider_name: ClassVar[str] = "observability"

    def __init__(self, *, metrics_probe: MetricsProbe | None = None) -> None:
        self._probe = metrics_probe
        self._calls = 0
        self._budget = 0

    def reset_budget(self, budget: int) -> None:
        self._budget = budget
        self._calls = 0

    def search(self, query: str, budget: int) -> list[Source]:
        if self._calls >= self._budget:
            raise TierBudgetExhausted(
                f"T2 budget {self._budget} exhausted after {self._calls} calls"
            )
        self._calls += 1
        if not self._probe:
            return []
        try:
            rows = self._probe(query)
        except Exception as exc:
            log.warning("t2_observability_probe_failed", error=str(exc))
            return []
        return [
            Source(
                id=f"metric:{r.get('metric', '')}",
                tier=self.tier,
                url=None,
                title=r.get("metric", "")[:200],
                chunk=(
                    f"{r.get('metric', '')} = {r.get('value', '')} "
                    f"{r.get('unit', '')} ({r.get('context', '')})"
                )[:2000],
                provider=self.provider_name,
            )
            for r in rows[:budget]
        ]

    def fetch(self, url: str) -> str:
        raise NotImplementedError(
            "T2 sources carry content in-band; fetch() is not supported"
        )
