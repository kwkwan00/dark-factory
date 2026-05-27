"""T4 — Academic / Research.

arXiv + conference proceedings. Used sparingly — the plan's tier-budget
defaults cap T4 at 2 sources per debate because academic sources are
the slowest to validate and rarely justify the cost.
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


AcademicSearch = Callable[[str, int], list[dict[str, Any]]]


class AcademicProvider(ResearchProvider):
    tier: ClassVar[SourceTier] = SourceTier.T4_ACADEMIC
    provider_name: ClassVar[str] = "arxiv"

    def __init__(self, *, search_fn: AcademicSearch | None = None) -> None:
        self._search_fn = search_fn
        self._calls = 0
        self._budget = 0

    def reset_budget(self, budget: int) -> None:
        self._budget = budget
        self._calls = 0

    def search(self, query: str, budget: int) -> list[Source]:
        if self._calls >= self._budget:
            raise TierBudgetExhausted(
                f"T4 budget {self._budget} exhausted after {self._calls} calls"
            )
        self._calls += 1
        if not self._search_fn:
            return []
        try:
            hits = self._search_fn(query, budget)
        except Exception as exc:
            log.warning("t4_academic_search_failed", error=str(exc))
            return []
        return [
            Source(
                id=f"arxiv:{h.get('id', '')}",
                tier=self.tier,
                url=h.get("url"),
                title=h.get("title", "")[:200],
                chunk=h.get("abstract", "")[:2000],
                provider=self.provider_name,
            )
            for h in hits[:budget]
        ]

    def fetch(self, url: str) -> str:
        # arXiv abstracts come back with chunk already populated.
        raise NotImplementedError(
            "T4 sources return abstracts in-band; fetch() is not supported"
        )
