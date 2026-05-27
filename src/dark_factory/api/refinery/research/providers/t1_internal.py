"""T1 — Internal Docs (PRDs / ADRs / postmortems / runbooks).

Wraps the existing ingest path's document corpus. Phase 8 ships a
minimal stub that returns empty results by default; operators plug in
their doc-source repos (Confluence / Notion / local markdown) by
injecting a ``doc_loader`` callable at construction time.
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


DocLoader = Callable[[str, int], list[dict[str, Any]]]
"""``(query, limit) → list[{id, title, chunk, url, source_tag}]``"""


class InternalDocsProvider(ResearchProvider):
    """T1 — internal-docs provider. Results carry a ``source_tag`` so
    the Analyst / Editor can surface which internal source a finding
    came from (ADR-042 vs postmortem-2025-03)."""

    tier: ClassVar[SourceTier] = SourceTier.T1_INTERNAL
    provider_name: ClassVar[str] = "internal-docs"

    def __init__(self, *, doc_loader: DocLoader | None = None) -> None:
        self._loader = doc_loader
        self._calls = 0
        self._budget = 0

    def reset_budget(self, budget: int) -> None:
        self._budget = budget
        self._calls = 0

    def search(self, query: str, budget: int) -> list[Source]:
        if self._calls >= self._budget:
            raise TierBudgetExhausted(
                f"T1 budget {self._budget} exhausted after {self._calls} calls"
            )
        self._calls += 1
        if not self._loader:
            return []
        try:
            rows = self._loader(query, budget)
        except Exception as exc:
            log.warning("t1_internal_docs_failed", error=str(exc))
            return []
        return [
            Source(
                id=f"internal:{r.get('id', '')}",
                tier=self.tier,
                url=r.get("url"),
                title=r.get("title", "")[:200],
                chunk=r.get("chunk", "")[:2000],
                provider=self.provider_name,
            )
            for r in rows
        ]

    def fetch(self, url: str) -> str:
        # Internal docs are returned with chunk already attached.
        raise NotImplementedError(
            "T1 sources carry content in-band; fetch() is not supported"
        )
