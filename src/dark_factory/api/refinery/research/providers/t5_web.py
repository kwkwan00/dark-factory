"""T5 — Public Web & Search.

Lowest trust tier. Results can only propagate downstream when
corroborated by a non-T5 tier (Editor enforces via
``guardrail_t5_propagation_rejected``). Raw T5 content never reaches
the Judge's prompt.
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


WebSearch = Callable[[str, int], list[dict[str, Any]]]
WebFetch = Callable[[str], str]


class WebSearchProvider(ResearchProvider):
    tier: ClassVar[SourceTier] = SourceTier.T5_WEB
    provider_name: ClassVar[str] = "web-search"

    def __init__(
        self,
        *,
        search_fn: WebSearch | None = None,
        fetch_fn: WebFetch | None = None,
    ) -> None:
        self._search_fn = search_fn
        self._fetch_fn = fetch_fn
        self._calls = 0
        self._budget = 0
        self._search_urls: set[str] = set()

    def reset_budget(self, budget: int) -> None:
        self._budget = budget
        self._calls = 0
        self._search_urls.clear()

    def search(self, query: str, budget: int) -> list[Source]:
        if self._calls >= self._budget:
            raise TierBudgetExhausted(
                f"T5 budget {self._budget} exhausted after {self._calls} calls"
            )
        self._calls += 1
        if not self._search_fn:
            return []
        try:
            hits = self._search_fn(query, budget)
        except Exception as exc:
            log.warning("t5_web_search_failed", error=str(exc))
            return []
        sources: list[Source] = []
        for h in hits[:budget]:
            url = h.get("url", "")
            if url:
                self._search_urls.add(url)
            sources.append(Source(
                id=f"web:{h.get('id', url)}",
                tier=self.tier,
                url=url,
                title=h.get("title", "")[:200],
                chunk=h.get("snippet", "")[:1200],
                provider=self.provider_name,
            ))
        return sources

    def fetch(self, url: str) -> str:
        """Agents cannot fetch arbitrary URLs — the URL must have come
        from a prior ``search()`` call on this provider instance. This
        stops a model from constructing a URL out of nothing."""

        if url not in self._search_urls:
            raise ValueError(
                f"T5 fetch denied — URL {url!r} was not in any prior "
                "search() result"
            )
        if not self._fetch_fn:
            return ""
        try:
            return self._fetch_fn(url)
        except Exception as exc:
            log.warning("t5_web_fetch_failed", url=url, error=str(exc))
            return ""
