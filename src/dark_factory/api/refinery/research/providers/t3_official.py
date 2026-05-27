"""T3 — Official / Vendor Documentation.

URL-allowlisted fetch: search returns candidate hits, fetch() accepts
only URLs that either appeared in a prior ``search()`` result or match
the operator's official-domain allowlist.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar
from urllib.parse import urlparse

import structlog

from dark_factory.api.refinery.contracts import Source, SourceTier
from dark_factory.api.refinery.research.base import (
    ResearchProvider,
    TierBudgetExhausted,
)

log = structlog.get_logger()


OfficialSearch = Callable[[str, list[str], int], list[dict[str, Any]]]
"""``(query, domain_allowlist, limit) → list[{id, url, title, snippet}]``"""

OfficialFetch = Callable[[str], str]
"""``(url) → text chunk``"""


class OfficialDocsProvider(ResearchProvider):
    tier: ClassVar[SourceTier] = SourceTier.T3_OFFICIAL
    provider_name: ClassVar[str] = "official-docs"

    def __init__(
        self,
        *,
        domain_allowlist: list[str] | None = None,
        search_fn: OfficialSearch | None = None,
        fetch_fn: OfficialFetch | None = None,
    ) -> None:
        self._allowlist = list(domain_allowlist or [])
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
                f"T3 budget {self._budget} exhausted after {self._calls} calls"
            )
        self._calls += 1
        if not self._search_fn:
            return []
        try:
            hits = self._search_fn(query, list(self._allowlist), budget)
        except Exception as exc:
            log.warning("t3_official_docs_search_failed", error=str(exc))
            return []
        sources: list[Source] = []
        for h in hits[:budget]:
            url = h.get("url", "")
            if not self._url_allowed(url):
                continue
            self._search_urls.add(url)
            sources.append(Source(
                id=f"official:{h.get('id', url)}",
                tier=self.tier,
                url=url,
                title=h.get("title", "")[:200],
                chunk=h.get("snippet", "")[:1200],
                provider=self.provider_name,
            ))
        return sources

    def fetch(self, url: str) -> str:
        """Fetch only URLs from our allowlist OR ones previously
        returned by our search(). Agents cannot construct arbitrary
        URLs from model output and fetch them."""

        if url not in self._search_urls and not self._url_allowed(url):
            raise ValueError(
                f"T3 fetch denied — URL {url!r} is neither from a prior "
                "search() result nor in the domain allowlist"
            )
        if not self._fetch_fn:
            return ""
        try:
            return self._fetch_fn(url)
        except Exception as exc:
            log.warning("t3_official_docs_fetch_failed", url=url, error=str(exc))
            return ""

    def _url_allowed(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        # Belt-and-brace: reject anything that isn't plain http(s).
        # ``javascript:``, ``data:``, ``file:`` etc. would otherwise
        # bypass the allowlist when their userinfo or hostname happens
        # to match an entry. The downstream ``fetch_fn`` typically
        # rejects these schemes too, but we don't rely on that here.
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        return any(host == d or host.endswith("." + d) for d in self._allowlist)
