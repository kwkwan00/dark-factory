"""ResearchProvider ABC — one provider per tier.

Guardrail enforcement (see plan's "Research agent (layered sourcing)"):

1. Every provider declares its tier at class level; the Editor uses the
   tier to trust-weight claims (and reject T5-only clusters).
2. ``fetch(url)`` must either accept only URLs previously returned by
   this provider's ``search()`` (tracked per-session) or match a tier-
   specific allowlist (T3 vendor docs). Agents cannot construct
   arbitrary URLs from model output.
3. Budget exhaustion raises ``TierBudgetExhausted`` — the caller
   continues with what it has.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from dark_factory.api.refinery.contracts import Source, SourceTier


class TierBudgetExhausted(Exception):
    """Raised when a research provider has exceeded its per-debate budget."""


class ResearchProvider(ABC):
    """One tier's source surface. Concrete implementations (one per tier
    under ``research/providers/``) stay small — search + fetch + budget."""

    tier: ClassVar[SourceTier]
    provider_name: ClassVar[str] = ""

    @abstractmethod
    def search(self, query: str, budget: int) -> list[Source]:
        """Return up to *budget* sources for *query*, each with
        ``tier=self.tier`` stamped. Raises ``TierBudgetExhausted`` if
        the debate's budget for this tier has been exceeded."""

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Fetch + return the content chunk for *url*. Implementations
        MUST enforce that *url* either appeared in an earlier ``search``
        result from this provider or matches a tier-specific allowlist."""
