"""QueryRewriter — role-specific expansion of the raw requirement query.

Phase 6 ships the ABC + an ``IdentityRewriter`` (pass-through) that
covers tests and the default operational path. An LLM-backed rewriter
can be added without touching any caller since the signature is fixed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class QueryRewriter(ABC):
    """Transform a raw requirement description into a role-specific
    retrieval query before hitting Qdrant / Neo4j."""

    @abstractmethod
    def rewrite(
        self,
        requirement_text: str,
        role: str,
        round_number: int = 0,
    ) -> str:
        """Return the rewritten query for *role* at the given round."""


class IdentityRewriter(QueryRewriter):
    """No-op rewriter — returns the input unchanged. The only rewriter
    currently wired; an LLM-backed rewriter was planned but never
    landed, so this is the de-facto production behaviour."""

    def rewrite(
        self,
        requirement_text: str,
        role: str,
        round_number: int = 0,
    ) -> str:
        return requirement_text
