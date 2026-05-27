"""Tier providers for the layered-sourcing Research agent.

Each provider implements ``ResearchProvider`` at one tier. Concrete
providers stay small — roughly 40-80 lines each — because the guardrail
machinery (tier tracking, URL allowlist, budget accounting) lives in
the base class and the agent.

Tier order + trust weights per plan's "Research agent (layered sourcing)"
section:

- T0 Structured: Neo4j + Qdrant memory — trust 1.00
- T1 Internal:   PRDs / ADRs / postmortems — trust 0.95
- T2 Observability: Prometheus + Postgres — trust 0.90
- T3 Official:   Vendor docs (URL-allowlisted) — trust 0.80
- T4 Academic:   arXiv / whitepapers — trust 0.60
- T5 Web:        General search — trust 0.30
"""

from dark_factory.api.refinery.research.providers.t0_structured import (
    InternalKnowledgeProvider,
)
from dark_factory.api.refinery.research.providers.t1_internal import (
    InternalDocsProvider,
)
from dark_factory.api.refinery.research.providers.t2_observability import (
    ObservabilityProvider,
)
from dark_factory.api.refinery.research.providers.t3_official import (
    OfficialDocsProvider,
)
from dark_factory.api.refinery.research.providers.t4_academic import (
    AcademicProvider,
)
from dark_factory.api.refinery.research.providers.t5_web import (
    WebSearchProvider,
)

__all__ = [
    "InternalKnowledgeProvider",
    "InternalDocsProvider",
    "ObservabilityProvider",
    "OfficialDocsProvider",
    "AcademicProvider",
    "WebSearchProvider",
]
