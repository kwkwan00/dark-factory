"""Refinery observability — DebateTrace, TraceContext, ObservabilityHub.

Trace records flow to three stores: Postgres (forensic, joinable),
Prometheus (time-series aggregates), and trace JSON under
``refinery/{result_id}/trace.json`` (archival full-fidelity). See the
plan's "Data-store layering" section for what lands where.
"""

from dark_factory.api.refinery.observability.context import TraceContext
from dark_factory.api.refinery.observability.hub import ObservabilityHub
from dark_factory.api.refinery.observability.trace import (
    CallRecord,
    CriticCall,
    DebateTrace,
    DecisionRecord,
    MemoryAuditEntry,
    RoundRecord,
)

__all__ = [
    "CallRecord",
    "CriticCall",
    "DebateTrace",
    "DecisionRecord",
    "MemoryAuditEntry",
    "ObservabilityHub",
    "RoundRecord",
    "TraceContext",
]
