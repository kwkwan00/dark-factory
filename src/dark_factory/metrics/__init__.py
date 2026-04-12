"""PostgreSQL-backed metrics and telemetry store.

This module collects structured metrics around pipeline runs, the agent
swarm, LLM calls, and DeepEval evaluations. Metrics are written from hot
paths via :class:`MetricsRecorder`, a non-blocking background writer that
drops events if Postgres is unavailable — the pipeline itself must never
be slowed or broken by metrics failures.

Public entry points:

- :class:`PostgresClient` — connection pool wrapper
- :class:`MetricsRepository` — typed read/write API
- :class:`MetricsRecorder` — background writer bound to the progress broker
- :func:`ensure_schema` — idempotent DDL installer

All writes go through the recorder; direct repository use is reserved for
REST endpoints (read-side) and tests.
"""

from dark_factory.metrics.client import PostgresClient
from dark_factory.metrics.recorder import MetricsRecorder
from dark_factory.metrics.repository import MetricsRepository
from dark_factory.metrics.schema import ensure_schema

__all__ = [
    "MetricsRecorder",
    "MetricsRepository",
    "PostgresClient",
    "ensure_schema",
]
