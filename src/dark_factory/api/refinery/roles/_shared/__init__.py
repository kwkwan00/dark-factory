"""Shared utilities for role modules (evidence formatters, prompt helpers).

These utilities are not role-secret — they're pure formatters over the
run-context evidence bag. Individual role prompts import only the
formatters they need.
"""

from dark_factory.api.refinery.roles._shared.formatters import (
    fmt_episodes,
    fmt_evals,
    fmt_gaps,
    fmt_learnings,
    fmt_requirements,
    fmt_traceability,
)

__all__ = [
    "fmt_episodes",
    "fmt_evals",
    "fmt_gaps",
    "fmt_learnings",
    "fmt_requirements",
    "fmt_traceability",
]
