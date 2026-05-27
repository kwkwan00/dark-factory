"""Memory-kind vocabulary helpers.

Any prompt that enumerates memory kinds renders from this helper so the
taxonomy stays in sync with ``MemoryKind`` automatically. Prompts that
hardcode "the five kinds" or "pattern, strategy, or past episode" go
stale every time we extend the enum.
"""

from __future__ import annotations

from dark_factory.api.refinery.contracts import MemoryKind


def render_memory_kinds() -> str:
    """Pipe-delimited list of every ``MemoryKind`` value, e.g.
    ``"decision | incident | pattern | constraint | conflict | hypothesis | anti_pattern"``."""

    return " | ".join(k.value for k in MemoryKind)
