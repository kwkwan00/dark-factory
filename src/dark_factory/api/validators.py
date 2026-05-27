"""Shared validation helpers for API route modules."""

from __future__ import annotations

import re


def validate_api_key(v: str | None) -> str | None:
    """Validate and normalise an API key field.

    - ``None`` and empty strings map to ``None`` (no override).
    - Rejects keys longer than 512 chars or containing whitespace.

    Used as a Pydantic ``field_validator`` in request models across
    multiple route modules.
    """
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    if len(v) > 512:
        raise ValueError("API key too long (max 512 chars)")
    if any(c.isspace() for c in v):
        raise ValueError("API key must not contain whitespace")
    return v


# ── Run-ID validation ─────────────────────────────────────────────────────

#: Regex pattern string for a valid run ID.
RUN_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,128}$"

#: Compiled regex for validating run IDs.
RUN_ID_RE = re.compile(RUN_ID_PATTERN)
