"""Shared JSON extraction for LLM responses.

Models occasionally wrap JSON output in ``json``-fenced code blocks
or surround it with prose. The single helper here extracts the first
JSON object from such responses tolerantly, so every parser
(rebuttal, cross-review, planner, …) shares one extraction surface
instead of each rolling its own near-identical regex.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

_FENCED_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_T = TypeVar("_T", bound=BaseModel)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the first top-level JSON object from ``raw``.

    Tolerant of:
    - ```json ... ``` fenced blocks
    - leading / trailing prose around a bare object
    - extra whitespace

    Returns ``None`` when the payload isn't a JSON object (including
    when it parses as a list, scalar, or fails to parse at all).
    """

    text = (raw or "").strip()
    fenced = _FENCED_PATTERN.search(text)
    if fenced:
        text = fenced.group(1)
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first:last + 1]

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_pydantic_json(
    raw: str,
    model_cls: type[_T],
    *,
    defaults: dict[str, Any] | None = None,
) -> _T | None:
    """Extract a JSON object from ``raw`` and validate it against
    ``model_cls``. Returns ``None`` on extraction failure or schema
    violation — callers fall through to their deterministic branch.

    ``defaults`` are merged via ``setdefault`` so missing optional
    fields (e.g. a Rebuttal's ``mode``) get sensible values without
    forcing the LLM to emit them.
    """

    payload = extract_json_object(raw)
    if payload is None:
        return None
    if defaults:
        for k, v in defaults.items():
            payload.setdefault(k, v)
    try:
        return model_cls.model_validate(payload)
    except Exception:  # pragma: no cover — defensive
        return None


__all__ = ["extract_json_object", "parse_pydantic_json"]
