"""Prompt loader — reads all agent prompts from ``prompts.yaml``.

Usage::

    from dark_factory.prompts import get_prompt

    system = get_prompt("swarm_planner", "system")
    user   = get_prompt("codegen", "user").format(spec_id=..., ...)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")


@functools.lru_cache(maxsize=1)
def _load_prompts() -> dict[str, dict[str, str]]:
    """Load and cache the prompts YAML (read once, cached forever)."""
    with open(_PROMPTS_FILE, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def get_prompt(section: str, key: str) -> str:
    """Return a prompt string from ``prompts.yaml``.

    Parameters
    ----------
    section:
        Top-level key, e.g. ``"swarm_planner"``, ``"codegen"``.
    key:
        Sub-key within the section, e.g. ``"system"``, ``"user"``.

    Returns
    -------
    str
        The prompt text. Trailing whitespace is stripped.

    Raises
    ------
    KeyError
        If the section or key does not exist.
    """
    prompts = _load_prompts()
    if section not in prompts:
        raise KeyError(f"Prompt section {section!r} not found in prompts.yaml")
    entry = prompts[section]
    if key not in entry:
        raise KeyError(
            f"Key {key!r} not found in prompt section {section!r}. "
            f"Available keys: {sorted(entry)}"
        )
    return entry[key].rstrip()


def reload_prompts() -> None:
    """Clear the cache so the next ``get_prompt`` re-reads the file.

    Useful in tests or after hot-editing the YAML at runtime.
    """
    _load_prompts.cache_clear()
