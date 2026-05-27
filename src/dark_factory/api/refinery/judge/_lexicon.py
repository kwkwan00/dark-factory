"""Shared lexicons used by both the rules gate and the fallback judge.

Keeping the vague-word catalog + measurable-criterion regexes in one
place prevents drift — adding a modifier to one scorer without the
other would produce inconsistent rule firings vs fallback scoring.
"""

from __future__ import annotations

import re

# Words that signal vagueness when not paired with a concrete measure.
VAGUE_WORDS: tuple[str, ...] = (
    "fast", "secure", "scalable", "robust", "simple", "intuitive",
    "modern", "efficient", "reliable", "flexible", "user-friendly",
)

VAGUE_RE = re.compile(
    r"\b(" + "|".join(VAGUE_WORDS) + r")\b",
    re.IGNORECASE,
)

# Numeric measures with units (ms, seconds, %, req/s, etc.) — the
# presence of one of these near a vague word defuses the rule.
MEASURE_RE = re.compile(
    r"\d+\s*(ms|s|sec|seconds|min|minutes|hours|%|percent|req|rps|"
    r"requests|users|rows|mb|gb|tb|ops|kb)",
    re.IGNORECASE,
)

# WHEN/THEN shapes that indicate a testable acceptance criterion.
WHEN_THEN_RE = re.compile(r"\bwhen\b.+\bthen\b", re.IGNORECASE)
GIVEN_WHEN_RE = re.compile(r"\bgiven\b.+\bwhen\b", re.IGNORECASE)

# Reversibility-dimension vocabulary. Phrases that indicate a draft
# bakes in irreversible-by-default operations vs. phrases that signal
# the author thought through rollback. Shared so a future rules-side
# reversibility rule sees the same signal set as the fallback judge.
IRREVERSIBLE_SIGNALS: tuple[str, ...] = (
    "drop column", "drop table", "destructive migration",
    "delete data", "purge", "public api", "ga release",
    "regulatory commitment", "contractual obligation",
)

ROLLBACK_SIGNALS: tuple[str, ...] = (
    "feature flag", "rollback", "reversible", "additive",
    "versioned api", "behind a flag", "shadow mode",
    "soft delete", "tombstone",
)
