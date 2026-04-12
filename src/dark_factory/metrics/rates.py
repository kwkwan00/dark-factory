"""Static model rate table + cost computation.

Rates are expressed as USD per **1,000 tokens**. They are approximate snapshots
from vendor pricing pages at the time of writing — update as needed. When a
model isn't recognised, :func:`compute_cost_usd` returns ``None`` rather than
guessing, so "unknown model → unknown cost" is visible in the metrics.

Keys are matched by longest-prefix to handle versioned model ids like
``claude-sonnet-4-6-20250101``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    """Per-1K-token prices in USD."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


# Rates are intentionally approximate and conservative. If a model isn't listed
# here, cost_usd is recorded as NULL and the UI shows "—".
_RATES: dict[str, Rate] = {
    # ── Anthropic ─────────────────────────────────────────────────────────
    "claude-opus-4": Rate(input=15.0 / 1000, output=75.0 / 1000,
                          cache_read=1.5 / 1000, cache_write=18.75 / 1000),
    "claude-sonnet-4": Rate(input=3.0 / 1000, output=15.0 / 1000,
                            cache_read=0.3 / 1000, cache_write=3.75 / 1000),
    "claude-haiku-4": Rate(input=0.8 / 1000, output=4.0 / 1000,
                           cache_read=0.08 / 1000, cache_write=1.0 / 1000),
    "claude-3-5-sonnet": Rate(input=3.0 / 1000, output=15.0 / 1000,
                              cache_read=0.3 / 1000, cache_write=3.75 / 1000),
    "claude-3-5-haiku": Rate(input=0.8 / 1000, output=4.0 / 1000,
                             cache_read=0.08 / 1000, cache_write=1.0 / 1000),
    "claude-3-opus": Rate(input=15.0 / 1000, output=75.0 / 1000),
    "claude-3-sonnet": Rate(input=3.0 / 1000, output=15.0 / 1000),
    "claude-3-haiku": Rate(input=0.25 / 1000, output=1.25 / 1000),
    # ── OpenAI (for DeepEval judge) ───────────────────────────────────────
    "gpt-5": Rate(input=5.0 / 1000, output=20.0 / 1000),
    "gpt-5.1": Rate(input=5.0 / 1000, output=20.0 / 1000),
    "gpt-5.4": Rate(input=5.0 / 1000, output=20.0 / 1000),
    "gpt-4o": Rate(input=2.5 / 1000, output=10.0 / 1000,
                   cache_read=1.25 / 1000),
    "gpt-4o-mini": Rate(input=0.15 / 1000, output=0.6 / 1000,
                        cache_read=0.075 / 1000),
    "gpt-4.1": Rate(input=2.0 / 1000, output=8.0 / 1000,
                    cache_read=0.5 / 1000),
    "gpt-4-turbo": Rate(input=10.0 / 1000, output=30.0 / 1000),
    "gpt-4": Rate(input=30.0 / 1000, output=60.0 / 1000),
    "gpt-3.5-turbo": Rate(input=0.5 / 1000, output=1.5 / 1000),
    "o1": Rate(input=15.0 / 1000, output=60.0 / 1000),
    "o1-mini": Rate(input=3.0 / 1000, output=12.0 / 1000),
    "o3": Rate(input=10.0 / 1000, output=40.0 / 1000),
    "o3-mini": Rate(input=1.1 / 1000, output=4.4 / 1000),
    # ── Embeddings (used by vector store) ─────────────────────────────────
    "text-embedding-3-large": Rate(input=0.13 / 1000, output=0.0),
    "text-embedding-3-small": Rate(input=0.02 / 1000, output=0.0),
}


def get_rate(model: str | None) -> Rate | None:
    """Look up the per-1K-token rate for a model name.

    Uses longest-prefix match so versioned ids like
    ``claude-sonnet-4-6-20250101`` map to ``claude-sonnet-4``.
    """
    if not model:
        return None
    name = model.lower()
    # Sort keys longest-first for longest-prefix matching.
    for key in sorted(_RATES.keys(), key=len, reverse=True):
        if name.startswith(key):
            return _RATES[key]
    return None


def compute_cost_usd(
    *,
    model: str | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> float | None:
    """Compute approximate USD cost for a single LLM call.

    Returns ``None`` when the model isn't in the rate table so the DB
    records a NULL rather than a zero (which would understate totals).
    """
    rate = get_rate(model)
    if rate is None:
        return None

    total = 0.0
    if input_tokens:
        total += (input_tokens / 1000.0) * rate.input
    if output_tokens:
        total += (output_tokens / 1000.0) * rate.output
    if cache_read_tokens:
        total += (cache_read_tokens / 1000.0) * rate.cache_read
    if cache_creation_tokens:
        total += (cache_creation_tokens / 1000.0) * rate.cache_write
    return round(total, 6)
