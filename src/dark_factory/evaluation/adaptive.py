"""Adaptive threshold computation based on eval score trends."""

from __future__ import annotations


def compute_adaptive_threshold(
    *,
    base_threshold: float,
    recent_scores: list[float],
    trend_raise_step: float = 0.05,
    trend_lower_step: float = 0.05,
    threshold_min: float = 0.3,
    threshold_max: float = 0.9,
) -> float:
    """Compute an adaptive eval threshold based on recent score trends.

    - If fewer than 3 scores, return ``base_threshold``.
    - If scores are trending up (second half mean > first half + 0.05), raise.
    - If scores are trending down or flat for 3+ evals, lower.
    - Clamped to ``[threshold_min, threshold_max]``.
    """
    if len(recent_scores) < 3:
        return base_threshold

    mid = len(recent_scores) // 2
    first_half = recent_scores[:mid]
    second_half = recent_scores[mid:]

    first_mean = sum(first_half) / len(first_half)
    second_mean = sum(second_half) / len(second_half)

    if second_mean > first_mean + 0.05:
        # Trending up — raise the bar
        threshold = base_threshold + trend_raise_step
    elif second_mean < first_mean - 0.05:
        # Trending down — lower to unblock
        threshold = base_threshold - trend_lower_step
    else:
        threshold = base_threshold

    return max(threshold_min, min(threshold, threshold_max))
