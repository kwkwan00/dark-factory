"""Adaptive role weighting — bounded multipliers on critique severity.

Mirrors the T3/T5 provider learning loop, but for the panel itself.
The signal is the Judge's own rebuttal ledger: every accepted /
rejected / deferred critique is recorded in
``refinery_critique_dispositions``. The aggregator computes a rolling
acceptance rate per (role, severity), which converts to a bounded
multiplier the Judge applies at synthesis time:

- A role whose blockers consistently get accepted → multiplier > 1.0
  (its critiques carry more weight in the next debate's synthesis).
- A role whose blockers consistently get rejected → multiplier < 1.0
  (attenuated, never zeroed).
- No data, or too little signal in the window → identity multiplier
  (1.0); the system keeps existing behaviour until evidence accrues.

Bounds are tight by design — the panel's Parnas discipline is that
prompts and roles stay stable; weighting is a lever, not a prompt
rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


_MIN_TOTAL_FOR_LEARNING = 8
"""Below this many disposition rows, ignore the role — too little
signal to trust the rate."""

_LOW_RATE_FLOOR = 0.20
"""Below this acceptance rate, a role gets a soft attenuation."""

_HIGH_RATE_CEILING = 0.65
"""Above this acceptance rate, a role gets a soft boost."""

_STEP = 0.10
"""How far each adjustment moves the multiplier per refresh."""

_DEFAULT_MULTIPLIER = 1.0
_MIN_MULTIPLIER = 0.60
_MAX_MULTIPLIER = 1.40
"""Hard bounds. A misbehaving role can be attenuated to 60% but never
silenced; a high-performing role can be amplified to 140% but never
allowed to dominate the synthesis."""


def _classify_acceptance(
    *,
    total: int,
    rate: float,
    min_total: int,
    low: float,
    high: float,
    step: float,
    min_mult: float,
    max_mult: float,
) -> tuple[float, str]:
    """Shared band-classifier for both per-role and per-(role, dim)
    weighting. Returns ``(multiplier, reason)``."""

    if total < min_total:
        return _DEFAULT_MULTIPLIER, "below_min_total"
    if rate >= high:
        return min(max_mult, _DEFAULT_MULTIPLIER + step), "high_acceptance"
    if rate <= low:
        return max(min_mult, _DEFAULT_MULTIPLIER - step), "low_acceptance"
    return _DEFAULT_MULTIPLIER, "neutral"


@dataclass
class RoleWeight:
    """Effective multiplier for one role at synthesis time, plus the
    audit trail so operators can see what drove the adjustment."""

    role: str
    multiplier: float
    accepted: int
    rejected: int
    deferred: int
    total: int
    acceptance_rate: float
    reason: str


def adjust_role_weights(
    *,
    role_stats: list[dict[str, Any]],
    severity: str = "blocker",
) -> dict[str, RoleWeight]:
    """Convert raw disposition stats into bounded multipliers.

    ``role_stats`` rows each carry ``role``, ``severity``, ``total``,
    ``accepted``, ``rejected``, ``deferred``, ``acceptance_rate``.
    Filtering by ``severity`` (default ``blocker``) limits the signal
    to the most load-bearing critiques — accepting a WARNING is much
    cheaper than accepting a BLOCKER, so they shouldn't count equally.

    Roles absent from the input get no entry in the result; callers
    that want identity behaviour for missing roles use
    ``weight_for(role, ...)``.
    """

    out: dict[str, RoleWeight] = {}
    for row in role_stats or []:
        if row.get("severity") != severity:
            continue
        role = row.get("role")
        if not isinstance(role, str) or not role:
            continue

        try:
            total = int(row.get("total") or 0)
            accepted = int(row.get("accepted") or 0)
            rejected = int(row.get("rejected") or 0)
            deferred = int(row.get("deferred") or 0)
        except (TypeError, ValueError):
            log.warning("refinery_role_weighting_bad_row", row=row)
            continue
        rate = (accepted / total) if total else 0.0
        multiplier, reason = _classify_acceptance(
            total=total, rate=rate,
            min_total=_MIN_TOTAL_FOR_LEARNING,
            low=_LOW_RATE_FLOOR, high=_HIGH_RATE_CEILING, step=_STEP,
            min_mult=_MIN_MULTIPLIER, max_mult=_MAX_MULTIPLIER,
        )

        out[role] = RoleWeight(
            role=role,
            multiplier=multiplier,
            accepted=accepted,
            rejected=rejected,
            deferred=deferred,
            total=total,
            acceptance_rate=rate,
            reason=reason,
        )
    return out


def weight_for(
    role: str,
    weights: dict[str, RoleWeight] | dict[str, float],
) -> float:
    """Identity-default lookup. Unknown roles get 1.0.

    Accepts the typed map (in-process path) or the flat
    ``{role: float}`` evidence-bag shape — the orchestrator stamps
    flat dicts onto the bag for JSON-friendly transit, so lookup must
    handle both."""

    w = weights.get(role)
    if w is None:
        return _DEFAULT_MULTIPLIER
    if isinstance(w, (int, float)):
        return float(w)
    if hasattr(w, "multiplier"):
        return float(w.multiplier)
    return _DEFAULT_MULTIPLIER


def load_role_weights(
    *,
    metrics_client: Any | None,
    window_days: int = 30,
    severity: str = "blocker",
    role_stats: list[dict[str, Any]] | None = None,
) -> dict[str, RoleWeight]:
    """Top-level entry point — fetch stats from Postgres (or take pre-
    computed ``role_stats`` for the dashboard endpoint that already
    has them) and convert to multipliers.

    Returns an empty dict on any failure path. The Judge's caller
    treats an empty dict as "all multipliers = 1.0" — the system never
    hard-fails because role weighting can't load.
    """

    if role_stats is None:
        if metrics_client is None:
            return {}
        try:
            from dark_factory.metrics.refinery_repository import (
                RefineryMetricsRepository,
            )
            repo = RefineryMetricsRepository(metrics_client)
            role_stats = repo.compute_role_acceptance_stats(
                window_days=window_days,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_role_weighting_load_failed", error=str(exc),
            )
            return {}

    return adjust_role_weights(role_stats=role_stats, severity=severity)


_DIM_MIN_TOTAL_FOR_LEARNING = 6
"""Per-(role, dimension) cells are 5× sparser than per-role cells, so
the trustworthy-signal floor is tighter — but identity-default still
applies until it's met."""

_DIM_MIN_MULTIPLIER = 0.50
_DIM_MAX_MULTIPLIER = 1.50
"""Per-dimension multiplier bounds. Slightly wider than the per-role
band on the high side (a role's blockers on ITS native dimension
should be allowed more sway than the flat per-role multiplier permits)
and tighter on the low side because we never want a single dim cell
to silence a role's domain-correct findings."""

_PRODUCT_MIN_MULTIPLIER = 0.50
_PRODUCT_MAX_MULTIPLIER = 1.60
"""Hard cap on ``role_weight × dim_weight``. Without this, the bounds
of the two factors compose multiplicatively and a role's true effective
weight could land at 0.30 (silenced) or 2.10 (dominant). Both are
outside what either loop's bounds were designed to express, so we
clamp the product to a sensible band that the synthesis prompt can
actually act on."""


@dataclass
class RoleDimensionWeight:
    """Effective multiplier for one (role, dimension) cell + audit trail.

    Sibling to :class:`RoleWeight` but split by dimension. Used when the
    flat per-role multiplier would average a role's ``risk_coverage``
    accuracy together with its ``feasibility`` accuracy and dilute both.
    """

    role: str
    dimension: str
    multiplier: float
    accepted: int
    rejected: int
    deferred: int
    total: int
    acceptance_rate: float
    reason: str


def adjust_role_dimension_weights(
    *,
    role_dim_stats: list[dict[str, Any]],
    severity: str = "blocker",
) -> dict[tuple[str, str], RoleDimensionWeight]:
    """Convert raw per-(role, dimension) disposition stats into bounded
    multipliers. Mirrors :func:`adjust_role_weights` but groups by an
    additional ``dimension`` axis.

    Cells absent from the input get no entry; callers that want
    identity behaviour for missing cells use
    :func:`dimension_weight_for`.
    """

    out: dict[tuple[str, str], RoleDimensionWeight] = {}
    for row in role_dim_stats or []:
        if row.get("severity") != severity:
            continue
        role = row.get("role")
        dimension = row.get("dimension")
        if (
            not isinstance(role, str) or not role
            or not isinstance(dimension, str) or not dimension
        ):
            continue

        try:
            total = int(row.get("total") or 0)
            accepted = int(row.get("accepted") or 0)
            rejected = int(row.get("rejected") or 0)
            deferred = int(row.get("deferred") or 0)
        except (TypeError, ValueError):
            log.warning("refinery_role_dim_weighting_bad_row", row=row)
            continue

        rate = (accepted / total) if total else 0.0
        multiplier, reason = _classify_acceptance(
            total=total, rate=rate,
            min_total=_DIM_MIN_TOTAL_FOR_LEARNING,
            low=_LOW_RATE_FLOOR, high=_HIGH_RATE_CEILING, step=_STEP,
            min_mult=_DIM_MIN_MULTIPLIER, max_mult=_DIM_MAX_MULTIPLIER,
        )

        out[(role, dimension)] = RoleDimensionWeight(
            role=role,
            dimension=dimension,
            multiplier=multiplier,
            accepted=accepted,
            rejected=rejected,
            deferred=deferred,
            total=total,
            acceptance_rate=rate,
            reason=reason,
        )
    return out


def dimension_weight_for(
    role: str,
    dimension: str,
    weights: dict[tuple[str, str], RoleDimensionWeight] | dict[str, float],
) -> float:
    """Identity-default lookup. Accepts the typed map (in-process path)
    or the flat ``{"role:dim": float}`` evidence-bag shape.

    Unknown ``(role, dimension)`` cells get 1.0 — never silently demote
    a role on a dimension we have no signal for.
    """

    if not weights:
        return _DEFAULT_MULTIPLIER

    # Flat string-keyed dict (evidence bag)
    flat_key = f"{role}:{dimension}"
    flat = weights.get(flat_key)
    if isinstance(flat, (int, float)):
        return float(flat)

    # Typed (role, dim) tuple-keyed dict
    entry = weights.get((role, dimension))
    if entry is not None and hasattr(entry, "multiplier"):
        return float(entry.multiplier)
    return _DEFAULT_MULTIPLIER


def combined_weight_for(
    role: str,
    dimension: str,
    *,
    role_weights: dict[str, RoleWeight] | dict[str, float] | None,
    role_dim_weights: (
        dict[tuple[str, str], RoleDimensionWeight] | dict[str, float] | None
    ),
) -> float:
    """Multiplicative composition of the flat per-role and the per-
    (role, dimension) weights, clamped to the product band.

    ``role_weight × dim_weight`` is the natural model — a role with
    high overall credibility AND high credibility on a specific
    dimension should compound. The clamp prevents the bounds of the
    two factors from composing into ranges neither was designed for.
    Identity (1.0 × 1.0) propagates trivially.
    """

    rw = weight_for(role, role_weights or {})
    dw = dimension_weight_for(role, dimension, role_dim_weights or {})
    product = rw * dw
    return max(_PRODUCT_MIN_MULTIPLIER, min(_PRODUCT_MAX_MULTIPLIER, product))


def load_role_dimension_weights(
    *,
    metrics_client: Any | None,
    window_days: int = 30,
    severity: str = "blocker",
    role_dim_stats: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str], RoleDimensionWeight]:
    """Top-level entry — fetch per-(role, dimension) stats from Postgres
    or take precomputed ones, then convert to bounded multipliers.

    Empty dict on any failure path. Callers treat empty as "every cell
    is identity (×1.0)" so the system never hard-fails because the
    loop can't load.
    """

    if role_dim_stats is None:
        if metrics_client is None:
            return {}
        try:
            from dark_factory.metrics.refinery_repository import (
                RefineryMetricsRepository,
            )
            repo = RefineryMetricsRepository(metrics_client)
            role_dim_stats = repo.compute_role_dimension_acceptance_stats(
                window_days=window_days, severity=severity,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_role_dim_weighting_load_failed", error=str(exc),
            )
            return {}

    return adjust_role_dimension_weights(
        role_dim_stats=role_dim_stats, severity=severity,
    )


def flatten_role_weights(
    weights: dict[str, RoleWeight],
) -> dict[str, float]:
    """JSON-friendly ``{role: float}`` dict for the evidence bag, with
    identity entries dropped — keeps the calibration prompt block
    short when most roles have no signal yet."""

    return {
        role: w.multiplier
        for role, w in weights.items()
        if abs(w.multiplier - _DEFAULT_MULTIPLIER) > 1e-6
    }


def flatten_role_dim_weights(
    weights: dict[tuple[str, str], RoleDimensionWeight],
) -> dict[str, float]:
    """Convert the typed map to a JSON-friendly ``{"role:dim": float}``
    dict for the evidence bag. Only non-default multipliers are
    included — identity is implicit at the lookup site, and keeping
    the payload small keeps the synthesis prompt's calibration block
    readable."""

    return {
        f"{rd.role}:{rd.dimension}": rd.multiplier
        for rd in weights.values()
        if abs(rd.multiplier - _DEFAULT_MULTIPLIER) > 1e-6
    }


__all__ = [
    "RoleDimensionWeight",
    "RoleWeight",
    "adjust_role_dimension_weights",
    "adjust_role_weights",
    "combined_weight_for",
    "dimension_weight_for",
    "flatten_role_dim_weights",
    "flatten_role_weights",
    "load_role_dimension_weights",
    "load_role_weights",
    "weight_for",
]
