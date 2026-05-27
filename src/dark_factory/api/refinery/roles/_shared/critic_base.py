"""Shared critic machinery — rubber-stamp validator, retry, placeholder.

Adversarial contract:

1. Every critic's prompt explicitly instructs it to challenge, not
   validate. The prompt preamble lives in
   ``roles/_shared/critic_prompts.py``.
2. The critic's output is parsed into a ``Critique`` and validated
   against the rubber-stamp guardrail (finding ≥ 20 chars, proposed_fix
   ≥ 10 chars OR severity=INFO with a non-empty search narrative).
3. A rejected critique gets ONE retry with an explicit "you returned a
   rubber-stamp" follow-up.
4. A second rubber-stamp becomes a placeholder Critique tagged
   ``rubber_stamp_retry_count=2`` so the synthesizer + trace see the
   fall-back; the round continues.

The adversarial-contract rule (``rule_no_rubber_stamp_rebuttal``) in
the rules gate enforces the complementary invariant on the Judge's
Rebuttal side.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import structlog

from dark_factory.api.refinery.contracts import (
    Critique,
    CritiqueDimension,
    Draft,
    RoleContext,
    Severity,
)

log = structlog.get_logger()

# Minimum finding + proposed_fix length for a non-INFO critique to pass
# the rubber-stamp validator. Tuned to reject common validation phrases
# ("looks good", "LGTM", "no concerns") while accepting terse-but-
# substantive findings like "missing CSRF token on POST /session".
_MIN_FINDING_CHARS = 20
_MIN_PROPOSED_FIX_CHARS = 10

# Obvious rubber-stamp phrases. Critics that emit text containing these
# at the head of the finding are rejected even if they pass the length
# check (phrases like "Looks good overall, no concerns worth raising,
# the design is fine" would otherwise sneak through).
_RUBBER_STAMP_PATTERNS = (
    re.compile(r"^\s*looks good\b", re.IGNORECASE),
    re.compile(r"^\s*lgtm\b", re.IGNORECASE),
    re.compile(r"^\s*no (?:concerns|issues)\b", re.IGNORECASE),
    re.compile(r"^\s*nothing to (?:add|flag)\b", re.IGNORECASE),
    re.compile(r"^\s*seems (?:fine|good|reasonable)\b", re.IGNORECASE),
    re.compile(r"^\s*all good\b", re.IGNORECASE),
)


class RubberStampRejected(ValueError):
    """Raised internally when a critique fails the adversarial-contract
    validator. Not a user-facing error — the base class catches this,
    retries, and falls through to a placeholder."""


def is_rubber_stamp(critique: Critique) -> bool:
    """Return True if *critique* looks like a validation rather than an
    adversarial challenge.

    Rules:
    - Severity=INFO is OK as long as the finding is non-empty (the critic
      is explicitly declaring it searched and found nothing substantive).
    - Otherwise, finding ≥ 20 chars AND proposed_fix ≥ 10 chars AND
      the finding doesn't start with a known rubber-stamp pattern.
    """

    finding = (critique.finding or "").strip()
    proposed_fix = (critique.proposed_fix or "").strip()

    # Empty findings are always rubber-stamps, even for INFO severity.
    if not finding:
        return True

    # INFO critics get a pass on the length rules as long as the finding
    # is a non-trivial narrative explaining what they searched for.
    if critique.severity == Severity.INFO:
        if len(finding) < 15:
            return True
        return any(p.search(finding) for p in _RUBBER_STAMP_PATTERNS)

    # Substantive critiques must clear both length gates and not open
    # with a known rubber-stamp cliche.
    if len(finding) < _MIN_FINDING_CHARS:
        return True
    if len(proposed_fix) < _MIN_PROPOSED_FIX_CHARS:
        return True
    if any(p.search(finding) for p in _RUBBER_STAMP_PATTERNS):
        return True
    return False


def make_placeholder_critique(
    *,
    role_name: str,
    dimension: CritiqueDimension,
    reason: str,
    retry_count: int = 2,
    error: str | None = None,
) -> Critique:
    """Produce the fall-back critique recorded when the critic can't be
    coerced into a substantive challenge. The synthesizer + trace still
    see a Critique object so the barrier reducer doesn't block; the
    placeholder just carries confidence=0 and severity=INFO so it
    doesn't influence the Judge's synthesis."""

    # Retry count is stamped on the metadata so tests and the rules
    # gate can assert the adversarial contract actually fired. The
    # Critique schema doesn't carry ``rubber_stamp_retry_count`` directly
    # (that's a CriticCall trace field), so we encode it into ``error``
    # when the placeholder fires specifically from a rubber-stamp retry
    # so downstream observers can parse it back out.
    tag = f"rubber_stamp_retry_count={retry_count}"
    return Critique(
        author_role=role_name,
        dimension=dimension,
        severity=Severity.INFO,
        finding=f"{role_name} produced no substantive finding after retry ({reason})",
        proposed_fix="",
        cited_evidence=[],
        confidence=0.0,
        error=error or tag,
    )


# A CritiqueGenerator is any callable that returns raw LLM-style JSON
# (or JSON-shaped dict) representing a Critique. Concrete critic roles
# wire this to the real OpenAI / Anthropic SDK in production and to
# stubs in tests.
CritiqueGenerator = Callable[[Draft, RoleContext, str], str]


def parse_critique_json(
    raw: str | dict,
    *,
    role_name: str,
    default_dimension: CritiqueDimension,
) -> Critique:
    """Parse a raw LLM JSON payload into a validated Critique.

    Accepts either a JSON string or a pre-parsed dict. The ``author_role``
    field is always rewritten to ``role_name`` — we don't trust the
    model to stamp its own role correctly.

    Raises ``RubberStampRejected`` when the parsed critique fails the
    adversarial-contract validator, so the caller's retry loop can
    distinguish "bad JSON" (ValueError) from "substantively empty"
    (RubberStampRejected).
    """

    payload: dict[str, Any]
    if isinstance(raw, dict):
        payload = dict(raw)
    else:
        # Tolerant extraction — some models wrap JSON in ```json fences
        # or prose. Strip the fences, find the first {...} block.
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                text = text[first_brace:last_brace + 1]
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"critic output was not valid JSON: {exc}") from exc

    # Coerce fields. Missing dimension defaults to the role's primary
    # dimension (Engineering → feasibility, Security → risk_coverage,
    # etc.) so a model that forgets the field doesn't fail validation.
    payload["author_role"] = role_name
    if "dimension" not in payload:
        payload["dimension"] = default_dimension.value
    if "severity" not in payload:
        payload["severity"] = Severity.INFO.value

    try:
        critique = Critique.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"critic output failed schema validation: {exc}") from exc

    if is_rubber_stamp(critique):
        raise RubberStampRejected(
            f"critique from {role_name} was a rubber-stamp: "
            f"severity={critique.severity.value} finding_len="
            f"{len(critique.finding)} fix_len={len(critique.proposed_fix)}"
        )

    return critique


def call_critic_with_retry(
    *,
    role_name: str,
    default_dimension: CritiqueDimension,
    draft: Draft,
    context: RoleContext,
    generate: CritiqueGenerator,
    max_retries: int = 1,
) -> Critique:
    """Run the critic with the adversarial retry contract.

    Flow:
    1. Call ``generate(draft, context, "")`` once.
    2. Parse + validate. On RubberStampRejected, retry up to
       ``max_retries`` times with a pointed "don't rubber-stamp"
       follow-up message.
    3. On final failure (rubber-stamp or schema), return a placeholder
       Critique tagged with the retry count.

    Exceptions from ``generate()`` itself are treated as transient:
    logged, one retry, then a placeholder with the exception message
    stamped in ``error``.
    """

    attempts = 0
    last_error: str | None = None
    followup = ""

    while attempts <= max_retries:
        try:
            raw = generate(draft, context, followup)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "refinery_critic_generate_raised",
                role=role_name, attempt=attempts, error=last_error,
            )
            attempts += 1
            followup = (
                "Your previous attempt raised an error. Return valid JSON "
                "matching the Critique schema, with a substantive finding "
                "(≥ 20 chars) and proposed_fix (≥ 10 chars)."
            )
            continue

        try:
            return parse_critique_json(
                raw,
                role_name=role_name,
                default_dimension=default_dimension,
            )
        except RubberStampRejected as exc:
            last_error = str(exc)
            log.info(
                "refinery_critic_rubber_stamp",
                role=role_name, attempt=attempts,
            )
            attempts += 1
            followup = (
                "Your previous response was a rubber-stamp / validation. "
                "Your job is to challenge this draft. Either identify a "
                "concrete flaw with severity WARNING or BLOCKER and a "
                "proposed fix ≥ 10 chars, OR return INFO severity with "
                "a substantive search-narrative (≥ 15 chars) explaining "
                "what you checked and why you found no genuine concern."
            )
        except ValueError as exc:
            # Malformed output — treat as transient and retry.
            last_error = f"parse failed: {exc}"
            log.warning(
                "refinery_critic_parse_failed",
                role=role_name, attempt=attempts, error=str(exc),
            )
            attempts += 1
            followup = (
                "Your previous response was not valid JSON. Return ONLY a "
                "JSON object matching the Critique schema."
            )

    # Exhausted retries — placeholder.
    return make_placeholder_critique(
        role_name=role_name,
        dimension=default_dimension,
        reason=last_error or "unknown",
        retry_count=attempts,
        error=last_error,
    )
