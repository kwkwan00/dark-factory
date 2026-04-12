"""Proxy endpoints that list available models from Anthropic and OpenAI.

The frontend Settings tab uses these to populate the Main LLM and Eval
Model dropdowns dynamically from whatever API keys the operator has
configured (either the server's env vars or a per-session override
typed into the Settings tab).

Why proxy through the backend?
- **CORS.** Browsers can't hit ``api.anthropic.com`` / ``api.openai.com``
  directly because those origins don't return ``Access-Control-Allow-Origin``
  for arbitrary domains.
- **Timeouts.** We control the timeout server-side so a wedged upstream
  can't hang the tab.
- **Key safety.** Keys travel in a POST body (not a URL), never land in
  access logs, and are never logged by this module.

Both endpoints accept the same request body:

    {"api_key": "sk-…"}  // optional; falls back to env var

And return the same shape:

    {
      "source": "live" | "fallback",
      "models": [
        {"id": "...", "display_name": "...", "created_at": "..."},
        ...
      ]
    }

``source="fallback"`` means the endpoint couldn't reach the upstream
(no key anywhere, or the /v1/models call failed in a way we chose to
degrade gracefully from) and returned a curated hardcoded list instead.
The frontend uses this to show a small "using defaults" indicator.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

log = structlog.get_logger()

router = APIRouter()


# ── Upstream config ────────────────────────────────────────────────────────

_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_ANTHROPIC_API_VERSION = "2023-06-01"

_OPENAI_MODELS_URL = "https://api.openai.com/v1/models"

_HTTP_TIMEOUT_SECONDS = 10.0


# ── Fallback lists ─────────────────────────────────────────────────────────
#
# Used when no API key is available or the upstream is unreachable.
# Kept in sync with the curated lists previously hardcoded in the
# frontend SettingsTab. The frontend merges the live list with these
# as a safety net so the dropdown always has something to render.

_ANTHROPIC_FALLBACK_MODELS = [
    {"id": "claude-opus-4-6", "display_name": "Claude Opus 4.6", "created_at": None},
    {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6", "created_at": None},
    {
        "id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "created_at": None,
    },
]

_OPENAI_FALLBACK_MODELS = [
    {"id": "gpt-5.4", "display_name": "gpt-5.4", "created_at": None},
    {"id": "gpt-5.1", "display_name": "gpt-5.1", "created_at": None},
    {"id": "gpt-5", "display_name": "gpt-5", "created_at": None},
    {"id": "gpt-4.1", "display_name": "gpt-4.1", "created_at": None},
    {"id": "gpt-4.1-mini", "display_name": "gpt-4.1-mini", "created_at": None},
    {"id": "gpt-4o", "display_name": "gpt-4o", "created_at": None},
    {"id": "o4-mini", "display_name": "o4-mini", "created_at": None},
    {"id": "o3-mini", "display_name": "o3-mini", "created_at": None},
    {"id": "o1", "display_name": "o1", "created_at": None},
]


# ── Request model ──────────────────────────────────────────────────────────


class ModelListRequest(BaseModel):
    """Body for the model-list endpoints.

    ``hide_input_in_errors=True`` ensures that validation errors on the
    ``api_key`` field do not echo the submitted value back to the
    client. Combined with the app-level validation exception handler,
    this prevents any pasted key from leaking into browser console logs.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    api_key: str | None = None

    @field_validator("api_key")
    @classmethod
    def _validate_key(cls, v: str | None) -> str | None:
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


# ── OpenAI chat-model filter ───────────────────────────────────────────────


def _is_chat_capable_openai_model(model_id: str) -> bool:
    """Return True for models that DeepEval can use as a judge.

    Keep only conversational / reasoning models and exclude embeddings,
    audio, image, and moderation endpoints. Also exclude ``-instruct``
    variants which use the legacy completions API that DeepEval doesn't
    speak. The filter is deliberately positive (opt-in by prefix) so
    new OpenAI launches that don't yet start with ``gpt-``/``o*`` won't
    accidentally flood the list.
    """
    mid = model_id.lower()

    # Negative exclusions first — these are the top few categories
    # OpenAI returns alongside chat models.
    if any(
        bad in mid
        for bad in (
            "embedding",
            "whisper",
            "tts",
            "audio",
            "dall-e",
            "dalle",
            "moderation",
            "image",
            "omni-moderation",
        )
    ):
        return False
    if mid.endswith("-instruct"):
        return False
    if mid.startswith("text-") or mid.startswith("davinci") or mid.startswith("babbage"):
        return False  # legacy completion models

    # Positive prefixes — chat/reasoning models we know about.
    return (
        mid.startswith("gpt-")
        or mid.startswith("chatgpt-")
        or mid.startswith("o1")
        or mid.startswith("o3")
        or mid.startswith("o4")
        or mid.startswith("o5")
    )


# ── Sort helpers ───────────────────────────────────────────────────────────


def _sort_models_newest_first(
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort models by created_at desc with None pushed to the end."""
    return sorted(
        models,
        key=lambda m: (m.get("created_at") is None, -(m.get("created_at") or 0)),
    )


# ── Anthropic ──────────────────────────────────────────────────────────────


def _fetch_anthropic_models(api_key: str) -> list[dict[str, Any]]:
    """Call ``GET /v1/models`` on api.anthropic.com.

    Raises HTTPException on auth or network failure. The Anthropic API
    paginates via ``has_more`` + ``last_id`` — we iterate until we've
    collected every model.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_API_VERSION,
    }

    collected: list[dict[str, Any]] = []
    next_cursor: str | None = None
    # Hard cap on pages to prevent infinite loops on a misbehaving
    # upstream. 10 pages × 1000 default limit is well beyond the real
    # model inventory.
    for _page in range(10):
        params: dict[str, Any] = {"limit": 1000}
        if next_cursor:
            params["after_id"] = next_cursor
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as client:
                resp = client.get(
                    _ANTHROPIC_MODELS_URL,
                    headers=headers,
                    params=params,
                )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Anthropic API unreachable: {exc}",
            ) from exc

        if resp.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Anthropic API key invalid or expired",
            )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Anthropic API returned {resp.status_code}: "
                    f"{resp.text[:200]}"
                ),
            )

        body = resp.json()
        for item in body.get("data", []):
            created_raw = item.get("created_at")
            # Anthropic returns ISO 8601 strings; keep both raw +
            # an integer epoch for sorting. Be tolerant if they ever
            # switch to epoch seconds directly.
            created_iso: str | None = None
            created_epoch: int | None = None
            if isinstance(created_raw, str):
                created_iso = created_raw
                try:
                    from datetime import datetime

                    created_epoch = int(
                        datetime.fromisoformat(
                            created_raw.replace("Z", "+00:00")
                        ).timestamp()
                    )
                except Exception:
                    created_epoch = None
            elif isinstance(created_raw, (int, float)):
                created_epoch = int(created_raw)

            collected.append(
                {
                    "id": item.get("id"),
                    "display_name": item.get("display_name") or item.get("id"),
                    "created_at": created_iso,
                    "_epoch": created_epoch,
                }
            )

        if not body.get("has_more"):
            break
        next_cursor = body.get("last_id")
        if not next_cursor:
            break

    # Sort newest-first by epoch, then move the _epoch helper out.
    collected.sort(
        key=lambda m: (m["_epoch"] is None, -(m["_epoch"] or 0))
    )
    return [
        {
            "id": m["id"],
            "display_name": m["display_name"],
            "created_at": m["created_at"],
        }
        for m in collected
        if m.get("id")
    ]


# ── OpenAI ─────────────────────────────────────────────────────────────────


def _fetch_openai_models(api_key: str) -> list[dict[str, Any]]:
    """Call ``GET /v1/models`` on api.openai.com and filter to chat models."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            resp = client.get(_OPENAI_MODELS_URL, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI API unreachable: {exc}",
        ) from exc

    if resp.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="OpenAI API key invalid or expired",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API returned {resp.status_code}: {resp.text[:200]}",
        )

    body = resp.json()
    models: list[dict[str, Any]] = []
    for item in body.get("data", []):
        mid = item.get("id")
        if not isinstance(mid, str):
            continue
        if not _is_chat_capable_openai_model(mid):
            continue
        created = item.get("created")
        models.append(
            {
                "id": mid,
                "display_name": mid,
                "created_at": (
                    int(created) if isinstance(created, (int, float)) else None
                ),
            }
        )

    # Sort newest-first (OpenAI ``created`` is epoch seconds).
    models.sort(key=lambda m: -(m["created_at"] or 0))

    # Normalise created_at back to ISO so the shape matches Anthropic's.
    from datetime import datetime, timezone

    for m in models:
        epoch = m["created_at"]
        if epoch is not None:
            m["created_at"] = datetime.fromtimestamp(
                epoch, tz=timezone.utc
            ).isoformat()
    return models


# ── Routes ─────────────────────────────────────────────────────────────────


def _resolve_key(body_key: str | None, env_name: str) -> str | None:
    """Pick the per-request key if provided, else fall back to env var."""
    if body_key:
        return body_key
    env_val = os.getenv(env_name)
    if env_val and env_val.strip():
        return env_val.strip()
    return None


@router.post("/models/anthropic")
def list_anthropic_models(body: ModelListRequest) -> dict[str, Any]:
    """List Claude models available to the configured API key.

    If no key is provided in the body AND the server has no
    ``ANTHROPIC_API_KEY`` env var set, returns a curated fallback list
    marked ``source="fallback"`` instead of erroring. This keeps the
    Settings tab functional even on a brand-new install.
    """
    key = _resolve_key(body.api_key, "ANTHROPIC_API_KEY")
    log.info("models_list_requested", provider="anthropic", has_key=bool(key))

    if not key:
        return {"source": "fallback", "models": _ANTHROPIC_FALLBACK_MODELS}

    models = _fetch_anthropic_models(key)
    return {"source": "live", "models": models}


@router.post("/models/openai")
def list_openai_models(body: ModelListRequest) -> dict[str, Any]:
    """List chat-capable OpenAI models available to the configured API key.

    Same fallback semantics as the Anthropic endpoint. The returned
    list is filtered to chat/reasoning models only (no embeddings,
    whisper, tts, dall-e, moderation, or legacy completion models) so
    the DeepEval dropdown doesn't show models the judge can't use.
    """
    key = _resolve_key(body.api_key, "OPENAI_API_KEY")
    log.info("models_list_requested", provider="openai", has_key=bool(key))

    if not key:
        return {"source": "fallback", "models": _OPENAI_FALLBACK_MODELS}

    models = _fetch_openai_models(key)
    return {"source": "live", "models": models}
