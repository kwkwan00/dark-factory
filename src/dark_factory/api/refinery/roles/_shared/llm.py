"""Shared single-shot LLM helper for the adversarial panel.

The adversarial seats (Product, Engineering, Security, Operations, Cost,
Judge) don't need a multi-turn tool-use loop — each one receives full
context in its prompt and produces a single structured response. This
helper picks the right SDK based on the model ID (Anthropic for
``claude-*``, OpenAI for everything else) and emits per-call events
(``refinery_llm_started`` / ``refinery_llm_ready`` /
``refinery_llm_failed``) to the global progress broker so the Agent
Log tab shows real per-agent activity.

Event payloads deliberately exclude prompt content and response
content — only model identity, provider, reasoning effort, and token
counts cross the AG-UI bridge. Full prompts and responses live in the
forensic trace store, not the live event stream.

Tests monkey-patch the concrete roles' ``_call_llm`` hook directly
and never reach this module; production code instantiates the panel
via the registry and every verb routes through here.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import structlog

from dark_factory.agents.tools import emit_progress

log = structlog.get_logger()

# DeepEval / OpenAI Responses don't accept ``xhigh`` directly. Map it
# to the highest tier the SDK understands. ``low`` / ``medium`` /
# ``high`` pass through unchanged.
_OPENAI_REASONING_MAP = {"xhigh": "high"}

# Anthropic effort enum across Claude 4.6+ models is
# ``low | medium | high | max`` — ``xhigh`` (the legacy Opus-4.7 name
# for the top tier) is rejected with a 400 by other models. Map it to
# ``max`` so refinery's role vocabulary keeps working everywhere.
_ANTHROPIC_REASONING_MAP = {"xhigh": "max"}

# Thread-local progress callback. The refinery SSE orchestrator sets
# this once per worker thread (see ``install_progress_callback`` below)
# so the LLM helper can echo each event into the per-run progress
# stream alongside the global broker. Tests don't install a callback;
# the helper falls back to broker-only when the slot is empty.
_thread_local = threading.local()

ProgressCallback = Callable[[dict], None]


@contextmanager
def install_progress_callback(cb: ProgressCallback | None):
    """Install ``cb`` as the active progress callback for the calling
    thread. Restores the previous slot value on exit so nested debate
    invocations on the same worker thread don't bleed callbacks."""

    prev = getattr(_thread_local, "progress_cb", None)
    _thread_local.progress_cb = cb
    try:
        yield
    finally:
        _thread_local.progress_cb = prev


def _emit(event_name: str, **payload: Any) -> None:
    """Emit ``event_name`` to the global broker and the thread-local
    refinery progress callback (when one is installed)."""

    emit_progress(event_name, **payload)
    cb = getattr(_thread_local, "progress_cb", None)
    if cb is not None:
        # Refinery SSE expects ``phase`` so its existing relay code
        # tags the event correctly. The broker version doesn't need
        # this — the broker keys on ``event``.
        try:
            cb({"phase": "refining", "event": event_name, **payload})
        except Exception:  # pragma: no cover — defensive
            log.exception("refinery_llm_progress_cb_failed")


def _provider_for(model: str) -> str:
    """Return ``"anthropic"`` for Claude models, else ``"openai"``."""

    return "anthropic" if model.lower().startswith("claude") else "openai"


def call_refinery_llm(
    prompt: str,
    *,
    model: str,
    reasoning_effort: str = "high",
    agent: str,
    feature: str = "refinery",
    max_output_tokens: int = 8192,
) -> str:
    """Run one single-shot LLM call for a refinery role.

    Parameters
    ----------
    prompt:
        The full role prompt (context already baked in).
    model:
        Model ID. Routes to the Anthropic SDK for ``claude-*`` and
        the OpenAI Responses API for everything else (e.g. ``gpt-5.4``).
    reasoning_effort:
        ``low`` / ``medium`` / ``high`` / ``xhigh``. Passed through
        verbatim to the provider; OpenAI Responses normalises ``xhigh``
        to ``high`` (its highest tier).
    agent:
        Role name (``engineering``, ``security``, ``operations``,
        ``cost``, ``product``, ``judge``). Emitted on every event so
        the agent log can colour-code per role.
    feature:
        Feature label for the global broker (defaults to ``refinery``).
    max_output_tokens:
        Per-call output cap.

    Returns
    -------
    The raw response text. Callers own parsing (JSON or otherwise).
    """

    provider = _provider_for(model)
    _emit(
        "refinery_llm_started",
        feature=feature,
        agent=agent,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
    )

    started = time.monotonic()
    try:
        if provider == "anthropic":
            text, tokens_in, tokens_out = _call_anthropic(
                prompt, model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
        else:
            text, tokens_in, tokens_out = _call_openai(
                prompt, model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        _emit(
            "refinery_llm_failed",
            feature=feature,
            agent=agent,
            model=model,
            provider=provider,
            latency_ms=latency_ms,
            error=str(exc),
        )
        log.warning(
            "refinery_llm_call_failed",
            agent=agent, model=model, provider=provider, error=str(exc),
        )
        raise

    latency_ms = int((time.monotonic() - started) * 1000)
    _emit(
        "refinery_llm_ready",
        feature=feature,
        agent=agent,
        model=model,
        provider=provider,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    return text


def _call_openai(
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> tuple[str, int, int]:
    """Single-shot OpenAI Responses call. Returns (text, tokens_in, tokens_out)."""

    import openai

    effort = _OPENAI_REASONING_MAP.get(reasoning_effort, reasoning_effort)
    client = openai.OpenAI()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": effort},
        temperature=1,
        max_output_tokens=max_output_tokens,
    )
    text = response.output_text or ""
    usage = getattr(response, "usage", None)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    return text, tokens_in, tokens_out


def _call_anthropic(
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> tuple[str, int, int]:
    """Single-shot Anthropic Messages call. Returns (text, tokens_in, tokens_out).

    Streaming is required by Anthropic when ``max_tokens`` is high or
    the call could exceed the 10-minute non-streaming limit, so we
    iterate ``text_stream`` to capture every chunk.
    """

    import anthropic

    client = anthropic.Anthropic()
    parts: list[str] = []
    tokens_in = 0
    tokens_out = 0
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Anthropic 4.6+ models (Opus 4.6, Sonnet 4.6) replaced the
    # fixed-budget thinking API with adaptive thinking gated by
    # ``output_config.effort``. The older ``{"type": "enabled",
    # "budget_tokens": N}`` shape is rejected with a 400 on these
    # models. Anthropic's effort enum is ``low|medium|high|max``;
    # refinery uses ``xhigh`` as its top-tier sentinel, so map it
    # through ``_ANTHROPIC_REASONING_MAP`` before forwarding. Lower
    # tiers skip thinking entirely so the call stays cheap.
    effort = _ANTHROPIC_REASONING_MAP.get(reasoning_effort, reasoning_effort)
    if effort in ("high", "max"):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}

    with client.messages.stream(**kwargs) as stream:
        for chunk in stream.text_stream:
            parts.append(chunk)
        final = stream.get_final_message()
        usage = getattr(final, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)

    return "".join(parts), tokens_in, tokens_out


def is_llm_available() -> bool:
    """Return ``True`` iff at least one provider SDK is importable."""

    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = [
    "call_refinery_llm",
    "install_progress_callback",
    "is_llm_available",
]
