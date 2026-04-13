"""Multi-turn agentic tool-use loop using the Anthropic SDK directly.

Replaces the Claude Agent SDK subprocess for file-creating deep agent tools.
Runs a synchronous messages.create() loop, dispatching tool calls to
``tool_handlers.execute_tool`` and feeding results back until the model
emits ``end_turn`` or the budget (turns / wall-clock) is exhausted.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import anthropic
import structlog

from dark_factory.llm.anthropic import _record_llm_call
from dark_factory.llm.tool_handlers import execute_tool, get_tool_schemas

# Optional progress callback type: (turn, max_turns, tool_names, text_preview) -> None
OnTurnCallback = Callable[[int, int, list[str], str], None]

log = structlog.get_logger()

_DEFAULT_MODEL = "claude-sonnet-4-6"


def run_agentic_loop(
    *,
    prompt: str,
    system: str | None = None,
    allowed_tools: list[str],
    sandbox_root: Path,
    max_turns: int = 20,
    timeout_seconds: float = 600.0,
    model: str | None = None,
    max_tokens: int = 16384,
    api_key: str | None = None,
    on_turn: OnTurnCallback | None = None,
) -> str:
    """Run a multi-turn agentic tool-use loop and return the final text.

    Parameters
    ----------
    prompt:
        The initial user message that kicks off the loop.
    system:
        Optional system prompt prepended to every API call.
    allowed_tools:
        Tool names to expose to the model (passed to
        ``get_tool_schemas``).
    sandbox_root:
        Root directory for tool execution (file writes are sandboxed
        here).
    max_turns:
        Maximum number of model invocations before the loop stops.
    timeout_seconds:
        Wall-clock budget for the entire loop.
    model:
        Anthropic model id.  Falls back to settings or
        ``claude-sonnet-4-6``.
    max_tokens:
        Per-turn max output tokens.
    api_key:
        Anthropic API key.  When ``None``, the SDK reads
        ``ANTHROPIC_API_KEY`` from the environment.
    """

    # Resolve model ---------------------------------------------------------
    if model is None:
        try:
            from dark_factory.config import load_settings

            model = load_settings().llm.model or _DEFAULT_MODEL
        except Exception:
            model = _DEFAULT_MODEL

    # Client & tools --------------------------------------------------------
    client = anthropic.Anthropic(api_key=api_key)
    tools = get_tool_schemas(allowed_tools)
    messages: list[dict] = [{"role": "user", "content": prompt}]

    wall_start = time.time()
    accumulated_text: list[str] = []

    for turn in range(1, max_turns + 1):
        # ---- timeout check ------------------------------------------------
        elapsed = time.time() - wall_start
        if elapsed >= timeout_seconds:
            log.warning(
                "agentic_loop_timeout",
                turn=turn,
                elapsed=round(elapsed, 1),
                timeout=timeout_seconds,
            )
            note = (
                f"\n[agentic loop timed out after {elapsed:.0f}s "
                f"on turn {turn}/{max_turns}]"
            )
            return (_join_text(accumulated_text) + note).strip()

        # ---- API call with single retry on transient errors ---------------
        response = _call_with_retry(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            system=system,
            timeout_seconds=timeout_seconds - elapsed,
            turn=turn,
        )

        # ---- metrics ------------------------------------------------------
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_create = getattr(usage, "cache_creation_input_tokens", None)

        # Collect tool names invoked this turn for logging
        tool_names_this_turn = [
            block.name
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        log.info(
            "agentic_turn",
            turn=turn,
            stop_reason=response.stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tools_called=tool_names_this_turn or None,
        )

        _record_llm_call(
            client="agentic",
            model=model,
            started_at=time.time() - (
                (getattr(response, "_request_duration", None) or 0)
                or 0
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_create,
            stop_reason=response.stop_reason,
            max_tokens_requested=max_tokens,
        )

        # ---- progress callback --------------------------------------------
        if on_turn is not None:
            # Extract a short text preview for the progress event
            text_preview = ""
            for block in response.content:
                if getattr(block, "type", None) == "text" and block.text.strip():
                    text_preview = block.text.strip()[:200]
                    break
            try:
                on_turn(turn, max_turns, tool_names_this_turn, text_preview)
            except Exception:
                pass  # Never let a callback failure break the loop

        # ---- extract text from this turn ----------------------------------
        for block in response.content:
            if getattr(block, "type", None) == "text":
                accumulated_text.append(block.text)

        # ---- handle stop reason -------------------------------------------
        if response.stop_reason == "end_turn":
            return _join_text(accumulated_text)

        if response.stop_reason == "max_tokens":
            log.warning(
                "agentic_max_tokens",
                turn=turn,
                output_tokens=output_tokens,
                max_tokens=max_tokens,
            )
            return _join_text(accumulated_text)

        if response.stop_reason == "tool_use":
            # Append assistant message with full content
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Execute each tool call and build result blocks
            tool_result_blocks: list[dict] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                result_text, is_error = _safe_execute_tool(
                    name=block.name,
                    tool_input=block.input,
                    sandbox_root=sandbox_root,
                )

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                })

            messages.append({
                "role": "user",
                "content": tool_result_blocks,
            })
            continue

        # Unknown stop reason — treat as terminal
        log.warning(
            "agentic_unexpected_stop",
            stop_reason=response.stop_reason,
            turn=turn,
        )
        return _join_text(accumulated_text)

    # Exhausted turn budget -------------------------------------------------
    note = f"\n[agentic loop exhausted turn budget ({max_turns} turns)]"
    return (_join_text(accumulated_text) + note).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _join_text(parts: list[str]) -> str:
    """Join accumulated text blocks, stripping leading/trailing whitespace."""
    return "\n".join(parts).strip()


def _safe_execute_tool(
    *,
    name: str,
    tool_input: dict,
    sandbox_root: Path,
) -> tuple[str, bool]:
    """Execute a tool call, catching exceptions so the loop continues.

    Returns (result_text, is_error).
    """
    try:
        return execute_tool(name, tool_input, sandbox_root)
    except Exception as exc:
        log.warning(
            "agentic_tool_error",
            tool=name,
            error=str(exc)[:500],
        )
        return (f"Error executing {name}: {exc}", True)


def _call_with_retry(
    *,
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int,
    system: str | None,
    timeout_seconds: float,
    turn: int,
) -> anthropic.types.Message:
    """Call messages.create with a single retry on transient errors."""

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
        "timeout": max(timeout_seconds, 30.0),
    }
    if system:
        kwargs["system"] = system

    last_error: Exception | None = None
    for attempt in range(2):
        started_at = time.time()
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_error = exc
            http_status = getattr(exc, "status_code", None)
            rate_limited = isinstance(exc, anthropic.RateLimitError) or (
                http_status == 429
            )

            is_transient = (
                rate_limited
                or isinstance(
                    exc,
                    (
                        anthropic.APIConnectionError,
                        anthropic.APITimeoutError,
                        anthropic.InternalServerError,
                    ),
                )
                or (isinstance(http_status, int) and http_status >= 500)
            )

            _record_llm_call(
                client="agentic",
                model=model,
                started_at=started_at,
                max_tokens_requested=max_tokens,
                http_status=http_status,
                rate_limited=rate_limited,
                retry_count=attempt,
                error=str(exc)[:500],
            )

            log.warning(
                "agentic_api_error",
                turn=turn,
                attempt=attempt + 1,
                transient=is_transient,
                error=str(exc)[:500],
            )

            if not is_transient:
                raise

    assert last_error is not None
    raise last_error
