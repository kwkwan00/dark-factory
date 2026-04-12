"""Anthropic LLM client implementation."""

from __future__ import annotations

import re
import time
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel

log = structlog.get_logger()

from dark_factory.llm.base import DEFAULT_LLM_TIMEOUT_SECONDS, LLMClient

T = TypeVar("T", bound=BaseModel)

# H6: regex to extract JSON from markdown code blocks or raw text.
# Still used by ClaudeAgentClient which doesn't have a tool_use equivalent;
# AnthropicClient itself now uses native tool_use for structured output and
# no longer depends on this path.
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _inline_refs(schema: dict) -> dict:
    """Recursively inline ``$defs`` / ``definitions`` references into a
    self-contained JSON Schema.

    Pydantic's ``model_json_schema()`` emits ``$ref`` pointers to a top-level
    ``$defs`` section for nested models (e.g. ``Spec.scenarios: list[Scenario]``).
    Anthropic's tool ``input_schema`` accepts JSON Schema but the safest,
    most portable form is fully-inlined with no external refs — older SDK
    versions and some model revisions get tripped up by ``$ref`` resolution.
    """
    if not isinstance(schema, dict):
        return schema
    schema = dict(schema)
    defs = dict(schema.pop("$defs", {}) or {})
    defs.update(schema.pop("definitions", {}) or {})

    def resolve(node: Any, seen: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, dict):
            if isinstance(node.get("$ref"), str):
                ref = node["$ref"]
                for prefix in ("#/$defs/", "#/definitions/"):
                    if ref.startswith(prefix):
                        name = ref[len(prefix):]
                        if name in seen:
                            # Self-referential schema — break the cycle with
                            # a permissive object type.
                            return {"type": "object"}
                        target = defs.get(name)
                        if target is not None:
                            return resolve(dict(target), seen | {name})
                        break
                return node
            return {k: resolve(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve(schema)


def _build_tool_schema(response_model: type[BaseModel]) -> dict:
    """Build an Anthropic-compatible ``input_schema`` for a Pydantic model."""
    return _inline_refs(response_model.model_json_schema())


def _extract_tool_input(final_message: Any) -> dict[str, Any] | None:
    """Find the ``tool_use`` content block in a streamed final message."""
    content = getattr(final_message, "content", None) or []
    for block in content:
        if getattr(block, "type", None) == "tool_use":
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    return None


def _record_llm_call(
    *,
    client: str,
    model: str,
    started_at: float,
    prompt_chars: int | None = None,
    completion_chars: int | None = None,
    system_prompt_chars: int | None = None,
    max_tokens_requested: int | None = None,
    temperature: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    time_to_first_token_seconds: float | None = None,
    retry_count: int = 0,
    stop_reason: str | None = None,
    http_status: int | None = None,
    rate_limited: bool = False,
    error: str | None = None,
) -> None:
    """Best-effort write to BOTH the Postgres recorder (if enabled) and
    the Prometheus LLM counters/histograms. Never raises.
    """
    latency_seconds = time.time() - started_at

    try:
        from dark_factory.metrics.rates import compute_cost_usd

        cost_usd = compute_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_input_tokens,
            cache_creation_tokens=cache_creation_input_tokens,
        )
    except Exception:  # pragma: no cover — defensive
        cost_usd = None

    # Prometheus (always on)
    try:
        from dark_factory.metrics.prometheus import observe_llm_call

        observe_llm_call(
            client=client,
            model=model,
            latency_seconds=latency_seconds,
            time_to_first_token_seconds=time_to_first_token_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cost_usd=cost_usd,
            retry_count=retry_count,
            error=error,
            rate_limited=rate_limited,
            http_status=http_status,
        )
    except Exception:  # pragma: no cover — defensive
        pass

    # Postgres (optional)
    try:
        from dark_factory.agents import tools as _tools_mod

        recorder = _tools_mod._metrics_recorder
        if recorder is None:
            return
        recorder.record_llm_call(
            client=client,
            model=model,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
            system_prompt_chars=system_prompt_chars,
            max_tokens_requested=max_tokens_requested,
            temperature=temperature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            latency_seconds=latency_seconds,
            time_to_first_token_seconds=time_to_first_token_seconds,
            retry_count=retry_count,
            stop_reason=stop_reason,
            http_status=http_status,
            rate_limited=rate_limited,
            cost_usd=cost_usd,
            error=error,
        )
    except Exception:  # pragma: no cover — defensive
        pass


def _extract_json(raw: str) -> str:
    """Extract JSON from LLM response, handling markdown fences and surrounding text.

    H8 fix: when there's no fenced block, scan for the LARGEST balanced JSON
    object (matching braces) instead of greedily grabbing from the first ``{``.
    This avoids cases like "Here is [1,2,3] of options..." where the first
    bracket is inside prose, not the actual structured payload.
    """
    text = raw.strip()
    # Prefer a fenced JSON block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # Otherwise find the first balanced {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 32768,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        log.debug("llm_complete", model=self.model, prompt_len=len(prompt))
        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else DEFAULT_LLM_TIMEOUT_SECONDS
        )
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            # H4 fix: bound the call explicitly. Without this, the
            # Anthropic SDK falls back to its own (long) default and
            # a hung call can block a worker thread indefinitely.
            "timeout": effective_timeout,
        }
        if system:
            kwargs["system"] = system

        # Use streaming mode — Anthropic requires it when max_tokens is high
        # or requests may exceed the 10-minute non-streaming limit.
        # Iterate text_stream explicitly to guarantee we capture every chunk.
        started_at = time.time()
        first_token_at: float | None = None
        parts: list[str] = []
        stop_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_read: int | None = None
        cache_create: int | None = None
        error: str | None = None
        http_status: int | None = None
        rate_limited = False

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if first_token_at is None and text:
                        first_token_at = time.time()
                    parts.append(text)
                # Log stop_reason in case of truncation
                final = stream.get_final_message()
                stop_reason = final.stop_reason
                input_tokens = getattr(final.usage, "input_tokens", None)
                output_tokens = getattr(final.usage, "output_tokens", None)
                # Prompt caching fields (available on recent Anthropic SDK versions)
                cache_read = getattr(final.usage, "cache_read_input_tokens", None)
                cache_create = getattr(final.usage, "cache_creation_input_tokens", None)
                if stop_reason not in (None, "end_turn"):
                    log.warning(
                        "llm_stop_reason",
                        stop_reason=stop_reason,
                        output_tokens=output_tokens,
                        max_tokens=self.max_tokens,
                    )
        except Exception as exc:
            error = str(exc)
            # Extract HTTP status + rate-limit flag from anthropic SDK exceptions
            http_status = getattr(exc, "status_code", None)
            rate_limited = isinstance(exc, anthropic.RateLimitError) or (
                http_status == 429
            )
            _record_llm_call(
                client="anthropic",
                model=self.model,
                prompt_chars=len(prompt),
                system_prompt_chars=len(system) if system else None,
                max_tokens_requested=self.max_tokens,
                started_at=started_at,
                http_status=http_status,
                rate_limited=rate_limited,
                error=error,
            )
            # Record as an incident too for the Incidents panel.
            try:
                from dark_factory.metrics.helpers import record_incident

                record_incident(
                    category="llm",
                    severity="error",
                    message=f"{type(exc).__name__}: {error}"[:500],
                    phase="llm_call",
                )
            except Exception:  # pragma: no cover
                pass
            raise

        completion = "".join(parts)
        ttft: float | None = None
        if first_token_at is not None:
            ttft = first_token_at - started_at
        _record_llm_call(
            client="anthropic",
            model=self.model,
            prompt_chars=len(prompt),
            completion_chars=len(completion),
            system_prompt_chars=len(system) if system else None,
            max_tokens_requested=self.max_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_create,
            time_to_first_token_seconds=ttft,
            stop_reason=stop_reason,
            started_at=started_at,
        )
        return completion

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        """Return a validated Pydantic model via Anthropic's native tool_use.

        This used to ask the model to "respond with ONLY valid JSON" and then
        parse the free-form streamed text. That approach was fragile — if the
        model stopped mid-string or emitted any malformed JSON the parser
        died with ``"EOF while parsing a string"``, even though the rest of
        the response would have been fine.

        The native tool_use path is bulletproof for this case:

        1. A synthetic tool named ``return_structured_output`` is declared
           with ``input_schema`` derived from the Pydantic model (with all
           ``$defs`` inlined so nested models like ``Spec → Scenario`` work).
        2. ``tool_choice`` is forced to that tool so the model MUST emit a
           tool_use block rather than free-form text.
        3. The Anthropic SDK's ``messages.stream()`` helper reassembles the
           ``input_json_delta`` events into a complete JSON blob before
           returning ``get_final_message()``. Truncation-mid-string errors
           that haunted the text-based parser are eliminated by construction.
        4. We pull the ``tool_use`` block out of the final message's content
           and hand it directly to ``response_model.model_validate``.

        A single retry covers transient SDK / network errors.
        """
        tool_name = "return_structured_output"
        tool_schema = _build_tool_schema(response_model)

        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else DEFAULT_LLM_TIMEOUT_SECONDS
        )

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": tool_name,
                    "description": (
                        f"Return the {response_model.__name__} payload as "
                        f"structured tool input. ALWAYS call this tool; "
                        f"do not reply with plain text."
                    ),
                    "input_schema": tool_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
            # H4 fix: explicit timeout so a hung structured completion
            # can't block a worker thread indefinitely. 300s default
            # (see DEFAULT_LLM_TIMEOUT_SECONDS) is well below the
            # deep-agent timeout (600s) because these calls are
            # meant to be fast.
            "timeout": effective_timeout,
        }
        if system:
            kwargs["system"] = system

        last_error: Exception | None = None
        for attempt in range(2):
            started_at = time.time()
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    final = stream.get_final_message()

                stop_reason = getattr(final, "stop_reason", None)
                input_tokens = getattr(final.usage, "input_tokens", None)
                output_tokens = getattr(final.usage, "output_tokens", None)
                cache_read = getattr(final.usage, "cache_read_input_tokens", None)
                cache_create = getattr(
                    final.usage, "cache_creation_input_tokens", None
                )

                tool_input = _extract_tool_input(final)
                if tool_input is None:
                    block_types = [
                        getattr(b, "type", "?")
                        for b in (getattr(final, "content", None) or [])
                    ]
                    raise RuntimeError(
                        f"Anthropic returned no tool_use block "
                        f"(stop_reason={stop_reason}, blocks={block_types})"
                    )

                _record_llm_call(
                    client="anthropic",
                    model=self.model,
                    prompt_chars=len(prompt),
                    system_prompt_chars=len(system) if system else None,
                    max_tokens_requested=self.max_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_input_tokens=cache_read,
                    cache_creation_input_tokens=cache_create,
                    stop_reason=stop_reason,
                    retry_count=attempt,
                    started_at=started_at,
                )

                return response_model.model_validate(tool_input)
            except Exception as exc:
                last_error = exc
                http_status = getattr(exc, "status_code", None)
                rate_limited = isinstance(exc, anthropic.RateLimitError) or (
                    http_status == 429
                )
                _record_llm_call(
                    client="anthropic",
                    model=self.model,
                    prompt_chars=len(prompt),
                    system_prompt_chars=len(system) if system else None,
                    max_tokens_requested=self.max_tokens,
                    started_at=started_at,
                    http_status=http_status,
                    rate_limited=rate_limited,
                    retry_count=attempt,
                    error=str(exc)[:500],
                )

                # Classify the error: only retry on transient failures
                # (rate limits, 5xx server errors, network glitches).
                # Auth errors, 4xx validation errors, and invalid-request
                # errors will never succeed on retry — bail immediately
                # rather than burning a second API call and doubling the
                # latency on permanent failures.
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
                log.warning(
                    "structured_tool_use_retry",
                    attempt=attempt + 1,
                    model=self.model,
                    response_model=response_model.__name__,
                    transient=is_transient,
                    error=str(exc)[:500],
                )
                if not is_transient:
                    # Permanent failure — don't retry, re-raise now with
                    # the original exception for the caller's stack.
                    raise

        assert last_error is not None
        raise last_error
