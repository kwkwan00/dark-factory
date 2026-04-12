"""Claude Agent SDK client implementation."""

from __future__ import annotations

import json
import time
from typing import TypeVar

import structlog

log = structlog.get_logger()


def _record_claude_code_call(
    *,
    model: str,
    started_at: float,
    prompt_chars: int | None = None,
    system_prompt_chars: int | None = None,
    completion_chars: int | None = None,
    error: str | None = None,
) -> None:
    """Best-effort write to BOTH Prometheus and the Postgres recorder.

    The Claude Agent SDK doesn't surface token usage cleanly from the Python
    query() helper, so this records latency/chars/error only. Cost can't be
    computed without token counts — the cost_usd column will be NULL for
    claude_code calls until the SDK adds structured usage to its messages.
    """
    latency_seconds = time.time() - started_at

    try:
        from dark_factory.metrics.prometheus import observe_llm_call

        observe_llm_call(
            client="claude_code",
            model=model,
            latency_seconds=latency_seconds,
            error=error,
        )
    except Exception:  # pragma: no cover
        pass

    try:
        from dark_factory.agents import tools as _tools_mod

        recorder = _tools_mod._metrics_recorder
        if recorder is None:
            return
        recorder.record_llm_call(
            client="claude_code",
            model=model,
            prompt_chars=prompt_chars,
            system_prompt_chars=system_prompt_chars,
            completion_chars=completion_chars,
            latency_seconds=latency_seconds,
            error=error,
        )
    except Exception:  # pragma: no cover
        pass

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from pydantic import BaseModel

from dark_factory.agents.background_loop import BackgroundLoop
from dark_factory.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class ClaudeAgentClient(LLMClient):
    """LLM client that delegates to the Claude Agent SDK."""

    def __init__(self, model: str | None = None, cwd: str | None = None) -> None:
        self.model = model
        self.cwd = cwd

    def _options(self, system: str | None = None) -> ClaudeAgentOptions:
        opts = ClaudeAgentOptions()
        if self.model:
            opts.model = self.model
        if system:
            opts.system_prompt = system
        if self.cwd:
            opts.cwd = self.cwd
        return opts

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        log.debug("claude_agent_complete", prompt_len=len(prompt))
        # Run on the long-lived background loop so subprocess cleanup
        # callbacks land on a loop that stays alive (avoids the
        # "Event loop is closed" error from per-call asyncio.run).
        # H4 fix: bound the BackgroundLoop call explicitly so a hung
        # SDK invocation can't stall a worker thread forever.
        from dark_factory.llm.base import DEFAULT_LLM_TIMEOUT_SECONDS

        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else DEFAULT_LLM_TIMEOUT_SECONDS
        )
        started_at = time.time()
        try:
            result = BackgroundLoop.get().run(
                self._complete_async(prompt, system),
                timeout=effective_timeout,
            )
        except Exception as exc:
            _record_claude_code_call(
                model=self.model or "claude-code-default",
                started_at=started_at,
                prompt_chars=len(prompt),
                system_prompt_chars=len(system) if system else None,
                error=str(exc),
            )
            raise
        _record_claude_code_call(
            model=self.model or "claude-code-default",
            started_at=started_at,
            prompt_chars=len(prompt),
            system_prompt_chars=len(system) if system else None,
            completion_chars=len(result) if isinstance(result, str) else None,
        )
        return result

    async def _complete_async(self, prompt: str, system: str | None = None) -> str:
        texts: list[str] = []
        async for message in query(prompt=prompt, options=self._options(system)):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)
            elif isinstance(message, ResultMessage) and message.result:
                texts.append(message.result)
        return "\n".join(texts)

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        schema = response_model.model_json_schema()
        structured_prompt = (
            f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        # H6 fix: robust JSON extraction with retry
        from dark_factory.llm.anthropic import _extract_json

        last_error: Exception | None = None
        for attempt in range(2):
            raw = self.complete(
                structured_prompt,
                system=system,
                timeout_seconds=timeout_seconds,
            )
            text = _extract_json(raw)
            try:
                return response_model.model_validate_json(text)
            except Exception as exc:
                last_error = exc
                log.warning("structured_parse_retry", attempt=attempt + 1, error=str(exc))
        assert last_error is not None
        raise last_error
