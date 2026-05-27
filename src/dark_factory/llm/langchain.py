"""LangChain-based LLM client using ChatAnthropic."""

from __future__ import annotations

import time
from typing import TypeVar

import structlog
from langchain_anthropic import ChatAnthropic

log = structlog.get_logger()
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from dark_factory.llm.anthropic import _backoff_sleep, _is_transient, _record_llm_call
from dark_factory.llm.base import DEFAULT_MODEL, LLMClient
from dark_factory.log import trace_methods

T = TypeVar("T", bound=BaseModel)


@trace_methods
class LangChainClient(LLMClient):
    """LLM client backed by LangChain's ChatAnthropic."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        kwargs: dict = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        self.llm = ChatAnthropic(**kwargs)
        self.model = model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        log.debug("langchain_complete", prompt_len=len(prompt))
        from dark_factory.llm.base import DEFAULT_LLM_TIMEOUT_SECONDS

        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else DEFAULT_LLM_TIMEOUT_SECONDS
        )
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        started_at = time.time()
        # H4 fix: langchain's ``invoke`` accepts a ``config={"timeout":
        # seconds}`` runnable config. Pass the effective timeout so
        # hung calls fail fast rather than blocking a worker thread.
        invoke_config = {"timeout": effective_timeout}
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = self.llm.invoke(messages, config=invoke_config)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                _record_llm_call(
                    client="langchain",
                    model=self.model,
                    started_at=started_at,
                    prompt_chars=len(prompt),
                    system_prompt_chars=len(system) if system else None,
                    error=str(exc),
                )
                if not _is_transient(exc):
                    raise
                if attempt < 1:
                    log.warning(
                        "langchain_complete_retry",
                        attempt=attempt + 1,
                        error=str(exc)[:200],
                    )
                    _backoff_sleep(attempt, exc)
        if last_exc is not None:
            raise last_exc
        content = response.content
        # LangChain sometimes returns a list of content chunks
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        usage = getattr(response, "usage_metadata", None) or {}
        _record_llm_call(
            client="langchain",
            model=self.model,
            started_at=started_at,
            prompt_chars=len(prompt),
            system_prompt_chars=len(system) if system else None,
            completion_chars=len(content) if isinstance(content, str) else None,
            input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
        )
        return content

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        # H6 parity fix: retry once on structured-output failure so this client
        # behaves like AnthropicClient / ClaudeAgentClient. `with_structured_output`
        # surfaces Pydantic validation errors from the parsing layer — a one-shot
        # failure on a flaky model is recoverable with a single retry.
        from dark_factory.llm.base import DEFAULT_LLM_TIMEOUT_SECONDS

        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else DEFAULT_LLM_TIMEOUT_SECONDS
        )
        structured_llm = self.llm.with_structured_output(response_model)
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        invoke_config = {"timeout": effective_timeout}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return structured_llm.invoke(messages, config=invoke_config)
            except Exception as exc:
                last_error = exc
                transient = _is_transient(exc)
                log.warning(
                    "structured_parse_retry",
                    client="langchain",
                    attempt=attempt + 1,
                    transient=transient,
                    error=str(exc),
                )
                if not transient:
                    raise
                if attempt < 1:
                    _backoff_sleep(attempt, exc)
        assert last_error is not None
        raise last_error
