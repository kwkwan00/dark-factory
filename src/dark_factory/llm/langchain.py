"""LangChain-based LLM client using ChatAnthropic."""

from __future__ import annotations

import json
from typing import TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from dark_factory.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class LangChainClient(LLMClient):
    """LLM client backed by LangChain's ChatAnthropic."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
        kwargs: dict = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        self.llm = ChatAnthropic(**kwargs)

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return response.content

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        structured_llm = self.llm.with_structured_output(response_model)
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        return structured_llm.invoke(messages)
