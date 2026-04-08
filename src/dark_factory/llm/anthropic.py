"""Anthropic LLM client implementation."""

from __future__ import annotations

import json
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from dark_factory.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        schema = response_model.model_json_schema()
        structured_prompt = (
            f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        raw = self.complete(structured_prompt, system=system)

        # Extract JSON from response (handle markdown code blocks)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        return response_model.model_validate_json(text)
