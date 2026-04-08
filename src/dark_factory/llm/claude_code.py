"""Claude Agent SDK client implementation."""

from __future__ import annotations

import asyncio
import json
from typing import TypeVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from pydantic import BaseModel

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

    def complete(self, prompt: str, system: str | None = None) -> str:
        return asyncio.run(self._complete_async(prompt, system))

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
