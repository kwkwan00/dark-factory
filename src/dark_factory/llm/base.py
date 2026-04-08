"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Provider-agnostic interface for LLM interactions."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and return the raw text response."""
        ...

    @abstractmethod
    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        """Send a prompt and return a validated Pydantic model."""
        ...
