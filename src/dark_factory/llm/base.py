"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# Default per-call timeout for non-deep-agent LLM completions. Deep
# agents have their own dedicated timeout (``DEEP_AGENT_TIMEOUT_SECONDS``,
# default 600s) because their work is longer-running by design. Normal
# structured completions should finish much faster; 300s gives generous
# headroom for a slow network or a heavy completion without letting a
# hung request stall a worker thread indefinitely.
DEFAULT_LLM_TIMEOUT_SECONDS = 300.0


class LLMClient(ABC):
    """Provider-agnostic interface for LLM interactions."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        """Send a prompt and return the raw text response.

        :param timeout_seconds: per-call timeout. When ``None``, the
            implementation uses :data:`DEFAULT_LLM_TIMEOUT_SECONDS`.
        """
        ...

    @abstractmethod
    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        """Send a prompt and return a validated Pydantic model.

        :param timeout_seconds: per-call timeout. When ``None``, the
            implementation uses :data:`DEFAULT_LLM_TIMEOUT_SECONDS`.
        """
        ...
