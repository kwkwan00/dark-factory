"""Abstract base class for pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dark_factory.models.domain import PipelineContext


class Stage(ABC):
    """A single step in the code generation pipeline."""

    name: str

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute this stage, returning an updated context."""
        ...

    def __repr__(self) -> str:
        return f"<Stage: {self.name}>"
