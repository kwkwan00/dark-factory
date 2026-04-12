"""Shared test fixtures."""

from __future__ import annotations

from typing import TypeVar

import pytest
from pydantic import BaseModel

from dark_factory.llm.base import LLMClient
from dark_factory.models.domain import PipelineContext, Priority, Requirement, Spec

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient(LLMClient):
    """LLM client that returns preconfigured responses for testing."""

    def __init__(self, responses: dict[type, BaseModel] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return "{}"

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        self.calls.append({"prompt": prompt, "system": system, "model": response_model})
        if response_model in self.responses:
            return self.responses[response_model]
        raise ValueError(f"No fake response configured for {response_model}")


@pytest.fixture
def sample_requirement() -> Requirement:
    return Requirement(
        id="req-001",
        title="User Authentication",
        description="The system shall support user authentication via email and password.",
        source_file="requirements/auth.md",
        priority=Priority.HIGH,
    )


@pytest.fixture
def sample_spec() -> Spec:
    return Spec(
        id="spec-req-001",
        title="Authentication Module",
        description="Implement email/password authentication with bcrypt hashing and JWT tokens.",
        requirement_ids=["req-001"],
        acceptance_criteria=[
            "Users can register with email and password",
            "Passwords are hashed with bcrypt",
            "Login returns a valid JWT token",
        ],
    )


@pytest.fixture
def sample_context(sample_requirement: Requirement) -> PipelineContext:
    return PipelineContext(
        input_path="tests/fixtures",
        requirements=[sample_requirement],
    )


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


# ── API test fixtures ──────────────────────────────────────────────────────────
# L7 fix: shared fixture with proper setup/teardown for API tests.
# The lifespan creates Neo4j clients — mock them to avoid needing a real DB.


@pytest.fixture
def api_client():
    """TestClient with mocked Neo4j/memory — safe to use without external services."""
    from unittest.mock import MagicMock, patch

    with (
        patch("dark_factory.graph.client.Neo4jClient") as _neo4j_cls,
        patch("dark_factory.memory.schema.init_memory_schema"),
        patch("dark_factory.memory.repository.MemoryRepository"),
    ):
        from dark_factory.api.app import app
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            yield client
        # TestClient exits lifespan here — cleanup runs automatically
