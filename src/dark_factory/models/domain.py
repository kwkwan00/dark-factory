"""Domain models that flow through the pipeline stages."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Requirement(BaseModel):
    """A single requirement extracted from a requirements document."""

    id: str
    title: str
    description: str
    source_file: str
    priority: Priority = Priority.MEDIUM
    tags: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    """A WHEN/THEN scenario used in OpenSpec specifications."""

    name: str
    when: str
    then: str


class Spec(BaseModel):
    """A specification derived from one or more requirements."""

    id: str
    title: str
    description: str
    requirement_ids: list[str]
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list, description="IDs of specs this depends on")
    scenarios: list[Scenario] = Field(default_factory=list)
    capability: str = ""


class CodeArtifact(BaseModel):
    """A generated code file."""

    id: str
    spec_id: str
    file_path: str
    language: str
    content: str


class TestCase(BaseModel):
    """A generated test for a code artifact."""

    id: str
    artifact_id: str
    test_type: str = "unit"  # unit, integration, eval
    file_path: str
    content: str


class PipelineContext(BaseModel):
    """Accumulator passed between pipeline stages."""

    input_path: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    specs: list[Spec] = Field(default_factory=list)
    artifacts: list[CodeArtifact] = Field(default_factory=list)
    tests: list[TestCase] = Field(default_factory=list)
