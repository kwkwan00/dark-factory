"""Tests for the ingest stage."""

from __future__ import annotations

import json
from pathlib import Path

from dark_factory.models.domain import PipelineContext
from dark_factory.stages.ingest import IngestStage


def test_ingest_json_file(tmp_path: Path) -> None:
    req_file = tmp_path / "reqs.json"
    req_file.write_text(
        json.dumps(
            [
                {
                    "id": "req-1",
                    "title": "Feature A",
                    "description": "Build feature A",
                    "source_file": "reqs.json",
                }
            ]
        )
    )

    stage = IngestStage()
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    assert len(result.requirements) == 1
    assert result.requirements[0].id == "req-1"


def test_ingest_text_file(tmp_path: Path) -> None:
    req_file = tmp_path / "feature.md"
    req_file.write_text("The system shall do X.")

    stage = IngestStage()
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    assert len(result.requirements) == 1
    assert result.requirements[0].title == "Feature"


def test_ingest_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("Requirement A")
    (tmp_path / "b.txt").write_text("Requirement B")

    stage = IngestStage()
    ctx = PipelineContext(input_path=str(tmp_path))
    result = stage.run(ctx)

    assert len(result.requirements) == 2


def test_ingest_openspec_directory(tmp_path: Path) -> None:
    """IngestStage auto-detects OpenSpec directory structure."""
    spec_dir = tmp_path / "specs" / "auth-login"
    spec_dir.mkdir(parents=True)
    (spec_dir / "spec.md").write_text(
        """\
## ADDED Requirements

### Requirement: Login Flow
Users log in with email and password.

#### Scenario: Valid credentials
**WHEN** the user provides correct email and password
**THEN** a session is created
"""
    )

    stage = IngestStage()
    ctx = PipelineContext(input_path=str(tmp_path))
    result = stage.run(ctx)

    assert len(result.requirements) == 1
    assert result.requirements[0].title == "Login Flow"
    assert "openspec" in result.requirements[0].tags
