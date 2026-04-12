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


def test_ingest_small_file_without_llm_single_requirement(tmp_path: Path) -> None:
    """Small files without an LLM produce a single requirement (unchanged behavior)."""
    req_file = tmp_path / "small.md"
    req_file.write_text("The system shall do X.")

    stage = IngestStage()  # no llm
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    assert len(result.requirements) == 1


def test_ingest_large_file_without_llm_single_requirement(tmp_path: Path) -> None:
    """Without an LLM, even large files fall back to single-requirement behavior."""
    req_file = tmp_path / "large.md"
    req_file.write_text("x" * 5000)

    stage = IngestStage()
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    assert len(result.requirements) == 1


def test_ingest_large_file_with_llm_splits_into_multiple(tmp_path: Path) -> None:
    """A large file with an LLM is split into multiple granular requirements."""
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    req_file.write_text("# PRD\n\n" + ("lots of content ... " * 200))

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(
                title="Feature A",
                description="System shall do A",
                priority="high",
                tags=["mvp"],
            ),
            _ExtractedRequirement(
                title="Feature B",
                description="System shall do B",
                priority="medium",
                tags=[],
            ),
            _ExtractedRequirement(
                title="Feature C",
                description="System shall do C",
                priority="low",
                tags=[],
            ),
        ]
    )

    stage = IngestStage(llm=fake_llm)
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    assert len(result.requirements) == 3
    assert result.requirements[0].title == "Feature A"
    assert result.requirements[0].priority.value == "high"
    assert result.requirements[1].title == "Feature B"
    assert result.requirements[2].title == "Feature C"
    # All should share the source file
    for req in result.requirements:
        assert req.source_file == str(req_file)
    # IDs should be unique
    assert len({r.id for r in result.requirements}) == 3


def test_ingest_llm_split_falls_back_on_error(tmp_path: Path) -> None:
    """If LLM splitting raises, the stage falls back to single-requirement."""
    from unittest.mock import MagicMock

    req_file = tmp_path / "prd.md"
    req_file.write_text("content " * 500)

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = RuntimeError("LLM down")

    stage = IngestStage(llm=fake_llm)
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    # Falls back to single requirement
    assert len(result.requirements) == 1


def test_ingest_strict_split_reraises_on_llm_error(tmp_path: Path) -> None:
    """M8: strict_split=True re-raises splitter exceptions instead of falling back."""
    from unittest.mock import MagicMock

    import pytest

    req_file = tmp_path / "prd.md"
    req_file.write_text("content " * 500)

    fake_llm = MagicMock()
    fake_llm.complete_structured.side_effect = RuntimeError("LLM down")

    stage = IngestStage(llm=fake_llm, strict_split=True)
    ctx = PipelineContext(input_path=str(req_file))
    with pytest.raises(RuntimeError, match="LLM down"):
        stage.run(ctx)


def test_ingest_llm_split_filters_blank_entries(tmp_path: Path) -> None:
    """M9: extracted requirements with blank titles or descriptions are dropped."""
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    req_file.write_text("x " * 1000)

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(title="Valid", description="has content"),
            _ExtractedRequirement(title="", description="no title"),
            _ExtractedRequirement(title="No description", description=""),
            _ExtractedRequirement(title="   ", description="whitespace only"),
        ]
    )

    stage = IngestStage(llm=fake_llm)
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)
    assert len(result.requirements) == 1
    assert result.requirements[0].title == "Valid"


def test_ingest_llm_split_ids_stable_across_reordered_llm_output(
    tmp_path: Path,
) -> None:
    """Previously the requirement id hashed ``path.name + i + title``,
    where ``i`` was the LLM splitter's output index. LLM ordering isn't
    bit-stable across calls, so re-running the pipeline on the same file
    produced different req_ids on every run — the downstream
    ``spec-{req_id}`` ids were also different, so the spec generation
    swarm would redo work on logically identical requirements and Neo4j
    would accumulate duplicate Spec nodes.

    The fix switches to content-based ids: ``sha256(title + '\\n' +
    description)`` (case-normalised). This test exercises the invariant:
    two ingest runs whose mocked splitter returns the SAME extracted
    requirements in a DIFFERENT order must produce identical ids per
    (title, description) pair.
    """
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    # Must exceed SPLIT_THRESHOLD_CHARS to take the LLM path.
    req_file.write_text("x " * 1000)

    items_in_order_a = [
        _ExtractedRequirement(title="Feature A", description="does A"),
        _ExtractedRequirement(title="Feature B", description="does B"),
        _ExtractedRequirement(title="Feature C", description="does C"),
    ]
    items_in_order_b = [
        _ExtractedRequirement(title="Feature C", description="does C"),
        _ExtractedRequirement(title="Feature A", description="does A"),
        _ExtractedRequirement(title="Feature B", description="does B"),
    ]

    # Run 1 — original order
    fake_llm_a = MagicMock()
    fake_llm_a.complete_structured.return_value = _RequirementList(
        requirements=items_in_order_a
    )
    stage_a = IngestStage(llm=fake_llm_a)
    ctx_a = PipelineContext(input_path=str(req_file))
    result_a = stage_a.run(ctx_a)

    # Run 2 — shuffled order (simulating LLM non-determinism)
    fake_llm_b = MagicMock()
    fake_llm_b.complete_structured.return_value = _RequirementList(
        requirements=items_in_order_b
    )
    stage_b = IngestStage(llm=fake_llm_b)
    ctx_b = PipelineContext(input_path=str(req_file))
    result_b = stage_b.run(ctx_b)

    # Both runs should produce 3 requirements with matching ids
    # keyed by (title, description). Compare by content to verify id
    # stability regardless of list ordering.
    ids_a = {(r.title, r.description): r.id for r in result_a.requirements}
    ids_b = {(r.title, r.description): r.id for r in result_b.requirements}
    assert ids_a == ids_b, (
        f"Requirement ids changed between runs despite identical content:\n"
        f"  run_a: {ids_a}\n  run_b: {ids_b}"
    )


def test_ingest_llm_split_ids_stable_under_case_and_whitespace(
    tmp_path: Path,
) -> None:
    """Content-based ids should be normalised: upper/lowercase and
    surrounding whitespace in the title or description don't change
    the id. Prevents ids from drifting when the LLM capitalises
    differently across runs."""
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    req_file.write_text("x " * 1000)

    fake_llm_a = MagicMock()
    fake_llm_a.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(title="Feature A", description="Does A")
        ]
    )
    fake_llm_b = MagicMock()
    fake_llm_b.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(title="  feature a  ", description="does a")
        ]
    )

    ctx_a = PipelineContext(input_path=str(req_file))
    ctx_b = PipelineContext(input_path=str(req_file))
    result_a = IngestStage(llm=fake_llm_a).run(ctx_a)
    result_b = IngestStage(llm=fake_llm_b).run(ctx_b)

    assert len(result_a.requirements) == 1
    assert len(result_b.requirements) == 1
    assert result_a.requirements[0].id == result_b.requirements[0].id


def test_ingest_llm_split_ids_differ_when_content_edited(
    tmp_path: Path,
) -> None:
    """Editing a requirement's description MUST produce a new id so the
    spec gets regenerated — the fix mustn't accidentally make ids too
    sticky."""
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    req_file.write_text("x " * 1000)

    fake_llm_a = MagicMock()
    fake_llm_a.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(title="Feature A", description="original")
        ]
    )
    fake_llm_b = MagicMock()
    fake_llm_b.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(title="Feature A", description="edited")
        ]
    )

    result_a = IngestStage(llm=fake_llm_a).run(
        PipelineContext(input_path=str(req_file))
    )
    result_b = IngestStage(llm=fake_llm_b).run(
        PipelineContext(input_path=str(req_file))
    )

    assert result_a.requirements[0].id != result_b.requirements[0].id


def test_ingest_llm_split_dedupes_identical_entries(tmp_path: Path) -> None:
    """M9: duplicate (title, description) pairs are dropped."""
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    req_file.write_text("x " * 1000)

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(title="Feature A", description="does A"),
            _ExtractedRequirement(title="Feature A", description="does A"),  # dupe
            _ExtractedRequirement(title="feature a", description="DOES A"),  # case dupe
            _ExtractedRequirement(title="Feature B", description="does B"),
        ]
    )

    stage = IngestStage(llm=fake_llm)
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)
    assert len(result.requirements) == 2
    titles = [r.title for r in result.requirements]
    assert "Feature A" in titles
    assert "Feature B" in titles


def test_ingest_llm_split_truncates_huge_document(tmp_path: Path) -> None:
    """M10: documents larger than MAX_SPLIT_INPUT_CHARS are truncated."""
    from unittest.mock import MagicMock

    from dark_factory.stages.ingest import (
        MAX_SPLIT_INPUT_CHARS,
        _ExtractedRequirement,
        _RequirementList,
    )

    req_file = tmp_path / "huge.md"
    # Well over the truncation threshold
    req_file.write_text("x" * (MAX_SPLIT_INPUT_CHARS + 10000))

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _RequirementList(
        requirements=[_ExtractedRequirement(title="F", description="d")]
    )

    stage = IngestStage(llm=fake_llm)
    ctx = PipelineContext(input_path=str(req_file))
    stage.run(ctx)

    # Inspect the prompt the LLM was called with — should be truncated
    prompt_arg = fake_llm.complete_structured.call_args.kwargs["prompt"]
    # The prompt contains overhead text too, so it's not exactly MAX_SPLIT_INPUT_CHARS
    # but the document content inside should be at most MAX_SPLIT_INPUT_CHARS
    assert "truncated" in prompt_arg.lower()


def test_ingest_llm_invalid_priority_defaults_to_medium(tmp_path: Path) -> None:
    """Invalid priority strings from the LLM default to MEDIUM."""
    from unittest.mock import MagicMock

    from dark_factory.models.domain import Priority
    from dark_factory.stages.ingest import _ExtractedRequirement, _RequirementList

    req_file = tmp_path / "prd.md"
    req_file.write_text("x " * 1000)

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = _RequirementList(
        requirements=[
            _ExtractedRequirement(
                title="Weird",
                description="...",
                priority="urgent-please",  # not a valid Priority value
            ),
        ]
    )

    stage = IngestStage(llm=fake_llm)
    ctx = PipelineContext(input_path=str(req_file))
    result = stage.run(ctx)

    assert result.requirements[0].priority == Priority.MEDIUM


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
