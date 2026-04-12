"""Security regression tests for path traversal and Cypher injection fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Path traversal in file tools ──────────────────────────────────────────────


def test_write_file_blocks_path_traversal(tmp_path: Path) -> None:
    """C3: write_file rejects paths that escape the output directory."""
    import dark_factory.agents.tools as tools_mod

    original = tools_mod._output_dir
    tools_mod._output_dir = tmp_path
    try:
        result = tools_mod.write_file.invoke(
            {"file_path": "../escaped.txt", "content": "pwned"}
        )
        assert "escapes" in result.lower()
        assert not (tmp_path.parent / "escaped.txt").exists()
    finally:
        tools_mod._output_dir = original


def test_write_file_blocks_absolute_path(tmp_path: Path) -> None:
    """C3: write_file rejects absolute paths outside the output dir."""
    import dark_factory.agents.tools as tools_mod

    original = tools_mod._output_dir
    tools_mod._output_dir = tmp_path
    try:
        result = tools_mod.write_file.invoke(
            {"file_path": "/tmp/evil.txt", "content": "pwned"}
        )
        assert "escapes" in result.lower()
    finally:
        tools_mod._output_dir = original


def test_write_file_allows_valid_paths(tmp_path: Path) -> None:
    """write_file still works for legitimate paths."""
    import dark_factory.agents.tools as tools_mod

    original = tools_mod._output_dir
    tools_mod._output_dir = tmp_path
    try:
        result = tools_mod.write_file.invoke(
            {"file_path": "src/main.py", "content": "print('hi')"}
        )
        assert "escapes" not in result.lower()
        assert (tmp_path / "src" / "main.py").exists()
        assert (tmp_path / "src" / "main.py").read_text() == "print('hi')"
    finally:
        tools_mod._output_dir = original


def test_read_file_blocks_path_traversal(tmp_path: Path) -> None:
    """C3: read_file rejects paths that escape the output directory."""
    import dark_factory.agents.tools as tools_mod

    # Create a file outside the output dir that should NOT be readable
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret data")

    original = tools_mod._output_dir
    tools_mod._output_dir = tmp_path
    try:
        result = tools_mod.read_file.invoke({"file_path": "../secret.txt"})
        assert "escapes" in result.lower()
        assert "secret data" not in result
    finally:
        tools_mod._output_dir = original
        outside.unlink(missing_ok=True)


def test_read_openspec_blocks_path_traversal(tmp_path: Path) -> None:
    """C4: read_openspec rejects capability values with path traversal."""
    import dark_factory.agents.tools as tools_mod

    original = tools_mod._openspec_root
    tools_mod._openspec_root = tmp_path
    try:
        result = tools_mod.read_openspec.invoke({"capability": "../../../etc"})
        assert "invalid" in result.lower() or "escape" in result.lower()
    finally:
        tools_mod._openspec_root = original


# ── Path traversal in codegen/testgen ─────────────────────────────────────────


def test_codegen_blocks_path_traversal(tmp_path: Path) -> None:
    """C5: codegen skips artifacts with paths that escape output_dir."""
    from dark_factory.models.domain import CodeArtifact, PipelineContext, Spec
    from dark_factory.stages.codegen import CodegenStage

    spec = Spec(
        id="spec-1",
        title="Test",
        description="A test spec",
        requirement_ids=["req-1"],
        acceptance_criteria=["criterion"],
    )
    malicious_artifact = CodeArtifact(
        id="a1",
        spec_id="spec-1",
        language="python",
        file_path="../../../etc/evil.py",
        content="pwned",
    )
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = malicious_artifact
    fake_repo = MagicMock()
    fake_repo.get_spec_with_context.return_value = "no deps"

    stage = CodegenStage(llm=fake_llm, repo=fake_repo, output_dir=str(tmp_path))
    ctx = PipelineContext(input_path="test", specs=[spec])
    stage.run(ctx)

    # The resolved malicious path should not have been written
    # (we check that nothing containing "evil.py" was created outside tmp_path)
    resolved_escape = (tmp_path / "../../../etc/evil.py").resolve()
    assert not resolved_escape.exists()


def test_testgen_blocks_path_traversal(tmp_path: Path) -> None:
    """C5: testgen skips test cases with paths that escape output_dir."""
    from dark_factory.models.domain import CodeArtifact, PipelineContext, Spec, TestCase
    from dark_factory.stages.testgen import TestgenStage

    spec = Spec(
        id="spec-1",
        title="Test",
        description="A test spec",
        requirement_ids=["req-1"],
        acceptance_criteria=["criterion"],
    )
    artifact = CodeArtifact(
        id="a1",
        spec_id="spec-1",
        language="python",
        file_path="src/main.py",
        content="def foo(): pass",
    )
    malicious_test = TestCase(
        id="t1",
        artifact_id="a1",
        framework="pytest",
        file_path="../../../tmp/evil_test.py",
        content="assert False",
    )
    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = malicious_test

    stage = TestgenStage(llm=fake_llm, output_dir=str(tmp_path))
    ctx = PipelineContext(input_path="test", specs=[spec], artifacts=[artifact])
    stage.run(ctx)

    resolved_escape = (tmp_path / "../../../tmp/evil_test.py").resolve()
    assert not resolved_escape.exists()


# ── Cypher injection mitigations ──────────────────────────────────────────────


def test_memory_boost_rejects_invalid_label() -> None:
    """C2: boost_relevance rejects labels not in the allowlist."""
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    repo = MemoryRepository(mock_client)
    # Malicious label that could be used for injection
    repo.boost_relevance("some-id", "Pattern; DROP DATABASE neo4j--")

    # Session should never be called for invalid labels
    mock_client.session.assert_not_called()


def test_memory_demote_rejects_invalid_label() -> None:
    """C2: demote_relevance rejects labels not in the allowlist."""
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    repo = MemoryRepository(mock_client)
    repo.demote_relevance("some-id", "NotAValidLabel")

    mock_client.session.assert_not_called()


def test_memory_boost_accepts_valid_labels() -> None:
    """C2: valid labels are accepted.

    H3 fix: boost_relevance now goes through ``session.execute_write``
    instead of ``session.run`` directly, so assert against the
    write-tx path. Four labels → four execute_write calls.
    """
    from dark_factory.memory.repository import MemoryRepository

    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client.session.return_value.__enter__.return_value = mock_session

    repo = MemoryRepository(mock_client)
    for label in ["Pattern", "Mistake", "Solution", "Strategy"]:
        repo.boost_relevance("id-1", label)

    assert mock_session.execute_write.call_count == 4


# ── Destructive function safety guards ───────────────────────────────────────


def test_clear_graph_requires_confirm() -> None:
    """M6: clear_graph raises without confirm=True."""
    from dark_factory.graph.schema import clear_graph

    with pytest.raises(ValueError, match="confirm=True"):
        clear_graph(MagicMock())


def test_clear_graph_with_confirm() -> None:
    """M6: clear_graph executes with confirm=True."""
    from dark_factory.graph.schema import clear_graph

    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client.session.return_value.__enter__.return_value = mock_session

    clear_graph(mock_client, confirm=True)
    mock_session.run.assert_called_once()
