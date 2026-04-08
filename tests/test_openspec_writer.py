"""Tests for the OpenSpec writer."""

from __future__ import annotations

from pathlib import Path

from dark_factory.models.domain import Scenario, Spec
from dark_factory.openspec.writer import (
    archive_change,
    init_openspec_dir,
    write_change_specs,
    write_design,
    write_proposal,
    write_spec_md,
    write_tasks,
)


def _sample_spec() -> Spec:
    return Spec(
        id="spec-abc123",
        title="User Authentication",
        description="Implement email/password authentication.",
        requirement_ids=["abc123"],
        acceptance_criteria=[
            "WHEN valid credentials THEN session created",
            "WHEN invalid password THEN error shown",
        ],
        scenarios=[
            Scenario(name="Successful login", when="valid credentials submitted", then="session created"),
            Scenario(name="Invalid password", when="wrong password submitted", then="error shown"),
        ],
        capability="user-auth",
    )


def test_init_openspec_dir(tmp_path: Path) -> None:
    root = tmp_path / "openspec"
    init_openspec_dir(root)
    assert (root / "specs").is_dir()
    assert (root / "changes").is_dir()


def test_write_spec_md(tmp_path: Path) -> None:
    spec = _sample_spec()
    path = write_spec_md(spec, tmp_path)

    assert path == tmp_path / "specs" / "user-auth" / "spec.md"
    assert path.exists()

    content = path.read_text()
    assert "### Requirement: User Authentication" in content
    assert "**WHEN** valid credentials submitted" in content
    assert "**THEN** session created" in content
    assert "#### Scenario: Successful login" in content


def test_write_proposal(tmp_path: Path) -> None:
    spec = _sample_spec()
    path = write_proposal("my-change", [spec], "Add user authentication", tmp_path)

    assert path.exists()
    content = path.read_text()
    assert "Add user authentication" in content
    assert "User Authentication" in content


def test_write_design(tmp_path: Path) -> None:
    spec = _sample_spec()
    path = write_design("my-change", [spec], tmp_path)

    assert path.exists()
    content = path.read_text()
    assert "User Authentication" in content
    assert "Goals" in content


def test_write_tasks(tmp_path: Path) -> None:
    spec = _sample_spec()
    path = write_tasks("my-change", [spec], tmp_path)

    assert path.exists()
    content = path.read_text()
    assert "- [ ]" in content
    assert "User Authentication" in content


def test_write_change_specs(tmp_path: Path) -> None:
    spec = _sample_spec()
    out_dir = write_change_specs("my-change", [spec], tmp_path)

    spec_file = out_dir / "user-auth" / "spec.md"
    assert spec_file.exists()
    assert "### Requirement:" in spec_file.read_text()


def test_archive_change(tmp_path: Path) -> None:
    change_dir = tmp_path / "changes" / "my-change"
    change_dir.mkdir(parents=True)
    (change_dir / "proposal.md").write_text("test")

    dest = archive_change("my-change", tmp_path)
    assert dest == tmp_path / "archive" / "my-change"
    assert (dest / "proposal.md").exists()
    assert not change_dir.exists()


def test_round_trip(tmp_path: Path) -> None:
    """Write a spec, then parse it back and verify fidelity."""
    from dark_factory.openspec.parser import parse_openspec_specs

    spec = _sample_spec()
    write_spec_md(spec, tmp_path)

    parsed = parse_openspec_specs(tmp_path)
    assert len(parsed) == 1
    p = parsed[0]
    assert p.title == spec.title
    assert p.capability == spec.capability
    assert len(p.scenarios) == len(spec.scenarios)
    assert p.scenarios[0].when == spec.scenarios[0].when
    assert p.scenarios[0].then == spec.scenarios[0].then
