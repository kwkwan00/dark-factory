"""Tests for the OpenSpec parser."""

from __future__ import annotations

from pathlib import Path

from dark_factory.openspec.parser import parse_openspec_dir, parse_openspec_specs, parse_spec_md


def _write_spec(tmp_path: Path, capability: str, content: str) -> Path:
    spec_dir = tmp_path / "specs" / capability
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / "spec.md"
    spec_file.write_text(content)
    return spec_file


SAMPLE_SPEC = """\
## ADDED Requirements

### Requirement: User Login
Users must be able to log in with email and password.

#### Scenario: Successful login
**WHEN** the user submits valid credentials
**THEN** a session token is returned

#### Scenario: Invalid password
**WHEN** the user submits a wrong password
**THEN** an error message is displayed
"""


def test_parse_spec_md(tmp_path: Path) -> None:
    spec_file = _write_spec(tmp_path, "auth-login", SAMPLE_SPEC)
    reqs = parse_spec_md(spec_file, capability="auth-login")

    assert len(reqs) == 1
    assert reqs[0].title == "User Login"
    assert "auth-login" in reqs[0].tags
    assert "openspec" in reqs[0].tags
    assert "valid credentials" in reqs[0].description
    assert "wrong password" in reqs[0].description


def test_parse_openspec_dir(tmp_path: Path) -> None:
    _write_spec(tmp_path, "auth-login", SAMPLE_SPEC)
    _write_spec(
        tmp_path,
        "user-profile",
        """\
## ADDED Requirements

### Requirement: View Profile
Users can view their profile page.

#### Scenario: Profile loads
**WHEN** the user navigates to /profile
**THEN** the profile data is displayed
""",
    )

    reqs = parse_openspec_dir(tmp_path)
    assert len(reqs) == 2
    titles = {r.title for r in reqs}
    assert titles == {"User Login", "View Profile"}


def test_parse_openspec_dir_empty(tmp_path: Path) -> None:
    (tmp_path / "specs").mkdir()
    assert parse_openspec_dir(tmp_path) == []


def test_parse_openspec_dir_no_specs_dir(tmp_path: Path) -> None:
    assert parse_openspec_dir(tmp_path) == []


def test_parse_openspec_specs(tmp_path: Path) -> None:
    _write_spec(tmp_path, "auth-login", SAMPLE_SPEC)
    specs = parse_openspec_specs(tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.title == "User Login"
    assert spec.capability == "auth-login"
    assert len(spec.scenarios) == 2
    assert spec.scenarios[0].name == "Successful login"
    assert spec.scenarios[0].when == "the user submits valid credentials"
    assert spec.scenarios[0].then == "a session token is returned"
    assert len(spec.acceptance_criteria) == 2


def test_parse_spec_md_no_scenarios(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path,
        "basic",
        """\
## ADDED Requirements

### Requirement: Simple Feature
Just a plain requirement with no scenarios.
""",
    )
    reqs = parse_spec_md(spec_file, capability="basic")
    assert len(reqs) == 1
    assert reqs[0].title == "Simple Feature"
    assert "Just a plain requirement" in reqs[0].description
