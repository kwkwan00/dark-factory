"""Parse OpenSpec spec.md files into domain models."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from dark_factory.models.domain import Requirement, Scenario, Spec


def parse_openspec_dir(root: Path) -> list[Requirement]:
    """Walk ``root/specs/*/spec.md`` and return a flat list of Requirements."""
    specs_dir = root / "specs"
    if not specs_dir.is_dir():
        return []

    requirements: list[Requirement] = []
    for capability_dir in sorted(specs_dir.iterdir()):
        if not capability_dir.is_dir():
            continue
        spec_file = capability_dir / "spec.md"
        if spec_file.exists():
            reqs = parse_spec_md(spec_file, capability=capability_dir.name)
            requirements.extend(reqs)
    return requirements


def parse_spec_md(path: Path, capability: str) -> list[Requirement]:
    """Extract Requirement models from an OpenSpec ``spec.md`` file.

    Each ``### Requirement: <name>`` block becomes a Requirement.
    Scenario WHEN/THEN pairs are included in the description so downstream
    LLM stages have full context.
    """
    text = path.read_text()
    requirements: list[Requirement] = []

    # Split on requirement headers
    req_pattern = re.compile(r"^### Requirement:\s*(.+)$", re.MULTILINE)
    req_splits = req_pattern.split(text)

    # req_splits: [preamble, name1, body1, name2, body2, ...]
    for i in range(1, len(req_splits), 2):
        name = req_splits[i].strip()
        body = req_splits[i + 1] if i + 1 < len(req_splits) else ""

        scenarios = _extract_scenarios(body)
        description = _build_description(body, scenarios)

        req_id = hashlib.sha256(f"{capability}/{name}".encode()).hexdigest()[:12]
        requirements.append(
            Requirement(
                id=req_id,
                title=name,
                description=description,
                source_file=str(path),
                tags=[capability, "openspec"],
            )
        )
    return requirements


def parse_openspec_specs(root: Path) -> list[Spec]:
    """Parse OpenSpec spec.md files directly into Spec models (for ``apply``)."""
    specs_dir = root / "specs"
    if not specs_dir.is_dir():
        return []

    specs: list[Spec] = []
    for capability_dir in sorted(specs_dir.iterdir()):
        if not capability_dir.is_dir():
            continue
        spec_file = capability_dir / "spec.md"
        if not spec_file.exists():
            continue

        capability = capability_dir.name
        text = spec_file.read_text()
        req_pattern = re.compile(r"^### Requirement:\s*(.+)$", re.MULTILINE)
        req_splits = req_pattern.split(text)

        for i in range(1, len(req_splits), 2):
            name = req_splits[i].strip()
            body = req_splits[i + 1] if i + 1 < len(req_splits) else ""
            scenarios = _extract_scenarios(body)

            # Strip scenario blocks to get the plain description
            desc_text = re.sub(
                r"####\s+Scenario:.*?(?=####|\Z)", "", body, flags=re.DOTALL
            ).strip()

            spec_id = hashlib.sha256(f"{capability}/{name}".encode()).hexdigest()[:12]
            spec = Spec(
                id=f"spec-{spec_id}",
                title=name,
                description=desc_text or name,
                requirement_ids=[spec_id],
                scenarios=scenarios,
                acceptance_criteria=[f"WHEN {s.when} THEN {s.then}" for s in scenarios],
                capability=capability,
            )
            specs.append(spec)
    return specs


# ── Private helpers ───────────────────────────────────────────────────


def _extract_scenarios(body: str) -> list[Scenario]:
    """Extract ``#### Scenario:`` blocks with WHEN/THEN pairs."""
    scenario_pattern = re.compile(
        r"^####\s+Scenario:\s*(.+)$", re.MULTILINE
    )
    splits = scenario_pattern.split(body)
    scenarios: list[Scenario] = []

    for i in range(1, len(splits), 2):
        scenario_name = splits[i].strip()
        scenario_body = splits[i + 1] if i + 1 < len(splits) else ""

        when_match = re.search(r"\*\*WHEN\*\*\s*(.+?)(?=\*\*THEN\*\*|\Z)", scenario_body, re.DOTALL)
        then_match = re.search(r"\*\*THEN\*\*\s*(.+?)(?=\*\*WHEN\*\*|####|\Z)", scenario_body, re.DOTALL)

        when_text = when_match.group(1).strip() if when_match else ""
        then_text = then_match.group(1).strip() if then_match else ""

        if when_text or then_text:
            scenarios.append(Scenario(name=scenario_name, when=when_text, then=then_text))

    return scenarios


def _build_description(body: str, scenarios: list[Scenario]) -> str:
    """Build a rich description including requirement text and scenarios."""
    # Get text before the first scenario block
    desc_text = re.sub(
        r"####\s+Scenario:.*?(?=####|\Z)", "", body, flags=re.DOTALL
    ).strip()

    parts = [desc_text] if desc_text else []
    for s in scenarios:
        parts.append(f"Scenario: {s.name} — WHEN {s.when} THEN {s.then}")

    return "\n".join(parts)
