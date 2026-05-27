"""Shared formatter for Phase A rule findings.

The Judge's ``score`` (via ``DeepEvalJudge``) and ``defend`` both need
to render rule findings in the same shape — score uses them as scoring
ground truth, defend uses them as already-accepted critiques. Sharing
the renderer prevents the two surfaces from drifting.
"""

from __future__ import annotations

from typing import Any


def render_rule_findings_block(findings: Any) -> str:
    """Pipe-delimited bullet list of rule findings. Empty-list / None /
    unrecognised shape → ``"(no rule findings)"``. Defensive against
    legacy dict shapes that pre-date ``RuleViolation``."""

    if not findings:
        return "(no rule findings)"
    if not isinstance(findings, list):
        return str(findings)

    lines: list[str] = []
    for f in findings:
        if isinstance(f, dict):
            sev = str(f.get("severity", "?"))
            dim = str(f.get("dimension", "?"))
            msg = str(f.get("finding", ""))
            fix = str(f.get("suggested_fix", ""))
        else:
            sev = getattr(f, "severity", "?")
            dim = getattr(f, "dimension", "?")
            msg = getattr(f, "finding", "")
            fix = getattr(f, "suggested_fix", "")
            sev = sev.value if hasattr(sev, "value") else str(sev)
            dim = dim.value if hasattr(dim, "value") else str(dim)
        line = f"- [{sev.upper()} / {dim}] {msg}"
        if fix:
            line += f" — fix: {fix}"
        lines.append(line)
    return "\n".join(lines)
