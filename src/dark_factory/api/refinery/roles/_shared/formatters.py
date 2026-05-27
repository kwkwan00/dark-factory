"""Evidence formatters used by multiple role prompts.

These are pure text formatters over the run-context evidence bag. They
were previously private helpers in ``refinery/prompts.py`` (the
``_fmt_*`` functions); extracted here so each role can pull the slice
it needs without re-implementing them.

The ``prompts.py`` module keeps thin wrappers around these for backward
compatibility with any caller that imported the old private names.
"""

from __future__ import annotations

import json


def fmt_requirements(reqs: list[dict]) -> str:
    lines = []
    for r in reqs:
        lines.append(
            f"- [{r.get('id', '?')}] {r.get('title', '?')} "
            f"(priority={r.get('priority', '?')}, tags={r.get('tags', [])})\n"
            f"  Description: {r.get('description', '?')}"
        )
    return "\n".join(lines)


def fmt_traceability(trace: dict | None) -> str:
    if not trace:
        return "No traceability data available."
    rows = trace.get("rows", [])
    if not rows:
        return "No traceability rows."
    lines = []
    for row in rows[:50]:
        req = row.get("requirement", {})
        status = row.get("overall_status", "?")
        specs = row.get("specs", [])
        spec_summary = ", ".join(
            f"{s.get('id', '?')}({s.get('capability', '?')}, "
            f"passed={s.get('all_passed', '?')})"
            for s in specs
        )
        lines.append(
            f"- Req [{req.get('id', '?')}] {req.get('title', '?')} → "
            f"status={status}, specs=[{spec_summary}]"
        )
    return "\n".join(lines)


def fmt_gaps(gaps: dict | None) -> str:
    if not gaps:
        return "No gap data available."
    sections = []
    for key in [
        "unimplemented_requirements",
        "specs_without_artifacts",
        "specs_failing_evals",
        "broken_dependencies",
        "capability_islands",
        "missing_episodes",
    ]:
        items = gaps.get(key, [])
        if items:
            sections.append(f"{key}: {len(items)} items")
            for item in items[:10]:
                sections.append(f"  - {json.dumps(item, default=str)[:200]}")
    return "\n".join(sections) if sections else "No gaps found."


def fmt_episodes(episodes: list) -> str:
    if not episodes:
        return "No episodes available."
    lines = []
    for ep in episodes[:20]:
        lines.append(
            f"- Feature: {ep.get('feature', '?')}, "
            f"outcome: {ep.get('outcome', '?')}, "
            f"summary: {(ep.get('summary') or '')[:200]}"
        )
    return "\n".join(lines)


def fmt_learnings(learnings: list) -> str:
    if not learnings:
        return "No procedural memories from this run."
    lines = []
    for m in learnings[:20]:
        lines.append(
            f"- [{m.get('type', '?')}] {(m.get('description') or '')[:150]} "
            f"(feature={m.get('source_feature', '?')})"
        )
    return "\n".join(lines)


def fmt_evals(evals: list) -> str:
    if not evals:
        return "No eval data available."
    lines = []
    for run_eval in evals[:5]:
        for spec_eval in (run_eval.get("specs") or [])[:10]:
            sid = spec_eval.get("spec_id", "?")
            for ev in (spec_eval.get("evals") or [])[:3]:
                score = ev.get("overall_score", "?")
                passed = ev.get("all_passed", "?")
                metrics = ev.get("metrics") or []
                failing = [m for m in metrics if not m.get("passed", True)]
                if failing:
                    reasons = "; ".join(
                        f"{m.get('name', '?')}: {(m.get('reason') or '')[:100]}"
                        for m in failing
                    )
                    lines.append(
                        f"- Spec {sid}: score={score}, passed={passed}, "
                        f"failures: [{reasons}]"
                    )
    return "\n".join(lines) if lines else "All evals passed."
