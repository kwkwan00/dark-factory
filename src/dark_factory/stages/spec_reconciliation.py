"""Stage 2b: Reconcile specs and requirements after spec generation.

Runs between spec generation and the knowledge-graph write to ensure
the specs form a coherent, complete, and correctly-linked dependency
graph before they are committed.

The reconciliation performs five checks:

1. **Requirement coverage** — every requirement must be implemented by
   at least one spec.  Uncovered requirements are flagged.
2. **Phantom requirement refs** — spec ``requirement_ids`` entries that
   don't match any known requirement are stripped.
3. **Cross-spec dependency completion** — an LLM pass reviews spec
   descriptions and acceptance criteria to surface implicit dependencies
   that the spec-gen planner missed, and removes refs to non-existent
   spec IDs.
4. **Circular dependency detection** — cycles in the dependency graph
   are detected and the back-edge is removed so topological ordering
   remains possible.
5. **Capability coherence** — specs that share a capability are
   validated for consistent scoping.

The stage is best-effort: if the LLM call fails the pipeline continues
with specs as-is (deterministic checks still run).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, Field

from dark_factory.agents.cancellation import PipelineCancelled
from dark_factory.log import trace_methods
from dark_factory.models.domain import PipelineContext, Requirement, Spec
from dark_factory.prompts import get_prompt
from dark_factory.stages.base import Stage

if TYPE_CHECKING:
    from dark_factory.llm.base import LLMClient

log = structlog.get_logger()


# ── Structured LLM output ────────────────────────────────────────────


class _SpecPatch(BaseModel):
    """LLM-proposed corrections for a single spec."""

    spec_id: str
    requirement_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    capability: str = ""


class _ReconciliationIssue(BaseModel):
    """A single issue detected during reconciliation."""

    severity: Literal["warning", "error"] = "warning"
    spec_id: str = ""
    message: str = ""


class _ReconciliationResult(BaseModel):
    """Full structured output from the reconciliation LLM call."""

    specs: list[_SpecPatch] = Field(default_factory=list)
    issues: list[_ReconciliationIssue] = Field(default_factory=list)


# ── Deterministic helpers ─────────────────────────────────────────────


def _detect_cycles(specs: list[Spec]) -> list[tuple[str, str]]:
    """Return back-edges that form cycles in the spec dependency graph.

    Uses iterative DFS with a colour map (white/grey/black).  Each
    returned tuple is ``(from_spec_id, to_spec_id)`` — the edge that
    should be removed to break the cycle.
    """
    adj: dict[str, list[str]] = {s.id: list(s.dependencies) for s in specs}
    all_ids = {s.id for s in specs}

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {sid: WHITE for sid in all_ids}
    back_edges: list[tuple[str, str]] = []

    for start in sorted(all_ids):
        if colour[start] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        colour[start] = GREY
        while stack:
            node, idx = stack[-1]
            neighbours = adj.get(node, [])
            if idx < len(neighbours):
                stack[-1] = (node, idx + 1)
                nbr = neighbours[idx]
                if nbr not in all_ids:
                    continue
                if colour[nbr] == WHITE:
                    colour[nbr] = GREY
                    stack.append((nbr, 0))
                elif colour[nbr] == GREY:
                    back_edges.append((node, nbr))
            else:
                colour[node] = BLACK
                stack.pop()

    return back_edges


def _strip_phantom_requirement_ids(
    specs: list[Spec], valid_req_ids: set[str],
) -> int:
    """Remove requirement_ids that don't exist.  Returns count removed."""
    removed = 0
    for spec in specs:
        original = spec.requirement_ids
        cleaned = [rid for rid in original if rid in valid_req_ids]
        if len(cleaned) < len(original):
            removed += len(original) - len(cleaned)
            spec.requirement_ids = cleaned
    return removed


def _strip_phantom_dependency_ids(specs: list[Spec]) -> int:
    """Remove dependency IDs that don't match any spec.  Returns count removed."""
    valid_ids = {s.id for s in specs}
    removed = 0
    for spec in specs:
        original = spec.dependencies
        cleaned = [d for d in original if d in valid_ids]
        if len(cleaned) < len(original):
            removed += len(original) - len(cleaned)
            spec.dependencies = cleaned
    return removed


def _find_uncovered_requirements(
    requirements: list[Requirement], specs: list[Spec],
) -> list[str]:
    """Return requirement IDs not referenced by any spec."""
    covered = set()
    for spec in specs:
        covered.update(spec.requirement_ids)
    return [r.id for r in requirements if r.id not in covered]


def _remove_back_edges(specs: list[Spec], back_edges: list[tuple[str, str]]) -> int:
    """Remove back-edges from specs to break cycles. Returns count removed."""
    edge_set = set(back_edges)
    removed = 0
    for spec in specs:
        original = spec.dependencies
        cleaned = [d for d in original if (spec.id, d) not in edge_set]
        if len(cleaned) < len(original):
            removed += len(original) - len(cleaned)
            spec.dependencies = cleaned
    return removed


def _find_capability_islands(specs: list[Spec]) -> list[dict[str, Any]]:
    """Find specs that share a capability but have no dependency path
    between them.

    Returns a list of ``{capability, island_a, island_b}`` dicts where
    ``island_a`` and ``island_b`` are groups of spec IDs within the same
    capability that are not connected by any dependency chain. These
    suggest a missing dependency edge.
    """
    # Group specs by capability (skip empty/singleton capabilities)
    by_cap: dict[str, list[str]] = {}
    for s in specs:
        if s.capability:
            by_cap.setdefault(s.capability, []).append(s.id)

    # Build adjacency (both directions for reachability)
    adj: dict[str, set[str]] = {s.id: set(s.dependencies) for s in specs}
    radj: dict[str, set[str]] = {s.id: set() for s in specs}
    for s in specs:
        for d in s.dependencies:
            if d in radj:
                radj[d].add(s.id)

    def _reachable(start: str, graph: dict[str, set[str]], scope: set[str]) -> set[str]:
        visited: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nbr in graph.get(node, ()):
                if nbr in scope and nbr not in visited:
                    stack.append(nbr)
        return visited

    islands: list[dict[str, Any]] = []
    for cap, spec_ids in by_cap.items():
        if len(spec_ids) < 2:
            continue
        scope = set(spec_ids)
        # Find connected components via bidirectional reachability
        remaining = set(spec_ids)
        components: list[set[str]] = []
        while remaining:
            seed = next(iter(remaining))
            forward = _reachable(seed, adj, scope)
            backward = _reachable(seed, radj, scope)
            component = (forward | backward) & scope
            components.append(component)
            remaining -= component
        if len(components) > 1:
            islands.append({
                "capability": cap,
                "components": [sorted(c) for c in components],
            })
    return islands


# ── LLM-assisted reconciliation ──────────────────────────────────────


_MAX_SPECS_IN_PROMPT = 120
"""Cap the number of specs sent to the LLM to keep the prompt within
context limits.  Specs beyond this limit still benefit from the
deterministic pass — only the LLM-assisted dependency discovery is
skipped for the tail."""


def _build_prompt(
    requirements: list[Requirement], specs: list[Spec],
) -> str:
    """Build the user prompt for the reconciliation LLM call."""
    req_lines = []
    for r in requirements:
        req_obj: dict[str, Any] = {
            "id": r.id, "title": r.title,
            "description": r.description[:500],
            "priority": r.priority.value if hasattr(r.priority, "value") else str(r.priority),
        }
        if r.tags:
            req_obj["tags"] = r.tags
        req_lines.append(json.dumps(req_obj))
    requirements_block = "\n".join(req_lines)

    capped = specs[:_MAX_SPECS_IN_PROMPT]
    spec_lines = []
    for s in capped:
        spec_obj: dict[str, Any] = {
            "id": s.id, "title": s.title,
            "description": s.description[:500],
            "requirement_ids": s.requirement_ids,
            "dependencies": s.dependencies,
            "acceptance_criteria": s.acceptance_criteria[:5],
            "capability": s.capability,
        }
        if s.scenarios:
            spec_obj["scenarios"] = [
                f"WHEN {sc.when} THEN {sc.then}" for sc in s.scenarios[:3]
            ]
        spec_lines.append(json.dumps(spec_obj))
    specs_block = "\n".join(spec_lines)
    if len(specs) > _MAX_SPECS_IN_PROMPT:
        specs_block += (
            f"\n\n(... {len(specs) - _MAX_SPECS_IN_PROMPT} additional specs "
            f"omitted for brevity — focus on the specs shown above)"
        )

    template = get_prompt("spec_reconciliation", "user")
    return template.format(
        requirements_block=requirements_block,
        specs_block=specs_block,
    )


def _apply_llm_patches(
    specs: list[Spec],
    patches: list[_SpecPatch],
    valid_req_ids: set[str],
) -> dict[str, Any]:
    """Apply LLM-suggested patches to specs.  Returns stats dict."""
    spec_map = {s.id: s for s in specs}
    valid_spec_ids = set(spec_map.keys())

    stats: dict[str, int] = {
        "req_ids_added": 0,
        "deps_added": 0,
        "capabilities_fixed": 0,
        "specs_patched": 0,
    }

    for patch in patches:
        spec = spec_map.get(patch.spec_id)
        if spec is None:
            continue

        changed = False

        # Reconcile requirement_ids — additive only.  The spec stage's
        # assignment is authoritative; the LLM can propose *new* links
        # but cannot strip links the spec stage assigned.
        new_req_ids = [r for r in patch.requirement_ids if r in valid_req_ids]
        old_set = set(spec.requirement_ids)
        added = set(new_req_ids) - old_set
        if added:
            spec.requirement_ids = spec.requirement_ids + sorted(added)
            stats["req_ids_added"] += len(added)
            changed = True

        # Reconcile dependencies — additive only.  Removals are handled
        # by the deterministic phantom-strip pass; the LLM can only
        # surface *missing* edges.
        new_deps = [d for d in patch.dependencies
                    if d in valid_spec_ids and d != spec.id]
        old_deps = set(spec.dependencies)
        deps_added = set(new_deps) - old_deps
        if deps_added:
            spec.dependencies = spec.dependencies + sorted(deps_added)
            stats["deps_added"] += len(deps_added)
            changed = True

        # Reconcile capability — the LLM can propose a corrected
        # capability when it detects grouping mismatches (e.g. two
        # specs that should share a capability but don't).
        if patch.capability and patch.capability != spec.capability:
            spec.capability = patch.capability
            stats["capabilities_fixed"] += 1
            changed = True

        if changed:
            stats["specs_patched"] += 1

    return stats


# ── Stage ─────────────────────────────────────────────────────────────


@trace_methods
class SpecReconciliationStage(Stage):
    """Reconcile specs and requirements after spec generation."""

    name = "spec_reconciliation"

    def __init__(
        self,
        llm: "LLMClient | None" = None,
    ) -> None:
        self.llm = llm
        # Populated after run() completes — summary for callers.
        self.summary: dict[str, Any] = {}

    def run(self, context: PipelineContext) -> PipelineContext:
        requirements = context.requirements
        specs = context.specs

        if not specs:
            log.info("spec_recon_skip_no_specs")
            return context

        valid_req_ids = {r.id for r in requirements}

        # ── Deterministic pass (always runs) ──────────────────────────

        # 1. Strip phantom requirement_ids
        phantom_reqs = _strip_phantom_requirement_ids(specs, valid_req_ids)
        if phantom_reqs:
            log.warning("spec_recon_phantom_req_ids_removed", count=phantom_reqs)

        # 2. Strip phantom dependency IDs
        phantom_deps = _strip_phantom_dependency_ids(specs)
        if phantom_deps:
            log.warning("spec_recon_phantom_dep_ids_removed", count=phantom_deps)

        # 3. Detect and break circular dependencies
        back_edges = _detect_cycles(specs)
        if back_edges:
            cycles_removed = _remove_back_edges(specs, back_edges)
            log.warning(
                "spec_recon_cycles_broken",
                back_edges=[(a, b) for a, b in back_edges],
                removed=cycles_removed,
            )

        # 4. Find uncovered requirements
        uncovered = _find_uncovered_requirements(requirements, specs)
        if uncovered:
            log.warning("spec_recon_uncovered_requirements", ids=uncovered)

        # 5. Detect capability islands (specs that share a capability
        #    but have no dependency path between them)
        capability_islands = _find_capability_islands(specs)
        if capability_islands:
            log.warning(
                "spec_recon_capability_islands",
                count=len(capability_islands),
                islands=capability_islands,
            )

        # ── LLM pass (best-effort) ───────────────────────────────────

        llm_issues: list[dict[str, str]] = []
        llm_stats: dict[str, Any] = {}

        if self.llm is not None:
            try:
                system_prompt = get_prompt("spec_reconciliation", "system")
                user_prompt = _build_prompt(requirements, specs)
                result = self.llm.complete_structured(
                    prompt=user_prompt,
                    response_model=_ReconciliationResult,
                    system=system_prompt,
                )

                # Apply patches
                llm_stats = _apply_llm_patches(
                    specs, result.specs, valid_req_ids,
                )
                llm_issues = [i.model_dump() for i in result.issues]

                # Re-run cycle detection after LLM may have added edges
                new_back_edges = _detect_cycles(specs)
                if new_back_edges:
                    extra_removed = _remove_back_edges(specs, new_back_edges)
                    log.warning(
                        "spec_recon_post_llm_cycles_broken",
                        removed=extra_removed,
                    )

                # Re-strip phantom deps the LLM may have introduced
                _strip_phantom_dependency_ids(specs)

                log.info("spec_recon_llm_complete", **llm_stats)

            except PipelineCancelled:
                raise
            except Exception as exc:
                log.warning("spec_recon_llm_failed", error=str(exc))

        # ── Summary ──────────────────────────────────────────────────

        # Re-check coverage after all fixes
        final_uncovered = _find_uncovered_requirements(requirements, specs)

        self.summary = {
            "total_specs": len(specs),
            "total_requirements": len(requirements),
            "phantom_req_ids_removed": phantom_reqs,
            "phantom_dep_ids_removed": phantom_deps,
            "cycles_broken": len(back_edges),
            "uncovered_requirements": final_uncovered,
            "capability_islands": capability_islands,
            "llm_issues": llm_issues,
            **llm_stats,
        }
        log.info("spec_reconciliation_complete", **{
            k: (len(v) if isinstance(v, list) else v)
            for k, v in self.summary.items()
        })

        # Metrics
        try:
            from dark_factory.metrics.helpers import record_spec_reconciliation

            record_spec_reconciliation(
                phantom_req_ids_removed=phantom_reqs,
                phantom_dep_ids_removed=phantom_deps,
                cycles_broken=len(back_edges),
                deps_added=llm_stats.get("deps_added", 0),
                req_ids_added=llm_stats.get("req_ids_added", 0),
                uncovered_requirements=len(final_uncovered),
                capability_islands=len(capability_islands),
            )
        except Exception:  # pragma: no cover — defensive
            pass

        return context
