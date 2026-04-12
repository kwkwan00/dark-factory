"""Stage 2: Convert requirements into specifications using an LLM.

Pipeline:
    (A) Planning phase — OPTIONAL (``enable_decomposition=True``)
        For each input ``Requirement``, an LLM planner decomposes it into
        a list of 3-8 smaller, independently-implementable sub-specs
        (``_PlannedSpec``). Sub-specs carry a title, technical purpose,
        and title-based dependencies on sibling sub-specs.

        Planner failures (LLM error, malformed JSON, empty list) fall back
        gracefully to a single-entry plan that matches the original
        requirement, so the pipeline never dies here.

    (B) Refinement phase — ALWAYS RUNS
        Each (parent requirement, planned sub-spec) pair runs through a
        small **architect/critic refinement loop**: the architect generates
        a full OpenSpec ``Spec``, the critic evaluates it with DeepEval,
        and if the score is below ``eval_threshold`` the architect refines
        using the critic's feedback. Up to ``max_handoffs`` iterations,
        early-exit on threshold. The best-scoring spec across attempts
        is returned.

    (C) Dependency resolution — OPTIONAL (only when decomposition is on)
        After all sub-specs are refined, a pass translates the planner's
        title-based ``depends_on`` references into generated spec ids and
        populates ``spec.dependencies`` on the resolved ``Spec`` objects.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from dark_factory.llm.base import LLMClient
from dark_factory.models.domain import PipelineContext, Requirement, Spec
from dark_factory.stages.base import Stage

if TYPE_CHECKING:
    from dark_factory.graph.repository import GraphRepository

log = structlog.get_logger()

SPEC_SYSTEM_PROMPT = """\
You are a software architect. Given a requirement, produce a detailed specification \
for spec-driven development using the OpenSpec format. Return valid JSON matching the \
provided schema."""

SPEC_USER_TEMPLATE = """\
Convert this requirement into a specification.

Requirement ID: {req_id}
Title: {title}
Description:
{description}

Return JSON with fields: id (string, use "spec-" prefix + requirement id), title, \
description (detailed technical spec), requirement_ids (list with the requirement id), \
acceptance_criteria (list of testable criteria), dependencies (list of other spec ids, \
empty if none), capability (kebab-case name for this capability, e.g. "user-auth"), \
scenarios (list of objects with fields: name, when, then — each describing a \
WHEN/THEN behavioral scenario)."""

SPEC_REFINE_TEMPLATE = """\
You previously generated this specification for requirement `{req_id}`. The
LLM-as-judge evaluation found weaknesses — refine the spec to address them.

Original requirement:
{title}: {description}

Previous spec attempt #{prev_attempt} (overall score: {prev_score:.2f}):
{previous_spec}

Evaluation feedback (per metric):
{feedback}

Generate an IMPROVED specification that directly addresses the lowest-scoring
metrics. Be more specific in acceptance criteria, add WHEN/THEN scenarios for
edge cases, and ensure the technical description is unambiguous. Return the
same JSON schema."""


# ── Decomposition planning prompts ───────────────────────────────────────────


SPEC_PLAN_SYSTEM_PROMPT = """\
You are a senior systems architect and requirements decomposer. You break a \
single product requirement into the smallest possible set of independently \
implementable, testable specifications.

Principles:
- Prefer MORE granular sub-specs over fewer. A good sub-spec does ONE thing well.
- Each sub-spec must be independently testable and buildable in isolation.
- Separate concerns: data models, validation, storage, business logic, HTTP \
surface, UI state, presentation, error handling, and cross-cutting infrastructure \
(auth, logging, config) should each get their own sub-spec where applicable.
- For UI features, separate state management, API calls, and presentation.
- For backend features, separate the HTTP handler, service logic, repository, \
and validation/schemas.
- Express dependencies between sub-specs by referencing EARLIER sibling titles \
exactly (case-sensitive). Sibling titles must be distinct and contain no colons.
- Do NOT generate full technical detail yet — the refinement phase will do that. \
Keep each sub-spec's description to 1-3 sentences naming scope boundaries.
- Return 3-8 sub-specs for typical requirements. Return exactly 1 only if the \
requirement is genuinely atomic.

Return valid JSON matching the provided schema."""

SPEC_PLAN_USER_TEMPLATE = """\
Decompose this requirement into the full set of granular sub-specifications \
needed to implement it.

Parent Requirement
==================
ID: {req_id}
Title: {title}
Description:
{description}

Rules:
- Return between 1 and 8 sub-specs. Aim for 3-8 unless the requirement is atomic.
- Each sub-spec must own ONE capability slice.
- Sub-spec titles must be distinct within this plan and contain no colons.
- Use `depends_on` to list titles of EARLIER sibling sub-specs this one requires. \
Empty list for roots.
- Use kebab-case for `capability`.

Return JSON with fields: parent_requirement_id (must equal "{req_id}") and specs \
(a list of objects with fields: title, description, capability, depends_on, rationale)."""

SPEC_USER_TEMPLATE_DECOMPOSED = """\
Convert this PLANNED sub-specification into a full OpenSpec specification.

Parent Requirement (context only — do NOT expand scope beyond the sub-spec slice)
==================================================================================
ID: {parent_req_id}
Title: {parent_req_title}
Description:
{parent_req_description}

Planned Sub-Spec Slice (stay strictly within this scope)
========================================================
Title: {planned_title}
Purpose: {planned_description}
Capability: {planned_capability}
Rationale: {planned_rationale}
Declared sibling dependencies (titles only): {planned_depends_on}

Return JSON with these fields:
- id: MUST equal "{target_spec_id}" exactly.
- title: MUST equal "{planned_title}" exactly.
- description: detailed technical spec for THIS slice only.
- requirement_ids: list containing "{parent_req_id}".
- acceptance_criteria: testable criteria SPECIFIC to this slice.
- dependencies: return an empty list. The pipeline will fill dependencies \
from the planner's sibling title references.
- capability: use "{planned_capability}" unless you can provide a better kebab-case name.
- scenarios: list of objects with fields name, when, then."""


# ── Internal planner models ──────────────────────────────────────────────────


class _PlannedSpec(BaseModel):
    """One decomposed sub-spec inside a ``_SpecPlan``.

    Private to this module: the planner's intermediate representation. Never
    leaves the stage and is never persisted.
    """

    title: str = Field(description="Specific, action-oriented sub-spec name")
    description: str = Field(description="Technical scope of this sub-spec (1-3 sentences)")
    capability: str = Field(default="", description="kebab-case capability name")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Titles of earlier sibling sub-specs this one depends on",
    )
    rationale: str = Field(
        default="",
        description="Why this is a separate sub-spec instead of being merged",
    )


class _SpecPlan(BaseModel):
    """A planner's decomposition of one parent requirement into sub-specs."""

    parent_requirement_id: str
    specs: list[_PlannedSpec] = Field(default_factory=list)


class SpecStage(Stage):
    name = "spec"

    def __init__(
        self,
        llm: LLMClient,
        vector_repo: object | None = None,
        max_parallel: int = 4,
        max_handoffs: int = 5,
        eval_threshold: float = 0.8,
        enable_decomposition: bool = False,
        max_specs_per_requirement: int = 12,
        *,
        graph_repo: "GraphRepository | None" = None,
        reuse_existing_specs: bool = False,
    ) -> None:
        """Create a SpecStage.

        ``enable_decomposition`` defaults to ``False`` so unit tests that mock
        ``llm.complete_structured`` to always return a ``Spec`` stay green.
        Production callers (``api/ag_ui_bridge.py``) pass
        ``enable_decomposition=settings.pipeline.enable_spec_decomposition``
        which defaults to ``True``.

        ``graph_repo`` + ``reuse_existing_specs`` enable the preflight
        skip path: before dispatching work units to the refinement
        thread pool, the stage queries Neo4j for ``:Spec`` nodes whose
        id matches a target id, loads them, and passes them through
        unchanged. This makes re-runs on an unchanged requirements
        directory effectively free. Both default to off so unit tests
        and callers that don't have a repo still work.
        """
        self.llm = llm
        self.vector_repo = vector_repo
        self.max_parallel = max(1, max_parallel)
        self.max_handoffs = max(1, max_handoffs)
        self.eval_threshold = eval_threshold
        self.enable_decomposition = enable_decomposition
        self.max_specs_per_requirement = max(1, max_specs_per_requirement)
        self.graph_repo = graph_repo
        self.reuse_existing_specs = reuse_existing_specs

    def run(self, context: PipelineContext) -> PipelineContext:
        from dark_factory.agents.cancellation import raise_if_cancelled
        from dark_factory.agents.tools import emit_progress

        raise_if_cancelled()

        requirements = context.requirements
        if not requirements:
            context.specs = []
            return context

        # ── Phase A: planning ────────────────────────────────────────────────
        # When decomposition is on, each requirement is first split into
        # multiple planned sub-specs. When off, we stay on the classic
        # "one requirement → one spec" path.
        plans: dict[str, _SpecPlan] = {}
        if self.enable_decomposition:
            plans = self._plan_all(requirements)

        # ── Phase B: flatten plans into refinement work units ────────────────
        # Each work unit is (parent_req, planned_sub_spec | None, target_spec_id).
        # When ``planned`` is None we use the classic prompt; when present we
        # use the decomposed prompt with parent-req context.
        work_units: list[tuple[Requirement, _PlannedSpec | None, str]] = []
        if plans:
            for req in requirements:
                plan = plans.get(req.id)
                if plan and plan.specs:
                    for i, planned in enumerate(plan.specs):
                        target_id = self._sub_spec_id(req, i)
                        work_units.append((req, planned, target_id))
                else:
                    # Defensive: should have been replaced by fallback already.
                    work_units.append((req, None, f"spec-{req.id}"))
        else:
            work_units = [(req, None, f"spec-{req.id}") for req in requirements]

        # ── Phase B.5: preflight skip ───────────────────────────────────────
        # Query Neo4j in bulk for any target spec ids that already exist
        # and pre-fill the results list for those indices. Re-running
        # the pipeline on unchanged requirements becomes a near-no-op
        # rather than re-spending the full LLM budget on spec gen.
        #
        # The check is gated by ``reuse_existing_specs`` and only runs
        # when a ``graph_repo`` was provided — unit tests that mock the
        # LLM and never touch Neo4j stay on the original path.
        specs: list[Spec | None] = [None] * len(work_units)
        reused_indices: set[int] = set()
        if (
            self.reuse_existing_specs
            and self.graph_repo is not None
            and work_units
        ):
            target_ids = [target_id for _, _, target_id in work_units]
            try:
                existing_ids = self.graph_repo.existing_spec_ids(target_ids)
            except Exception as exc:
                # If Neo4j is flaky we do NOT fail the run — we just
                # skip the optimisation and do the full work.
                log.warning("spec_preflight_check_failed", error=str(exc))
                existing_ids = set()

            if existing_ids:
                try:
                    existing_specs = {
                        s.id: s
                        for s in self.graph_repo.get_specs(list(existing_ids))
                    }
                except Exception as exc:
                    log.warning(
                        "spec_preflight_load_failed", error=str(exc)
                    )
                    existing_specs = {}

                for idx, (req, _planned, target_id) in enumerate(work_units):
                    existing = existing_specs.get(target_id)
                    if existing is None:
                        continue
                    specs[idx] = existing
                    reused_indices.add(idx)
                    emit_progress(
                        "spec_gen_skipped",
                        requirement_id=req.id,
                        requirement_title=req.title,
                        target_spec_id=target_id,
                        reason="spec_already_exists",
                        index=idx,
                        total=len(work_units),
                    )

        pending_units = [
            (idx, req, planned, target_id)
            for idx, (req, planned, target_id) in enumerate(work_units)
            if idx not in reused_indices
        ]
        workers = min(self.max_parallel, max(1, len(pending_units)))

        emit_progress(
            "spec_gen_layer_started",
            total=len(requirements),
            parallel=workers,
            max_handoffs=self.max_handoffs,
            eval_threshold=self.eval_threshold,
            decomposition_enabled=self.enable_decomposition,
            planned_sub_specs=len(work_units),
            reused_from_graph=len(reused_indices),
            pending=len(pending_units),
        )
        log.info(
            "spec_generation_starting",
            requirements=len(requirements),
            work_units=len(work_units),
            reused=len(reused_indices),
            pending=len(pending_units),
            parallel=workers,
            max_handoffs=self.max_handoffs,
            decomposition_enabled=self.enable_decomposition,
        )

        # ── Phase C: refine in parallel ──────────────────────────────────────
        failures: list[tuple[int, Exception]] = []
        if pending_units:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._refine_spec,
                        idx,
                        req,
                        len(work_units),
                        planned,
                        target_id,
                    ): idx
                    for idx, req, planned, target_id in pending_units
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        specs[idx] = future.result()
                    except Exception as exc:
                        req, planned, target_id = work_units[idx]
                        log.error(
                            "spec_gen_failed",
                            requirement_id=req.id,
                            target_spec_id=target_id,
                            sub_spec_title=planned.title if planned else None,
                            index=idx,
                            error=str(exc),
                        )
                        failures.append((idx, exc))
                        emit_progress(
                            "spec_gen_failed",
                            requirement_id=req.id,
                            target_spec_id=target_id,
                            sub_spec_title=planned.title if planned else None,
                            index=idx,
                            error=str(exc),
                        )
        # else: every target was already in Neo4j — nothing to submit.

        completed = [s for s in specs if s is not None]

        # ── Phase D: dependency resolution (decomposition only) ──────────────
        if plans:
            self._resolve_dependencies(work_units, specs, plans)

        emit_progress(
            "spec_gen_layer_completed",
            total=len(completed),
            failed=len(failures),
        )

        if failures and not completed:
            raise failures[0][1]

        log.info("spec_complete", count=len(completed), failed=len(failures))
        context.specs = completed
        return context

    # ── Planning phase ───────────────────────────────────────────────────────

    def _plan_all(self, requirements: list[Requirement]) -> dict[str, _SpecPlan]:
        """Plan every requirement in parallel. Planner failures fall back to
        single-spec plans so the pipeline always has something to refine."""
        from dark_factory.agents.cancellation import is_cancelled, raise_if_cancelled

        raise_if_cancelled()
        plans: dict[str, _SpecPlan] = {}
        workers = min(self.max_parallel, len(requirements))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._plan_requirement, req): req
                for req in requirements
            }
            for future in as_completed(futures):
                # Cancellation mid-planning: cancel remaining futures and
                # install fallback plans for the uncompleted requirements
                # so downstream refinement still has something to consume
                # (it will then hit its own cancel checks and stop quickly).
                if is_cancelled():
                    for pending_future, pending_req in futures.items():
                        if pending_future is future or pending_future.done():
                            continue
                        pending_future.cancel()
                        plans[pending_req.id] = self._single_spec_fallback_plan(pending_req)
                    # Drain the current future and bail out.
                    req = futures[future]
                    try:
                        plans[req.id] = future.result()
                    except Exception:
                        plans[req.id] = self._single_spec_fallback_plan(req)
                    raise_if_cancelled()

                req = futures[future]
                try:
                    plans[req.id] = future.result()
                except Exception as exc:
                    # `_plan_requirement` already catches and falls back, but
                    # double-safety: any unexpected exception still gets a
                    # fallback so we never lose a requirement.
                    log.error(
                        "spec_plan_unexpected_error",
                        requirement_id=req.id,
                        error=str(exc),
                    )
                    plans[req.id] = self._single_spec_fallback_plan(req)
        return plans

    def _plan_requirement(self, req: Requirement) -> _SpecPlan:
        """Ask the LLM to decompose a single requirement into sub-specs.

        On any failure (LLM error, Pydantic validation, empty result) a
        fallback single-spec plan is returned so the downstream refinement
        phase can still produce a spec for this requirement.
        """
        from dark_factory.agents.tools import emit_progress
        from dark_factory.metrics.helpers import record_decomposition_stats

        emit_progress(
            "spec_plan_started",
            requirement_id=req.id,
            requirement_title=req.title,
        )

        prompt = SPEC_PLAN_USER_TEMPLATE.format(
            req_id=req.id,
            title=req.title,
            description=req.description,
        )

        try:
            plan = self.llm.complete_structured(
                prompt=prompt,
                system=SPEC_PLAN_SYSTEM_PROMPT,
                response_model=_SpecPlan,
            )
        except Exception as exc:
            log.warning(
                "spec_plan_failed",
                requirement_id=req.id,
                error=str(exc),
            )
            emit_progress(
                "spec_plan_failed",
                requirement_id=req.id,
                requirement_title=req.title,
                error=str(exc),
                fallback="single-spec",
            )
            record_decomposition_stats(
                requirement_id=req.id,
                requirement_title=req.title,
                planned_sub_specs_count=1,
                fallback=True,
                empty_result=False,
                truncated=False,
                depends_on_declared=0,
            )
            return self._single_spec_fallback_plan(req)

        if not plan.specs:
            log.warning(
                "spec_plan_empty",
                requirement_id=req.id,
            )
            emit_progress(
                "spec_plan_failed",
                requirement_id=req.id,
                requirement_title=req.title,
                error="planner returned no sub-specs",
                fallback="single-spec",
            )
            record_decomposition_stats(
                requirement_id=req.id,
                requirement_title=req.title,
                planned_sub_specs_count=1,
                fallback=True,
                empty_result=True,
                truncated=False,
                depends_on_declared=0,
            )
            return self._single_spec_fallback_plan(req)

        # Dedupe by title (case-insensitive), preserving order
        seen_titles: set[str] = set()
        unique_specs: list[_PlannedSpec] = []
        for planned in plan.specs:
            key = planned.title.strip().lower()
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            # Drop self-referential deps (a sub-spec can't depend on itself)
            planned.depends_on = [
                d for d in planned.depends_on
                if d.strip().lower() != key
            ]
            unique_specs.append(planned)

        # Cap at max_specs_per_requirement
        original_len = len(unique_specs)
        truncated = original_len > self.max_specs_per_requirement
        if truncated:
            log.warning(
                "spec_plan_truncated",
                requirement_id=req.id,
                original=original_len,
                kept=self.max_specs_per_requirement,
            )
            unique_specs = unique_specs[: self.max_specs_per_requirement]

        if not unique_specs:
            # Post-dedup the plan is empty — treat as failure
            emit_progress(
                "spec_plan_failed",
                requirement_id=req.id,
                requirement_title=req.title,
                error="planner returned only duplicates/blanks",
                fallback="single-spec",
            )
            record_decomposition_stats(
                requirement_id=req.id,
                requirement_title=req.title,
                planned_sub_specs_count=1,
                fallback=True,
                empty_result=True,
                truncated=False,
                depends_on_declared=0,
            )
            return self._single_spec_fallback_plan(req)

        plan.specs = unique_specs

        declared_deps = sum(len(p.depends_on) for p in unique_specs)
        record_decomposition_stats(
            requirement_id=req.id,
            requirement_title=req.title,
            planned_sub_specs_count=len(unique_specs),
            fallback=False,
            empty_result=False,
            truncated=truncated,
            depends_on_declared=declared_deps,
        )

        emit_progress(
            "spec_plan_completed",
            requirement_id=req.id,
            requirement_title=req.title,
            sub_spec_count=len(unique_specs),
            titles=[s.title for s in unique_specs],
        )
        log.info(
            "spec_plan_completed",
            requirement_id=req.id,
            sub_spec_count=len(unique_specs),
        )
        return plan

    @staticmethod
    def _single_spec_fallback_plan(req: Requirement) -> _SpecPlan:
        """Minimal plan used when the planner fails or returns nothing."""
        return _SpecPlan(
            parent_requirement_id=req.id,
            specs=[
                _PlannedSpec(
                    title=req.title,
                    description=req.description,
                    capability="",
                    depends_on=[],
                    rationale="fallback — planner failed or returned no sub-specs",
                )
            ],
        )

    @staticmethod
    def _sub_spec_id(req: Requirement, index: int) -> str:
        """Deterministic sub-spec id: ``spec-<parent_req_id>-<NN>``."""
        return f"spec-{req.id}-{index:02d}"

    # ── Dependency resolution ───────────────────────────────────────────────

    def _resolve_dependencies(
        self,
        work_units: list[tuple[Requirement, _PlannedSpec | None, str]],
        specs: list[Spec | None],
        plans: dict[str, _SpecPlan],
    ) -> None:
        """Translate planner title-based deps into generated spec ids.

        The planner expresses dependencies as sibling titles; after every
        sub-spec has been refined into a real ``Spec`` we look each title up
        in the (parent_req_id, title) index and populate ``spec.dependencies``.

        Unknown titles are logged and silently dropped — a planner hallucination
        of a nonexistent sibling should not crash the pipeline.
        """
        from dark_factory.agents.tools import emit_progress

        # Build (parent_req_id, normalised_title) → generated spec id.
        title_to_id: dict[tuple[str, str], str] = {}
        for idx, (req, planned, target_id) in enumerate(work_units):
            spec = specs[idx]
            if spec is None or planned is None:
                continue
            title_to_id[(req.id, planned.title.strip().lower())] = spec.id

        # Per-parent tally for the summary event.
        per_parent: dict[str, dict[str, int]] = {}

        for idx, (req, planned, _target_id) in enumerate(work_units):
            spec = specs[idx]
            if spec is None or planned is None:
                continue
            tally = per_parent.setdefault(
                req.id, {"resolved": 0, "unresolved": 0}
            )
            existing = set(spec.dependencies)
            for dep_title in planned.depends_on:
                key = (req.id, dep_title.strip().lower())
                dep_id = title_to_id.get(key)
                if dep_id is None:
                    tally["unresolved"] += 1
                    log.debug(
                        "spec_dep_unresolved",
                        requirement_id=req.id,
                        spec_id=spec.id,
                        unknown_title=dep_title,
                    )
                    continue
                if dep_id in existing or dep_id == spec.id:
                    continue
                spec.dependencies.append(dep_id)
                existing.add(dep_id)
                tally["resolved"] += 1

        for req_id, counts in per_parent.items():
            emit_progress(
                "spec_plan_resolved",
                requirement_id=req_id,
                resolved=counts["resolved"],
                unresolved=counts["unresolved"],
            )

    # ── Architect/critic refinement loop ─────────────────────────────────────

    def _refine_spec(
        self,
        index: int,
        req: Requirement,
        total: int,
        planned: _PlannedSpec | None = None,
        target_spec_id: str | None = None,
    ) -> Spec:
        """Run up to ``max_handoffs`` generate→evaluate→refine iterations.
        Returns the best-scoring spec.

        When ``planned`` is not None, the first attempt uses the decomposed
        prompt template and the resulting spec's ``id`` / ``requirement_ids``
        are forced to match the target sub-spec id / parent requirement id.
        Subsequent refinement iterations use the generic refine template
        regardless of the decomposition path.
        """
        from dark_factory.agents.cancellation import raise_if_cancelled
        from dark_factory.agents.tools import emit_progress

        # Pre-flight cancel check so work items still queued in the thread
        # pool don't spend any LLM budget once cancel has been requested.
        raise_if_cancelled()

        emit_progress(
            "spec_gen_started",
            requirement_id=req.id,
            requirement_title=req.title,
            index=index,
            total=total,
            max_handoffs=self.max_handoffs,
            sub_spec_title=planned.title if planned else None,
            target_spec_id=target_spec_id,
        )

        best_spec: Spec | None = None
        best_score: float = -1.0
        last_feedback: str = ""
        attempt = 0

        for attempt in range(1, self.max_handoffs + 1):
            # Cancellation check between refinement attempts. Inside a
            # single attempt the LLM call is blocking and can't be
            # interrupted, but we catch it the moment the attempt returns.
            raise_if_cancelled()

            # ── Architect: generate or refine ──
            if attempt == 1 or best_spec is None:
                if planned is not None:
                    prompt = SPEC_USER_TEMPLATE_DECOMPOSED.format(
                        parent_req_id=req.id,
                        parent_req_title=req.title,
                        parent_req_description=req.description,
                        planned_title=planned.title,
                        planned_description=planned.description,
                        planned_capability=planned.capability or "",
                        planned_rationale=planned.rationale or "",
                        planned_depends_on=", ".join(planned.depends_on) or "(none)",
                        target_spec_id=target_spec_id or f"spec-{req.id}",
                    )
                else:
                    prompt = SPEC_USER_TEMPLATE.format(
                        req_id=req.id,
                        title=req.title,
                        description=req.description,
                    )
                role = "architect"
            else:
                prompt = SPEC_REFINE_TEMPLATE.format(
                    req_id=req.id,
                    title=req.title,
                    description=req.description,
                    prev_attempt=attempt - 1,
                    prev_score=best_score,
                    previous_spec=best_spec.model_dump_json(indent=2),
                    feedback=last_feedback,
                )
                role = "refiner"

            try:
                spec = self.llm.complete_structured(
                    prompt=prompt,
                    system=SPEC_SYSTEM_PROMPT,
                    response_model=Spec,
                )
            except Exception as exc:
                log.warning(
                    "spec_attempt_llm_failed",
                    requirement_id=req.id,
                    target_spec_id=target_spec_id,
                    sub_spec_title=planned.title if planned else None,
                    attempt=attempt,
                    error=str(exc),
                )
                # If we have any best so far, keep it; otherwise re-raise
                if best_spec is not None:
                    break
                raise

            # When running decomposed, force the id / requirement_ids so a
            # model that ignores the "MUST equal" instruction in the prompt
            # can't break dependency resolution.
            if planned is not None:
                if target_spec_id:
                    spec.id = target_spec_id
                if req.id not in spec.requirement_ids:
                    spec.requirement_ids = [req.id, *spec.requirement_ids]
                # Planner-sourced deps are installed by _resolve_dependencies
                # later. Prevent the architect from fabricating ids it has
                # never seen.
                spec.dependencies = []

            # Back-fill criteria from scenarios
            if not spec.acceptance_criteria and spec.scenarios:
                spec.acceptance_criteria = [
                    f"WHEN {s.when} THEN {s.then}" for s in spec.scenarios
                ]

            # ── Critic: evaluate ──
            score, feedback, raw_results = self._evaluate_spec(
                req, spec, sub_spec_title=planned.title if planned else None
            )

            # Emit a structured rubric event with each metric's score so the
            # Logs tab can display the per-metric breakdown of every attempt.
            if raw_results:
                metrics_payload = [
                    {
                        "name": name,
                        "score": float(r.get("score", 0.0) or 0.0),
                        "passed": bool(r.get("passed", False)),
                        "reason": (r.get("reason") or "")[:300],
                    }
                    for name, r in raw_results.items()
                ]
                emit_progress(
                    "eval_rubric",
                    feature="spec-generation",  # phase indicator
                    requirement_id=req.id,
                    requirement_title=req.title,
                    target_spec_id=target_spec_id,
                    sub_spec_title=planned.title if planned else None,
                    attempt=attempt,
                    max_handoffs=self.max_handoffs,
                    role=role,
                    metrics=metrics_payload,
                    avg_score=score,
                    threshold=self.eval_threshold,
                )
                # Prometheus: one observation per metric per attempt.
                try:
                    from dark_factory.metrics.prometheus import observe_eval_rubric

                    for m in metrics_payload:
                        observe_eval_rubric(
                            metric_name=m["name"],
                            score=m["score"],
                            passed=m["passed"],
                        )
                except Exception:  # pragma: no cover — defensive
                    pass

            log.info(
                "spec_handoff",
                requirement_id=req.id,
                target_spec_id=target_spec_id,
                sub_spec_title=planned.title if planned else None,
                attempt=attempt,
                role=role,
                score=score,
            )
            emit_progress(
                "spec_handoff",
                requirement_id=req.id,
                requirement_title=req.title,
                target_spec_id=target_spec_id,
                sub_spec_title=planned.title if planned else None,
                index=index,
                attempt=attempt,
                max_handoffs=self.max_handoffs,
                role=role,
                score=score,
                threshold=self.eval_threshold,
            )

            # Track the best spec
            if score > best_score:
                best_score = score
                best_spec = spec
                last_feedback = feedback

            # Early exit when threshold is met
            if score >= self.eval_threshold:
                log.info(
                    "spec_threshold_met",
                    requirement_id=req.id,
                    target_spec_id=target_spec_id,
                    sub_spec_title=planned.title if planned else None,
                    attempt=attempt,
                    score=score,
                )
                try:
                    from dark_factory.metrics.prometheus import (
                        observe_spec_attempts_to_pass,
                    )

                    observe_spec_attempts_to_pass(attempt)
                except Exception:  # pragma: no cover — defensive
                    pass
                break

        # ── Vector index the BEST spec only ──
        if best_spec is not None and self.vector_repo:
            try:
                self.vector_repo.upsert_spec(spec=best_spec)
            except Exception as exc:
                log.warning(
                    "vector_spec_index_failed",
                    spec_id=best_spec.id,
                    error=str(exc),
                )

        emit_progress(
            "spec_gen_completed",
            requirement_id=req.id,
            spec_id=best_spec.id if best_spec else None,
            spec_title=best_spec.title if best_spec else None,
            sub_spec_title=planned.title if planned else None,
            index=index,
            total=total,
            final_score=best_score,
            attempts=attempt,
        )

        if best_spec is None:
            raise RuntimeError(f"All spec attempts failed for {req.id}")
        return best_spec

    def _evaluate_spec(
        self,
        req: Requirement,
        spec: Spec,
        *,
        sub_spec_title: str | None = None,
    ) -> tuple[float, str, dict[str, dict]]:
        """Run DeepEval on a spec and return ``(avg_score, feedback_text, raw_results)``.

        ``raw_results`` is the per-metric dict from deepeval, used by the
        caller to emit a structured ``eval_rubric`` progress event.
        ``sub_spec_title`` is passed through to the eval helper purely
        for log disambiguation when decomposition is on.
        """
        try:
            from dark_factory.evaluation.metrics import evaluate_generated_spec

            results = evaluate_generated_spec(
                requirement_title=req.title,
                requirement_description=req.description,
                spec_json=spec.model_dump_json(),
                target_spec_id=spec.id,
                sub_spec_title=sub_spec_title,
            )
        except Exception as exc:
            log.warning(
                "spec_eval_skipped",
                requirement=req.id,
                target_spec_id=spec.id,
                sub_spec_title=sub_spec_title,
                error=str(exc),
            )
            return 0.0, f"Evaluation failed: {exc}", {}

        if not results:
            return 0.0, "No metrics returned", {}

        # Defensive access: a DeepEval judge can return a metric row
        # without a ``score`` field when the upstream LLM errors (e.g.
        # returns a malformed critique). Skip such rows rather than
        # raising KeyError mid-loop and crashing the entire spec stage.
        scores = [
            float(r["score"])
            for r in results.values()
            if isinstance(r, dict) and isinstance(r.get("score"), (int, float))
        ]
        if not scores:
            return 0.0, "No scorable metrics returned", results
        avg = sum(scores) / len(scores)
        feedback_lines = [
            f"- {name}: score={r.get('score', 0.0):.2f}, passed={r.get('passed', False)}"
            + (f", reason: {r['reason']}" if r.get("reason") else "")
            for name, r in results.items()
            if isinstance(r, dict)
        ]
        feedback = "\n".join(feedback_lines)
        return avg, feedback, results
