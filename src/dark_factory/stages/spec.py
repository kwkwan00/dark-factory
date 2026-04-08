"""Stage 2: Convert requirements into specifications using an LLM."""

from __future__ import annotations

import structlog

from dark_factory.llm.base import LLMClient
from dark_factory.models.domain import PipelineContext, Spec
from dark_factory.stages.base import Stage

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


class SpecStage(Stage):
    name = "spec"

    def __init__(self, llm: LLMClient, vector_repo: object | None = None) -> None:
        self.llm = llm
        self.vector_repo = vector_repo

    def run(self, context: PipelineContext) -> PipelineContext:
        specs: list[Spec] = []

        for req in context.requirements:
            log.info("generating_spec", requirement_id=req.id)
            prompt = SPEC_USER_TEMPLATE.format(
                req_id=req.id,
                title=req.title,
                description=req.description,
            )
            spec = self.llm.complete_structured(
                prompt=prompt,
                system=SPEC_SYSTEM_PROMPT,
                response_model=Spec,
            )
            # Back-fill acceptance_criteria from scenarios if empty
            if not spec.acceptance_criteria and spec.scenarios:
                spec.acceptance_criteria = [
                    f"WHEN {s.when} THEN {s.then}" for s in spec.scenarios
                ]
            specs.append(spec)

            # Run AI evaluation on the generated spec
            try:
                from dark_factory.evaluation.metrics import evaluate_generated_spec

                eval_results = evaluate_generated_spec(
                    requirement_title=req.title,
                    requirement_description=req.description,
                    spec_json=spec.model_dump_json(),
                )
                for metric_name, result in eval_results.items():
                    log.info(
                        "spec_eval_result",
                        requirement=req.id,
                        metric=metric_name,
                        score=result["score"],
                        passed=result["passed"],
                    )
            except Exception as exc:
                log.warning("spec_eval_skipped", requirement=req.id, error=str(exc))

            # Auto-index spec in Qdrant for semantic search
            if self.vector_repo:
                try:
                    self.vector_repo.upsert_spec(spec=spec)
                except Exception as exc:
                    log.warning("vector_spec_index_failed", spec_id=spec.id, error=str(exc))

        log.info("spec_complete", count=len(specs))
        context.specs = specs
        return context
