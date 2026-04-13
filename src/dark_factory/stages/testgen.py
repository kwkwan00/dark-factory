"""Stage 5: Generate evaluation tests for the generated code."""

from __future__ import annotations

from pathlib import Path

import structlog

from dark_factory.llm.base import LLMClient
from dark_factory.models.domain import PipelineContext, Spec, TestCase
from dark_factory.stages.base import Stage

log = structlog.get_logger()

from dark_factory.prompts import get_prompt

TESTGEN_SYSTEM_PROMPT = get_prompt("testgen", "system")
TESTGEN_USER_TEMPLATE = get_prompt("testgen", "user")


class TestgenStage(Stage):
    name = "testgen"

    def __init__(self, llm: LLMClient, output_dir: str) -> None:
        self.llm = llm
        self.output_dir = Path(output_dir)

    def run(self, context: PipelineContext) -> PipelineContext:
        specs_by_id: dict[str, Spec] = {s.id: s for s in context.specs}
        tests: list[TestCase] = []

        for artifact in context.artifacts:
            log.info("generating_tests", artifact_id=artifact.id)
            spec = specs_by_id.get(artifact.spec_id)
            criteria = spec.acceptance_criteria if spec else []

            prompt = TESTGEN_USER_TEMPLATE.format(
                artifact_id=artifact.id,
                spec_title=spec.title if spec else "Unknown",
                file_path=artifact.file_path,
                language=artifact.language,
                criteria="\n".join(f"- {c}" for c in criteria),
                code=artifact.content,
            )
            test_case = self.llm.complete_structured(
                prompt=prompt,
                system=TESTGEN_SYSTEM_PROMPT,
                response_model=TestCase,
            )
            tests.append(test_case)

            # C5 fix: validate path stays within output directory
            out_path = (self.output_dir / test_case.file_path).resolve()
            if not out_path.is_relative_to(self.output_dir.resolve()):
                log.warning("testgen_path_traversal_blocked", path=test_case.file_path)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(test_case.content)
            log.info("wrote_test", path=str(out_path))

            # Run AI evaluation on the generated tests
            try:
                from dark_factory.evaluation.metrics import evaluate_generated_tests

                eval_results = evaluate_generated_tests(
                    spec_title=spec.title if spec else "Unknown",
                    acceptance_criteria=criteria,
                    source_code=artifact.content,
                    test_code=test_case.content,
                )
                for metric_name, result in eval_results.items():
                    log.info(
                        "eval_result",
                        artifact=artifact.id,
                        metric=metric_name,
                        score=result["score"],
                        passed=result["passed"],
                    )
            except Exception as exc:
                log.warning("eval_skipped", artifact=artifact.id, error=str(exc))

        context.tests = tests
        log.info("testgen_complete", count=len(tests))
        return context
