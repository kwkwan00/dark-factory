"""Stage 4: Generate application code from the knowledge graph."""

from __future__ import annotations

from pathlib import Path

import structlog

from dark_factory.graph.repository import GraphRepository
from dark_factory.llm.base import LLMClient
from dark_factory.models.domain import CodeArtifact, PipelineContext
from dark_factory.stages.base import Stage

log = structlog.get_logger()

CODEGEN_SYSTEM_PROMPT = """\
You are a senior software engineer. Given a specification and its dependency context \
from a knowledge graph, generate clean, well-structured application code. \
Return valid JSON matching the provided schema."""

CODEGEN_USER_TEMPLATE = """\
Generate application code for this specification.

Spec ID: {spec_id}
Title: {title}
Description:
{description}

Acceptance Criteria:
{criteria}

Dependency Context:
{dep_context}

Return JSON with fields: id (string, use "code-" prefix + spec id), spec_id, \
file_path (appropriate path for the generated module), language, content (the full source code)."""


class CodegenStage(Stage):
    name = "codegen"

    def __init__(self, llm: LLMClient, repo: GraphRepository, output_dir: str, vector_repo: object | None = None) -> None:
        self.llm = llm
        self.repo = repo
        self.output_dir = Path(output_dir)
        self.vector_repo = vector_repo

    def run(self, context: PipelineContext) -> PipelineContext:
        artifacts: list[CodeArtifact] = []

        for spec in context.specs:
            log.info("generating_code", spec_id=spec.id)
            dep_context = self.repo.get_spec_with_context(spec.id)

            prompt = CODEGEN_USER_TEMPLATE.format(
                spec_id=spec.id,
                title=spec.title,
                description=spec.description,
                criteria="\n".join(f"- {c}" for c in spec.acceptance_criteria),
                dep_context=dep_context or "No dependencies.",
            )
            artifact = self.llm.complete_structured(
                prompt=prompt,
                system=CODEGEN_SYSTEM_PROMPT,
                response_model=CodeArtifact,
            )
            artifacts.append(artifact)

            # Write generated code to disk (C5 fix: validate path stays in output dir)
            out_path = (self.output_dir / artifact.file_path).resolve()
            if not out_path.is_relative_to(self.output_dir.resolve()):
                log.warning("codegen_path_traversal_blocked", path=artifact.file_path)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(artifact.content)
            log.info("wrote_code", path=str(out_path))

            # Auto-index code in Qdrant for semantic search
            if self.vector_repo:
                try:
                    self.vector_repo.upsert_code(artifact=artifact)
                except Exception as exc:
                    log.warning("vector_code_index_failed", artifact_id=artifact.id, error=str(exc))

        context.artifacts = artifacts
        log.info("codegen_complete", count=len(artifacts))
        return context
