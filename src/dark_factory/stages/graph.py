"""Stage 3: Populate the Neo4j knowledge graph with requirements and specs."""

from __future__ import annotations

import structlog

from dark_factory.graph.repository import GraphRepository
from dark_factory.models.domain import PipelineContext
from dark_factory.stages.base import Stage

log = structlog.get_logger()


class GraphStage(Stage):
    name = "graph"

    def __init__(self, repo: GraphRepository) -> None:
        self.repo = repo

    def run(self, context: PipelineContext) -> PipelineContext:
        for req in context.requirements:
            log.info("graph_upsert_requirement", id=req.id)
            self.repo.upsert_requirement(req)

        for spec in context.specs:
            log.info("graph_upsert_spec", id=spec.id)
            self.repo.upsert_spec(spec)
            for req_id in spec.requirement_ids:
                self.repo.link_spec_to_requirement(spec.id, req_id)
            for dep_id in spec.dependencies:
                self.repo.link_spec_dependency(spec.id, dep_id)

        log.info("graph_complete", requirements=len(context.requirements), specs=len(context.specs))
        return context
