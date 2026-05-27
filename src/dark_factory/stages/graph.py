"""Stage 3: Populate the Neo4j knowledge graph with requirements and specs."""

from __future__ import annotations

import time

import structlog

from dark_factory.agents.cancellation import raise_if_cancelled
from dark_factory.graph.repository import GraphRepository
from dark_factory.log import trace_methods
from dark_factory.models.domain import PipelineContext
from dark_factory.stages.base import Stage

log = structlog.get_logger()


@trace_methods
class GraphStage(Stage):
    name = "graph"

    def __init__(self, repo: GraphRepository) -> None:
        self.repo = repo

    def run(self, context: PipelineContext) -> PipelineContext:
        raise_if_cancelled()
        started = time.monotonic()
        implements_edges = 0
        depends_on_edges = 0

        for req in context.requirements:
            raise_if_cancelled()
            log.info("graph_upsert_requirement", id=req.id)
            self.repo.upsert_requirement(req)

        for spec in context.specs:
            raise_if_cancelled()
            log.info("graph_upsert_spec", id=spec.id)
            self.repo.upsert_spec(spec)
            for req_id in spec.requirement_ids:
                self.repo.link_spec_to_requirement(spec.id, req_id)
                implements_edges += 1
            for dep_id in spec.dependencies:
                self.repo.link_spec_dependency(spec.id, dep_id)
                depends_on_edges += 1

        duration = time.monotonic() - started
        log.info("graph_complete", requirements=len(context.requirements), specs=len(context.specs))

        try:
            from dark_factory.metrics.helpers import record_graph_write

            record_graph_write(
                requirements_written=len(context.requirements),
                specs_written=len(context.specs),
                implements_edges=implements_edges,
                depends_on_edges=depends_on_edges,
                duration_seconds=duration,
            )
        except Exception:  # pragma: no cover — defensive
            pass

        return context
