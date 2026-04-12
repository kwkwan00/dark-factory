"""Neo4j schema setup: constraints and indexes."""

from __future__ import annotations

import structlog

from dark_factory.graph.client import Neo4jClient

log = structlog.get_logger()

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT req_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT spec_id IF NOT EXISTS FOR (s:Spec) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT code_id IF NOT EXISTS FOR (c:CodeArtifact) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT test_id IF NOT EXISTS FOR (t:TestCase) REQUIRE t.id IS UNIQUE",
    "CREATE INDEX req_title IF NOT EXISTS FOR (r:Requirement) ON (r.title)",
    "CREATE INDEX spec_title IF NOT EXISTS FOR (s:Spec) ON (s.title)",
]


def init_schema(client: Neo4jClient) -> None:
    """Create all constraints and indexes."""
    with client.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            session.run(stmt)
    log.info("graph_schema_initialized", statements=len(SCHEMA_STATEMENTS))


def clear_graph(client: Neo4jClient, *, confirm: bool = False) -> None:
    """Delete all nodes and relationships. Requires confirm=True to execute."""
    if not confirm:
        raise ValueError("clear_graph requires confirm=True to prevent accidental data loss")
    with client.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    log.warning("graph_cleared")
