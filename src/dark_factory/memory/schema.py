"""Schema initialisation for the procedural memory graph."""

from __future__ import annotations

from dark_factory.graph.client import Neo4jClient

MEMORY_SCHEMA_STATEMENTS = [
    # Uniqueness constraints
    "CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT mistake_id IF NOT EXISTS FOR (m:Mistake) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT solution_id IF NOT EXISTS FOR (s:Solution) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT strategy_id IF NOT EXISTS FOR (st:Strategy) REQUIRE st.id IS UNIQUE",
    # Indexes for common lookups
    "CREATE INDEX pattern_agent IF NOT EXISTS FOR (p:Pattern) ON (p.agent)",
    "CREATE INDEX mistake_error_type IF NOT EXISTS FOR (m:Mistake) ON (m.error_type)",
    "CREATE INDEX strategy_agent IF NOT EXISTS FOR (st:Strategy) ON (st.agent)",
    "CREATE INDEX pattern_feature IF NOT EXISTS FOR (p:Pattern) ON (p.source_feature)",
    "CREATE INDEX mistake_feature IF NOT EXISTS FOR (m:Mistake) ON (m.source_feature)",
    "CREATE INDEX pattern_run IF NOT EXISTS FOR (p:Pattern) ON (p.run_id)",
    "CREATE INDEX mistake_run IF NOT EXISTS FOR (m:Mistake) ON (m.run_id)",
    "CREATE INDEX solution_run IF NOT EXISTS FOR (s:Solution) ON (s.run_id)",
    "CREATE INDEX strategy_run IF NOT EXISTS FOR (st:Strategy) ON (st.run_id)",
    # Eval result and run tracking
    "CREATE CONSTRAINT eval_result_id IF NOT EXISTS FOR (e:EvalResult) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT run_id IF NOT EXISTS FOR (r:Run) REQUIRE r.id IS UNIQUE",
    "CREATE INDEX eval_result_spec IF NOT EXISTS FOR (e:EvalResult) ON (e.spec_id)",
    "CREATE INDEX eval_result_run IF NOT EXISTS FOR (e:EvalResult) ON (e.run_id)",
    "CREATE INDEX eval_result_type IF NOT EXISTS FOR (e:EvalResult) ON (e.eval_type)",
    "CREATE INDEX run_timestamp IF NOT EXISTS FOR (r:Run) ON (r.timestamp)",
]


def init_memory_schema(client: Neo4jClient) -> None:
    """Create constraints and indexes in the memory database."""
    with client.session() as session:
        for stmt in MEMORY_SCHEMA_STATEMENTS:
            session.run(stmt)


def clear_memory(client: Neo4jClient) -> None:
    """Delete all nodes and relationships in the memory database."""
    with client.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
