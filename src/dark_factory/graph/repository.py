"""Domain-specific Neo4j operations. All Cypher lives here."""

from __future__ import annotations

import json

from dark_factory.graph.client import Neo4jClient
from dark_factory.models.domain import Requirement, Spec


class GraphRepository:
    """CRUD operations on the knowledge graph."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def upsert_requirement(self, req: Requirement) -> None:
        with self.client.session() as session:
            session.run(
                """
                MERGE (r:Requirement {id: $id})
                SET r.title = $title,
                    r.description = $description,
                    r.source_file = $source_file,
                    r.priority = $priority,
                    r.tags = $tags
                """,
                id=req.id,
                title=req.title,
                description=req.description,
                source_file=req.source_file,
                priority=req.priority.value,
                tags=req.tags,
            )

    def upsert_spec(self, spec: Spec) -> None:
        with self.client.session() as session:
            session.run(
                """
                MERGE (s:Spec {id: $id})
                SET s.title = $title,
                    s.description = $description,
                    s.acceptance_criteria = $acceptance_criteria,
                    s.capability = $capability,
                    s.scenarios = $scenarios
                """,
                id=spec.id,
                title=spec.title,
                description=spec.description,
                acceptance_criteria=spec.acceptance_criteria,
                capability=spec.capability,
                scenarios=json.dumps([s.model_dump() for s in spec.scenarios]),
            )

    def link_spec_to_requirement(self, spec_id: str, req_id: str) -> None:
        with self.client.session() as session:
            session.run(
                """
                MATCH (s:Spec {id: $spec_id})
                MATCH (r:Requirement {id: $req_id})
                MERGE (s)-[:IMPLEMENTS]->(r)
                """,
                spec_id=spec_id,
                req_id=req_id,
            )

    def link_spec_dependency(self, spec_id: str, dep_id: str) -> None:
        with self.client.session() as session:
            session.run(
                """
                MATCH (s:Spec {id: $spec_id})
                MATCH (d:Spec {id: $dep_id})
                MERGE (s)-[:DEPENDS_ON]->(d)
                """,
                spec_id=spec_id,
                dep_id=dep_id,
            )

    def get_spec_with_context(self, spec_id: str) -> str | None:
        """Get a spec and its dependency tree as a formatted string for LLM context."""
        with self.client.session() as session:
            result = session.run(
                """
                MATCH (s:Spec {id: $spec_id})
                OPTIONAL MATCH (s)-[:DEPENDS_ON]->(dep:Spec)
                OPTIONAL MATCH (s)-[:IMPLEMENTS]->(req:Requirement)
                RETURN s, collect(DISTINCT dep) AS deps, collect(DISTINCT req) AS reqs
                """,
                spec_id=spec_id,
            )
            record = result.single()
            if not record:
                return None

            lines = []
            for req in record["reqs"]:
                lines.append(f"Requirement [{req['id']}]: {req['title']} - {req['description']}")
            for dep in record["deps"]:
                lines.append(f"Dependency [{dep['id']}]: {dep['title']} - {dep['description']}")

            return "\n".join(lines) if lines else None

    def get_feature_groups(self) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
        """Return specs grouped by capability and inter-group dependency edges.

        Returns:
            groups: ``{capability: [spec_id, ...]}``
            group_deps: ``{capability: {dependent_capability, ...}}``

        Specs with empty capability get a singleton group keyed by spec_id.
        """
        with self.client.session() as session:
            result = session.run(
                """
                MATCH (s:Spec)
                OPTIONAL MATCH (s)-[:DEPENDS_ON]->(dep:Spec)
                RETURN s.id AS id, s.capability AS capability,
                       collect(DISTINCT dep.id) AS dep_ids,
                       collect(DISTINCT dep.capability) AS dep_caps
                """
            )

            groups: dict[str, list[str]] = {}
            # Map spec_id -> its group key (capability or spec_id)
            spec_to_group: dict[str, str] = {}
            raw_dep_caps: dict[str, set[str]] = {}  # group_key -> dep group_keys

            for record in result:
                spec_id = record["id"]
                capability = record["capability"] or spec_id
                spec_to_group[spec_id] = capability
                groups.setdefault(capability, []).append(spec_id)

            # Second pass: build inter-group dependencies
            result2 = session.run(
                """
                MATCH (s:Spec)-[:DEPENDS_ON]->(dep:Spec)
                RETURN s.id AS id, s.capability AS cap,
                       dep.id AS dep_id, dep.capability AS dep_cap
                """
            )
            for record in result2:
                src_group = record["cap"] or record["id"]
                dep_group = record["dep_cap"] or record["dep_id"]
                if src_group != dep_group:
                    raw_dep_caps.setdefault(src_group, set()).add(dep_group)

        return groups, raw_dep_caps
