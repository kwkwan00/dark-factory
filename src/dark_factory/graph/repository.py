"""Domain-specific Neo4j operations. All Cypher lives here."""

from __future__ import annotations

import json

from dark_factory.graph.client import Neo4jClient
from dark_factory.log import trace_methods
from dark_factory.models.domain import Requirement, Spec


@trace_methods
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

    def existing_spec_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ``ids`` that already exist as ``:Spec`` nodes.

        Used by :meth:`SpecStage.run` to skip the full refinement loop for
        targets that are already persisted — preventing the spec
        generation swarm from redoing work on re-runs of the same
        requirements directory. Empty input returns an empty set without
        opening a session.
        """
        if not ids:
            return set()
        with self.client.session() as session:
            result = session.run(
                "MATCH (s:Spec) WHERE s.id IN $ids RETURN s.id AS id",
                ids=list(ids),
            )
            return {r["id"] for r in result}

    def get_specs(self, ids: list[str]) -> list[Spec]:
        """Reconstruct Spec domain models from Neo4j for the given ids.

        Used by the preflight skip path: we need the existing Specs to
        continue flowing through ``context.specs`` into the downstream
        graph stage (and from there into the swarm) without re-running
        the refinement loop. Scenarios are stored as a JSON string on the
        node (see :meth:`upsert_spec`) and decoded here. Nodes that
        don't exist are silently dropped — the caller is expected to
        intersect with :meth:`existing_spec_ids` first.
        """
        if not ids:
            return []

        from dark_factory.models.domain import Scenario

        with self.client.session() as session:
            result = session.run(
                """
                MATCH (s:Spec) WHERE s.id IN $ids
                OPTIONAL MATCH (s)-[:IMPLEMENTS]->(r:Requirement)
                OPTIONAL MATCH (s)-[:DEPENDS_ON]->(d:Spec)
                RETURN s,
                       collect(DISTINCT r.id) AS req_ids,
                       collect(DISTINCT d.id) AS dep_ids
                """,
                ids=list(ids),
            )
            specs: list[Spec] = []
            for record in result:
                node = record["s"]
                if node is None:
                    continue
                props = dict(node)

                # Scenarios are JSON-encoded on write; tolerate bad data
                # by falling back to an empty list so a corrupt row
                # can't take out the entire preflight pass.
                raw_scenarios = props.get("scenarios", "[]")
                scenarios: list[Scenario] = []
                if isinstance(raw_scenarios, str) and raw_scenarios:
                    try:
                        scenarios_data = json.loads(raw_scenarios)
                        if isinstance(scenarios_data, list):
                            for item in scenarios_data:
                                if isinstance(item, dict):
                                    try:
                                        scenarios.append(Scenario(**item))
                                    except Exception:
                                        continue
                    except json.JSONDecodeError:
                        pass

                specs.append(
                    Spec(
                        id=props["id"],
                        title=props.get("title") or "",
                        description=props.get("description") or "",
                        requirement_ids=[r for r in (record["req_ids"] or []) if r],
                        acceptance_criteria=list(props.get("acceptance_criteria") or []),
                        dependencies=[d for d in (record["dep_ids"] or []) if d],
                        scenarios=scenarios,
                        capability=props.get("capability") or "",
                    )
                )
            return specs

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

    def get_requirement(self, req_id: str) -> Requirement | None:
        """Return a single Requirement by ID, or None if not found."""
        with self.client.session() as session:
            result = session.run(
                "MATCH (r:Requirement {id: $id}) RETURN r",
                id=req_id,
            )
            record = result.single()
            if record is None:
                return None
            props = dict(record["r"])
            from dark_factory.models.domain import Priority
            return Requirement(
                id=props["id"],
                title=props.get("title") or "",
                description=props.get("description") or "",
                source_file=props.get("source_file") or "",
                priority=Priority(props.get("priority") or "medium"),
                tags=list(props.get("tags") or []),
            )

    def get_all_requirements(self) -> list[Requirement]:
        """Return all Requirement nodes, ordered by priority then ID."""
        with self.client.session() as session:
            result = session.run(
                "MATCH (r:Requirement) RETURN r ORDER BY r.priority, r.id"
            )
            from dark_factory.models.domain import Priority
            reqs: list[Requirement] = []
            for record in result:
                props = dict(record["r"])
                reqs.append(Requirement(
                    id=props["id"],
                    title=props.get("title") or "",
                    description=props.get("description") or "",
                    source_file=props.get("source_file") or "",
                    priority=Priority(props.get("priority") or "medium"),
                    tags=list(props.get("tags") or []),
                ))
            return reqs

    def get_feature_groups(self) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
        """Return specs grouped by capability and inter-group dependency edges.

        Returns:
            groups: ``{capability: [spec_id, ...]}``
            group_deps: ``{capability: {dependent_capability, ...}}``

        Specs with empty capability get a singleton group keyed by spec_id.
        Uses a single Neo4j query to fetch both grouping and dependency data.
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
            raw_dep_caps: dict[str, set[str]] = {}

            for record in result:
                spec_id = record["id"]
                capability = record["capability"] or spec_id
                groups.setdefault(capability, []).append(spec_id)

                # Build inter-group dependencies from the same result set
                dep_ids = record["dep_ids"] or []
                dep_caps = record["dep_caps"] or []
                for dep_id, dep_cap in zip(dep_ids, dep_caps):
                    dep_group = dep_cap or dep_id
                    if capability != dep_group:
                        raw_dep_caps.setdefault(capability, set()).add(dep_group)

        return groups, raw_dep_caps
