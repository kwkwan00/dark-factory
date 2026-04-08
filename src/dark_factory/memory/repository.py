"""CRUD and search operations on the procedural memory graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from dark_factory.graph.client import Neo4jClient

if TYPE_CHECKING:
    from dark_factory.vector.repository import VectorRepository

log = structlog.get_logger()


class MemoryRepository:
    """Read/write procedural memories (patterns, mistakes, solutions, strategies)."""

    def __init__(self, client: Neo4jClient, vector_repo: VectorRepository | None = None) -> None:
        self.client = client
        self.vector_repo = vector_repo

    def _vector_upsert(self, node_id: str, memory_type: str, description: str,
                        secondary_text: str, source_feature: str, source_spec_id: str,
                        agent: str) -> None:
        """Best-effort upsert to Qdrant alongside Neo4j."""
        if self.vector_repo is None:
            return
        try:
            self.vector_repo.upsert_memory(
                node_id=node_id, memory_type=memory_type,
                description=description, secondary_text=secondary_text,
                source_feature=source_feature, source_spec_id=source_spec_id,
                agent=agent, relevance_score=0.5,
            )
        except Exception as exc:
            log.warning("vector_upsert_failed", node_id=node_id, error=str(exc))

    # ── Write ────────────────────────────────────────────────────────

    def record_pattern(
        self,
        *,
        description: str,
        context: str,
        source_feature: str,
        agent: str,
        source_spec_id: str = "",
        run_id: str = "",
    ) -> str:
        node_id = f"pattern-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (p:Pattern {
                    id: $id, description: $description, context: $context,
                    source_feature: $source_feature, source_spec_id: $source_spec_id,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_applied: 0,
                    created_at: $now, updated_at: $now
                })
                """,
                id=node_id, description=description, context=context,
                source_feature=source_feature, source_spec_id=source_spec_id,
                agent=agent, run_id=run_id, now=now,
            )
        self._vector_upsert(node_id, "pattern", description, context,
                            source_feature, source_spec_id, agent)
        return node_id

    def record_mistake(
        self,
        *,
        description: str,
        error_type: str,
        trigger_context: str,
        source_feature: str,
        agent: str,
        source_spec_id: str = "",
        run_id: str = "",
    ) -> str:
        node_id = f"mistake-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (m:Mistake {
                    id: $id, description: $description, error_type: $error_type,
                    trigger_context: $trigger_context,
                    source_feature: $source_feature, source_spec_id: $source_spec_id,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_seen: 1,
                    created_at: $now, updated_at: $now
                })
                """,
                id=node_id, description=description, error_type=error_type,
                trigger_context=trigger_context,
                source_feature=source_feature, source_spec_id=source_spec_id,
                agent=agent, run_id=run_id, now=now,
            )
        self._vector_upsert(node_id, "mistake", description, trigger_context,
                            source_feature, source_spec_id, agent)
        return node_id

    def record_solution(
        self,
        *,
        description: str,
        source_feature: str,
        agent: str,
        mistake_id: str = "",
        code_snippet: str = "",
        source_spec_id: str = "",
        run_id: str = "",
    ) -> str:
        node_id = f"solution-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (s:Solution {
                    id: $id, description: $description, code_snippet: $code_snippet,
                    source_feature: $source_feature, source_spec_id: $source_spec_id,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_applied: 0,
                    created_at: $now, updated_at: $now
                })
                """,
                id=node_id, description=description, code_snippet=code_snippet,
                source_feature=source_feature, source_spec_id=source_spec_id,
                agent=agent, run_id=run_id, now=now,
            )
            if mistake_id:
                session.run(
                    """
                    MATCH (m:Mistake {id: $mistake_id})
                    MATCH (s:Solution {id: $solution_id})
                    MERGE (m)-[:RESOLVED_BY]->(s)
                    """,
                    mistake_id=mistake_id, solution_id=node_id,
                )
        self._vector_upsert(node_id, "solution", description, code_snippet,
                            source_feature, source_spec_id, agent)
        return node_id

    def record_strategy(
        self,
        *,
        description: str,
        applicability: str,
        source_feature: str,
        agent: str,
        run_id: str = "",
    ) -> str:
        node_id = f"strategy-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (st:Strategy {
                    id: $id, description: $description, applicability: $applicability,
                    source_feature: $source_feature,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_applied: 0,
                    created_at: $now, updated_at: $now
                })
                """,
                id=node_id, description=description, applicability=applicability,
                source_feature=source_feature, agent=agent, run_id=run_id, now=now,
            )
        self._vector_upsert(node_id, "strategy", description, applicability,
                            source_feature, "", agent)
        return node_id

    def boost_relevance(self, node_id: str, label: str, delta: float = 0.1) -> None:
        """Increment relevance_score and usage counter for a memory node."""
        counter = "times_applied" if label != "Mistake" else "times_seen"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                f"""
                MATCH (n:{label} {{id: $id}})
                SET n.relevance_score = CASE
                    WHEN n.relevance_score + $delta > 1.0 THEN 1.0
                    ELSE n.relevance_score + $delta
                END,
                n.{counter} = n.{counter} + 1,
                n.updated_at = $now
                """,
                id=node_id, delta=delta, now=now,
            )

    def demote_relevance(self, node_id: str, label: str, delta: float = 0.05) -> None:
        """Decrease relevance_score, floored at 0.0."""
        counter = "times_applied" if label != "Mistake" else "times_seen"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                f"""
                MATCH (n:{label} {{id: $id}})
                SET n.relevance_score = CASE
                    WHEN n.relevance_score - $delta < 0.0 THEN 0.0
                    ELSE n.relevance_score - $delta
                END,
                n.{counter} = n.{counter} + 1,
                n.updated_at = $now
                """,
                id=node_id, delta=delta, now=now,
            )

    # ── Eval result persistence ──────────────────────────────────────

    def record_eval_result(
        self,
        *,
        spec_id: str,
        feature_name: str,
        eval_type: str,
        metrics: dict,
        run_id: str,
        recalled_memory_ids: list[str] | None = None,
    ) -> str:
        """Persist an evaluation result. Returns node ID."""
        node_id = f"eval-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        scores = [m.get("score", 0) for m in metrics.values() if isinstance(m, dict)]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        all_passed = all(m.get("passed", False) for m in metrics.values() if isinstance(m, dict))

        with self.client.session() as session:
            session.run(
                """
                CREATE (e:EvalResult {
                    id: $id, spec_id: $spec_id, feature_name: $feature_name,
                    eval_type: $eval_type, metrics: $metrics,
                    overall_score: $overall_score, all_passed: $all_passed,
                    run_id: $run_id, recalled_memory_ids: $recalled_ids,
                    timestamp: $now
                })
                """,
                id=node_id, spec_id=spec_id, feature_name=feature_name,
                eval_type=eval_type, metrics=json.dumps(metrics, default=str),
                overall_score=overall_score, all_passed=all_passed,
                run_id=run_id, recalled_ids=json.dumps(recalled_memory_ids or []),
                now=now,
            )
            if run_id:
                session.run(
                    """
                    MATCH (e:EvalResult {id: $eid})
                    MATCH (r:Run {id: $rid})
                    MERGE (e)-[:EVALUATED_IN]->(r)
                    """,
                    eid=node_id, rid=run_id,
                )
        return node_id

    # ── Run lifecycle ────────────────────────────────────────────────

    def create_run(self, *, spec_count: int, feature_count: int) -> str:
        """Create a new Run node with status='running'. Returns run_id."""
        run_id = f"run-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (r:Run {
                    id: $id, timestamp: $now, status: 'running',
                    spec_count: $spec_count, feature_count: $feature_count,
                    pass_rate: 0.0, mean_eval_scores: '{}',
                    worst_features: '[]', duration_seconds: 0.0
                })
                """,
                id=run_id, now=now, spec_count=spec_count, feature_count=feature_count,
            )
        return run_id

    def complete_run(
        self,
        *,
        run_id: str,
        status: str,
        pass_rate: float,
        mean_eval_scores: dict,
        worst_features: list[dict],
        duration_seconds: float,
    ) -> None:
        """Update a Run node with final aggregated stats."""
        with self.client.session() as session:
            session.run(
                """
                MATCH (r:Run {id: $id})
                SET r.status = $status,
                    r.pass_rate = $pass_rate,
                    r.mean_eval_scores = $mean_eval_scores,
                    r.worst_features = $worst_features,
                    r.duration_seconds = $duration_seconds
                """,
                id=run_id, status=status, pass_rate=pass_rate,
                mean_eval_scores=json.dumps(mean_eval_scores, default=str),
                worst_features=json.dumps(worst_features, default=str),
                duration_seconds=duration_seconds,
            )

    # ── Eval history queries ─────────────────────────────────────────

    def get_eval_history(
        self, *, spec_id: str, eval_type: str | None = None, limit: int = 10,
    ) -> list[dict]:
        """Return recent eval results for a spec, newest first."""
        cypher = "MATCH (e:EvalResult {spec_id: $spec_id})"
        params: dict = {"spec_id": spec_id, "limit": limit}
        if eval_type:
            cypher = f"MATCH (e:EvalResult {{spec_id: $spec_id, eval_type: $eval_type}})"
            params["eval_type"] = eval_type
        cypher += " RETURN e ORDER BY e.timestamp DESC LIMIT $limit"
        return self._run_search(cypher, params, "e")

    def get_spec_eval_trend(self, *, spec_id: str, window: int = 5) -> list[float]:
        """Return the last N overall_score values for a spec, oldest first."""
        with self.client.session() as session:
            result = session.run(
                """
                MATCH (e:EvalResult {spec_id: $spec_id})
                RETURN e.overall_score AS score
                ORDER BY e.timestamp DESC LIMIT $window
                """,
                spec_id=spec_id, window=window,
            )
            scores = [record["score"] for record in result if record["score"] is not None]
        return list(reversed(scores))

    def get_run_history(self, *, limit: int = 5) -> list[dict]:
        """Return recent pipeline runs, newest first."""
        return self._run_search(
            "MATCH (r:Run) RETURN r ORDER BY r.timestamp DESC LIMIT $limit",
            {"limit": limit}, "r",
        )

    # ── Decay and feedback ───────────────────────────────────────────

    def decay_all_relevance(self, factor: float = 0.95) -> int:
        """Multiply all memory node relevance_scores by factor. Returns count updated."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n:Pattern OR n:Mistake OR n:Solution OR n:Strategy
                SET n.relevance_score = n.relevance_score * $factor,
                    n.updated_at = $now
                RETURN count(n) AS cnt
                """,
                factor=factor, now=now,
            )
            record = result.single()
            return record["cnt"] if record else 0

    def apply_eval_feedback(
        self,
        *,
        recalled_memory_ids: list[str],
        all_passed: bool,
        boost_delta: float = 0.1,
        demote_delta: float = 0.05,
    ) -> None:
        """After an eval, boost recalled memories if passed, demote if failed."""
        for mem_id in recalled_memory_ids:
            label = self._detect_label(mem_id)
            if not label:
                continue
            if all_passed:
                self.boost_relevance(mem_id, label, delta=boost_delta)
            else:
                self.demote_relevance(mem_id, label, delta=demote_delta)

    def _detect_label(self, node_id: str) -> str | None:
        """Detect the label of a memory node from its ID prefix."""
        for prefix, label in [
            ("pattern-", "Pattern"),
            ("mistake-", "Mistake"),
            ("solution-", "Solution"),
            ("strategy-", "Strategy"),
        ]:
            if node_id.startswith(prefix):
                return label
        return None

    # ── Search ───────────────────────────────────────────────────────

    def search_patterns(
        self, *, keywords: str, agent: str | None = None, limit: int = 5,
    ) -> list[dict]:
        cypher = """
            MATCH (p:Pattern)
            WHERE toLower(p.description) CONTAINS toLower($kw)
               OR toLower(p.context) CONTAINS toLower($kw)
        """
        params: dict = {"kw": keywords, "limit": limit}
        if agent:
            cypher += " AND p.agent = $agent"
            params["agent"] = agent
        cypher += " RETURN p ORDER BY p.relevance_score DESC LIMIT $limit"
        return self._run_search(cypher, params, "p")

    def search_mistakes(
        self, *, keywords: str, error_type: str | None = None, limit: int = 5,
    ) -> list[dict]:
        cypher = """
            MATCH (m:Mistake)
            WHERE toLower(m.description) CONTAINS toLower($kw)
               OR toLower(m.trigger_context) CONTAINS toLower($kw)
        """
        params: dict = {"kw": keywords, "limit": limit}
        if error_type:
            cypher += " AND m.error_type = $error_type"
            params["error_type"] = error_type
        cypher += " RETURN m ORDER BY m.relevance_score DESC LIMIT $limit"
        return self._run_search(cypher, params, "m")

    def search_solutions(self, *, keywords: str, limit: int = 5) -> list[dict]:
        cypher = """
            MATCH (s:Solution)
            WHERE toLower(s.description) CONTAINS toLower($kw)
               OR toLower(s.code_snippet) CONTAINS toLower($kw)
            RETURN s ORDER BY s.relevance_score DESC LIMIT $limit
        """
        return self._run_search(cypher, {"kw": keywords, "limit": limit}, "s")

    def get_strategies(self, *, keywords: str, limit: int = 3) -> list[dict]:
        cypher = """
            MATCH (st:Strategy)
            WHERE toLower(st.description) CONTAINS toLower($kw)
               OR toLower(st.applicability) CONTAINS toLower($kw)
            RETURN st ORDER BY st.relevance_score DESC LIMIT $limit
        """
        return self._run_search(cypher, {"kw": keywords, "limit": limit}, "st")

    def get_related_memories(
        self, *, feature_name: str, spec_id: str = "", limit: int = 10,
    ) -> list[dict]:
        """Return all memory nodes related to a feature or spec."""
        cypher = """
            CALL {
                MATCH (p:Pattern) WHERE p.source_feature = $feat
                    OR p.source_spec_id = $spec RETURN p AS n, p.relevance_score AS score
                UNION ALL
                MATCH (m:Mistake) WHERE m.source_feature = $feat
                    OR m.source_spec_id = $spec RETURN m AS n, m.relevance_score AS score
                UNION ALL
                MATCH (s:Solution) WHERE s.source_feature = $feat
                    OR s.source_spec_id = $spec RETURN s AS n, s.relevance_score AS score
                UNION ALL
                MATCH (st:Strategy) WHERE st.source_feature = $feat
                    RETURN st AS n, st.relevance_score AS score
            }
            RETURN n ORDER BY score DESC LIMIT $limit
        """
        return self._run_search(
            cypher, {"feat": feature_name, "spec": spec_id or "", "limit": limit}, "n",
        )

    # ── Helpers ──────────────────────────────────────────────────────

    # ── Cross-feature learning ─────────────────────────────────────

    def get_run_learnings(
        self, run_id: str, exclude_feature: str = "", limit: int = 20,
    ) -> list[dict]:
        """Return all memories recorded during this run, for cross-feature briefing."""
        cypher = """
            CALL {
                MATCH (p:Pattern) WHERE p.run_id = $run_id
                    RETURN p AS n, 'pattern' AS type, p.created_at AS ts
                UNION ALL
                MATCH (m:Mistake) WHERE m.run_id = $run_id
                    RETURN m AS n, 'mistake' AS type, m.created_at AS ts
                UNION ALL
                MATCH (s:Solution) WHERE s.run_id = $run_id
                    RETURN s AS n, 'solution' AS type, s.created_at AS ts
                UNION ALL
                MATCH (st:Strategy) WHERE st.run_id = $run_id
                    RETURN st AS n, 'strategy' AS type, st.created_at AS ts
            }
        """
        params: dict = {"run_id": run_id, "limit": limit}
        if exclude_feature:
            cypher += " WHERE n.source_feature <> $exclude"
            params["exclude"] = exclude_feature
        cypher += " RETURN n, type ORDER BY ts ASC LIMIT $limit"

        with self.client.session() as session:
            result = session.run(cypher, **params)
            return [
                {"type": record["type"], **dict(record["n"])}
                for record in result
            ]

    def _run_search(self, cypher: str, params: dict, var: str) -> list[dict]:
        with self.client.session() as session:
            result = session.run(cypher, **params)
            return [dict(record[var]) for record in result]
