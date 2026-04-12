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


# Maps memory_type keywords (used by the Qdrant payload and the
# dedup helper) to the corresponding Neo4j node label (used by the
# boost/demote cypher dispatch). The two namespaces are different
# by convention — ``memory_type`` is lowercase and plural-friendly,
# ``label`` is the Neo4j PascalCase node label.
_MEMORY_TYPE_TO_LABEL: dict[str, str] = {
    "pattern": "Pattern",
    "mistake": "Mistake",
    "solution": "Solution",
    "strategy": "Strategy",
}


class MemoryRepository:
    """Read/write procedural memories (patterns, mistakes, solutions, strategies)."""

    def __init__(
        self,
        client: Neo4jClient,
        vector_repo: VectorRepository | None = None,
        *,
        dedup_threshold: float = 0.92,
    ) -> None:
        self.client = client
        self.vector_repo = vector_repo
        # Tier A: write-time dedup. See memory/dedup_writer.py. The
        # helper is cheap to construct (just holds refs) so we build
        # one unconditionally; it becomes a no-op when the vector
        # repo is missing or the threshold is set to 0.0 in Settings.
        from dark_factory.memory.dedup_writer import MemoryDedupHelper

        self.dedup_helper = MemoryDedupHelper(
            vector_repo=vector_repo,
            threshold=dedup_threshold,
        )

    def set_dedup_threshold(self, threshold: float) -> None:
        """Live-update the dedup threshold. Called by the Settings
        PATCH handler so operators can tune dedup without a restart."""
        self.dedup_helper.threshold = max(0.0, min(1.0, threshold))

    def _try_dedup_and_boost(
        self,
        *,
        memory_type: str,
        query_text: str,
        source_feature: str,
        boost_delta: float = 0.05,
    ) -> str | None:
        """Run the dedup check for a candidate memory.

        Returns the existing memory id if a near-duplicate was found
        (and boosts its relevance + bumps times_applied), or ``None``
        if the caller should create a new node. Never raises — any
        failure falls through to "no match".

        Instrumentation: fires either ``memory_writes_total{outcome=deduped}``
        + ``memory_relevance_adjustments_total{direction=boost}`` on
        a hit, or nothing on a miss (the caller records the
        ``outcome=created`` counter after the Neo4j insert succeeds).
        """
        match = self.dedup_helper.find_existing_match(
            memory_type=memory_type,
            query_text=query_text,
            source_feature=source_feature,
        )
        if match is None:
            return None

        matched_id = match.get("id", "")
        if not matched_id:
            return None

        # Boost the existing memory's relevance + bump its
        # times_applied counter so frequently-rediscovered memories
        # earn their weight faster. Tolerate boost failures — a
        # transient Neo4j hiccup shouldn't cause us to create a
        # duplicate instead.
        label = _MEMORY_TYPE_TO_LABEL.get(memory_type)
        if label is not None:
            try:
                self.boost_relevance(matched_id, label, delta=boost_delta)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "memory_dedup_boost_failed",
                    matched_id=matched_id,
                    error=str(exc),
                )

        try:
            from dark_factory.metrics.prometheus import (
                observe_memory_relevance_adjustment,
                observe_memory_write,
            )

            observe_memory_write(memory_type=memory_type, outcome="deduped")
            observe_memory_relevance_adjustment(
                memory_type=memory_type, direction="boost", count=1
            )
        except Exception:  # pragma: no cover — defensive
            pass

        return matched_id

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
        # Tier A: dedup check first. If a near-duplicate Pattern exists
        # in the same feature, boost it and return its id rather than
        # creating a duplicate. The helper is a no-op when the vector
        # repo is missing or threshold=0.0.
        existing = self._try_dedup_and_boost(
            memory_type="pattern",
            query_text=f"{description}\n{context}",
            source_feature=source_feature,
        )
        if existing is not None:
            return existing

        node_id = f"pattern-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (p:Pattern {
                    id: $id, description: $description, context: $context,
                    source_feature: $source_feature, source_spec_id: $source_spec_id,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_applied: 0,
                    times_recalled: 0,
                    created_at: $now, updated_at: $now
                })
                """,
                id=node_id, description=description, context=context,
                source_feature=source_feature, source_spec_id=source_spec_id,
                agent=agent, run_id=run_id, now=now,
            )
        self._vector_upsert(node_id, "pattern", description, context,
                            source_feature, source_spec_id, agent)
        _metric_memory_op(
            operation="create",
            memory_type="pattern",
            memory_id=node_id,
            source_feature=source_feature,
            run_id=run_id or None,
        )
        try:
            from dark_factory.metrics.prometheus import observe_memory_write

            observe_memory_write(memory_type="pattern", outcome="created")
        except Exception:  # pragma: no cover — defensive
            pass
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
        existing = self._try_dedup_and_boost(
            memory_type="mistake",
            query_text=f"{description}\n{trigger_context}",
            source_feature=source_feature,
        )
        if existing is not None:
            # Bump the existing Mistake's times_seen counter so
            # "how often have we tripped on this?" is accurate.
            try:
                with self.client.session() as session:
                    session.run(
                        "MATCH (m:Mistake {id: $id}) "
                        "SET m.times_seen = coalesce(m.times_seen, 0) + 1, "
                        "    m.updated_at = $now",
                        id=existing,
                        now=datetime.now(tz=timezone.utc).isoformat(),
                    )
            except Exception:  # pragma: no cover — defensive
                pass
            return existing

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
                    times_recalled: 0,
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
        _metric_memory_op(
            operation="create",
            memory_type="mistake",
            memory_id=node_id,
            source_feature=source_feature,
            run_id=run_id or None,
        )
        try:
            from dark_factory.metrics.prometheus import observe_memory_write

            observe_memory_write(memory_type="mistake", outcome="created")
        except Exception:  # pragma: no cover — defensive
            pass
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
        existing = self._try_dedup_and_boost(
            memory_type="solution",
            query_text=f"{description}\n{code_snippet}",
            source_feature=source_feature,
        )
        if existing is not None:
            # Even when we're deduping, honour the caller's
            # mistake_id linkage — if this solution is being recorded
            # as the fix for a NEW mistake, add the RESOLVED_BY edge
            # even to the boosted existing solution node.
            if mistake_id:
                try:
                    with self.client.session() as session:
                        session.run(
                            """
                            MATCH (m:Mistake {id: $mistake_id})
                            MATCH (s:Solution {id: $solution_id})
                            MERGE (m)-[:RESOLVED_BY]->(s)
                            """,
                            mistake_id=mistake_id, solution_id=existing,
                        )
                except Exception:  # pragma: no cover — defensive
                    pass
            return existing

        node_id = f"solution-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (s:Solution {
                    id: $id, description: $description, code_snippet: $code_snippet,
                    source_feature: $source_feature, source_spec_id: $source_spec_id,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_applied: 0,
                    times_recalled: 0,
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
        _metric_memory_op(
            operation="create",
            memory_type="solution",
            memory_id=node_id,
            source_feature=source_feature,
            run_id=run_id or None,
        )
        try:
            from dark_factory.metrics.prometheus import observe_memory_write

            observe_memory_write(memory_type="solution", outcome="created")
        except Exception:  # pragma: no cover — defensive
            pass
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
        existing = self._try_dedup_and_boost(
            memory_type="strategy",
            query_text=f"{description}\n{applicability}",
            source_feature=source_feature,
        )
        if existing is not None:
            return existing

        node_id = f"strategy-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                CREATE (st:Strategy {
                    id: $id, description: $description, applicability: $applicability,
                    source_feature: $source_feature,
                    agent: $agent, run_id: $run_id, relevance_score: 0.5, times_applied: 0,
                    times_recalled: 0,
                    created_at: $now, updated_at: $now
                })
                """,
                id=node_id, description=description, applicability=applicability,
                source_feature=source_feature, agent=agent, run_id=run_id, now=now,
            )
        self._vector_upsert(node_id, "strategy", description, applicability,
                            source_feature, "", agent)
        _metric_memory_op(
            operation="create",
            memory_type="strategy",
            memory_id=node_id,
            source_feature=source_feature,
            run_id=run_id or None,
        )
        try:
            from dark_factory.metrics.prometheus import observe_memory_write

            observe_memory_write(memory_type="strategy", outcome="created")
        except Exception:  # pragma: no cover — defensive
            pass
        return node_id

    # M4 fix: derive _VALID_LABELS from the single source of truth
    # at class body evaluation time so the two stay in sync. The
    # set is used by the Cypher dispatch functions further down;
    # ``_MEMORY_TYPE_TO_LABEL`` is used by the dedup helper and the
    # write paths. Previously they were two independent hardcoded
    # constants and drift risk was real (Tier A introduced the
    # dict; this consolidation pins them together).
    _VALID_LABELS = frozenset(_MEMORY_TYPE_TO_LABEL.values())

    # H1 fix: pre-built Cypher per label, eliminating f-string interpolation
    # of user-facing values into query text. Even though label is validated
    # against _VALID_LABELS, a future refactor bypassing that check would
    # reintroduce injection risk. Dict-dispatch removes the footgun entirely.
    _BOOST_CYPHER: dict[str, str] = {
        "Pattern": """
            MATCH (n:Pattern {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score + $delta > 1.0 THEN 1.0
                ELSE n.relevance_score + $delta
            END,
            n.times_applied = n.times_applied + 1,
            n.updated_at = $now
        """,
        "Mistake": """
            MATCH (n:Mistake {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score + $delta > 1.0 THEN 1.0
                ELSE n.relevance_score + $delta
            END,
            n.times_seen = n.times_seen + 1,
            n.updated_at = $now
        """,
        "Solution": """
            MATCH (n:Solution {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score + $delta > 1.0 THEN 1.0
                ELSE n.relevance_score + $delta
            END,
            n.times_applied = n.times_applied + 1,
            n.updated_at = $now
        """,
        "Strategy": """
            MATCH (n:Strategy {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score + $delta > 1.0 THEN 1.0
                ELSE n.relevance_score + $delta
            END,
            n.times_applied = n.times_applied + 1,
            n.updated_at = $now
        """,
    }

    _DEMOTE_CYPHER: dict[str, str] = {
        "Pattern": """
            MATCH (n:Pattern {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score - $delta < 0.0 THEN 0.0
                ELSE n.relevance_score - $delta
            END,
            n.times_applied = n.times_applied + 1,
            n.updated_at = $now
        """,
        "Mistake": """
            MATCH (n:Mistake {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score - $delta < 0.0 THEN 0.0
                ELSE n.relevance_score - $delta
            END,
            n.times_seen = n.times_seen + 1,
            n.updated_at = $now
        """,
        "Solution": """
            MATCH (n:Solution {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score - $delta < 0.0 THEN 0.0
                ELSE n.relevance_score - $delta
            END,
            n.times_applied = n.times_applied + 1,
            n.updated_at = $now
        """,
        "Strategy": """
            MATCH (n:Strategy {id: $id})
            SET n.relevance_score = CASE
                WHEN n.relevance_score - $delta < 0.0 THEN 0.0
                ELSE n.relevance_score - $delta
            END,
            n.times_applied = n.times_applied + 1,
            n.updated_at = $now
        """,
    }

    def boost_relevance(self, node_id: str, label: str, delta: float = 0.1) -> None:
        """Increment relevance_score and usage counter for a memory node.

        H3 fix: wrapped in ``session.execute_write`` so Neo4j's
        per-node write lock serialises concurrent boost/demote calls
        on the same node. Two workers that both call
        ``boost_relevance`` on the same id no longer lose updates via
        a MATCH/SET race — the second call's ``MATCH`` blocks until
        the first's transaction commits, then reads the freshly-boosted
        value before applying its own increment.
        """
        cypher = self._BOOST_CYPHER.get(label)
        if cypher is None:
            log.warning("boost_invalid_label", label=label, node_id=node_id)
            return
        now = datetime.now(tz=timezone.utc).isoformat()

        def _tx(tx) -> None:
            tx.run(cypher, id=node_id, delta=delta, now=now)

        with self.client.session() as session:
            # Some tests pass MagicMock sessions that don't implement
            # ``execute_write`` — fall back to ``run`` so existing unit
            # tests keep passing without mocking the tx helper.
            exec_write = getattr(session, "execute_write", None)
            if exec_write is not None:
                exec_write(_tx)
            else:
                session.run(cypher, id=node_id, delta=delta, now=now)
        _metric_memory_op(
            operation="boost",
            memory_type=label.lower(),
            memory_id=node_id,
            delta=delta,
        )

    def demote_relevance(self, node_id: str, label: str, delta: float = 0.05) -> None:
        """Decrease relevance_score, floored at 0.0.

        H3 fix: same write-transaction wrapping as ``boost_relevance``
        so concurrent feedback signals on the same memory node are
        serialised by Neo4j's per-node write lock.
        """
        cypher = self._DEMOTE_CYPHER.get(label)
        if cypher is None:
            log.warning("demote_invalid_label", label=label, node_id=node_id)
            return
        now = datetime.now(tz=timezone.utc).isoformat()

        def _tx(tx) -> None:
            tx.run(cypher, id=node_id, delta=delta, now=now)

        with self.client.session() as session:
            exec_write = getattr(session, "execute_write", None)
            if exec_write is not None:
                exec_write(_tx)
            else:
                session.run(cypher, id=node_id, delta=delta, now=now)
        _metric_memory_op(
            operation="demote",
            memory_type=label.lower(),
            memory_id=node_id,
            delta=delta,
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

    def update_run_counts(
        self,
        *,
        run_id: str,
        spec_count: int | None = None,
        feature_count: int | None = None,
    ) -> None:
        """Bump the spec/feature counts on a running Run node as the pipeline
        discovers them. Used by the bridge after Phase 2 (spec gen) and the
        orchestrator after Phase 3 (graph) to keep the Run History entry
        accurate while the pipeline is still running."""
        sets = []
        params: dict = {"id": run_id}
        if spec_count is not None:
            sets.append("r.spec_count = $spec_count")
            params["spec_count"] = spec_count
        if feature_count is not None:
            sets.append("r.feature_count = $feature_count")
            params["feature_count"] = feature_count
        if not sets:
            return
        cypher = f"MATCH (r:Run {{id: $id}}) SET {', '.join(sets)}"
        with self.client.session() as session:
            session.run(cypher, **params)

    def mark_run_failed(self, *, run_id: str, error: str) -> None:
        """Mark a Run node as failed (e.g. when the pipeline errors out
        before completing the swarm phase). Stores the error message in
        the worst_features payload for visibility."""
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            session.run(
                """
                MATCH (r:Run {id: $id})
                SET r.status = 'error',
                    r.worst_features = $error_payload,
                    r.duration_seconds = coalesce(r.duration_seconds, 0.0)
                """,
                id=run_id,
                error_payload=json.dumps([{"feature": "(pipeline)", "score": 0.0, "reason": error[:500]}]),
            )
        log.warning("run_marked_failed", run_id=run_id, error=error[:200], at=now_iso)

    # ── Episodic memory ──────────────────────────────────────────────
    #
    # Episodes are per-feature autobiographical records written at the
    # end of every feature swarm. See ``dark_factory.memory.episodes``
    # for the full data model + synthesis flow.

    def write_episode(self, episode: "Episode") -> None:  # noqa: F821
        """Create or update an Episode node linked to its Run.

        Idempotent on the episode's content-addressed id (hash of
        run_id + feature), so re-writing the same episode updates
        the existing node rather than creating duplicates.
        """
        # Late import to break the circular episodes → repository
        # reference — the episodes module imports types FROM this
        # module via TYPE_CHECKING, so we have to defer in the same
        # direction.
        from dark_factory.memory.episodes import Episode as _Episode  # noqa: F401

        # Flatten key_events + tool_calls_summary to JSON for storage
        # (Neo4j property values must be primitives or arrays of
        # primitives — nested dicts get rejected).
        key_events_json = json.dumps(
            [ke.model_dump() for ke in episode.key_events], default=str
        )
        tool_calls_json = json.dumps(episode.tool_calls_summary, default=str)
        eval_scores_json = json.dumps(episode.final_eval_scores, default=str)

        with self.client.session() as session:
            session.run(
                """
                MERGE (ep:Episode {id: $id})
                SET ep.run_id = $run_id,
                    ep.feature = $feature,
                    ep.outcome = $outcome,
                    ep.summary = $summary,
                    ep.turns_used = $turns_used,
                    ep.duration_seconds = $duration_seconds,
                    ep.spec_ids = $spec_ids,
                    ep.agents_visited = $agents_visited,
                    ep.key_events_json = $key_events_json,
                    ep.tool_calls_json = $tool_calls_json,
                    ep.eval_scores_json = $eval_scores_json,
                    ep.started_at = $started_at,
                    ep.ended_at = $ended_at
                WITH ep
                OPTIONAL MATCH (r:Run {id: $run_id})
                FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END |
                    MERGE (ep)-[:PRODUCED_IN]->(r)
                )
                """,
                id=episode.id,
                run_id=episode.run_id,
                feature=episode.feature,
                outcome=episode.outcome,
                summary=episode.summary,
                turns_used=episode.turns_used,
                duration_seconds=episode.duration_seconds,
                spec_ids=list(episode.spec_ids),
                agents_visited=list(episode.agents_visited),
                key_events_json=key_events_json,
                tool_calls_json=tool_calls_json,
                eval_scores_json=eval_scores_json,
                started_at=episode.started_at.isoformat(),
                ended_at=episode.ended_at.isoformat(),
            )
        log.info(
            "episode_written",
            episode_id=episode.id,
            feature=episode.feature,
            run_id=episode.run_id,
            outcome=episode.outcome,
        )

    def get_episodes_for_run(
        self,
        *,
        run_id: str,
        feature: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return all episodes for a run (optionally filtered by feature).

        Sorted by started_at descending so newest-first lists populate
        the UI naturally. The returned dicts carry the Neo4j
        properties directly, so JSON-encoded fields
        (``key_events_json``, ``tool_calls_json``, ``eval_scores_json``)
        need to be decoded by the caller if the structured values
        are needed.
        """
        cypher = "MATCH (ep:Episode {run_id: $run_id})"
        params: dict = {"run_id": run_id, "limit": limit}
        if feature:
            cypher = "MATCH (ep:Episode {run_id: $run_id, feature: $feature})"
            params["feature"] = feature
        cypher += " RETURN ep ORDER BY ep.started_at DESC LIMIT $limit"
        return self._run_search(cypher, params, "ep")

    def search_episodes_keyword(
        self,
        *,
        keywords: str,
        feature: str | None = None,
        outcome: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Keyword-match episodes by summary / feature / outcome.

        Used as the Neo4j half of the ``recall_episodes`` hybrid RRF
        merge. Case-insensitive substring match over the summary
        field — not as powerful as full-text indexing, but
        deterministic and works without the APOC extension.
        """
        clauses = []
        params: dict = {"limit": limit}
        if feature:
            clauses.append("ep.feature = $feature")
            params["feature"] = feature
        if outcome and outcome.lower() != "any":
            clauses.append("ep.outcome = $outcome")
            params["outcome"] = outcome.lower()
        if keywords and keywords.strip():
            clauses.append(
                "toLower(ep.summary) CONTAINS toLower($kw) "
                "OR toLower(ep.feature) CONTAINS toLower($kw)"
            )
            params["kw"] = keywords.strip()
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cypher = (
            "MATCH (ep:Episode) "
            f"{where} "
            "RETURN ep ORDER BY ep.started_at DESC LIMIT $limit"
        )
        return self._run_search(cypher, params, "ep")

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

    def list_evals_by_run(self, *, run_limit: int = 20) -> list[dict]:
        """Return all eval results grouped by run → spec → attempts.

        Used by the browse-first Eval Scores tab to show every evaluation
        across pipeline runs without requiring the user to type a spec ID.

        Result shape::

            [
                {
                    "run_id": "run-...",
                    "timestamp": "...",
                    "status": "success" | "partial" | "error" | "running",
                    "pass_rate": 0.85,
                    "specs": [
                        {
                            "spec_id": "spec-...",
                            "feature_name": "auth",
                            "evals": [
                                {
                                    "id": "eval-...",
                                    "eval_type": "spec" | "test",
                                    "overall_score": 0.85,
                                    "all_passed": True,
                                    "timestamp": "...",
                                    "metrics": [
                                        {"name": "...", "score": 0.85, "passed": True, "reason": "..."}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        Runs are ordered newest first. Specs and evals within a run are
        ordered by timestamp ascending (chronological within the run).
        Orphaned evals (no run_id) are grouped under ``"(orphaned)"`` at the end.
        """
        with self.client.session() as session:
            # Pull recent runs first so we know what to surface
            run_records = session.run(
                """
                MATCH (r:Run)
                RETURN r.id AS id,
                       r.timestamp AS timestamp,
                       r.status AS status,
                       coalesce(r.pass_rate, 0.0) AS pass_rate,
                       coalesce(r.spec_count, 0) AS spec_count
                ORDER BY r.timestamp DESC
                LIMIT $limit
                """,
                limit=run_limit,
            )
            runs_by_id: dict[str, dict] = {}
            ordered_run_ids: list[str] = []
            for r in run_records:
                rid = r["id"]
                ordered_run_ids.append(rid)
                runs_by_id[rid] = {
                    "run_id": rid,
                    "timestamp": r["timestamp"] or "",
                    "status": r["status"] or "?",
                    "pass_rate": r["pass_rate"],
                    "spec_count": r["spec_count"],
                    "specs": {},  # spec_id -> spec entry (dict during build)
                }

            # Pull all eval results — we'll filter and group by run
            eval_records = session.run(
                """
                MATCH (e:EvalResult)
                RETURN e.id AS id,
                       e.spec_id AS spec_id,
                       e.feature_name AS feature_name,
                       e.eval_type AS eval_type,
                       coalesce(e.overall_score, 0.0) AS overall_score,
                       coalesce(e.all_passed, false) AS all_passed,
                       e.run_id AS run_id,
                       e.timestamp AS timestamp,
                       e.metrics AS metrics
                ORDER BY e.timestamp ASC
                """
            )

            orphaned: dict | None = None
            for er in eval_records:
                rid = er["run_id"] or ""
                if rid and rid in runs_by_id:
                    target = runs_by_id[rid]
                else:
                    if orphaned is None:
                        orphaned = {
                            "run_id": "(orphaned)",
                            "timestamp": "",
                            "status": "unknown",
                            "pass_rate": 0.0,
                            "spec_count": 0,
                            "specs": {},
                        }
                    target = orphaned

                sid = er["spec_id"] or "(unknown)"
                if sid not in target["specs"]:
                    target["specs"][sid] = {
                        "spec_id": sid,
                        "feature_name": er["feature_name"] or "",
                        "evals": [],
                    }

                # Parse the metrics JSON blob into a list[{name, score, passed, reason}]
                metrics_list: list[dict] = []
                metrics_raw = er["metrics"]
                if metrics_raw:
                    try:
                        parsed = json.loads(metrics_raw)
                        if isinstance(parsed, dict):
                            for name, m in parsed.items():
                                if isinstance(m, dict):
                                    metrics_list.append(
                                        {
                                            "name": name,
                                            "score": float(m.get("score", 0.0) or 0.0),
                                            "passed": bool(m.get("passed", False)),
                                            "reason": (m.get("reason") or "")[:500],
                                        }
                                    )
                                else:
                                    metrics_list.append({"name": name, "score": 0.0, "passed": False, "reason": ""})
                    except Exception:
                        pass

                target["specs"][sid]["evals"].append(
                    {
                        "id": er["id"],
                        "eval_type": er["eval_type"] or "spec",
                        "overall_score": float(er["overall_score"] or 0.0),
                        "all_passed": bool(er["all_passed"]),
                        "timestamp": er["timestamp"] or "",
                        "metrics": metrics_list,
                    }
                )

        # Convert specs dicts to lists, preserving newest-first run order
        output: list[dict] = []
        for rid in ordered_run_ids:
            entry = runs_by_id[rid]
            entry["specs"] = list(entry["specs"].values())
            output.append(entry)
        if orphaned is not None:
            orphaned["specs"] = list(orphaned["specs"].values())
            output.append(orphaned)
        return output

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

    # ── Memory stats (Tier A observability) ─────────────────────────

    def get_memory_stats(self) -> dict:
        """Return counts + relevance distribution per memory type.

        Powers the Memory section of the Metrics tab. Each entry
        carries the node count, mean / median / min / max relevance
        score, and a 10-bucket histogram of relevance distribution
        that the frontend renders as a bar chart.

        Shape::

            {
                "Pattern": {
                    "count": 342,
                    "mean_relevance": 0.61,
                    "median_relevance": 0.58,
                    "min_relevance": 0.02,
                    "max_relevance": 0.99,
                    "histogram": [12, 34, 56, 78, 90, 45, 23, 12, 4, 0],
                },
                "Mistake": {...},
                ...
            }

        Labels without any nodes are still included with zero
        counts so the frontend can render placeholders without
        tripping on missing keys.
        """
        labels = ["Pattern", "Mistake", "Solution", "Strategy", "Episode"]
        stats: dict[str, dict] = {}

        with self.client.session() as session:
            for label in labels:
                # The Episode label doesn't carry relevance_score
                # today (Stage 3 didn't add one), so we only project
                # count for it. Everything else gets the full stats.
                if label == "Episode":
                    record = session.run(
                        "MATCH (n:Episode) RETURN count(n) AS cnt"
                    ).single()
                    count = int(record["cnt"]) if record else 0
                    stats[label] = {
                        "count": count,
                        "mean_relevance": 0.0,
                        "median_relevance": 0.0,
                        "min_relevance": 0.0,
                        "max_relevance": 0.0,
                        "histogram": [0] * 10,
                    }
                    continue

                # Fetch all relevance scores for this label and
                # compute everything client-side. Neo4j 5 has
                # percentileCont for median but the implementation
                # varies by version; doing it in Python is trivial
                # and version-independent.
                cypher = (
                    f"MATCH (n:{label}) "
                    "RETURN coalesce(n.relevance_score, 0.5) AS s"
                )
                scores = [
                    float(row["s"]) for row in session.run(cypher) if row is not None
                ]
                count = len(scores)
                if count == 0:
                    stats[label] = {
                        "count": 0,
                        "mean_relevance": 0.0,
                        "median_relevance": 0.0,
                        "min_relevance": 0.0,
                        "max_relevance": 0.0,
                        "histogram": [0] * 10,
                    }
                    continue

                scores_sorted = sorted(scores)
                mean = sum(scores) / count
                if count % 2 == 1:
                    median = scores_sorted[count // 2]
                else:
                    median = (
                        scores_sorted[count // 2 - 1]
                        + scores_sorted[count // 2]
                    ) / 2

                # 10-bucket histogram on [0.0, 1.0]; relevance is
                # clamped there by boost/demote so anything outside
                # the range (shouldn't exist) gets clipped.
                histogram = [0] * 10
                for s in scores:
                    bucket = min(9, max(0, int(s * 10)))
                    histogram[bucket] += 1

                stats[label] = {
                    "count": count,
                    "mean_relevance": round(mean, 4),
                    "median_relevance": round(median, 4),
                    "min_relevance": round(scores_sorted[0], 4),
                    "max_relevance": round(scores_sorted[-1], 4),
                    "histogram": histogram,
                }

        return stats

    def get_top_recalled_memories(
        self,
        *,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Return the top N memories by ``times_recalled``.

        ``times_recalled`` is a counter incremented by
        ``increment_recall_counts`` on every successful recall. Old
        nodes that pre-date Tier A have the field absent — they get
        coalesce-default 0 and rank last. Powers the "most-used
        memories" table in the Memory metrics dashboard.
        """
        labels = (
            [_MEMORY_TYPE_TO_LABEL.get(memory_type, "Pattern")]
            if memory_type
            else ["Pattern", "Mistake", "Solution", "Strategy"]
        )
        out: list[dict] = []
        with self.client.session() as session:
            for label in labels:
                cypher = (
                    f"MATCH (n:{label}) "
                    "RETURN n.id AS id, "
                    "       n.description AS description, "
                    "       n.source_feature AS source_feature, "
                    "       coalesce(n.relevance_score, 0.5) AS relevance_score, "
                    "       coalesce(n.times_recalled, 0) AS times_recalled, "
                    "       coalesce(n.times_applied, 0) AS times_applied, "
                    "       labels(n)[0] AS label "
                    "ORDER BY coalesce(n.times_recalled, 0) DESC, "
                    "         coalesce(n.relevance_score, 0.5) DESC "
                    "LIMIT $limit"
                )
                for row in session.run(cypher, limit=limit):
                    out.append(
                        {
                            "id": row["id"],
                            "description": row["description"],
                            "source_feature": row["source_feature"],
                            "relevance_score": float(row["relevance_score"]),
                            "times_recalled": int(row["times_recalled"]),
                            "times_applied": int(row["times_applied"]),
                            "memory_type": row["label"].lower() if row["label"] else "",
                        }
                    )
        # If we queried all labels, re-sort and cap — the per-label
        # queries each return up to ``limit`` so we need one more
        # pass to produce a global top-N.
        out.sort(
            key=lambda r: (r["times_recalled"], r["relevance_score"]),
            reverse=True,
        )
        return out[:limit]

    def increment_recall_counts(self, memory_ids: list[str]) -> int:
        """Bump ``times_recalled`` on each node by 1.

        Called from the ``recall_memories`` agent tool after it hands
        its result to the LLM. Tolerates unknown ids (they're just
        missed by the MATCH). Returns the number of nodes updated.
        """
        if not memory_ids:
            return 0
        ids = [mid for mid in memory_ids if mid]
        if not ids:
            return 0
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.client.session() as session:
            result = session.run(
                """
                UNWIND $ids AS target_id
                MATCH (n)
                WHERE n.id = target_id
                  AND (n:Pattern OR n:Mistake OR n:Solution OR n:Strategy)
                SET n.times_recalled = coalesce(n.times_recalled, 0) + 1,
                    n.last_recalled_at = $now
                RETURN count(n) AS cnt
                """,
                ids=ids,
                now=now,
            )
            record = result.single()
            return int(record["cnt"]) if record else 0

    def get_recall_effectiveness(self, *, days: int = 7) -> dict:
        """Return aggregate recall feedback stats over the last N days.

        Shape::

            {
                "window_days": 7,
                "boosted": 124,       # eval-pass-attributed boosts
                "demoted": 37,        # eval-fail-attributed demotes
                "decays": 8,          # background 5% decay events
                "total_recalls": 410, # memories returned by recall_*
                "boost_rate": 0.30,   # boosted / total_recalls
            }

        Reads from the ``memory_operations`` Postgres table populated
        by ``_metric_memory_op``. Returns zeros when Postgres isn't
        enabled (the metrics store is optional).
        """
        try:
            from dark_factory.metrics.helpers import fetch_memory_effectiveness
        except Exception:
            return {
                "window_days": days,
                "boosted": 0,
                "demoted": 0,
                "decays": 0,
                "total_recalls": 0,
                "boost_rate": 0.0,
            }
        try:
            return fetch_memory_effectiveness(days=days)
        except Exception as exc:
            log.warning("recall_effectiveness_query_failed", error=str(exc))
            return {
                "window_days": days,
                "boosted": 0,
                "demoted": 0,
                "decays": 0,
                "total_recalls": 0,
                "boost_rate": 0.0,
            }

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
            count = record["cnt"] if record else 0
        # NOTE: ``delta`` is overloaded here — for decay operations it carries
        # the *multiplier* (e.g. 0.95), not an additive score change.
        # ``count`` is the number of affected nodes. Consumers of the
        # ``memory_operations`` table should interpret ``delta`` based on
        # the ``operation`` column.
        _metric_memory_op(operation="decay", count=count, delta=factor)
        return count

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

    def list_memories(
        self,
        *,
        memory_type: str = "all",
        limit: int = 100,
    ) -> list[dict]:
        """Browse all memories without a keyword filter, ordered by relevance.

        Used by the Memory tab dashboard to show what's in procedural memory
        without requiring users to know what to search for.

        ``memory_type`` is one of ``"all"``, ``"pattern"``, ``"mistake"``,
        ``"solution"``, ``"strategy"``. Returns each entry tagged with its
        type so the UI can render mixed-type lists.
        """
        # H1 fix: pre-built Cypher per memory_type, no f-string interpolation
        # of caller-controlled values into query text.
        per_type_cypher: dict[str, str] = {
            "pattern": (
                "MATCH (n:Pattern) "
                "RETURN n ORDER BY coalesce(n.relevance_score, 0.0) DESC "
                "LIMIT $limit"
            ),
            "mistake": (
                "MATCH (n:Mistake) "
                "RETURN n ORDER BY coalesce(n.relevance_score, 0.0) DESC "
                "LIMIT $limit"
            ),
            "solution": (
                "MATCH (n:Solution) "
                "RETURN n ORDER BY coalesce(n.relevance_score, 0.0) DESC "
                "LIMIT $limit"
            ),
            "strategy": (
                "MATCH (n:Strategy) "
                "RETURN n ORDER BY coalesce(n.relevance_score, 0.0) DESC "
                "LIMIT $limit"
            ),
        }

        if memory_type == "all":
            # UNION ALL across every label, ordered by score across the union
            cypher = """
                CALL {
                    MATCH (p:Pattern)
                    RETURN p AS n, 'pattern' AS type, p.relevance_score AS score
                    UNION ALL
                    MATCH (m:Mistake)
                    RETURN m AS n, 'mistake' AS type, m.relevance_score AS score
                    UNION ALL
                    MATCH (s:Solution)
                    RETURN s AS n, 'solution' AS type, s.relevance_score AS score
                    UNION ALL
                    MATCH (st:Strategy)
                    RETURN st AS n, 'strategy' AS type, st.relevance_score AS score
                }
                RETURN n, type, coalesce(score, 0.0) AS score
                ORDER BY score DESC LIMIT $limit
            """
            with self.client.session() as session:
                result = session.run(cypher, limit=limit)
                items = []
                for record in result:
                    node = dict(record["n"])
                    node["type"] = record["type"]
                    items.append(node)
                return items

        cypher = per_type_cypher.get(memory_type)
        if cypher is None:
            return []
        with self.client.session() as session:
            result = session.run(cypher, limit=limit)
            items = []
            for record in result:
                node = dict(record["n"])
                node["type"] = memory_type
                items.append(node)
            return items

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


def _metric_memory_op(**fields) -> None:
    """Best-effort forward to the metrics recorder. Never raises."""
    try:
        from dark_factory.metrics.helpers import record_memory_operation

        record_memory_operation(**fields)
    except Exception:  # pragma: no cover — defensive
        pass
