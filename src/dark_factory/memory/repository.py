"""CRUD and search operations on the procedural memory graph."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from dark_factory.graph.client import Neo4jClient
from dark_factory.log import trace_methods

if TYPE_CHECKING:
    from dark_factory.vector.repository import VectorRepository

log = structlog.get_logger()


# Maps memory_type keywords (used by the Qdrant payload and the
# dedup helper) to the corresponding Neo4j node label (used by the
# boost/demote cypher dispatch). The two namespaces are different
# by convention — ``memory_type`` is lowercase and plural-friendly,
# ``label`` is the Neo4j PascalCase node label.
MEMORY_KIND_TO_LABEL: dict[str, str] = {
    # Swarm-produced kinds.
    "pattern": "Pattern",
    "mistake": "Mistake",
    "solution": "Solution",
    "strategy": "Strategy",
    # Refinery institutional-memory kinds. Live alongside the swarm
    # kinds in the same collection; role-filter policies scope each
    # role's recall so refinery kinds don't leak to swarm callers by
    # default (see swarm_memory_kinds_enabled config).
    "decision": "Decision",
    "constraint": "Constraint",
    "conflict": "Conflict",
    "incident": "Mistake",  # alias — refinery "incident" re-uses :Mistake label
    "hypothesis": "Hypothesis",
    "anti_pattern": "AntiPattern",
}

# Backward-compat alias for module-internal callers; new code should
# import MEMORY_KIND_TO_LABEL from this module directly.
_MEMORY_TYPE_TO_LABEL = MEMORY_KIND_TO_LABEL


_RELEVANCE_LOCK_COUNT = 64
"""Hash-bucket count for the per-node relevance write lock pool. 64 is
enough that contention is rare with ~5 concurrent debates × 4 critic
roles, while keeping the per-process lock count bounded at startup."""


@trace_methods
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
        # Track which memories were already dedup-boosted in this run
        # so the same memory isn't boosted N times when N features
        # independently rediscover it.
        self._dedup_boosted_this_run: set[str] = set()

        # Per-node serialisation for boost/demote. Neo4j's
        # ``execute_write`` already serialises within Neo4j, but the
        # post-commit Qdrant payload sync runs OUTSIDE that lock — so
        # concurrent boost+demote on the same node could land Qdrant
        # writes out of order, leaving Neo4j and Qdrant at different
        # scores. Bucketing node ids into a small fixed lock pool
        # keeps that critical section serialised across both stores
        # without unbounded lock-dict growth.
        self._relevance_lock_pool: list[threading.Lock] = [
            threading.Lock() for _ in range(_RELEVANCE_LOCK_COUNT)
        ]

    def set_dedup_threshold(self, threshold: float) -> None:
        """Live-update the dedup threshold. Called by the Settings
        PATCH handler so operators can tune dedup without a restart."""
        self.dedup_helper.threshold = max(0.0, min(1.0, threshold))

    def reset_run_state(self) -> None:
        """Clear per-run tracking. Call at the start of each pipeline run."""
        self._dedup_boosted_this_run.clear()

    def _try_dedup_and_boost(
        self,
        *,
        memory_type: str,
        query_text: str,
        source_feature: str,
        boost_delta: float = 0.05,
        match_cross_feature: bool = False,
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
            match_cross_feature=match_cross_feature,
        )
        if match is None:
            return None

        matched_id = match.get("id", "")
        if not matched_id:
            return None

        # Boost once per run per memory. Without this cap, N features
        # rediscovering the same pattern would boost it N times (+N*delta)
        # in a single run with no eval evidence.
        label = _MEMORY_TYPE_TO_LABEL.get(memory_type)
        if label is not None and matched_id not in self._dedup_boosted_this_run:
            self._dedup_boosted_this_run.add(matched_id)
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

    def _emit_create_metrics(
        self, memory_type: str, node_id: str, source_feature: str, run_id: str,
    ) -> None:
        """Emit metrics for a newly created memory node. Never raises."""
        _metric_memory_op(
            operation="create",
            memory_type=memory_type,
            memory_id=node_id,
            source_feature=source_feature,
            run_id=run_id or None,
        )
        try:
            from dark_factory.metrics.prometheus import observe_memory_write

            observe_memory_write(memory_type=memory_type, outcome="created")
        except Exception:  # pragma: no cover — defensive
            pass

    def _vector_upsert(self, node_id: str, memory_type: str, description: str,
                        secondary_text: str, source_feature: str,
                        source_spec_id: str = "",
                        agent: str = "", run_id: str = "") -> None:
        """Best-effort upsert to Qdrant alongside Neo4j."""
        if self.vector_repo is None:
            return
        try:
            self.vector_repo.upsert_memory(
                node_id=node_id, memory_type=memory_type,
                description=description, secondary_text=secondary_text,
                source_feature=source_feature, source_spec_id=source_spec_id,
                agent=agent, relevance_score=0.5, run_id=run_id,
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
            match_cross_feature=True,
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
                            source_feature, source_spec_id, agent, run_id)
        self._emit_create_metrics("pattern", node_id, source_feature, run_id)
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
                            source_feature, source_spec_id, agent, run_id)
        self._emit_create_metrics("mistake", node_id, source_feature, run_id)
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
                            source_feature, source_spec_id, agent, run_id)
        self._emit_create_metrics("solution", node_id, source_feature, run_id)
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
            match_cross_feature=True,
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
                            source_feature, "", agent, run_id)
        self._emit_create_metrics("strategy", node_id, source_feature, run_id)
        return node_id

    # ═══════════════════════════════════════════════════════════════
    # Institutional-memory types (refinery-produced)
    # ═══════════════════════════════════════════════════════════════

    def record_refinery_memories_atomic(
        self,
        specs: list[dict[str, Any]],
    ) -> list[str]:
        """Write multiple refinery memories inside ONE Neo4j transaction.

        Used by ``PATCH /api/graph/requirements/{req_id}`` to honour the
        atomic-write-back contract: the requirement upsert and all
        selected suggested-memory writes commit or roll back together.

        Each spec is a dict with keys matching the ``_record_refinery_memory``
        kwargs (``memory_type``, ``summary``, ``body``, ``kind_props``,
        ``source_role``, ``source_requirement_id``, ``rationale``,
        ``provenance_refinery_run_id``, ``run_id``).

        Behaviour:
        - Dedup runs outside the Neo4j transaction (reads the existing
          vector-search index); any hits get boosted and return the
          existing id.
        - For the NEW nodes, a single ``session.execute_write`` opens
          one transaction and commits all CREATEs. Any node's Cypher
          error rolls back the whole batch; no partial state.
        - Qdrant upserts happen AFTER the Neo4j commit. If Qdrant fails,
          we issue a compensating DETACH DELETE for the just-created
          Neo4j nodes so the two stores don't diverge. The
          compensating delete is best-effort and logged.

        Returns the list of memory ids in the same order as ``specs``
        — mixing existing (deduped) and newly created ids.
        """

        from dark_factory.metrics.prometheus import (
            observe_refinery_memory_write_back_failure,
        )

        # Phase 1: dedup + id assignment outside the transaction.
        resolved: list[tuple[str, dict[str, Any] | None]] = []
        for spec in specs:
            memory_type = spec["memory_type"]
            label = _MEMORY_TYPE_TO_LABEL.get(memory_type)
            if label is None:
                raise ValueError(f"unknown refinery memory_type: {memory_type!r}")
            feature = spec.get("source_requirement_id") or "refinery"
            existing = self._try_dedup_and_boost(
                memory_type=memory_type,
                query_text=f"{spec['summary']}\n{spec.get('body', '')}",
                source_feature=feature,
                match_cross_feature=True,
            )
            if existing is not None:
                resolved.append((existing, None))
                continue
            node_id = f"{memory_type}-{uuid4().hex[:8]}"
            now = datetime.now(tz=timezone.utc).isoformat()
            props = {
                "id": node_id,
                "summary": spec["summary"],
                "body": spec.get("body", ""),
                "source_requirement_id": spec.get("source_requirement_id", ""),
                "source_role": spec.get("source_role", ""),
                "rationale": spec.get("rationale", ""),
                "provenance_refinery_run_id": spec.get(
                    "provenance_refinery_run_id", "",
                ),
                "run_id": spec.get("run_id", ""),
                "relevance_score": 0.5,
                "times_recalled": 0,
                "created_at": now,
                "updated_at": now,
                **spec.get("kind_props", {}),
            }
            resolved.append((node_id, {
                "label": label, "props": props, "memory_type": memory_type,
                "feature": feature,
                "source_role": spec.get("source_role", ""),
                "run_id": spec.get("run_id", ""),
                "summary": spec["summary"],
                "body": spec.get("body", ""),
            }))

        new_writes = [(nid, payload) for nid, payload in resolved if payload]
        if not new_writes:
            return [nid for nid, _ in resolved]

        # Phase 2: write all new nodes in one Neo4j transaction.
        def _tx(tx):
            for _nid, payload in new_writes:
                tx.run(
                    f"CREATE (n:{payload['label']} $props)",
                    props=payload["props"],
                )

        with self.client.session() as session:
            session.execute_write(_tx)

        # Phase 3: Qdrant upserts outside the Neo4j transaction. On
        # failure, compensate by deleting the just-created Neo4j nodes.
        try:
            for _nid, payload in new_writes:
                self._vector_upsert(
                    payload["props"]["id"],
                    payload["memory_type"],
                    payload["summary"], payload["body"],
                    payload["feature"], "",
                    payload["source_role"] or "judge",
                    payload["run_id"],
                )
                self._emit_create_metrics(
                    payload["memory_type"],
                    payload["props"]["id"],
                    payload["feature"],
                    payload["run_id"],
                )
        except Exception as exc:
            log.warning(
                "refinery_memory_atomic_qdrant_failed_rolling_back_neo4j",
                error=str(exc),
            )
            observe_refinery_memory_write_back_failure()
            # Compensating delete — best effort; log and continue either way.
            try:
                with self.client.session() as session:
                    session.execute_write(
                        lambda tx: tx.run(
                            "UNWIND $ids AS mid "
                            "MATCH (n) WHERE n.id = mid "
                            "DETACH DELETE n",
                            ids=[nid for nid, _ in new_writes],
                        )
                    )
            except Exception as cleanup_exc:  # pragma: no cover — defensive
                log.error(
                    "refinery_memory_atomic_cleanup_failed",
                    error=str(cleanup_exc),
                )
            raise

        return [nid for nid, _ in resolved]

    def _record_refinery_memory(
        self,
        *,
        memory_type: str,
        summary: str,
        body: str,
        source_role: str,
        kind_props: dict[str, Any],
        source_requirement_id: str = "",
        rationale: str = "",
        provenance_refinery_run_id: str = "",
        run_id: str = "",
    ) -> str:
        """Shared write path for refinery memories.

        Label resolved from ``_MEMORY_TYPE_TO_LABEL``. Dedup via
        ``_try_dedup_and_boost`` — a semantic duplicate within the same
        source feature is boosted rather than re-created. Returns the
        node id (either new or an existing one after boost).
        """

        label = _MEMORY_TYPE_TO_LABEL.get(memory_type)
        if label is None:
            raise ValueError(f"unknown refinery memory_type: {memory_type!r}")

        feature = source_requirement_id or "refinery"
        existing = self._try_dedup_and_boost(
            memory_type=memory_type,
            query_text=f"{summary}\n{body}",
            source_feature=feature,
            match_cross_feature=True,
        )
        if existing is not None:
            return existing

        node_id = f"{memory_type}-{uuid4().hex[:8]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        props: dict[str, Any] = {
            "id": node_id,
            "summary": summary,
            "body": body,
            "source_requirement_id": source_requirement_id,
            "source_role": source_role,
            "rationale": rationale,
            "provenance_refinery_run_id": provenance_refinery_run_id,
            "run_id": run_id,
            "relevance_score": 0.5,
            "times_recalled": 0,
            "created_at": now,
            "updated_at": now,
            **kind_props,
        }
        # Label interpolation is safe because it's resolved from the
        # trusted static ``_MEMORY_TYPE_TO_LABEL`` dict — never user input.
        with self.client.session() as session:
            session.run(f"CREATE (n:{label} $props)", props=props)
        self._vector_upsert(
            node_id, memory_type, summary, body,
            feature, "", source_role, run_id,
        )
        self._emit_create_metrics(memory_type, node_id, feature, run_id)
        return node_id

    def record_decision(
        self,
        *,
        summary: str,
        body: str,
        context: str = "",
        source_requirement_id: str = "",
        source_role: str = "",
        decision_alternatives: list[str] | None = None,
        rationale: str = "",
        provenance_refinery_run_id: str = "",
        run_id: str = "",
    ) -> str:
        """Record a Decision memory — why a particular choice was made
        over alternatives. Emitted when a synthesis Rebuttal accepts or
        rejects a critique with a rationale that generalises beyond the
        current requirement."""

        return self._record_refinery_memory(
            memory_type="decision",
            summary=summary, body=body,
            source_role=source_role,
            kind_props={
                "context": context,
                "decision_alternatives": list(decision_alternatives or []),
            },
            source_requirement_id=source_requirement_id,
            rationale=rationale,
            provenance_refinery_run_id=provenance_refinery_run_id,
            run_id=run_id,
        )

    def record_constraint(
        self,
        *,
        summary: str,
        body: str,
        constraint_domain: str = "system",
        applicability: str = "",
        source_requirement_id: str = "",
        source_role: str = "",
        rationale: str = "",
        provenance_refinery_run_id: str = "",
        run_id: str = "",
    ) -> str:
        """Record a Constraint memory — a system or business limitation
        that future requirements must respect. Emitted when a critic
        cites a concrete limit (stack pin, API rate, compliance rule)
        in a BLOCKER critique."""

        return self._record_refinery_memory(
            memory_type="constraint",
            summary=summary, body=body,
            source_role=source_role,
            kind_props={
                "constraint_domain": constraint_domain,
                "applicability": applicability,
            },
            source_requirement_id=source_requirement_id,
            rationale=rationale,
            provenance_refinery_run_id=provenance_refinery_run_id,
            run_id=run_id,
        )

    def record_anti_pattern(
        self,
        *,
        summary: str,
        body: str,
        alternative: str = "",
        harm: str = "",
        applicability: str = "",
        source_requirement_id: str = "",
        source_role: str = "",
        rationale: str = "",
        provenance_refinery_run_id: str = "",
        run_id: str = "",
    ) -> str:
        """Record an Anti-pattern memory — recurring negative guidance
        ("don't structure auth this way") paired with the recommended
        alternative. Distinct from Mistake (point-event failure) and
        Pattern (positive recurrent guidance). Emitted when a critic
        flags a draft as repeating a known harmful approach AND names a
        better alternative."""

        return self._record_refinery_memory(
            memory_type="anti_pattern",
            summary=summary, body=body,
            source_role=source_role,
            kind_props={
                "alternative": alternative,
                "harm": harm,
                "applicability": applicability,
            },
            source_requirement_id=source_requirement_id,
            rationale=rationale,
            provenance_refinery_run_id=provenance_refinery_run_id,
            run_id=run_id,
        )

    def record_hypothesis(
        self,
        *,
        summary: str,
        body: str,
        verification_query: str,
        status: str = "open",
        source_requirement_id: str = "",
        source_role: str = "",
        rationale: str = "",
        provenance_refinery_run_id: str = "",
        run_id: str = "",
    ) -> str:
        """Record a Hypothesis memory — an untested guess the panel
        flagged for verification. Carries the verification query the
        next debate's Research agent should target as a Librarian
        prompt; status moves open → verified | refuted as future
        debates resolve it."""

        return self._record_refinery_memory(
            memory_type="hypothesis",
            summary=summary, body=body,
            source_role=source_role,
            kind_props={
                "verification_query": verification_query,
                "status": status,
            },
            source_requirement_id=source_requirement_id,
            rationale=rationale,
            provenance_refinery_run_id=provenance_refinery_run_id,
            run_id=run_id,
        )

    def record_conflict(
        self,
        *,
        summary: str,
        body: str,
        conflict_parties: list[str] | None = None,
        conflict_resolution: str | None = None,
        cause: str = "disagreement",
        source_requirement_id: str = "",
        rationale: str = "",
        provenance_refinery_run_id: str = "",
        run_id: str = "",
    ) -> str:
        """Record a Conflict memory — a recurring disagreement worth
        preserving so future debates start with the tradeoff context
        already loaded. Emitted when a finalize round has high
        disagreement OR the reconcile node fires on short-circuit."""

        return self._record_refinery_memory(
            memory_type="conflict",
            summary=summary, body=body,
            source_role="judge",
            kind_props={
                "conflict_parties": list(conflict_parties or []),
                "conflict_resolution": conflict_resolution or "",
                "cause": cause,
            },
            source_requirement_id=source_requirement_id,
            rationale=rationale,
            provenance_refinery_run_id=provenance_refinery_run_id,
            run_id=run_id,
        )

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
    # Boost and demote Cypher now also set last_feedback_at so that
    # decay_all_relevance can skip recently-active memories.
    _BOOST_CYPHER: dict[str, str] = {
        label: f"""
            MATCH (n:{label} {{id: $id}})
            SET n.relevance_score = CASE
                WHEN n.relevance_score + $delta > 1.0 THEN 1.0
                ELSE n.relevance_score + $delta
            END,
            n.{counter} = coalesce(n.{counter}, 0) + 1,
            n.last_feedback_at = $now,
            n.updated_at = $now
        """
        for label, counter in [
            ("Pattern", "times_applied"),
            ("Mistake", "times_seen"),
            ("Solution", "times_applied"),
            ("Strategy", "times_applied"),
            # Phase-9 refinery kinds — counter rides on times_recalled,
            # which is the only universal counter on these nodes.
            ("Decision", "times_recalled"),
            ("Constraint", "times_recalled"),
            ("Conflict", "times_recalled"),
            ("Hypothesis", "times_recalled"),
            ("AntiPattern", "times_recalled"),
        ]
    }

    _DEMOTE_CYPHER: dict[str, str] = {
        label: f"""
            MATCH (n:{label} {{id: $id}})
            SET n.relevance_score = CASE
                WHEN n.relevance_score - $delta < 0.0 THEN 0.0
                ELSE n.relevance_score - $delta
            END,
            n.last_feedback_at = $now,
            n.updated_at = $now
        """
        for label in (
            "Pattern", "Mistake", "Solution", "Strategy",
            "Decision", "Constraint", "Conflict", "Hypothesis",
            "AntiPattern",
        )
    }

    def _sync_qdrant_relevance(self, node_id: str, new_score: float) -> None:
        """Best-effort sync of relevance_score to the Qdrant payload."""
        if self.vector_repo is None:
            return
        try:
            self.vector_repo.update_relevance_score(
                node_id=node_id, new_score=new_score,
            )
        except Exception as exc:
            log.debug("qdrant_relevance_sync_failed", node_id=node_id, error=str(exc))

    _READ_RELEVANCE_CYPHER: dict[str, str] = {
        label: f"MATCH (n:{label} {{id: $id}}) RETURN n.relevance_score AS score"
        for label in (
            "Pattern", "Mistake", "Solution", "Strategy",
            "Decision", "Constraint", "Conflict", "Hypothesis",
            "AntiPattern",
        )
    }

    def _read_relevance(self, node_id: str, label: str) -> float | None:
        """Read the current relevance_score from Neo4j (post-write)."""
        cypher = self._READ_RELEVANCE_CYPHER.get(label)
        if cypher is None:
            return None
        try:
            with self.client.session() as session:
                result = session.run(cypher, id=node_id)
                record = result.single()
                if record and record["score"] is not None:
                    return float(record["score"])
        except Exception:
            pass
        return None

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
        self._mutate_relevance(
            node_id=node_id, label=label, delta=delta,
            cypher=cypher, op="boost",
        )

    def demote_relevance(self, node_id: str, label: str, delta: float = 0.05) -> None:
        """Decrease relevance_score, floored at 0.0.

        Concurrent feedback signals on the same memory node serialise
        through ``_mutate_relevance``'s per-node Python lock, which
        wraps both the Neo4j commit and the Qdrant payload sync — so
        the two stores can't drift under boost+demote contention.
        """
        cypher = self._DEMOTE_CYPHER.get(label)
        if cypher is None:
            log.warning("demote_invalid_label", label=label, node_id=node_id)
            return
        self._mutate_relevance(
            node_id=node_id, label=label, delta=delta,
            cypher=cypher, op="demote",
        )

    def _mutate_relevance(
        self,
        *,
        node_id: str,
        label: str,
        delta: float,
        cypher: str,
        op: str,
    ) -> None:
        """Run a relevance-mutation Cypher + Qdrant payload sync under
        a per-node lock. Bucketed via hash so concurrent writes on
        different nodes don't block each other.
        """

        now = datetime.now(tz=timezone.utc).isoformat()

        def _tx(tx) -> None:
            tx.run(cypher, id=node_id, delta=delta, now=now)

        bucket = self._relevance_lock_pool[
            hash(node_id) % len(self._relevance_lock_pool)
        ]
        with bucket:
            with self.client.session() as session:
                exec_write = getattr(session, "execute_write", None)
                if exec_write is not None:
                    exec_write(_tx)
                else:
                    session.run(cypher, id=node_id, delta=delta, now=now)
            new_score = self._read_relevance(node_id, label)
            if new_score is not None:
                self._sync_qdrant_relevance(node_id, new_score)

        _metric_memory_op(
            operation=op,
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

    def mark_run_cancelled(self, *, run_id: str, duration_seconds: float = 0.0) -> None:
        """Mark a Run node as cancelled by the user."""
        with self.client.session() as session:
            session.run(
                """
                MATCH (r:Run {id: $id})
                SET r.status = 'cancelled',
                    r.duration_seconds = $duration
                """,
                id=run_id,
                duration=duration_seconds,
            )
        log.info("run_marked_cancelled", run_id=run_id)

    def delete_run(self, *, run_id: str) -> dict[str, int]:
        """Delete a Run and all linked data from Neo4j.

        Removes the Run node, its Episodes (+ APPLIED edges),
        EvalResults, and any memories scoped to this run. Returns
        a dict of counts per node type deleted.
        """
        counts: dict[str, int] = {}

        def _delete_all(tx) -> dict[str, int]:
            # Episodes linked to this run (+ their APPLIED edges)
            r = tx.run(
                "MATCH (ep:Episode {run_id: $id}) DETACH DELETE ep RETURN count(ep) AS c",
                id=run_id,
            ).single()
            ep_count = r["c"] if r else 0

            # EvalResults linked to this run
            r = tx.run(
                "MATCH (e:EvalResult {run_id: $id}) DETACH DELETE e RETURN count(e) AS c",
                id=run_id,
            ).single()
            eval_count = r["c"] if r else 0

            # Memories scoped to this run
            r = tx.run(
                """
                MATCH (n) WHERE n.run_id = $id
                  AND (n:Pattern OR n:Mistake OR n:Solution OR n:Strategy)
                DETACH DELETE n RETURN count(n) AS c
                """,
                id=run_id,
            ).single()
            mem_count = r["c"] if r else 0

            # The Run node itself
            r = tx.run(
                "MATCH (r:Run {id: $id}) DETACH DELETE r RETURN count(r) AS c",
                id=run_id,
            ).single()
            run_count = r["c"] if r else 0

            return {
                "episodes": ep_count,
                "eval_results": eval_count,
                "memories": mem_count,
                "runs": run_count,
            }

        with self.client.session() as session:
            exec_write = getattr(session, "execute_write", None)
            if exec_write is not None:
                counts = exec_write(_delete_all)
            else:
                # Fallback for older Neo4j driver versions
                counts = _delete_all(session)

        # Remove episode vectors from Qdrant
        if self.vector_repo is not None:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                self.vector_repo._client.client.delete(
                    collection_name=self.vector_repo._client.collection_name("episodes"),
                    points_selector=Filter(
                        must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
                    ),
                )
            except Exception as exc:
                log.warning("delete_run_qdrant_episodes_failed", error=str(exc))

            # Remove memory vectors scoped to this run
            try:
                self.vector_repo._client.client.delete(
                    collection_name=self.vector_repo._client.collection_name("memories"),
                    points_selector=Filter(
                        must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
                    ),
                )
            except Exception as exc:
                log.warning("delete_run_qdrant_memories_failed", error=str(exc))

        log.info("run_deleted", run_id=run_id, counts=counts)
        return counts

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
                    ep.recalled_memory_ids = $recalled_memory_ids,
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
                recalled_memory_ids=list(episode.recalled_memory_ids),
                started_at=episode.started_at.isoformat(),
                ended_at=episode.ended_at.isoformat(),
            )

            # Create APPLIED edges from the episode to each recalled
            # memory node (Pattern, Mistake, Solution, Strategy, or
            # prior Episode).  These edges enable graph traversal like
            # "which episodes used this pattern?" and "which patterns
            # came from runs that succeeded?".
            recalled = list(episode.recalled_memory_ids)
            if recalled:
                session.run(
                    """
                    MATCH (ep:Episode {id: $ep_id})
                    UNWIND $mem_ids AS mem_id
                    CALL {
                        WITH ep, mem_id
                        OPTIONAL MATCH (m) WHERE m.id = mem_id
                            AND (m:Pattern OR m:Mistake OR m:Solution
                                 OR m:Strategy OR m:Episode)
                        FOREACH (_ IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (ep)-[:APPLIED]->(m)
                        )
                    }
                    """,
                    ep_id=episode.id,
                    mem_ids=recalled,
                )

        log.info(
            "episode_written",
            episode_id=episode.id,
            feature=episode.feature,
            run_id=episode.run_id,
            outcome=episode.outcome,
            recalled_memories=len(episode.recalled_memory_ids),
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
            count = int(record["cnt"]) if record else 0

        # Sync last_recalled_at to Qdrant so filtered searches can
        # distinguish recently-active memories from dormant ones.
        # Batched: single set_payload call with all point IDs.
        if self.vector_repo is not None and ids:
            try:
                point_ids = [self.vector_repo._to_point_id(mid) for mid in ids]
                self.vector_repo._client.client.set_payload(
                    collection_name=self.vector_repo._client.collection_name("memories"),
                    payload={"last_recalled_at": now},
                    points=point_ids,
                )
            except Exception:
                pass  # best-effort

        return count

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

    def decay_all_relevance(
        self, factor: float = 0.95, grace_days: int = 7,
    ) -> int:
        """Multiply stale memory relevance_scores by *factor*.

        Memories that received boost/demote feedback within
        *grace_days* are skipped — only memories that have gone stale
        (no recent eval signal) decay.  Returns count updated.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        cutoff = (
            datetime.now(tz=timezone.utc)
            - __import__("datetime").timedelta(days=grace_days)
        ).isoformat()
        with self.client.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE (n:Pattern OR n:Mistake OR n:Solution OR n:Strategy)
                  AND (n.last_feedback_at IS NULL OR n.last_feedback_at < $cutoff)
                SET n.relevance_score = n.relevance_score * $factor,
                    n.updated_at = $now
                RETURN n.id AS id, n.relevance_score AS score
                """,
                factor=factor, now=now, cutoff=cutoff,
            )
            rows = [(r["id"], r["score"]) for r in result if r["id"]]
        count = len(rows)

        # Bulk-sync decayed scores to Qdrant so vector search ranking
        # stays consistent with Neo4j.
        # TODO(perf): This is N+1 — each update_relevance_score makes a
        # separate Qdrant set_payload RPC. A bulk_update_relevance helper
        # on VectorRepository that batches all (point_id, score) pairs into
        # a single Qdrant batch_update call would reduce round-trips from
        # O(N) to O(1). Acceptable for now since decay runs infrequently
        # (typically once per pipeline run).
        if self.vector_repo is not None and rows:
            for node_id, score in rows:
                try:
                    self.vector_repo.update_relevance_score(
                        node_id=node_id, new_score=float(score),
                    )
                except Exception:
                    pass  # best-effort — one failure shouldn't stop the batch

        _metric_memory_op(operation="decay", count=count, delta=factor)
        return count

    def prune_low_relevance(self, threshold: float = 0.05) -> int:
        """Delete memory nodes with relevance_score below *threshold*.

        Also removes the corresponding Qdrant points. Returns the
        number of nodes deleted. Call after ``decay_all_relevance`` to
        garbage-collect memories that have decayed below usefulness.
        """
        with self.client.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE (n:Pattern OR n:Mistake OR n:Solution OR n:Strategy)
                  AND n.relevance_score < $threshold
                WITH n, n.id AS nid
                DETACH DELETE n
                RETURN nid
                """,
                threshold=threshold,
            )
            deleted_ids = [r["nid"] for r in result if r["nid"]]

        # Remove from Qdrant — batched into a single delete call.
        if self.vector_repo is not None and deleted_ids:
            try:
                from qdrant_client.models import PointIdsList

                point_ids = [self.vector_repo._to_point_id(nid) for nid in deleted_ids]
                self.vector_repo._client.client.delete(
                    collection_name=self.vector_repo._client.collection_name("memories"),
                    points_selector=PointIdsList(points=point_ids),
                )
            except Exception:
                pass  # best-effort

        count = len(deleted_ids)
        if count:
            log.info("memory_pruned", count=count, threshold=threshold)
            _metric_memory_op(operation="prune", count=count, delta=threshold)
        return count

    # Labels that support relevance scoring (have _BOOST/_DEMOTE Cypher).
    # Episodes are detected by _detect_label but don't have relevance_score
    # on the Neo4j node — they're immutable records whose usefulness is
    # tracked via APPLIED edges instead.
    _SCORABLE_LABELS = frozenset({"Pattern", "Mistake", "Solution", "Strategy"})

    # Labels where eval feedback has a clear causal signal.
    # Pattern is excluded: a recalled pattern may be irrelevant to the
    # eval outcome, so boosting/demoting it adds noise. Patterns earn
    # relevance through dedup boosts (rediscovery) and lose it through
    # decay — both signals that don't require eval causation.
    _EVAL_FEEDBACK_LABELS = frozenset({"Mistake", "Solution", "Strategy"})

    def apply_eval_feedback(
        self,
        *,
        recalled_memory_ids: list[str],
        pass_rate: float,
        boost_delta: float = 0.1,
        demote_delta: float = 0.1,
    ) -> None:
        """Apply partial-credit feedback to recalled memories.

        Instead of all-or-nothing, the delta is scaled by the pass rate:
        - pass_rate=1.0 → full boost_delta
        - pass_rate=0.0 → full demote_delta
        - pass_rate=0.6 → net boost of 0.6*boost - 0.4*demote

        Only Mistake, Solution, and Strategy nodes receive feedback.
        Patterns are excluded because the causal link between recalling
        a pattern and an eval outcome is too noisy.
        """
        if not recalled_memory_ids:
            return
        for mem_id in recalled_memory_ids:
            label = self._detect_label(mem_id)
            if not label or label not in self._EVAL_FEEDBACK_LABELS:
                continue
            if pass_rate >= 0.5:
                scaled_delta = boost_delta * pass_rate
                self.boost_relevance(mem_id, label, delta=scaled_delta)
            else:
                scaled_delta = demote_delta * (1.0 - pass_rate)
                self.demote_relevance(mem_id, label, delta=scaled_delta)

    # Derive prefix→label map from _MEMORY_TYPE_TO_LABEL so the two
    # cannot diverge.  Episode is added manually since it isn't a
    # relevance-scored memory type (no entry in _MEMORY_TYPE_TO_LABEL).
    _PREFIX_TO_LABEL: dict[str, str] = {
        f"{k}-": v for k, v in _MEMORY_TYPE_TO_LABEL.items()
    }
    _PREFIX_TO_LABEL["ep-"] = "Episode"

    def _detect_label(self, node_id: str) -> str | None:
        """Detect the label of a memory node from its ID prefix."""
        for prefix, label in self._PREFIX_TO_LABEL.items():
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
            # UNION ALL across every label, ordered by score across the union.
            # For Mistakes, also fetch the linked Solution via RESOLVED_BY.
            cypher = """
                CALL () {
                    MATCH (p:Pattern)
                    RETURN p AS n, 'pattern' AS type, p.relevance_score AS score,
                           null AS resolved_by
                    UNION ALL
                    MATCH (m:Mistake)
                    OPTIONAL MATCH (m)-[:RESOLVED_BY]->(sol:Solution)
                    RETURN m AS n, 'mistake' AS type, m.relevance_score AS score,
                           sol AS resolved_by
                    UNION ALL
                    MATCH (s:Solution)
                    RETURN s AS n, 'solution' AS type, s.relevance_score AS score,
                           null AS resolved_by
                    UNION ALL
                    MATCH (st:Strategy)
                    RETURN st AS n, 'strategy' AS type, st.relevance_score AS score,
                           null AS resolved_by
                }
                RETURN n, type, coalesce(score, 0.0) AS score, resolved_by
                ORDER BY score DESC LIMIT $limit
            """
            with self.client.session() as session:
                result = session.run(cypher, limit=limit)
                return self._enrich_memory_rows(result)

        if memory_type == "mistake":
            cypher = (
                "MATCH (n:Mistake) "
                "OPTIONAL MATCH (n)-[:RESOLVED_BY]->(sol:Solution) "
                "RETURN n, 'mistake' AS type, coalesce(n.relevance_score, 0.0) AS score, "
                "       sol AS resolved_by "
                "ORDER BY score DESC LIMIT $limit"
            )
            with self.client.session() as session:
                result = session.run(cypher, limit=limit)
                return self._enrich_memory_rows(result)

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

    @staticmethod
    def _enrich_memory_rows(result) -> list[dict]:
        """Build memory list with resolved_by linkage for Mistakes."""
        items = []
        for record in result:
            node = dict(record["n"])
            node["type"] = record["type"]
            sol_node = record.get("resolved_by")
            if sol_node is not None and hasattr(sol_node, "get"):
                node["resolved_by"] = {
                    "id": sol_node.get("id"),
                    "description": sol_node.get("description"),
                }
            items.append(node)
        return items

    def check_duplicate(
        self,
        *,
        memory_type: str,
        description: str,
        context: str = "",
    ) -> dict | None:
        """Check if a memory is a semantic duplicate of an existing one.

        Read-only — does NOT write or boost anything. Returns a dict
        with ``id``, ``description``, and ``score`` if a near-duplicate
        exists above the configured threshold, or ``None`` if no match.
        """
        return self.dedup_helper.find_existing_match(
            memory_type=memory_type,
            query_text=f"{description}\n{context}",
            source_feature="",
            match_cross_feature=True,
        )

    def get_related_memories(
        self, *, feature_name: str, spec_id: str = "", limit: int = 10,
    ) -> list[dict]:
        """Return all memory nodes related to a feature.

        For Mistake nodes, the RESOLVED_BY solution (if any) is
        attached inline as a ``resolved_by`` dict so the agent sees
        the problem AND the fix in a single recall hit.

        The ``spec_id`` parameter is accepted for backward compatibility
        but is ignored — memories are never spec-specific.
        """
        cypher = """
            CALL () {
                MATCH (p:Pattern) WHERE p.source_feature = $feat
                    RETURN p AS n, p.relevance_score AS score, null AS solution
                UNION ALL
                MATCH (m:Mistake) WHERE m.source_feature = $feat
                    OPTIONAL MATCH (m)-[:RESOLVED_BY]->(sol:Solution)
                    RETURN m AS n, m.relevance_score AS score, sol AS solution
                UNION ALL
                MATCH (s:Solution) WHERE s.source_feature = $feat
                    RETURN s AS n, s.relevance_score AS score, null AS solution
                UNION ALL
                MATCH (st:Strategy) WHERE st.source_feature = $feat
                    RETURN st AS n, st.relevance_score AS score, null AS solution
            }
            RETURN n, solution ORDER BY score DESC LIMIT $limit
        """
        with self.client.session() as session:
            result = session.run(
                cypher,
                feat=feature_name, limit=limit,
            )
            memories = []
            for record in result:
                mem = dict(record["n"])
                sol_node = record["solution"]
                if sol_node is not None:
                    mem["resolved_by"] = dict(sol_node)
                memories.append(mem)
            return memories

    def delete_memory_node(self, memory_id: str) -> int:
        """Delete a single memory node and its Qdrant vector.

        Returns the number of Neo4j nodes deleted (0 or 1).
        Encapsulates both the Neo4j DETACH DELETE and the Qdrant
        point removal so callers don't need to reach into internals.
        """
        label = self._detect_label(memory_id)
        if label is None:
            return 0

        with self.client.session() as session:
            result = session.run(
                f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n RETURN count(n) AS c",
                id=memory_id,
            )
            record = result.single()
            deleted = record["c"] if record else 0

        if self.vector_repo is not None and deleted:
            try:
                from qdrant_client.models import PointIdsList

                point_id = self.vector_repo._to_point_id(memory_id)
                self.vector_repo._client.client.delete(
                    collection_name=self.vector_repo._client.collection_name("memories"),
                    points_selector=PointIdsList(points=[point_id]),
                )
            except Exception:
                pass

        return deleted

    def get_run_by_id(self, run_id: str) -> dict | None:
        """Fetch a single Run node by its id. Returns None if not found."""
        with self.client.session() as session:
            result = session.run(
                "MATCH (r:Run {id: $id}) RETURN r",
                id=run_id,
            )
            record = result.single()
            if record is None:
                return None
            return dict(record["r"])

    # ── Helpers ──────────────────────────────────────────────────────

    # ── Cross-feature learning ─────────────────────────────────────

    def get_run_learnings(
        self, run_id: str, exclude_feature: str = "", limit: int = 20,
    ) -> list[dict]:
        """Return all memories recorded during this run, for cross-feature briefing."""
        cypher = """
            CALL () {
                MATCH (p:Pattern) WHERE p.run_id = $run_id
                    AND ($exclude = '' OR p.source_feature <> $exclude)
                    RETURN p AS n, 'pattern' AS type, p.created_at AS ts
                UNION ALL
                MATCH (m:Mistake) WHERE m.run_id = $run_id
                    AND ($exclude = '' OR m.source_feature <> $exclude)
                    RETURN m AS n, 'mistake' AS type, m.created_at AS ts
                UNION ALL
                MATCH (s:Solution) WHERE s.run_id = $run_id
                    AND ($exclude = '' OR s.source_feature <> $exclude)
                    RETURN s AS n, 'solution' AS type, s.created_at AS ts
                UNION ALL
                MATCH (st:Strategy) WHERE st.run_id = $run_id
                    AND ($exclude = '' OR st.source_feature <> $exclude)
                    RETURN st AS n, 'strategy' AS type, st.created_at AS ts
            }
            RETURN n, type ORDER BY ts ASC LIMIT $limit
        """
        params: dict = {"run_id": run_id, "limit": limit, "exclude": exclude_feature}

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
