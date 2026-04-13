"""Parent orchestrator: one swarm subagent per feature in the knowledge graph.

The orchestrator queries Neo4j for features (specs grouped by capability),
computes a topological execution order respecting DEPENDS_ON relationships,
and dispatches each feature to an isolated swarm instance.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from dark_factory.agents.swarm import (
    MAX_HANDOFFS,
    FeatureResult,
    build_feature_swarm,
    init_swarm_context,
)
from dark_factory.config import Settings
from dark_factory.graph.repository import GraphRepository

log = structlog.get_logger()


# ── State schema ─────────────────────────────────────────────────────


class OrchestratorState(TypedDict, total=False):
    # Inputs
    all_spec_ids: list[str]

    # Computed by plan node
    feature_groups: dict[str, list[str]]
    group_deps: dict[str, list[str]]
    execution_order: list[list[str]]

    # Accumulated by execute_layer
    completed_features: list[FeatureResult]
    current_layer: int

    # Run tracking
    run_id: str
    run_start_time: float

    # Strategy adjustment
    strategy_overrides: dict[str, Any]
    layer_pass_rates: list[float]

    # Self-healing: per-layer retry tracking
    layer_retries: dict[int, int]  # layer_idx → attempt count
    retry_layer: int | None  # set by reflection to re-run a layer

    # Final output
    all_artifacts: list[dict[str, Any]]
    all_tests: list[dict[str, Any]]
    pass_rate: float
    mean_eval_scores: dict[str, float]
    worst_features: list[dict[str, Any]]


# ── Topological sort ─────────────────────────────────────────────────


def topological_layers(
    groups: dict[str, list[str]],
    group_deps: dict[str, set[str]],
) -> list[list[str]]:
    """Return features in dependency order, grouped into parallelisable layers.

    Uses a two-stage algorithm:

    1. **Tarjan's SCC**. The feature dependency graph is frequently
       cyclic because the decomposition planner is an LLM and produces
       capability-level deps that aren't transitively consistent (e.g.
       ``auth → dashboard`` AND ``dashboard → auth``). We find all
       strongly connected components — each SCC with more than one
       member is a cycle.

    2. **Kahn's on the condensation**. The SCC condensation is
       guaranteed acyclic, so we Kahn's-layer it normally. When we
       emit a layer, each SCC's members are expanded into that
       layer's feature list (members of the same SCC run in parallel
       because by definition they cannot be ordered).

    Any collapsed cycle is logged as a warning and emitted as a
    progress event so the operator can see which capabilities got
    merged. This is a pragmatic safety net — the alternative
    (raising ``ValueError``) left operators staring at stack traces
    with no recovery path when the planner produced a circular spec.
    """
    all_nodes = set(groups.keys())
    if not all_nodes:
        return []

    # Build ``deps_adj[src] = {nodes src depends on}`` restricted to
    # nodes that are actually in ``groups``. Self-loops are dropped
    # because a node depending on itself is never informative and
    # would force it into its own multi-member SCC.
    deps_adj: dict[str, set[str]] = {n: set() for n in all_nodes}
    for node, deps in group_deps.items():
        if node not in all_nodes:
            continue
        for dep in deps:
            if dep in all_nodes and dep != node:
                deps_adj[node].add(dep)

    # Stage 1: Tarjan's SCC. Returns a list of components; components
    # of size > 1 are cycles.
    sccs = _tarjan_scc(all_nodes, deps_adj)

    # Report any collapsed cycles so the operator can see them without
    # grepping logs. emit_progress is best-effort — if the broker is
    # unavailable (e.g. under unit test) we silently skip it.
    cycles = [sorted(scc) for scc in sccs if len(scc) > 1]
    if cycles:
        log.warning(
            "dependency_cycles_collapsed",
            cycle_count=len(cycles),
            cycles=cycles,
        )
        try:
            from dark_factory.agents.tools import emit_progress

            emit_progress(
                "orchestrator_cycles_collapsed",
                cycle_count=len(cycles),
                cycles=cycles,
            )
        except Exception:  # pragma: no cover — best-effort telemetry
            pass

    # Stage 2: condense into SCC-level DAG and Kahn's-layer it.
    node_to_scc: dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = i

    # ``scc_deps[i]`` = set of SCC indices that SCC ``i`` depends on.
    scc_deps: dict[int, set[int]] = {i: set() for i in range(len(sccs))}
    for src, deps in deps_adj.items():
        src_scc = node_to_scc[src]
        for dep in deps:
            dep_scc = node_to_scc[dep]
            if src_scc != dep_scc:
                scc_deps[src_scc].add(dep_scc)

    # Kahn's on the condensation. An SCC is ready when all its
    # outgoing edges (to other SCCs it depends on) have been emitted.
    scc_in_degree: dict[int, int] = {i: len(scc_deps[i]) for i in range(len(sccs))}
    scc_dependents: dict[int, list[int]] = {i: [] for i in range(len(sccs))}
    for src_scc, deps in scc_deps.items():
        for dep_scc in deps:
            scc_dependents[dep_scc].append(src_scc)

    queue: deque[int] = deque(
        sorted(i for i in range(len(sccs)) if scc_in_degree[i] == 0)
    )
    layers: list[list[str]] = []
    while queue:
        current_layer_sccs = list(queue)
        # Expand each SCC into its members. Within a layer everything
        # runs in parallel, so intra-SCC order doesn't matter — we
        # just sort for deterministic output.
        layer_nodes: list[str] = []
        for scc_idx in current_layer_sccs:
            layer_nodes.extend(sccs[scc_idx])
        layers.append(sorted(layer_nodes))

        next_queue: list[int] = []
        for scc_idx in current_layer_sccs:
            for dep_scc in scc_dependents[scc_idx]:
                scc_in_degree[dep_scc] -= 1
                if scc_in_degree[dep_scc] == 0:
                    next_queue.append(dep_scc)
        queue = deque(sorted(next_queue))

    return layers


def _tarjan_scc(
    nodes: set[str],
    adj: dict[str, set[str]],
) -> list[set[str]]:
    """Tarjan's strongly connected components algorithm.

    Returns the SCCs of the directed graph ``(nodes, adj)`` where
    ``adj[u] = {v : edge u → v}``. Each returned set is one SCC;
    singleton SCCs represent ordinary DAG nodes, multi-member SCCs
    are cycles.

    Iterative implementation to avoid Python's ~1000-frame default
    recursion limit on pathological inputs (swarms with thousands of
    specs are plausible once decomposition is aggressive).
    """
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    scc_stack: list[str] = []
    result: list[set[str]] = []
    index_counter = 0

    for start in sorted(nodes):
        if start in indices:
            continue

        # Initialise the starting node
        indices[start] = index_counter
        lowlinks[start] = index_counter
        index_counter += 1
        scc_stack.append(start)
        on_stack.add(start)

        # Work stack entries: (node, neighbour_iterator)
        work_stack: list[tuple[str, Any]] = [
            (start, iter(sorted(adj.get(start, ()))))
        ]

        while work_stack:
            v, it = work_stack[-1]
            try:
                w = next(it)
            except StopIteration:
                # All neighbours visited — pop v and maybe emit its SCC.
                work_stack.pop()
                if lowlinks[v] == indices[v]:
                    scc: set[str] = set()
                    while True:
                        w_popped = scc_stack.pop()
                        on_stack.discard(w_popped)
                        scc.add(w_popped)
                        if w_popped == v:
                            break
                    result.append(scc)
                # Propagate v's lowlink to its parent (Tarjan's
                # back-edge relaxation for the iterative variant).
                if work_stack:
                    parent = work_stack[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[v])
                continue

            if w not in indices:
                # Descend into w
                indices[w] = index_counter
                lowlinks[w] = index_counter
                index_counter += 1
                scc_stack.append(w)
                on_stack.add(w)
                work_stack.append((w, iter(sorted(adj.get(w, ())))))
            elif w in on_stack:
                # Back-edge into the current SCC — relax v's lowlink.
                lowlinks[v] = min(lowlinks[v], indices[w])
            # else: cross-edge into a finished SCC; ignore.

    return result


# ── Graph nodes ──────────────────────────────────────────────────────


def make_plan_node(repo: GraphRepository):
    """Create the plan node that queries the graph and computes execution order."""

    def plan_node(state: OrchestratorState) -> dict:
        groups, group_deps = repo.get_feature_groups()

        # Filter to only requested spec_ids if provided
        requested = set(state.get("all_spec_ids", []))
        if requested:
            groups = {
                k: [sid for sid in v if sid in requested]
                for k, v in groups.items()
            }
            groups = {k: v for k, v in groups.items() if v}

        order = topological_layers(groups, group_deps)

        log.info(
            "orchestrator_plan",
            features=len(groups),
            layers=len(order),
            order=order,
        )
        return {
            "feature_groups": groups,
            "group_deps": group_deps,
            "execution_order": order,
            "completed_features": [],
            "current_layer": 0,
            "strategy_overrides": {},
            "layer_pass_rates": [],
        }

    return plan_node


def make_execute_layer_node(
    model: "str | Any",
    max_parallel: int = 4,
    default_max_handoffs: int = MAX_HANDOFFS,
    *,
    enable_episodic_memory: bool = True,
    settings: "Settings | None" = None,
    agent_models: dict[str, Any] | None = None,
):
    """Create the execute_layer node with cross-feature learning and parallel execution.

    ``model`` accepts either a langchain ``BaseChatModel`` instance
    (the preferred path — constructed via
    :func:`~dark_factory.agents.swarm.build_chat_model` with explicit
    ``max_tokens`` and ``timeout``) or a provider string like
    ``"anthropic:claude-sonnet-4-6"`` (the old path, still accepted
    so existing call sites and tests don't break).
    """

    def execute_layer_node(state: OrchestratorState) -> dict:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime, timezone

        from dark_factory.agents.cancellation import (
            is_cancelled,
            raise_if_cancelled,
        )
        from dark_factory.agents.swarm import run_feature_swarm
        from dark_factory.agents.tools import _memory_repo, _vector_repo

        # Cancellation check at the top of every layer — if the user hit
        # cancel between layers, unwind immediately before spending any
        # more LLM budget on the next wave of features.
        raise_if_cancelled()

        layer_idx = state.get("current_layer", 0)
        order = state.get("execution_order", [])
        groups = state.get("feature_groups", {})
        completed = list(state.get("completed_features", []))
        overrides = state.get("strategy_overrides", {})
        run_id = state.get("run_id", "")
        # Strategy adjust can override the default per-layer (e.g. shrink budget
        # after a poor pass-rate); otherwise use the configured default.
        max_handoffs = overrides.get("max_handoffs", default_max_handoffs)

        if layer_idx >= len(order):
            return {"current_layer": layer_idx}

        layer = order[layer_idx]
        # C6 fix: include 'skipped' so transitive failures propagate.
        # If A fails → B is skipped → C (depends on B) must also be skipped.
        failed_features = {
            r["feature"] for r in completed if r["status"] in ("error", "skipped")
        }

        # ── Episode writer setup ─────────────────────────────────────
        # Lazily construct the EpisodeWriter once per layer invocation
        # so we pay the import + LLM client cost only when episodic
        # memory is actually enabled. All references are captured in
        # a closure so the per-feature callers can use them without
        # re-threading settings through the call chain.
        episode_writer = None
        episode_llm = None
        if enable_episodic_memory and _memory_repo is not None and run_id:
            try:
                from dark_factory.memory.episodes import EpisodeWriter
                from dark_factory.ui.helpers import build_llm

                # Pull the embeddings service directly off the vector
                # repo (if one is installed). Matches the pattern used
                # by recall_memories.
                embedder = None
                if _vector_repo is not None:
                    embedder = getattr(_vector_repo, "_embeddings", None)

                episode_llm = build_llm(settings) if settings is not None else None
                episode_writer = EpisodeWriter(
                    memory_repo=_memory_repo,
                    vector_repo=_vector_repo,
                    embeddings=embedder,
                )
            except Exception as exc:
                log.warning(
                    "episode_writer_init_failed",
                    error=str(exc),
                )
                episode_writer = None
                episode_llm = None

        # Track the wall-clock start of each feature so the Episode's
        # started_at reflects the actual start, not the end minus
        # stats['duration_seconds'] (which can be off when the swarm
        # includes queue-wait time). Map feature_name → datetime.
        feature_start_times: dict[str, datetime] = {}

        def _write_episode(feature_result: FeatureResult) -> None:
            """Best-effort: synthesise + persist an Episode for a
            completed FeatureResult. Never raises."""
            if episode_writer is None:
                return
            try:
                from dark_factory.memory.episodes import (
                    episode_from_feature_result,
                )

                feature_name = feature_result.get("feature", "?")
                started_at = feature_start_times.get(feature_name)
                episode = episode_from_feature_result(
                    run_id=run_id,
                    feature_result=dict(feature_result),
                    started_at=started_at,
                    progress_events=[],
                    llm=episode_llm,
                )
                episode_writer.write(episode)
            except Exception as exc:
                # Any failure in the episode-write path is swallowed —
                # losing an episode is acceptable; breaking the
                # pipeline because we couldn't log one is not.
                log.warning(
                    "episode_write_best_effort_failed",
                    feature=feature_result.get("feature"),
                    run_id=run_id,
                    error=str(exc),
                )

        # Build cross-feature briefing from this run's learnings so far
        run_context = ""
        if _memory_repo and run_id:
            try:
                learnings = _memory_repo.get_run_learnings(run_id, limit=20)
                if learnings:
                    lines = []
                    for mem in learnings:
                        mtype = mem.get("type", "?")
                        desc = mem.get("description", "")[:100]
                        feat = mem.get("source_feature", "?")
                        lines.append(f"[{mtype.upper()} from {feat}] {desc}")
                    run_context = "\n".join(lines)[:2000]
            except Exception as exc:
                # Memory repo failures here are non-fatal: we proceed with an
                # empty run_context rather than aborting the layer. Log so a
                # flaky memory DB shows up in diagnostics instead of silently
                # degrading cross-feature learning quality.
                log.warning(
                    "run_learnings_fetch_failed",
                    run_id=run_id,
                    error=str(exc),
                )

        # Collect features to run (filtering skipped)
        to_run: list[tuple[str, list[str]]] = []
        for feature_name in layer:
            spec_ids = groups.get(feature_name, [])
            if not spec_ids:
                continue
            deps_of_feature = _get_deps_from_completed(feature_name, state)
            if deps_of_feature & failed_features:
                log.warning("feature_skipped", feature=feature_name, reason="dependency_failed")
                from dark_factory.agents.tools import emit_progress as _emit

                _emit(
                    "feature_skipped",
                    feature=feature_name,
                    reason=f"dependency in {sorted(deps_of_feature & failed_features)} failed",
                )
                completed.append(
                    FeatureResult(
                        feature=feature_name,
                        spec_ids=spec_ids,
                        status="skipped",
                        artifacts=[],
                        tests=[],
                        error=f"Skipped: dependency in {deps_of_feature & failed_features} failed",
                        eval_scores={},
                    )
                )
                continue
            to_run.append((feature_name, spec_ids))

        from dark_factory.agents.tools import emit_progress as _emit

        _emit(
            "layer_started",
            layer=layer_idx + 1,
            total_layers=len(order),
            features=[f for f, _ in to_run],
        )

        # H10 fix: build the swarm graph ONCE per layer, not per feature.
        # All features in a layer share the same strategy_overrides, so the
        # compiled graph can be reused across parallel workers.
        compiled_for_layer = build_feature_swarm(
            model, strategy_overrides=overrides, models=agent_models,
        )

        # Run features in parallel within the layer
        def _run_one(feature_name: str, spec_ids: list[str]) -> FeatureResult:
            from dark_factory.agents.tools import (
                _thread_local,
                get_inflight_deep_agent_count,
            )

            # A1 fix: reset the H6 in-flight counter at the start of
            # every feature. ThreadPoolExecutor workers are reused
            # across features within a layer; without this reset, a
            # counter drift from a prior feature's crash path would
            # produce false-positive "inflight agents on crash"
            # signals for every subsequent feature scheduled on the
            # same worker thread. One line to keep the diagnostic
            # accurate.
            _thread_local.inflight_deep_agents = 0

            log.info("feature_swarm_starting", feature=feature_name, specs=len(spec_ids))
            _emit("feature_started", feature=feature_name, spec_count=len(spec_ids))
            # Capture wall-clock start for Episode.started_at — the
            # swarm stats carry duration but not start timestamp.
            feature_start_times[feature_name] = datetime.now(timezone.utc)
            try:
                result = run_feature_swarm(
                    compiled_for_layer, spec_ids, feature_name,
                    run_context=run_context,
                    max_handoffs=max_handoffs,
                )
            except Exception:
                # B1 fix: was ``except BaseException`` which caught
                # KeyboardInterrupt / SystemExit / CancelledError and
                # counted clean shutdown events as worker crashes. The
                # H6 diagnostic should only fire on genuine crashes
                # (any subclass of Exception); Python's BaseException
                # subclasses are reserved for shutdown flow and must
                # propagate untouched.
                #
                # H6: if the worker is about to die with a deep-agent
                # call still in-flight, bump the Prometheus counter so
                # dashboards can answer "are we leaking subprocesses?".
                # The SDK's own async cleanup still runs on
                # BackgroundLoop — this is the diagnostic signal that
                # tells operators when the cleanup path is being
                # exercised under fault conditions.
                if get_inflight_deep_agent_count() > 0:
                    try:
                        from dark_factory.metrics.prometheus import (
                            worker_crashes_with_inflight_agents_total,
                        )

                        worker_crashes_with_inflight_agents_total.labels(
                            feature=feature_name
                        ).inc()
                    except Exception:  # pragma: no cover — defensive
                        pass
                    log.warning(
                        "feature_worker_crashed_with_inflight_deep_agents",
                        feature=feature_name,
                        inflight=get_inflight_deep_agent_count(),
                    )
                raise
            # Forward swarm stats into the feature_completed payload so the
            # metrics recorder denormalises them into swarm_feature_events.
            stats = result.get("stats") or {}
            _emit(
                "feature_completed",
                feature=feature_name,
                status=result["status"],
                artifacts=len(result.get("artifacts", [])),
                tests=len(result.get("tests", [])),
                error=result.get("error"),
                layer=layer_idx + 1,
                duration_seconds=stats.get("duration_seconds"),
                agent_transitions=stats.get("agent_transitions"),
                unique_agents_visited=stats.get("unique_agents_visited"),
                planner_calls=stats.get("planner_calls"),
                coder_calls=stats.get("coder_calls"),
                reviewer_calls=stats.get("reviewer_calls"),
                tester_calls=stats.get("tester_calls"),
                tool_call_count=stats.get("tool_call_count"),
                tool_failure_count=stats.get("tool_failure_count"),
                deep_agent_invocations=stats.get("deep_agent_invocations"),
                subprocess_spawn_count=stats.get("subprocess_spawn_count"),
                worker_crash_count=stats.get("worker_crash_count"),
            )
            return result

        workers = min(max_parallel, len(to_run)) if to_run else 1
        # Manual executor lifecycle so we can pass ``cancel_futures=True``
        # on the cancellation path. The ``with`` statement calls
        # ``shutdown(wait=True)`` without cancel_futures, which would
        # block on every already-running worker — exactly the hang the
        # kill-switch is supposed to prevent. Python 3.9+ supports
        # ``cancel_futures`` on shutdown.
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {
                executor.submit(_run_one, fname, sids): fname
                for fname, sids in to_run
            }
            for future in as_completed(futures):
                # Cancellation mid-layer: mark any features whose futures
                # haven't completed yet as skipped so the result shape is
                # complete, and stop waiting for them. ``as_completed``
                # doesn't support early break + cancel cleanly, so we
                # best-effort cancel and continue draining the iterator.
                if is_cancelled():
                    for pending_future, pending_name in futures.items():
                        if pending_future is future or pending_future.done():
                            continue
                        if pending_future.cancel():
                            _skipped = FeatureResult(
                                feature=pending_name,
                                spec_ids=groups.get(pending_name, []),
                                status="skipped",
                                artifacts=[],
                                tests=[],
                                error="Cancelled by user",
                                eval_scores={},
                            )
                            completed.append(_skipped)
                            _write_episode(_skipped)
                    # Let the current future be consumed, then break.
                    # C1 fix: catch worker exceptions so a single crash doesn't
                    # stall executor.shutdown() with subprocess holds.
                    fname = futures[future]
                    try:
                        _result = future.result()
                        completed.append(_result)
                        _write_episode(_result)
                    except Exception as exc:
                        log.warning(
                            "feature_worker_crashed_during_cancel",
                            feature=fname,
                            error=str(exc),
                        )
                        _cancelled = FeatureResult(
                            feature=fname,
                            spec_ids=groups.get(fname, []),
                            status="skipped",
                            artifacts=[],
                            tests=[],
                            error="Cancelled by user",
                            eval_scores={},
                        )
                        completed.append(_cancelled)
                        _write_episode(_cancelled)
                    break

                # C1 fix: catch worker exceptions so a single crash doesn't
                # stall executor.shutdown() with subprocess holds.
                fname = futures[future]
                try:
                    _result = future.result()
                    completed.append(_result)
                    _write_episode(_result)
                except Exception as exc:
                    log.error("feature_worker_crashed", feature=fname, error=str(exc))
                    try:
                        from dark_factory.metrics.helpers import record_incident
                        from dark_factory.metrics.prometheus import observe_worker_crash

                        observe_worker_crash()
                        record_incident(
                            category="pipeline",
                            severity="error",
                            message=f"feature worker crashed: {exc}"[:500],
                            phase="swarm",
                            feature=fname,
                        )
                    except Exception:  # pragma: no cover — defensive
                        pass
                    _crashed = FeatureResult(
                        feature=fname,
                        spec_ids=groups.get(fname, []),
                        status="error",
                        artifacts=[],
                        tests=[],
                        error=f"Worker crashed: {exc}",
                        eval_scores={},
                    )
                    completed.append(_crashed)
                    _write_episode(_crashed)
                    _emit(
                        "feature_completed",
                        feature=fname,
                        status="error",
                        artifacts=0,
                        tests=0,
                        error=f"Worker crashed: {exc}",
                        layer=layer_idx + 1,
                        worker_crash_count=1,
                    )
        finally:
            # cancel_futures=True (Python 3.9+) cancels anything still
            # queued but not yet started — critical for the kill-switch
            # path because the default shutdown(wait=True) would block
            # on every in-flight worker until it finishes its current
            # LLM call. On the normal completion path this is a no-op.
            executor.shutdown(wait=True, cancel_futures=True)

        _emit("layer_completed", layer=layer_idx + 1)

        return {
            "completed_features": completed,
            "current_layer": layer_idx + 1,
        }

    return execute_layer_node


def _run_reflection(
    layer_idx: int,
    total_layers: int,
    layer_results: list[FeatureResult],
    retry_attempt: int,
    max_retries: int,
    overrides: dict[str, Any],
) -> dict | None:
    """Call the reflection LLM to diagnose failures and decide on retry.

    Returns a parsed JSON dict with ``retryable_features``, ``strategy``,
    etc., or ``None`` if reflection is unavailable or fails.
    """
    failed = [r for r in layer_results if r["status"] in ("error", "skipped")]
    if not failed:
        return None

    try:
        import json as _json

        from dark_factory.llm.anthropic import AnthropicClient
        from dark_factory.prompts import get_prompt

        failed_details = "\n".join(
            f"- {r['feature']} [{r['status']}]: {r.get('error', 'no error message')}"
            for r in failed
        )
        succeeded = sum(1 for r in layer_results if r["status"] == "success")
        total = len(layer_results)

        prompt = get_prompt("reflection", "user").format(
            layer_idx=layer_idx + 1,
            total_layers=total_layers,
            pass_rate=succeeded / total if total else 0,
            succeeded=succeeded,
            total=total,
            retry_attempt=retry_attempt,
            max_retries=max_retries,
            failed_details=failed_details,
            strategy_overrides=_json.dumps(overrides) if overrides else "none",
        )
        system = get_prompt("reflection", "system")

        # Use a fast model for reflection to minimise cost.
        from dark_factory.agents.tools import _resolve_deep_model

        model = _resolve_deep_model("deep_analysis")
        client = AnthropicClient(model=model) if model else AnthropicClient()
        raw = client.complete(prompt, system=system, timeout_seconds=30)

        # Extract JSON from the response (handle markdown fences).
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return _json.loads(text)
    except Exception as exc:
        log.warning("reflection_failed", error=str(exc))
        return None


def make_adjust_strategy_node(threshold: float = 0.5, max_layer_retries: int = 1):
    """Create a node that adjusts agent strategy between layers.

    When a layer has failures and retries remain, runs a reflection LLM
    call to diagnose root causes and decide whether to retry the layer
    with adjusted strategy.
    """

    def adjust_strategy_node(state: OrchestratorState) -> dict:
        completed = state.get("completed_features", [])
        order = state.get("execution_order", [])
        current_layer = state.get("current_layer", 0)
        overrides = dict(state.get("strategy_overrides", {}))
        layer_rates = list(state.get("layer_pass_rates", []))
        layer_retries = dict(state.get("layer_retries", {}))

        # Compute pass rate for the just-completed layer (current_layer was already incremented)
        prev_layer_idx = current_layer - 1
        retry_layer: int | None = None

        if 0 <= prev_layer_idx < len(order):
            layer_features = set(order[prev_layer_idx])
            layer_results = [r for r in completed if r["feature"] in layer_features]
            succeeded = sum(1 for r in layer_results if r["status"] == "success")
            total = len(layer_results) if layer_results else 1
            layer_rate = succeeded / total
            layer_rates.append(layer_rate)

            log.info("layer_pass_rate", layer=prev_layer_idx, rate=round(layer_rate, 2))

            if layer_rate < threshold:
                attempt = layer_retries.get(prev_layer_idx, 0)

                # ── Self-healing: reflection + retry ──────────────────
                if attempt < max_layer_retries and layer_rate < 1.0:
                    from dark_factory.agents.tools import emit_progress as _emit

                    _emit(
                        "reflection_started",
                        layer=prev_layer_idx + 1,
                        attempt=attempt + 1,
                        max_retries=max_layer_retries,
                    )

                    reflection = _run_reflection(
                        layer_idx=prev_layer_idx,
                        total_layers=len(order),
                        layer_results=layer_results,
                        retry_attempt=attempt + 1,
                        max_retries=max_layer_retries,
                        overrides=overrides,
                    )

                    if reflection and reflection.get("retryable_features"):
                        retryable = set(reflection["retryable_features"])
                        diagnosis = reflection.get("diagnosis", "")
                        strategy = reflection.get("strategy", {})

                        log.info(
                            "reflection_retry",
                            layer=prev_layer_idx,
                            retryable=sorted(retryable),
                            diagnosis=diagnosis[:200],
                        )

                        # Apply strategy adjustments from reflection
                        if strategy.get("force_claude_agent"):
                            overrides["force_claude_agent"] = True
                        if strategy.get("max_handoffs"):
                            overrides["max_handoffs"] = strategy["max_handoffs"]
                        if strategy.get("prompt_hint"):
                            overrides["prompt_hint"] = strategy["prompt_hint"]

                        # Remove failed results for retryable features so
                        # they get re-run. Keep terminal failures.
                        completed = [
                            r for r in completed
                            if r["feature"] not in retryable
                            or r["feature"] not in layer_features
                        ]

                        layer_retries[prev_layer_idx] = attempt + 1
                        retry_layer = prev_layer_idx

                        _emit(
                            "reflection_completed",
                            layer=prev_layer_idx + 1,
                            diagnosis=diagnosis[:500],
                            retryable=sorted(retryable),
                            terminal=sorted(
                                set(reflection.get("terminal_features", []))
                            ),
                        )
                    else:
                        # Reflection declined retry or unavailable.
                        log.info("reflection_no_retry", layer=prev_layer_idx)
                        overrides["force_claude_agent"] = True
                        overrides["max_handoffs"] = 30

                        _emit(
                            "reflection_completed",
                            layer=prev_layer_idx + 1,
                            diagnosis=reflection.get("diagnosis", "Reflection declined retry") if reflection else "Reflection unavailable",
                            retryable=[],
                            terminal=[],
                        )
                else:
                    # No retries left — fall back to basic strategy adjustment.
                    log.warning("strategy_adjustment", reason="low_pass_rate", rate=layer_rate)
                    overrides["force_claude_agent"] = True
                    overrides["max_handoffs"] = 30

            elif layer_rate >= threshold and overrides.get("force_claude_agent"):
                # Performance recovered — relax overrides
                log.info("strategy_relaxed", reason="pass_rate_recovered")
                overrides.pop("force_claude_agent", None)
                overrides.pop("max_handoffs", None)
                overrides.pop("prompt_hint", None)

        updates: dict[str, Any] = {
            "strategy_overrides": overrides,
            "layer_pass_rates": layer_rates,
            "layer_retries": layer_retries,
            "completed_features": completed,
        }

        # If reflection decided to retry, rewind current_layer.
        if retry_layer is not None:
            updates["current_layer"] = retry_layer

        return updates

    return adjust_strategy_node


def aggregate_node(state: OrchestratorState) -> dict:
    """Merge all feature results and compute aggregate scoring."""
    completed = state.get("completed_features", [])
    all_artifacts: list[dict] = []
    all_tests: list[dict] = []

    for result in completed:
        all_artifacts.extend(result.get("artifacts", []))
        all_tests.extend(result.get("tests", []))

    succeeded = sum(1 for r in completed if r["status"] == "success")
    failed = sum(1 for r in completed if r["status"] == "error")
    skipped = sum(1 for r in completed if r["status"] == "skipped")
    total = succeeded + failed + skipped
    pass_rate = succeeded / total if total else 0.0

    mean_eval_scores = _compute_mean_eval_scores(completed)
    worst_features = _compute_worst_features(completed, limit=3)

    log.info(
        "orchestrator_complete",
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        pass_rate=round(pass_rate, 2),
        mean_eval_scores=mean_eval_scores,
        worst_features=[w["feature"] for w in worst_features],
    )
    return {
        "all_artifacts": all_artifacts,
        "all_tests": all_tests,
        "pass_rate": pass_rate,
        "mean_eval_scores": mean_eval_scores,
        "worst_features": worst_features,
    }


def _compute_mean_eval_scores(completed: list[FeatureResult]) -> dict[str, float]:
    """Compute mean eval scores per metric across all features."""
    all_scores: dict[str, list[float]] = {}
    for result in completed:
        for spec_id, metrics in result.get("eval_scores", {}).items():
            if isinstance(metrics, dict):
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "score" in metric_data:
                        all_scores.setdefault(metric_name, []).append(metric_data["score"])
    return {k: round(sum(v) / len(v), 3) for k, v in all_scores.items() if v}


def _compute_worst_features(completed: list[FeatureResult], limit: int = 3) -> list[dict]:
    """Identify worst-performing features by status and eval scores."""
    scored: list[tuple[str, float, str]] = []
    for result in completed:
        if result["status"] == "error":
            scored.append((result["feature"], 0.0, result.get("error", "error")))
        elif result["status"] == "skipped":
            scored.append((result["feature"], 0.0, "skipped"))
        elif result.get("eval_scores"):
            all_metric_scores = []
            for metrics in result["eval_scores"].values():
                if isinstance(metrics, dict):
                    for m in metrics.values():
                        if isinstance(m, dict) and "score" in m:
                            all_metric_scores.append(m["score"])
            mean = sum(all_metric_scores) / len(all_metric_scores) if all_metric_scores else 1.0
            if mean < 0.8:
                scored.append((result["feature"], mean, "low_eval_score"))

    scored.sort(key=lambda x: x[1])
    return [{"feature": f, "score": s, "reason": r} for f, s, r in scored[:limit]]


# ── Routing ──────────────────────────────────────────────────────────


def check_done(state: OrchestratorState) -> Literal["execute_layer", "aggregate"]:
    layer_idx = state.get("current_layer", 0)
    order = state.get("execution_order", [])
    if layer_idx >= len(order):
        return "aggregate"
    # Self-healing: if adjust_strategy rewound current_layer, we re-enter
    # execute_layer for the retried layer.
    return "execute_layer"


# ── Helpers ──────────────────────────────────────────────────────────


def _get_deps_from_completed(feature: str, state: OrchestratorState) -> set[str]:
    """Get the set of features that ``feature`` depends on."""
    group_deps = state.get("group_deps", {})
    return set(group_deps.get(feature, []))


# ── Builder ──────────────────────────────────────────────────────────


def build_orchestrator(settings: Settings, repo: GraphRepository) -> StateGraph:
    """Build the orchestrator graph (does not compile)."""
    # Build the chat model via the swarm helper so ``max_tokens`` and
    # ``timeout`` are set explicitly. Without this, every per-feature
    # swarm in Phase 4 would use langchain's default 1024-token
    # ceiling and large ``write_file`` calls would truncate mid-
    # content, producing ``Error invoking tool 'write_file'`` errors.
    from dark_factory.agents.swarm import build_chat_model

    # Build per-agent models from routing config. If no overrides are
    # configured, every agent gets the same default model.
    routing = settings.model_routing
    default_model_id = settings.llm.model
    default_model = build_chat_model(settings)

    agent_models: dict[str, Any] = {}
    for role in ("planner", "coder", "reviewer", "tester"):
        override = routing.resolve(role, default_model_id)
        if override != default_model_id:
            agent_models[role] = build_chat_model(settings, model_override=override)

    max_parallel = settings.pipeline.max_parallel_features
    default_max_handoffs = settings.pipeline.max_codegen_handoffs

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", make_plan_node(repo))
    graph.add_node(
        "execute_layer",
        make_execute_layer_node(
            default_model,
            max_parallel=max_parallel,
            default_max_handoffs=default_max_handoffs,
            enable_episodic_memory=settings.pipeline.enable_episodic_memory,
            settings=settings,
            agent_models=agent_models or None,
        ),
    )
    graph.add_node("adjust_strategy", make_adjust_strategy_node(
        threshold=settings.evaluation.strategy_threshold,
        max_layer_retries=settings.pipeline.max_layer_retries,
    ))
    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_layer")
    graph.add_edge("execute_layer", "adjust_strategy")
    graph.add_conditional_edges(
        "adjust_strategy", check_done, ["execute_layer", "aggregate"]
    )
    graph.add_edge("aggregate", END)

    return graph


def run_orchestrator(settings: Settings, spec_ids: list[str]) -> OrchestratorState:
    """Build and run the orchestrator. Entry point for CLI."""
    from dark_factory.agents.tools import get_current_run_id, set_current_run_id

    from pathlib import Path

    from dark_factory.agents.tools import set_output_dir, set_run_storage

    repo, neo4j_client, memory_client = init_swarm_context(settings)
    try:
        # H4 fix: reuse the memory repo already created by init_swarm_context
        # instead of creating a duplicate without vector support.
        from dark_factory.agents.tools import _memory_repo as shared_memory_repo

        # If the bridge already created a Run History entry early, reuse it
        # rather than creating a duplicate. Otherwise (CLI usage or memory
        # disabled), create one here.
        run_id = get_current_run_id()
        if not run_id and memory_client and settings.memory.enabled and shared_memory_repo:
            run_id = shared_memory_repo.create_run(
                spec_count=len(spec_ids),
                feature_count=0,
            )
            set_current_run_id(run_id)

        # Always run memory decay (it's a separate concern from create_run)
        if memory_client and settings.memory.enabled and shared_memory_repo:
            decayed = shared_memory_repo.decay_all_relevance(factor=settings.evaluation.decay_factor)
            log.info("memory_decayed", count=decayed, factor=settings.evaluation.decay_factor)

            # If the bridge created the run with spec_count=0, bump it now
            # to the actual count so the History tab shows accurate numbers
            try:
                shared_memory_repo.update_run_counts(
                    run_id=run_id, spec_count=len(spec_ids)
                )
            except Exception as exc:
                log.warning("update_run_counts_in_orchestrator_failed", error=str(exc))

        # Generate a run ID for output namespacing even if memory is disabled
        if not run_id:
            from datetime import datetime, timezone
            from uuid import uuid4

            run_id = f"run-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4]}"
            set_current_run_id(run_id)

        # Namespace output directory by run ID
        run_output_dir = Path(settings.pipeline.output_dir) / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        set_output_dir(run_output_dir)
        log.info("run_output_dir", path=str(run_output_dir))

        # Wire up durable storage for this run
        from dark_factory.storage.backend import RunStorage, get_storage
        run_storage = RunStorage(
            get_storage(local_root=Path(settings.pipeline.output_dir)),
            run_id,
        )
        set_run_storage(run_storage)

        start_time = time.time()
        graph = build_orchestrator(settings, repo)
        compiled = graph.compile()
        result = compiled.invoke({
            "all_spec_ids": spec_ids,
            "run_id": run_id,
            "run_start_time": start_time,
        })

        # Complete the run with aggregate stats
        if memory_client and settings.memory.enabled and run_id and shared_memory_repo:
            completed = result.get("completed_features", [])
            succeeded = sum(1 for r in completed if r["status"] == "success")
            total = len(completed)
            shared_memory_repo.complete_run(
                run_id=run_id,
                status="success" if succeeded == total else "partial",
                pass_rate=result.get("pass_rate", 0.0),
                mean_eval_scores=result.get("mean_eval_scores", {}),
                worst_features=result.get("worst_features", []),
                duration_seconds=time.time() - start_time,
            )

        return result
    finally:
        neo4j_client.close()
        if memory_client:
            memory_client.close()
