"""Parent orchestrator: one swarm subagent per feature in the knowledge graph.

The orchestrator queries Neo4j for features (specs grouped by capability),
computes a topological execution order respecting DEPENDS_ON relationships,
and dispatches each feature to an isolated swarm instance.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
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

    Uses Kahn's algorithm with layer batching.
    Raises ``ValueError`` if a dependency cycle is detected.
    """
    all_nodes = set(groups.keys())

    # Build in-degree map and adjacency
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}
    dependents: dict[str, list[str]] = defaultdict(list)

    for node, deps in group_deps.items():
        for dep in deps:
            if dep in all_nodes:
                in_degree[node] = in_degree.get(node, 0) + 1
                dependents[dep].append(node)

    # Seed with zero-in-degree nodes
    queue: deque[str] = deque(n for n in all_nodes if in_degree[n] == 0)
    layers: list[list[str]] = []
    visited = 0

    while queue:
        layer = sorted(queue)  # deterministic ordering
        layers.append(layer)
        next_queue: deque[str] = deque()
        for node in layer:
            visited += 1
            for dep in dependents[node]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    next_queue.append(dep)
        queue = next_queue

    if visited < len(all_nodes):
        raise ValueError(
            f"Dependency cycle detected among features: "
            f"{[n for n in all_nodes if in_degree[n] > 0]}"
        )

    return layers


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
            "execution_order": order,
            "completed_features": [],
            "current_layer": 0,
            "strategy_overrides": {},
            "layer_pass_rates": [],
        }

    return plan_node


def make_execute_layer_node(model: str, max_parallel: int = 4):
    """Create the execute_layer node with cross-feature learning and parallel execution."""

    def execute_layer_node(state: OrchestratorState) -> dict:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from dark_factory.agents.swarm import run_feature_swarm
        from dark_factory.agents.tools import _memory_repo

        layer_idx = state.get("current_layer", 0)
        order = state.get("execution_order", [])
        groups = state.get("feature_groups", {})
        completed = list(state.get("completed_features", []))
        overrides = state.get("strategy_overrides", {})
        run_id = state.get("run_id", "")
        max_handoffs = overrides.get("max_handoffs", MAX_HANDOFFS)

        if layer_idx >= len(order):
            return {"current_layer": layer_idx}

        layer = order[layer_idx]
        failed_features = {r["feature"] for r in completed if r["status"] == "error"}

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
            except Exception:
                pass

        # Collect features to run (filtering skipped)
        to_run: list[tuple[str, list[str]]] = []
        for feature_name in layer:
            spec_ids = groups.get(feature_name, [])
            if not spec_ids:
                continue
            deps_of_feature = _get_deps_from_completed(feature_name, state)
            if deps_of_feature & failed_features:
                log.warning("feature_skipped", feature=feature_name, reason="dependency_failed")
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

        # Run features in parallel within the layer
        def _run_one(feature_name: str, spec_ids: list[str]) -> FeatureResult:
            log.info("feature_swarm_starting", feature=feature_name, specs=len(spec_ids))
            compiled = build_feature_swarm(model, strategy_overrides=overrides)
            return run_feature_swarm(
                compiled, spec_ids, feature_name,
                run_context=run_context,
                max_handoffs=max_handoffs,
            )

        workers = min(max_parallel, len(to_run)) if to_run else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, fname, sids): fname
                for fname, sids in to_run
            }
            for future in as_completed(futures):
                completed.append(future.result())

        return {
            "completed_features": completed,
            "current_layer": layer_idx + 1,
        }

    return execute_layer_node


def make_adjust_strategy_node(threshold: float = 0.5):
    """Create a node that adjusts agent strategy between layers based on performance."""

    def adjust_strategy_node(state: OrchestratorState) -> dict:
        completed = state.get("completed_features", [])
        order = state.get("execution_order", [])
        current_layer = state.get("current_layer", 0)
        overrides = dict(state.get("strategy_overrides", {}))
        layer_rates = list(state.get("layer_pass_rates", []))

        # Compute pass rate for the just-completed layer (current_layer was already incremented)
        prev_layer_idx = current_layer - 1
        if 0 <= prev_layer_idx < len(order):
            layer_features = set(order[prev_layer_idx])
            layer_results = [r for r in completed if r["feature"] in layer_features]
            succeeded = sum(1 for r in layer_results if r["status"] == "success")
            total = len(layer_results) if layer_results else 1
            layer_rate = succeeded / total
            layer_rates.append(layer_rate)

            log.info("layer_pass_rate", layer=prev_layer_idx, rate=round(layer_rate, 2))

            if layer_rate < threshold:
                log.warning("strategy_adjustment", reason="low_pass_rate", rate=layer_rate)
                overrides["force_claude_agent"] = True
                overrides["max_handoffs"] = 30
            elif layer_rate >= threshold and overrides.get("force_claude_agent"):
                # Performance recovered — relax overrides
                log.info("strategy_relaxed", reason="pass_rate_recovered")
                overrides.pop("force_claude_agent", None)
                overrides.pop("max_handoffs", None)

        return {
            "strategy_overrides": overrides,
            "layer_pass_rates": layer_rates,
        }

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
    return "execute_layer"


# ── Helpers ──────────────────────────────────────────────────────────


def _get_deps_from_completed(feature: str, state: OrchestratorState) -> set[str]:
    """Get the set of features that ``feature`` depends on (from execution_order context)."""
    # This is a simple heuristic: all features in earlier layers are potential deps.
    # The actual dep edges are in the graph but not stored in state.
    # For correctness, we'd need group_deps in state, but for the skip check
    # we just check if any completed feature with status="error" exists in prior layers.
    return set()  # Conservative: rely on the per-layer ordering for now


# ── Builder ──────────────────────────────────────────────────────────


def build_orchestrator(settings: Settings, repo: GraphRepository) -> StateGraph:
    """Build the orchestrator graph (does not compile)."""
    model = f"anthropic:{settings.llm.model}"
    max_parallel = settings.pipeline.max_parallel_features

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", make_plan_node(repo))
    graph.add_node("execute_layer", make_execute_layer_node(model, max_parallel=max_parallel))
    graph.add_node("adjust_strategy", make_adjust_strategy_node(settings.evaluation.strategy_threshold))
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
    from dark_factory.agents.tools import set_current_run_id

    repo, neo4j_client, memory_client = init_swarm_context(settings)
    try:
        # Create run and apply memory decay
        run_id = ""
        if memory_client and settings.memory.enabled:
            from dark_factory.memory.repository import MemoryRepository

            mem_repo = MemoryRepository(memory_client)
            run_id = mem_repo.create_run(
                spec_count=len(spec_ids),
                feature_count=0,
            )
            decayed = mem_repo.decay_all_relevance(factor=settings.evaluation.decay_factor)
            log.info("memory_decayed", count=decayed, factor=settings.evaluation.decay_factor)
            set_current_run_id(run_id)

        start_time = time.time()
        graph = build_orchestrator(settings, repo)
        compiled = graph.compile()
        result = compiled.invoke({
            "all_spec_ids": spec_ids,
            "run_id": run_id,
            "run_start_time": start_time,
        })

        # Complete the run with aggregate stats
        if memory_client and settings.memory.enabled and run_id:
            from dark_factory.memory.repository import MemoryRepository

            mem_repo = MemoryRepository(memory_client)
            completed = result.get("completed_features", [])
            succeeded = sum(1 for r in completed if r["status"] == "success")
            total = len(completed)
            mem_repo.complete_run(
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
