"""LangGraph swarm harness: iterative multi-agent code generation.

Four specialised agents (Planner, Coder, Reviewer, Tester) hand off to each
other via ``langgraph-swarm``'s ``create_swarm`` and ``create_handoff_tool``.
The swarm processes a featured list of specs with up to 50 handoffs.

Each agent has access to procedural memory — patterns, mistakes, solutions,
and strategies learned from past runs — stored in a separate Neo4j database.

This module exposes both:
- High-level ``build_swarm`` / ``run_swarm`` for standalone usage.
- Low-level ``init_swarm_context`` / ``build_feature_swarm`` / ``run_feature_swarm``
  so the orchestrator can spawn one swarm per feature.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from typing_extensions import TypedDict

from dark_factory.agents.tools import (
    CODEGEN_TOOLS,
    EVAL_HISTORY_TOOLS,
    EVAL_TOOLS,
    GRAPH_READ_TOOLS,
    MEMORY_READ_TOOLS,
    MEMORY_WRITE_MISTAKE,
    MEMORY_WRITE_PATTERN,
    MEMORY_WRITE_STRATEGY,
    TESTGEN_TOOLS,
    VECTOR_SEARCH_TOOLS,
    claude_agent_codegen,
    read_file,
    set_current_feature,
    set_current_run_id,
    set_eval_config,
    set_graph_repo,
    set_memory_repo,
    set_openspec_root,
    set_output_dir,
    set_vector_repo,
)
from dark_factory.config import Settings
from dark_factory.graph.client import Neo4jClient
from dark_factory.graph.repository import GraphRepository

log = structlog.get_logger()

MAX_HANDOFFS = 50


# ── Feature result ───────────────────────────────────────────────────


class FeatureResult(TypedDict):
    """Result from a single feature swarm execution."""

    feature: str
    spec_ids: list[str]
    status: str  # "success" | "error" | "skipped"
    artifacts: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    error: str | None
    eval_scores: dict[str, Any]


# ── Agent builders ───────────────────────────────────────────────────


def _build_planner(model: str):
    handoff_to_coder = create_handoff_tool(
        agent_name="coder",
        description="Hand off to the Coder agent to generate or fix code for a spec.",
    )
    from dark_factory.agents.tools import evaluate_spec

    return create_agent(
        model,
        tools=[*GRAPH_READ_TOOLS, evaluate_spec, *EVAL_HISTORY_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_STRATEGY, handoff_to_coder],
        system_prompt=(
            "You are a project planner coordinating spec-driven code generation. "
            "You receive a list of spec IDs via the user message. Check which are "
            "already done, pick the next unfinished spec, query the knowledge graph "
            "for context, then hand off to the Coder by calling transfer_to_coder.\n\n"
            "EVALUATION: Before handing off each spec to the Coder, call evaluate_spec "
            "to verify correctness, coherence, instruction following, and safety. "
            "Also call query_eval_history to check how this spec performed in past "
            "runs — if scores are trending down, flag the spec for extra attention.\n\n"
            "MEMORY: Before planning, call recall_memories with the feature name to "
            "retrieve known strategies and patterns from past runs. If you discover a "
            "useful planning approach during this run, record it with record_strategy.\n\n"
            "If ALL specs are complete, respond with DONE and do NOT call any "
            "handoff tool."
        ),
        name="planner",
    )


def _build_coder(model: str, prompt_suffix: str = ""):
    handoff_to_reviewer = create_handoff_tool(
        agent_name="reviewer",
        description="Hand off to the Reviewer agent after generating code.",
    )
    return create_agent(
        model,
        tools=[*CODEGEN_TOOLS, *VECTOR_SEARCH_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_PATTERN, claude_agent_codegen, handoff_to_reviewer],
        system_prompt=(
            "You are a senior software engineer generating production-quality code. "
            "Use get_spec_context to understand dependencies.\n\n"
            "RAG: Before generating code, call search_similar_specs to find similar "
            "past specs and search_similar_code to find reference implementations. "
            "Use these as context for higher quality code generation.\n\n"
            "MEMORY: Call recall_memories to check for known patterns and past "
            "mistakes related to this spec. Apply any relevant patterns. If you "
            "discover a reusable pattern, record it with record_pattern.\n\n"
            "You have two strategies for code generation:\n"
            "1. Use write_file to write code directly for simple specs.\n"
            "2. Use claude_agent_codegen for complex specs — it delegates to the "
            "Claude Agent SDK which has built-in file editing, shell, and search "
            "tools for sophisticated multi-file code generation. Pass the spec "
            "context and any fix instructions to it.\n\n"
            "If you receive reviewer feedback, fix the issues described. "
            "Prefer claude_agent_codegen for fixes since it can read existing code "
            "and make targeted edits.\n\n"
            "When code is complete, hand off to the Reviewer by calling "
            "transfer_to_reviewer."
        ) + (f"\n\n{prompt_suffix}" if prompt_suffix else ""),
        name="coder",
    )


def _build_reviewer(model: str):
    handoff_to_coder = create_handoff_tool(
        agent_name="coder",
        description="Hand off back to the Coder to fix issues found during review.",
    )
    handoff_to_tester = create_handoff_tool(
        agent_name="tester",
        description="Hand off to the Tester after approving the generated code.",
    )
    return create_agent(
        model,
        tools=[*GRAPH_READ_TOOLS, read_file, *EVAL_TOOLS, *EVAL_HISTORY_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_MISTAKE, handoff_to_coder, handoff_to_tester],
        system_prompt=(
            "You are a code reviewer. Read the generated code using read_file, "
            "check it against the spec's acceptance criteria from the knowledge graph.\n\n"
            "EVALUATION: Use evaluate_tests to run AI evaluation on the generated code. "
            "Call query_eval_history to compare against past scores for this spec — "
            "if a metric has been consistently failing, focus your review there.\n\n"
            "MEMORY: Check recall_memories for known mistakes related to this spec "
            "before reviewing. If you find issues, record each one with record_mistake "
            "(include error_type like 'missing_validation', 'async_mismatch'). When a "
            "previously reported issue is fixed, record the fix with record_solution "
            "and link it to the original mistake_id.\n\n"
            "If the code passes review, hand off to Tester by calling transfer_to_tester.\n"
            "If it needs fixes, describe the issues and hand off back to Coder by "
            "calling transfer_to_coder."
        ),
        name="reviewer",
    )


def _build_tester(model: str):
    handoff_to_planner = create_handoff_tool(
        agent_name="planner",
        description="Hand off back to the Planner after writing tests.",
    )
    return create_agent(
        model,
        tools=[*TESTGEN_TOOLS, *EVAL_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_MISTAKE, handoff_to_planner],
        system_prompt=(
            "You are a QA engineer. Write thorough tests for the current spec's "
            "code artifact. Use get_spec_context to understand acceptance criteria, "
            "read_file to inspect the source, then write_file to create tests.\n\n"
            "EVALUATION: After writing tests, you MUST call evaluate_tests to run "
            "AI evaluation on the generated tests. Pass the spec title, acceptance "
            "criteria (as JSON array), the source code, and the test code. If any "
            "metric scores below 0.5, revise the tests and re-evaluate before "
            "handing off. Record evaluation failures as mistakes in memory.\n\n"
            "MEMORY: Check recall_memories before writing tests to avoid known "
            "pitfalls. If tests reveal failures, record the root cause with "
            "record_mistake. If you identify the fix needed, record it with "
            "record_solution.\n\n"
            "When tests pass evaluation, hand off back to Planner by calling "
            "transfer_to_planner."
        ),
        name="tester",
    )


# ── Low-level API (used by orchestrator) ─────────────────────────────


def init_swarm_context(
    settings: Settings,
) -> tuple[GraphRepository, Neo4jClient, Neo4jClient | None]:
    """One-time setup: create Neo4j clients and set module-level tool state.

    Returns ``(repo, neo4j_client, memory_client)``.  Caller owns both client lifecycles.
    ``memory_client`` is ``None`` if memory is disabled.
    """
    neo4j_client = Neo4jClient(settings.neo4j)
    repo = GraphRepository(neo4j_client)

    set_graph_repo(repo)
    set_output_dir(settings.pipeline.output_dir)
    set_openspec_root(settings.openspec.root_dir)
    set_eval_config(settings.evaluation)

    # Qdrant vector search setup
    vector_repo = None
    if settings.qdrant.enabled:
        try:
            from dark_factory.vector.client import QdrantClientWrapper
            from dark_factory.vector.collections import ensure_collections
            from dark_factory.vector.embeddings import EmbeddingService
            from dark_factory.vector.repository import VectorRepository

            qdrant_client = QdrantClientWrapper(settings.qdrant)
            if qdrant_client.is_available():
                ensure_collections(qdrant_client)
                embedding_svc = EmbeddingService(model=settings.qdrant.embedding_model)
                vector_repo = VectorRepository(qdrant_client, embedding_svc)
                log.info("qdrant_connected", url=settings.qdrant.url)
            else:
                log.warning("qdrant_unavailable_falling_back_to_neo4j")
        except Exception as exc:
            log.warning("qdrant_init_failed", error=str(exc))
    set_vector_repo(vector_repo)

    # Memory setup (with optional vector dual-write)
    memory_client = None
    if settings.memory.enabled:
        from dark_factory.config import Neo4jConfig
        from dark_factory.memory.repository import MemoryRepository
        from dark_factory.memory.schema import init_memory_schema

        mem_config = Neo4jConfig(
            uri=settings.neo4j.uri,
            database=settings.memory.database,
            user=settings.neo4j.user,
            password=settings.neo4j.password,
        )
        memory_client = Neo4jClient(mem_config)
        init_memory_schema(memory_client)
        memory_repo = MemoryRepository(memory_client, vector_repo=vector_repo)
        set_memory_repo(memory_repo)
    else:
        set_memory_repo(None)

    return repo, neo4j_client, memory_client


def _strategy_suffix(overrides: dict[str, Any] | None) -> str:
    """Build extra prompt instructions from strategy overrides."""
    if not overrides:
        return ""
    parts = []
    if overrides.get("force_claude_agent"):
        parts.append(
            "STRATEGY OVERRIDE: Use claude_agent_codegen for ALL specs, "
            "not just complex ones. Direct write_file is disabled for this layer."
        )
    if "max_handoffs" in overrides:
        parts.append(f"STRATEGY OVERRIDE: Handoff budget reduced to {overrides['max_handoffs']}.")
    return "\n\n".join(parts)


def build_feature_swarm(model: str, strategy_overrides: dict[str, Any] | None = None) -> Any:
    """Build and compile a single swarm instance.

    Does NOT create a Neo4j client — assumes ``init_swarm_context`` was called.
    ``strategy_overrides`` modifies agent prompts when performance is poor.
    Returns the compiled LangGraph graph.
    """
    suffix = _strategy_suffix(strategy_overrides)

    workflow = create_swarm(
        agents=[
            _build_planner(model),
            _build_coder(model, prompt_suffix=suffix),
            _build_reviewer(model),
            _build_tester(model),
        ],
        default_active_agent="planner",
    )
    return workflow.compile()


def run_feature_swarm(
    compiled: Any,
    spec_ids: list[str],
    feature_name: str,
    run_context: str = "",
    max_handoffs: int = MAX_HANDOFFS,
) -> FeatureResult:
    """Invoke a compiled swarm with an isolated message history.

    Each call gets a fresh message list, so the handoff budget resets.
    Sets ``_current_feature`` so memory write tools auto-tag the feature.
    ``run_context`` is prepended as cross-feature learnings from earlier features.
    """
    set_current_feature(feature_name)
    content_parts = []
    if run_context:
        content_parts.append(f"== Learnings from earlier features ==\n{run_context}\n")
    content_parts.append(
        f"Feature: {feature_name}\n"
        f"Process these specs through code generation:\n"
        f"Spec IDs: {spec_ids}\n\n"
        f"For each spec: generate code, review it, write tests. "
        f"Maximum {max_handoffs} handoffs allowed."
    )
    initial = {
        "messages": [{"role": "user", "content": "\n".join(content_parts)}],
    }
    try:
        result = compiled.invoke(initial)
        log.info(
            "feature_swarm_complete",
            feature=feature_name,
            message_count=len(result.get("messages", [])),
        )
        return FeatureResult(
            feature=feature_name,
            spec_ids=spec_ids,
            status="success",
            artifacts=[],
            tests=[],
            error=None,
            eval_scores={},
        )
    except Exception as exc:
        log.error("feature_swarm_failed", feature=feature_name, error=str(exc))
        return FeatureResult(
            feature=feature_name,
            spec_ids=spec_ids,
            status="error",
            artifacts=[],
            tests=[],
            error=str(exc),
            eval_scores={},
        )


# ── High-level API (standalone usage) ────────────────────────────────


def build_swarm(settings: Settings) -> tuple[Any, Neo4jClient, Neo4jClient | None]:
    """Build the swarm. Returns ``(compiled_graph, neo4j_client, memory_client)``."""
    _repo, neo4j_client, memory_client = init_swarm_context(settings)
    model = f"anthropic:{settings.llm.model}"
    compiled = build_feature_swarm(model)
    return compiled, neo4j_client, memory_client


def run_swarm(settings: Settings, spec_ids: list[str]) -> dict:
    """Build, run, and return final state of the swarm."""
    compiled, neo4j_client, memory_client = build_swarm(settings)
    try:
        result = run_feature_swarm(compiled, spec_ids, feature_name="all")
        log.info("swarm_complete", total_specs=len(spec_ids), status=result["status"])
        return result
    finally:
        neo4j_client.close()
        if memory_client:
            memory_client.close()
