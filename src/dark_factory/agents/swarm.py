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

from dark_factory.prompts import get_prompt
from dark_factory.agents.tools import (
    CODEGEN_TOOLS,
    DEEP_PLANNER_TOOLS,
    DEEP_REVIEWER_TOOLS,
    DEEP_TESTER_TOOLS,
    EPISODE_READ_TOOLS,
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


class FeatureResult(TypedDict, total=False):
    """Result from a single feature swarm execution."""

    feature: str
    spec_ids: list[str]
    status: str  # "success" | "error" | "skipped"
    artifacts: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    error: str | None
    eval_scores: dict[str, Any]
    # Telemetry (optional, only populated when the swarm runs)
    stats: dict[str, Any]


# ── Message inspection helpers ───────────────────────────────────────


def _get_tool_calls(msg: Any) -> list[dict[str, Any]]:
    """Extract tool_calls from an AIMessage. Returns [] if none."""
    raw = getattr(msg, "tool_calls", None)
    if not raw:
        return []
    normalized: list[dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict):
            normalized.append(tc)
        else:
            normalized.append({
                "name": getattr(tc, "name", ""),
                "args": getattr(tc, "args", {}),
                "id": getattr(tc, "id", ""),
            })
    return normalized


def _is_tool_message(msg: Any) -> bool:
    """Detect a ToolMessage by class name + duck typing."""
    cls_name = type(msg).__name__
    return cls_name == "ToolMessage" or (
        hasattr(msg, "tool_call_id") and hasattr(msg, "name")
    )


def _is_ai_message(msg: Any) -> bool:
    """Detect an AIMessage (or AIMessageChunk) by class name."""
    cls_name = type(msg).__name__
    return cls_name in ("AIMessage", "AIMessageChunk")


def _extract_text_content(content: Any) -> str:
    """Extract human-readable text from a LangChain message.content payload.

    Handles both the simple string format and the multi-block list format
    that some providers (e.g. Anthropic) use:

        [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
        return "\n".join(p for p in parts if p).strip()
    if content is None:
        return ""
    return str(content).strip()


# Decision text length cap for logs (full text is in the agent's state)
_DECISION_TEXT_MAX = 500


# Roles to count with per-role columns in swarm_feature_events.
_KNOWN_ROLES = {"planner", "coder", "reviewer", "tester"}

# Tool names that invoke a deep sub-agent via the Claude Agent SDK.
_DEEP_AGENT_TOOLS = {
    "claude_agent_codegen",
    "deep_dependency_analysis",
    "deep_risk_assessment",
    "deep_security_review",
    "deep_performance_review",
    "deep_spec_compliance_review",
    "deep_unit_test_gen",
    "deep_integration_test_gen",
    "deep_edge_case_test_gen",
}


class _SwarmStats:
    """Mutable per-feature statistics accumulator.

    Populated while `run_feature_swarm` streams LangGraph updates; dumped
    into `FeatureResult.stats` at the end so the orchestrator can forward
    the rollup to the metrics recorder.
    """

    def __init__(self, feature: str) -> None:
        import time as _time

        self.feature = feature
        self.started_at = _time.time()
        self.ended_at: float | None = None

        self.agent_transitions = 0
        self.unique_agents: set[str] = set()
        self.role_calls: dict[str, int] = {r: 0 for r in _KNOWN_ROLES}

        self.tool_call_count = 0
        self.tool_failure_count = 0
        self.deep_agent_invocations = 0

        # Per-agent rollup fields for the agent_stats table.
        # {agent: {activations, tool_calls, decisions, handoffs_in, handoffs_out}}
        self.per_agent: dict[str, dict[str, int]] = {}

        # Correlate tool_call → tool_result by tool_call_id so we can measure
        # per-call latency. Holds (start_ts, tool_name, agent).
        self.pending_tool_calls: dict[str, tuple[float, str, str]] = {}

        self.worker_crashed = False

    def _agent_bucket(self, agent: str) -> dict[str, int]:
        b = self.per_agent.get(agent)
        if b is None:
            b = {
                "activations": 0,
                "tool_calls": 0,
                "decisions": 0,
                "handoffs_in": 0,
                "handoffs_out": 0,
            }
            self.per_agent[agent] = b
        return b

    def note_agent_active(self, agent: str) -> None:
        if not agent:
            return
        self.unique_agents.add(agent)
        self._agent_bucket(agent)["activations"] += 1
        if agent in self.role_calls:
            self.role_calls[agent] += 1

    def note_decision(self, agent: str) -> None:
        if not agent:
            return
        self._agent_bucket(agent)["decisions"] += 1

    def note_handoff(self, from_agent: str, to_agent: str) -> None:
        if from_agent:
            self._agent_bucket(from_agent)["handoffs_out"] += 1
        if to_agent:
            self._agent_bucket(to_agent)["handoffs_in"] += 1
        try:
            from dark_factory.metrics.prometheus import observe_agent_handoff

            observe_agent_handoff(from_agent=from_agent, to_agent=to_agent)
        except Exception:  # pragma: no cover — defensive
            pass

    def note_tool_call(self, tool: str, agent: str, tool_call_id: str) -> None:
        import time as _time

        self.tool_call_count += 1
        if tool in _DEEP_AGENT_TOOLS:
            self.deep_agent_invocations += 1
            try:
                from dark_factory.metrics.prometheus import observe_deep_agent_invocation

                observe_deep_agent_invocation(tool)
            except Exception:  # pragma: no cover — defensive
                pass
        if agent:
            self._agent_bucket(agent)["tool_calls"] += 1
        if tool_call_id:
            self.pending_tool_calls[tool_call_id] = (_time.time(), tool, agent)

    def note_tool_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        result: str,
        is_error: bool,
        run_id: str | None,
    ) -> None:
        import time as _time

        from dark_factory.metrics.helpers import record_tool_call

        latency: float | None = None
        agent: str | None = None
        pending = self.pending_tool_calls.pop(tool_call_id, None)
        if pending is not None:
            started_at, pending_tool, pending_agent = pending
            latency = _time.time() - started_at
            if tool_name == "?" or not tool_name:
                tool_name = pending_tool
            agent = pending_agent
        if is_error:
            self.tool_failure_count += 1
        record_tool_call(
            tool=tool_name or "?",
            feature=self.feature,
            agent=agent,
            success=not is_error,
            latency_seconds=latency,
            args_chars=None,
            result_chars=len(result) if isinstance(result, str) else None,
            error=(result[:500] if is_error and isinstance(result, str) else None),
            run_id=run_id,
        )

    def finalize(self) -> dict[str, Any]:
        import time as _time

        self.ended_at = _time.time()
        return {
            "feature": self.feature,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": max(0.0, self.ended_at - self.started_at),
            "agent_transitions": self.agent_transitions,
            "unique_agents_visited": len(self.unique_agents),
            "planner_calls": self.role_calls.get("planner", 0),
            "coder_calls": self.role_calls.get("coder", 0),
            "reviewer_calls": self.role_calls.get("reviewer", 0),
            "tester_calls": self.role_calls.get("tester", 0),
            "tool_call_count": self.tool_call_count,
            "tool_failure_count": self.tool_failure_count,
            "deep_agent_invocations": self.deep_agent_invocations,
            # subprocess_spawn_count ≈ deep agent invocations (each one spawns
            # a Claude SDK subprocess).
            "subprocess_spawn_count": self.deep_agent_invocations,
            "worker_crash_count": 1 if self.worker_crashed else 0,
            "per_agent": dict(self.per_agent),
        }


def _emit_message_events(
    msg: Any,
    feature_name: str,
    current_agent: str,
    stats: _SwarmStats | None = None,
    run_id: str | None = None,
) -> None:
    """Emit progress events for a single LangGraph message.

    Event mapping:
    - ToolMessage                        → ``tool_result``
    - AIMessage content (non-empty text) → ``agent_decision`` (the agent's
                                            reasoning before its next action)
    - AIMessage tool_calls               → ``tool_call`` per call, or
                                            ``agent_handoff`` for transfer_to_*
    - HumanMessage / SystemMessage       → no event (input/setup, not a decision)

    If ``stats`` is provided, also updates per-agent and per-tool counters
    and writes ``tool_calls`` rows with per-call latency.
    """
    from dark_factory.agents.tools import emit_progress

    # ToolMessage → tool_result
    if _is_tool_message(msg):
        tool_name = getattr(msg, "name", "") or "?"
        content = getattr(msg, "content", "")
        result_str = str(content) if content is not None else ""
        result_preview = result_str[:200] if result_str else ""
        tool_call_id = getattr(msg, "tool_call_id", "") or ""
        # LangChain ToolMessage has `status` attribute on newer versions.
        status = getattr(msg, "status", None)
        is_error = status == "error" or result_str.lower().startswith("error")
        emit_progress(
            "tool_result",
            feature=feature_name,
            agent=current_agent,
            tool=tool_name,
            result_preview=result_preview,
        )
        if stats is not None:
            stats.note_tool_result(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                result=result_str,
                is_error=bool(is_error),
                run_id=run_id,
            )
        return

    # AIMessage: emit decision text (if any) AND tool calls (if any).
    if _is_ai_message(msg):
        text = _extract_text_content(getattr(msg, "content", ""))
        if text:
            if stats is not None:
                stats.note_decision(current_agent)
            truncated = text[:_DECISION_TEXT_MAX]
            if len(text) > _DECISION_TEXT_MAX:
                truncated += "…"
            emit_progress(
                "agent_decision",
                feature=feature_name,
                agent=current_agent,
                text=truncated,
            )

        for tc in _get_tool_calls(msg):
            tool_name = tc.get("name", "") or ""
            args = tc.get("args", {})
            tool_call_id = tc.get("id", "") or ""
            args_preview = str(args)[:120] if args else ""

            if tool_name.startswith("transfer_to_"):
                target = tool_name[len("transfer_to_"):]
                if stats is not None:
                    stats.note_handoff(current_agent, target)
                emit_progress(
                    "agent_handoff",
                    feature=feature_name,
                    from_agent=current_agent,
                    to_agent=target,
                )
            else:
                if stats is not None:
                    stats.note_tool_call(tool_name, current_agent, tool_call_id)
                emit_progress(
                    "tool_call",
                    feature=feature_name,
                    agent=current_agent,
                    tool=tool_name,
                    args_preview=args_preview,
                )


# ── Agent builders ───────────────────────────────────────────────────


def _build_planner(model: "str | Any"):
    handoff_to_coder = create_handoff_tool(
        agent_name="coder",
        description="Hand off to the Coder agent to generate or fix code for a spec.",
    )
    from dark_factory.agents.tools import evaluate_spec

    return create_agent(
        model,
        tools=[
            *GRAPH_READ_TOOLS,
            evaluate_spec,
            *EVAL_HISTORY_TOOLS,
            *EPISODE_READ_TOOLS,
            *DEEP_PLANNER_TOOLS,
            *MEMORY_READ_TOOLS,
            *MEMORY_WRITE_STRATEGY,
            handoff_to_coder,
        ],
        system_prompt=get_prompt("swarm_planner", "system"),
        name="planner",
    )


def _build_coder(model: "str | Any", prompt_suffix: str = ""):
    handoff_to_reviewer = create_handoff_tool(
        agent_name="reviewer",
        description="Hand off to the Reviewer agent after generating code.",
    )
    return create_agent(
        model,
        tools=[*CODEGEN_TOOLS, *VECTOR_SEARCH_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_PATTERN, claude_agent_codegen, handoff_to_reviewer],
        system_prompt=get_prompt("swarm_coder", "system") + (f"\n\n{prompt_suffix}" if prompt_suffix else ""),
        name="coder",
    )


def _build_reviewer(model: "str | Any"):
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
        tools=[*GRAPH_READ_TOOLS, read_file, *EVAL_TOOLS, *EVAL_HISTORY_TOOLS, *DEEP_REVIEWER_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_MISTAKE, handoff_to_coder, handoff_to_tester],
        system_prompt=get_prompt("swarm_reviewer", "system"),
        name="reviewer",
    )


def _build_tester(model: "str | Any"):
    handoff_to_planner = create_handoff_tool(
        agent_name="planner",
        description="Hand off back to the Planner after writing tests.",
    )
    return create_agent(
        model,
        tools=[*TESTGEN_TOOLS, *EVAL_TOOLS, *DEEP_TESTER_TOOLS, *MEMORY_READ_TOOLS, *MEMORY_WRITE_MISTAKE, handoff_to_planner],
        system_prompt=get_prompt("swarm_tester", "system"),
        name="tester",
    )


# ── Low-level API (used by orchestrator) ─────────────────────────────


def init_swarm_context(
    settings: Settings,
) -> tuple[GraphRepository, Neo4jClient, Neo4jClient | None]:
    """One-time setup: create Neo4j clients and set module-level tool state.

    Returns ``(repo, neo4j_client, memory_client)``.

    H6 fix: when the FastAPI app lifespan has already installed a shared
    memory_repo and vector_repo via ``set_memory_repo`` / ``set_vector_repo``,
    this function reuses them instead of building new Neo4j drivers per
    pipeline run. CLI/standalone callers (where the globals are unset)
    still get fresh clients as before.
    """
    from dark_factory.agents import tools as _tools_mod

    neo4j_client = Neo4jClient(settings.neo4j)
    repo = GraphRepository(neo4j_client)

    set_graph_repo(repo)
    set_output_dir(settings.pipeline.output_dir)
    set_openspec_root(settings.openspec.root_dir)
    set_eval_config(settings.evaluation)

    # Reuse the lifespan-managed vector_repo if it exists
    vector_repo = _tools_mod._vector_repo
    if vector_repo is None and settings.qdrant.enabled:
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

    # Reuse the lifespan-managed memory_repo if it exists, else create one
    memory_client = None
    if _tools_mod._memory_repo is not None:
        # Already installed by app lifespan — nothing to do here. The
        # memory_client lifecycle is owned by the lifespan.
        pass
    elif settings.memory.enabled:
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


# Cached callback-handler class built on first use. See the comment
# on ``_get_stop_reason_handler`` below for why we defer the
# ``BaseCallbackHandler`` subclassing.
_STOP_REASON_HANDLER_CLASS: Any = None


def _get_stop_reason_handler() -> Any:
    """Return a cached ``BaseCallbackHandler`` subclass that logs a
    WARN when Claude stops for any reason other than normal turn
    completion.

    We subclass ``BaseCallbackHandler`` lazily (first time the
    function is called) rather than at module import time for two
    reasons:

    1. **Import hygiene**: ``langchain_core.callbacks`` is a
       moderately heavy import. The swarm module is loaded from
       places that don't need callback support (tests, CLI entry
       points), so deferring keeps the import surface small.
    2. **Test isolation**: importing ``BaseCallbackHandler`` at
       module level would create a concrete dependency between
       ``swarm`` import and ``langchain_core`` layout. Lazy import
       lets us catch ``ImportError`` and fall back to ``None`` when
       the package is missing, which matters for non-anthropic
       providers.

    The inner class is a proper subclass (required by
    ``ChatAnthropic``'s Pydantic validator — a duck-typed class
    fails the ``isinstance(BaseCallbackHandler)`` check).
    """
    global _STOP_REASON_HANDLER_CLASS
    if _STOP_REASON_HANDLER_CLASS is not None:
        return _STOP_REASON_HANDLER_CLASS

    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except Exception:  # pragma: no cover — defensive
        return None

    class _SwarmStopReasonLogger(BaseCallbackHandler):
        """LangChain callback that logs a WARN when Claude stops for
        any reason other than normal turn completion.

        Our own :class:`~dark_factory.llm.anthropic.AnthropicClient`
        already logs this at ``llm/anthropic.py:286``, but the
        swarm's ``ChatAnthropic`` route was silent. Surfacing
        ``stop_reason="max_tokens"`` via this callback means a
        future budget-exhaustion failure shows up in structured
        logs immediately instead of only manifesting as a
        downstream tool validation error (e.g. the
        ``Error invoking tool 'write_file'`` pattern that prompted
        this fix).
        """

        def on_llm_end(self, response, **kwargs: Any) -> None:
            try:
                generations = getattr(response, "generations", None) or []
                for generation_list in generations:
                    for gen in generation_list:
                        info = getattr(gen, "generation_info", None) or {}
                        stop = info.get("stop_reason")
                        if stop and stop not in (
                            "end_turn",
                            "stop_sequence",
                            "tool_use",
                        ):
                            log.warning(
                                "swarm_llm_stop_reason",
                                stop_reason=stop,
                                hint=(
                                    "Claude stopped the completion "
                                    "before the normal end of turn. "
                                    "If this is 'max_tokens', raise "
                                    "settings.pipeline.max_llm_tokens "
                                    "or break the artifact into "
                                    "smaller write_file calls."
                                ),
                            )
            except Exception:  # pragma: no cover — defensive
                pass

    _STOP_REASON_HANDLER_CLASS = _SwarmStopReasonLogger
    return _STOP_REASON_HANDLER_CLASS


_PROGRESS_HANDLER_CLASS: Any = None


def _get_progress_handler(feature_name: str) -> Any:
    """Return a callback handler that emits ``agent_llm_start``
    progress events so the Agent Logs tab shows real-time activity
    during each agent's LLM call — not just after the node completes.

    Without this, the planner (and every other agent) appears silent
    in the live log for the entire duration of its LLM call because
    LangGraph's ``stream_mode="updates"`` only yields after the full
    node finishes.
    """
    global _PROGRESS_HANDLER_CLASS

    if _PROGRESS_HANDLER_CLASS is not None:
        return _PROGRESS_HANDLER_CLASS(feature_name)

    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except Exception:  # pragma: no cover
        return None

    class _SwarmProgressHandler(BaseCallbackHandler):
        """Emit a progress event when any swarm agent starts an LLM call."""

        def __init__(self, feature: str) -> None:
            super().__init__()
            self._feature = feature

        def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
            try:
                from dark_factory.agents.tools import emit_progress

                # Try to extract the agent name from the invocation params
                invocation = kwargs.get("invocation_params", {}) or {}
                model = invocation.get("model", "") or invocation.get("model_name", "") or ""
                metadata = kwargs.get("metadata", {}) or {}
                agent = metadata.get("langgraph_node", "") or ""

                emit_progress(
                    "agent_llm_start",
                    feature=self._feature,
                    agent=agent or None,
                    model=model or None,
                )
            except Exception:
                pass

    _PROGRESS_HANDLER_CLASS = _SwarmProgressHandler
    return _PROGRESS_HANDLER_CLASS(feature_name)


def build_chat_model(settings: Settings, model_override: str | None = None) -> Any:
    """Construct a langchain chat model.

    Parameters
    ----------
    settings:
        Global settings (provides default model, provider, token limits).
    model_override:
        If given, use this model id instead of ``settings.llm.model``.
        Used by multi-model routing to give each swarm agent its own model.

    **Why this function exists**: passing a bare model string like
    ``"anthropic:claude-sonnet-4-6"`` to ``create_agent`` makes
    langchain build a ``ChatAnthropic`` with the library default of
    ``max_tokens=1024``. That ceiling is too small to hold a real
    ``write_file`` tool call — any dashboard, component file, or
    multi-hundred-line artifact runs Claude into the token wall
    mid-stream, the tool_use JSON is truncated, Pydantic rejects the
    malformed kwargs, and LangGraph raises
    ``Error invoking tool 'write_file' with kwargs {...}``. Happened
    routinely on utilization-dashboard-style files.

    The fix is to construct the ``ChatAnthropic`` instance ourselves
    with explicit ``max_tokens`` + ``timeout`` + a stop-reason
    callback for observability, then pass the instance to
    ``create_agent``. ``create_agent`` accepts both strings and
    ``BaseChatModel`` instances — the latter is the documented way
    to customise model parameters.

    Non-Anthropic providers fall back to the original string-based
    init so this change is provider-scoped.
    """
    model_id = model_override or settings.llm.model

    if settings.llm.provider != "anthropic":
        # Keep the old behaviour for any future provider. String
        # shortcut + langchain defaults.
        return f"{settings.llm.provider}:{model_id}"

    # Lazy import so non-anthropic deployments don't need the package.
    from langchain_anthropic import ChatAnthropic

    from dark_factory.llm.base import DEFAULT_LLM_TIMEOUT_SECONDS

    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": settings.pipeline.max_llm_tokens,
        "timeout": DEFAULT_LLM_TIMEOUT_SECONDS,
    }

    # Attach the stop-reason logger if langchain_core is importable.
    # Missing callback support is non-fatal — the model still works,
    # just without the extra observability hook.
    handler_class = _get_stop_reason_handler()
    if handler_class is not None:
        try:
            kwargs["callbacks"] = [handler_class()]
        except Exception:  # pragma: no cover — defensive
            pass

    return ChatAnthropic(**kwargs)


def build_feature_swarm(
    model: "str | Any",
    strategy_overrides: dict[str, Any] | None = None,
    *,
    models: dict[str, Any] | None = None,
) -> Any:
    """Build and compile a single swarm instance.

    Does NOT create a Neo4j client — assumes ``init_swarm_context`` was called.
    ``strategy_overrides`` modifies agent prompts when performance is poor.
    Returns the compiled LangGraph graph.

    Parameters
    ----------
    model:
        Default model for all agents (langchain ``BaseChatModel`` or string).
    models:
        Optional per-agent model map with keys ``"planner"``, ``"coder"``,
        ``"reviewer"``, ``"tester"``. Missing keys fall back to *model*.
    """
    suffix = _strategy_suffix(strategy_overrides)
    m = models or {}

    workflow = create_swarm(
        agents=[
            _build_planner(m.get("planner", model)),
            _build_coder(m.get("coder", model), prompt_suffix=suffix),
            _build_reviewer(m.get("reviewer", model)),
            _build_tester(m.get("tester", model)),
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
    """Stream a compiled swarm with an isolated message history.

    Each call gets a fresh message list, so the handoff budget resets.
    Sets ``_current_feature`` so memory write tools auto-tag the feature.
    ``run_context`` is prepended as cross-feature learnings from earlier features.

    Uses ``compiled.stream()`` so each agent-node transition emits a
    progress event via ``emit_progress``, which the AG-UI bridge forwards
    to the SSE stream for real-time visibility in the UI.
    """
    from dark_factory.agents.cancellation import (
        PipelineCancelled,
        is_cancelled,
        raise_if_cancelled,
    )
    from dark_factory.agents.tools import emit_progress, get_current_run_id

    # Pre-flight cancel check: if a cancel was requested while this worker
    # was waiting on the thread pool, skip it entirely rather than spending
    # any LLM budget.
    raise_if_cancelled()

    set_current_feature(feature_name)
    run_id = get_current_run_id() or None
    stats = _SwarmStats(feature_name)
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
    # M15 fix: enforce a hard recursion limit on the swarm so a runaway
    # agent loop can't blow past max_handoffs
    stream_config: dict[str, Any] = {"recursion_limit": max_handoffs * 4}

    # Inject a progress callback so agent LLM calls emit real-time
    # events to the Agent Logs tab (without this, the log is silent
    # until each graph node fully completes its LLM call).
    progress_handler = _get_progress_handler(feature_name)
    if progress_handler is not None:
        stream_config["callbacks"] = [progress_handler]

    # Track tools called during this run so we can decide success
    tool_calls_made: dict[str, int] = {}

    def _track_tool(msg: Any) -> None:
        for tc in _get_tool_calls(msg):
            name = tc.get("name", "")
            if name and not name.startswith("transfer_to_"):
                tool_calls_made[name] = tool_calls_made.get(name, 0) + 1

    try:
        # Stream node updates and inspect each chunk's messages so we can
        # emit fine-grained events that distinguish:
        #   - agent_active: which agent is currently doing work
        #   - agent_decision: text reasoning from an agent
        #   - tool_call:    agent invoked a tool
        #   - tool_result:  tool returned a result
        #   - agent_handoff: explicit handoff between agents (transfer_to_*)
        handoff_count = 0
        last_agent: str = ""
        # H2 fix: dedupe by python id() so the full-history state shape
        # doesn't cause N² re-emission of already-seen messages.
        seen_message_ids: set[int] = set()

        for chunk in compiled.stream(
            initial, stream_mode="updates", config=stream_config
        ):
            # Cancellation check between swarm stream chunks. This is the
            # tightest loop where checking is cheap (once per LangGraph
            # update, typically once per tool call or agent turn) and where
            # stopping saves the most downstream cost — each iteration
            # avoided is one fewer LLM round-trip.
            if is_cancelled():
                log.warning(
                    "feature_swarm_cancelled",
                    feature=feature_name,
                    handoffs_so_far=handoff_count,
                )
                raise PipelineCancelled(
                    f"swarm cancelled for feature={feature_name}"
                )

            if not isinstance(chunk, dict):
                continue
            for node_name, state_update in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue

                # Track current agent (skip generic tool/internal nodes)
                is_agent_node = node_name not in ("tools", "tool_node")
                if is_agent_node and node_name != last_agent:
                    last_agent = node_name
                    handoff_count += 1
                    stats.agent_transitions = handoff_count
                    stats.note_agent_active(node_name)
                    emit_progress(
                        "agent_active",
                        feature=feature_name,
                        agent=node_name,
                        messages=handoff_count,
                    )
                    log.debug(
                        "swarm_agent_active",
                        feature=feature_name,
                        agent=node_name,
                        handoff=handoff_count,
                    )

                # Inspect the messages added in this update for tool calls and results
                if isinstance(state_update, dict):
                    messages = state_update.get("messages", [])
                    if isinstance(messages, list):
                        for msg in messages:
                            mid = id(msg)
                            if mid in seen_message_ids:
                                continue
                            seen_message_ids.add(mid)
                            _track_tool(msg)
                            _emit_message_events(
                                msg,
                                feature_name,
                                last_agent or node_name,
                                stats=stats,
                                run_id=run_id,
                            )

        log.info(
            "feature_swarm_complete",
            feature=feature_name,
            handoffs=handoff_count,
            tool_calls=sum(tool_calls_made.values()),
        )

        # C2 fix: derive a real success signal from tool activity. A swarm
        # that never wrote any code or generated any tests is NOT a success,
        # regardless of whether the stream completed without raising.
        wrote_code = (
            tool_calls_made.get("write_file", 0) > 0
            or tool_calls_made.get("claude_agent_codegen", 0) > 0
        )
        _emit_agent_stats(stats, run_id=run_id)
        if not wrote_code:
            return FeatureResult(
                feature=feature_name,
                spec_ids=spec_ids,
                status="error",
                artifacts=[],
                tests=[],
                error="Swarm completed without writing any code (no write_file or claude_agent_codegen calls).",
                eval_scores={},
                stats=stats.finalize(),
            )

        return FeatureResult(
            feature=feature_name,
            spec_ids=spec_ids,
            status="success",
            artifacts=[],
            tests=[],
            error=None,
            eval_scores={},
            stats=stats.finalize(),
        )
    except PipelineCancelled as exc:
        # Kill-switch path: return a skipped result rather than an error
        # so the orchestrator's aggregate doesn't flag this as a crash.
        log.warning("feature_swarm_cancelled_clean", feature=feature_name)
        _emit_agent_stats(stats, run_id=run_id)
        return FeatureResult(
            feature=feature_name,
            spec_ids=spec_ids,
            status="skipped",
            artifacts=[],
            tests=[],
            error="Cancelled by user",
            eval_scores={},
            stats=stats.finalize(),
        )
    except Exception as exc:
        log.error("feature_swarm_failed", feature=feature_name, error=str(exc))
        stats.worker_crashed = True
        _emit_agent_stats(stats, run_id=run_id)
        return FeatureResult(
            feature=feature_name,
            spec_ids=spec_ids,
            status="error",
            artifacts=[],
            tests=[],
            error=str(exc),
            eval_scores={},
            stats=stats.finalize(),
        )
    finally:
        # L16 fix: clear thread-local state so a reused worker thread
        # doesn't carry stale feature/recall context into the next swarm.
        set_current_feature("")


def _emit_agent_stats(stats: _SwarmStats, *, run_id: str | None) -> None:
    """Flush per-agent rollup rows to the metrics recorder."""
    try:
        from dark_factory.metrics.helpers import record_agent_stats

        for agent, counters in stats.per_agent.items():
            record_agent_stats(
                run_id=run_id,
                feature=stats.feature,
                agent=agent,
                activations=counters.get("activations", 0),
                tool_calls=counters.get("tool_calls", 0),
                decisions=counters.get("decisions", 0),
                handoffs_in=counters.get("handoffs_in", 0),
                handoffs_out=counters.get("handoffs_out", 0),
            )
    except Exception:  # pragma: no cover — defensive
        pass


# ── High-level API (standalone usage) ────────────────────────────────


def build_swarm(settings: Settings) -> tuple[Any, Neo4jClient, Neo4jClient | None]:
    """Build the swarm. Returns ``(compiled_graph, neo4j_client, memory_client)``."""
    _repo, neo4j_client, memory_client = init_swarm_context(settings)
    # Construct the chat model via ``build_chat_model`` so ``max_tokens``
    # and ``timeout`` are set explicitly — langchain's default of 1024
    # max_tokens would truncate large ``write_file`` calls mid-content
    # and surface as tool validation errors.
    model = build_chat_model(settings)
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
