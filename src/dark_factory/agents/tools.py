"""LangChain tools that expose the knowledge graph and file system to agents."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from dark_factory.graph.repository import GraphRepository
    from dark_factory.memory.repository import MemoryRepository

# Module-level references set by the pipeline before agent invocation.
_graph_repo: GraphRepository | None = None
_output_dir: Path | None = None
_openspec_root: Path | None = None
_memory_repo: MemoryRepository | None = None
_current_run_id: str = ""
_eval_config: Any = None  # EvaluationConfig, lazy to avoid circular import
_vector_repo: Any = None  # VectorRepository, lazy to avoid circular import

# Thread-local state for per-feature isolation during parallel execution
_thread_local = threading.local()


def set_graph_repo(repo: GraphRepository) -> None:
    global _graph_repo
    _graph_repo = repo


def set_output_dir(path: str | Path) -> None:
    global _output_dir
    _output_dir = Path(path)


def set_openspec_root(path: str | Path) -> None:
    global _openspec_root
    _openspec_root = Path(path)


def set_memory_repo(repo: MemoryRepository | None) -> None:
    global _memory_repo
    _memory_repo = repo


def set_current_feature(name: str) -> None:
    _thread_local.current_feature = name


def get_current_feature() -> str:
    return getattr(_thread_local, "current_feature", "")


def set_current_run_id(run_id: str) -> None:
    global _current_run_id
    _current_run_id = run_id


def set_eval_config(config: Any) -> None:
    global _eval_config
    _eval_config = config


def add_recalled_memory_ids(ids: list[str]) -> None:
    if not hasattr(_thread_local, "recalled_memory_ids"):
        _thread_local.recalled_memory_ids = []
    _thread_local.recalled_memory_ids.extend(ids)


def get_recalled_memory_ids() -> list[str]:
    return getattr(_thread_local, "recalled_memory_ids", [])


def clear_recalled_memories() -> None:
    _thread_local.recalled_memory_ids = []


def set_vector_repo(repo: Any) -> None:
    global _vector_repo
    _vector_repo = repo


# ── Knowledge-graph tools ──────────────────────────────────────────────


@tool
def query_graph(cypher: str) -> str:
    """Run a read-only Cypher query against the Neo4j knowledge graph and return results as JSON."""
    if _graph_repo is None:
        return "Error: graph repository not initialised."
    with _graph_repo.client.session() as session:
        result = session.run(cypher)
        records = [dict(r) for r in result]
        return json.dumps(records, default=str)


@tool
def get_spec_context(spec_id: str) -> str:
    """Get a specification and its full dependency/requirement context from the knowledge graph."""
    if _graph_repo is None:
        return "Error: graph repository not initialised."
    ctx = _graph_repo.get_spec_with_context(spec_id)
    return ctx or "No context found for this spec."


@tool
def list_specs() -> str:
    """List all specifications currently stored in the knowledge graph."""
    if _graph_repo is None:
        return "Error: graph repository not initialised."
    with _graph_repo.client.session() as session:
        result = session.run("MATCH (s:Spec) RETURN s.id AS id, s.title AS title ORDER BY s.id")
        records = [{"id": r["id"], "title": r["title"]} for r in result]
        return json.dumps(records, default=str)


@tool
def list_requirements() -> str:
    """List all requirements currently stored in the knowledge graph."""
    if _graph_repo is None:
        return "Error: graph repository not initialised."
    with _graph_repo.client.session() as session:
        result = session.run(
            "MATCH (r:Requirement) RETURN r.id AS id, r.title AS title, r.priority AS priority ORDER BY r.id"
        )
        records = [dict(r) for r in result]
        return json.dumps(records, default=str)


# ── File-system tools ──────────────────────────────────────────────────


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the output directory. Returns the absolute path written."""
    base = _output_dir or Path("./output")
    out = base / file_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    return str(out.resolve())


@tool
def read_file(file_path: str) -> str:
    """Read a file from the output directory."""
    base = _output_dir or Path("./output")
    target = base / file_path
    if not target.exists():
        return f"Error: {target} does not exist."
    return target.read_text()


# ── OpenSpec tools ────────────────────────────────────────────────────


@tool
def write_openspec(capability: str, spec_json: str) -> str:
    """Write a specification as an OpenSpec spec.md file. spec_json must be valid Spec JSON."""
    from dark_factory.models.domain import Spec
    from dark_factory.openspec.writer import write_spec_md

    root = _openspec_root or Path("./openspec")
    spec = Spec.model_validate_json(spec_json)
    if not spec.capability:
        spec.capability = capability
    path = write_spec_md(spec, root)
    return f"Wrote OpenSpec: {path}"


@tool
def read_openspec(capability: str) -> str:
    """Read an OpenSpec spec.md file for a given capability."""
    root = _openspec_root or Path("./openspec")
    spec_file = root / "specs" / capability / "spec.md"
    if not spec_file.exists():
        return f"Error: no spec found at {spec_file}"
    return spec_file.read_text()


# ── Claude Agent SDK tool ─────────────────────────────────────────────


@tool
def claude_agent_codegen(spec_context: str, instructions: str) -> str:
    """Delegate code generation to the Claude Agent SDK which has built-in
    file editing, shell, and search capabilities. Provide the spec context
    and any specific instructions (e.g. fix feedback). Returns the agent's
    output describing what was generated."""
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    cwd = str(_output_dir or Path("./output"))
    prompt = (
        f"Generate production-quality code based on this specification:\n\n"
        f"{spec_context}\n\n"
        f"Instructions: {instructions}\n\n"
        f"Write all files to the current working directory. Use Read/Write/Edit "
        f"tools to create well-structured code."
    )
    opts = ClaudeAgentOptions(
        cwd=cwd,
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
        permission_mode="acceptEdits",
        max_turns=25,
    )

    async def _run():
        texts: list[str] = []
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, ResultMessage) and message.result:
                texts.append(message.result)
        return "\n".join(texts) if texts else "Code generation completed."

    return asyncio.run(_run())


# ── Procedural memory tools ───────────────────────────────────────────

_MEMORY_DISABLED_MSG = "Memory system is disabled."


@tool
def recall_memories(feature_name: str, spec_id: str = "") -> str:
    """Retrieve relevant procedural memories using hybrid semantic + keyword search.
    Call this before starting work on a feature or spec."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG

    # Neo4j structured search (always)
    neo4j_results = _memory_repo.get_related_memories(
        feature_name=feature_name, spec_id=spec_id, limit=20,
    )

    # Qdrant semantic search (if available)
    vector_results: list[dict] = []
    if _vector_repo:
        try:
            vector_results = _vector_repo.search_memories(
                query_text=feature_name, source_feature=feature_name, limit=20,
            )
        except Exception:
            pass  # graceful degradation

    # Hybrid merge via Reciprocal Rank Fusion
    if vector_results:
        from dark_factory.vector.merge import hybrid_merge
        memories = hybrid_merge(neo4j_results, vector_results, limit=10)
    else:
        memories = neo4j_results[:10]

    if not memories:
        return "No relevant memories found."
    # Track recalled memory IDs for eval feedback loop
    ids = [m.get("id", "") for m in memories if m.get("id")]
    if ids:
        add_recalled_memory_ids(ids)
    return json.dumps(memories, default=str)


@tool
def search_memory(keywords: str, memory_type: str = "all") -> str:
    """Search procedural memory semantically. Falls back to keyword search
    if vector search is unavailable. memory_type: 'pattern', 'mistake',
    'solution', 'strategy', or 'all'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG

    # Try semantic search first
    if _vector_repo:
        try:
            mem_type_filter = None if memory_type == "all" else memory_type
            results = _vector_repo.search_memories(
                query_text=keywords, memory_type=mem_type_filter, limit=10,
            )
            if results:
                return json.dumps(results, default=str)
        except Exception:
            pass  # fall through to Neo4j keyword search

    # Fallback: Neo4j keyword search
    results: list[dict] = []
    if memory_type in ("pattern", "all"):
        results.extend(_memory_repo.search_patterns(keywords=keywords))
    if memory_type in ("mistake", "all"):
        results.extend(_memory_repo.search_mistakes(keywords=keywords))
    if memory_type in ("solution", "all"):
        results.extend(_memory_repo.search_solutions(keywords=keywords))
    if memory_type in ("strategy", "all"):
        results.extend(_memory_repo.get_strategies(keywords=keywords))
    if not results:
        return "No memories found matching those keywords."
    return json.dumps(results, default=str)


@tool
def record_pattern(description: str, context: str) -> str:
    """Record a reusable coding pattern learned during this run.
    Example: 'When generating auth modules, always include rate limiting'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    node_id = _memory_repo.record_pattern(
        description=description, context=context,
        source_feature=get_current_feature(), agent="coder",
        run_id=_current_run_id,
    )
    return f"Pattern recorded: {node_id}"


@tool
def record_mistake(description: str, error_type: str, trigger_context: str) -> str:
    """Record a mistake found during review or testing.
    Example: 'Using sync I/O in async handler caused test failures'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    node_id = _memory_repo.record_mistake(
        description=description, error_type=error_type,
        trigger_context=trigger_context,
        source_feature=get_current_feature(), agent="reviewer",
        run_id=_current_run_id,
    )
    return f"Mistake recorded: {node_id}"


@tool
def record_solution(description: str, mistake_id: str = "", code_snippet: str = "") -> str:
    """Record a solution. Optionally link to a mistake_id it resolves."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    node_id = _memory_repo.record_solution(
        description=description, mistake_id=mistake_id,
        code_snippet=code_snippet,
        source_feature=get_current_feature(), agent="reviewer",
        run_id=_current_run_id,
    )
    return f"Solution recorded: {node_id}"


@tool
def record_strategy(description: str, applicability: str) -> str:
    """Record a planning/execution strategy.
    Example: 'For specs with >3 dependencies, query all dep contexts first'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    node_id = _memory_repo.record_strategy(
        description=description, applicability=applicability,
        source_feature=get_current_feature(), agent="planner",
        run_id=_current_run_id,
    )
    return f"Strategy recorded: {node_id}"


# ── AI evaluation tool ────────────────────────────────────────────────


def _auto_persist_eval(results: dict, spec_id: str, eval_type: str) -> None:
    """Persist eval result to memory and apply feedback to recalled memories."""
    if _memory_repo is None or not _current_run_id:
        return
    _memory_repo.record_eval_result(
        spec_id=spec_id,
        feature_name=get_current_feature(),
        eval_type=eval_type,
        metrics=results,
        run_id=_current_run_id,
        recalled_memory_ids=list(get_recalled_memory_ids()),
    )
    all_passed = all(m.get("passed", False) for m in results.values() if isinstance(m, dict))
    boost = _eval_config.boost_delta if _eval_config else 0.1
    demote = _eval_config.demote_delta if _eval_config else 0.05
    _memory_repo.apply_eval_feedback(
        recalled_memory_ids=list(get_recalled_memory_ids()),
        all_passed=all_passed,
        boost_delta=boost,
        demote_delta=demote,
    )
    clear_recalled_memories()


def _get_adaptive_threshold(spec_id: str) -> float:
    """Compute adaptive threshold from eval history if enabled."""
    base = _eval_config.base_threshold if _eval_config else 0.5
    if not _eval_config or not _eval_config.adaptive or _memory_repo is None:
        return base
    trend = _memory_repo.get_spec_eval_trend(spec_id=spec_id, window=_eval_config.trend_window)
    if len(trend) < 3:
        return base
    from dark_factory.evaluation.adaptive import compute_adaptive_threshold
    return compute_adaptive_threshold(
        base_threshold=base,
        recent_scores=trend,
        threshold_min=_eval_config.threshold_min,
        threshold_max=_eval_config.threshold_max,
    )


@tool
def evaluate_spec(
    requirement_title: str,
    requirement_description: str,
    spec_json: str,
    spec_id: str = "",
) -> str:
    """Evaluate a generated specification for correctness, coherence,
    instruction following, and safety/ethics using deepeval AI metrics.
    Returns a JSON report with scores and pass/fail for each metric.
    Results are auto-persisted to memory and feed back into relevance scoring."""
    from dark_factory.evaluation.metrics import evaluate_generated_spec

    threshold = _get_adaptive_threshold(spec_id) if spec_id else (
        _eval_config.base_threshold if _eval_config else 0.5
    )
    results = evaluate_generated_spec(
        requirement_title=requirement_title,
        requirement_description=requirement_description,
        spec_json=spec_json,
        threshold=threshold,
    )
    _auto_persist_eval(results, spec_id=spec_id or get_current_feature(), eval_type="spec")
    return json.dumps(results, default=str)


@tool
def evaluate_tests(
    spec_title: str,
    acceptance_criteria_json: str,
    source_code: str,
    test_code: str,
    spec_id: str = "",
) -> str:
    """Evaluate generated tests for correctness, coherence, and completeness
    using deepeval AI metrics. Returns a JSON report with scores and pass/fail
    for each metric. Results are auto-persisted and feed back into memory."""
    from dark_factory.evaluation.metrics import evaluate_generated_tests

    threshold = _get_adaptive_threshold(spec_id) if spec_id else (
        _eval_config.base_threshold if _eval_config else 0.5
    )
    criteria = json.loads(acceptance_criteria_json) if acceptance_criteria_json else []
    results = evaluate_generated_tests(
        spec_title=spec_title,
        acceptance_criteria=criteria,
        source_code=source_code,
        test_code=test_code,
        threshold=threshold,
    )
    _auto_persist_eval(results, spec_id=spec_id or get_current_feature(), eval_type="test")
    return json.dumps(results, default=str)


# ── Eval history tools ────────────────────────────────────────────────


@tool
def query_eval_history(spec_id: str, eval_type: str = "", limit: int = 5) -> str:
    """Query past evaluation results for a spec. Shows score trends
    so you can see if quality is improving or degrading across runs."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    history = _memory_repo.get_eval_history(
        spec_id=spec_id, eval_type=eval_type or None, limit=limit,
    )
    if not history:
        return "No evaluation history found for this spec."
    return json.dumps(history, default=str)


@tool
def query_run_history(limit: int = 5) -> str:
    """Query recent pipeline run summaries, including pass rates and mean scores."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    runs = _memory_repo.get_run_history(limit=limit)
    if not runs:
        return "No run history found."
    return json.dumps(runs, default=str)


# ── Vector search / RAG tools ────────────────────────────────────────


@tool
def search_similar_specs(description: str, limit: int = 5) -> str:
    """Find specs similar to the given description using semantic search.
    Useful for finding reference implementations before generating new code."""
    if _vector_repo is None:
        return "Vector search is not available."
    try:
        results = _vector_repo.search_similar_specs(query_text=description, limit=limit)
    except Exception:
        return "Vector search failed."
    if not results:
        return "No similar specs found."
    return json.dumps(results, default=str)


@tool
def search_similar_code(description: str, language: str = "", limit: int = 5) -> str:
    """Find code artifacts similar to the given description. Returns file paths
    and content previews of past generated code for reference."""
    if _vector_repo is None:
        return "Vector search is not available."
    try:
        results = _vector_repo.search_similar_code(
            query_text=description, language=language or None, limit=limit,
        )
    except Exception:
        return "Vector search failed."
    if not results:
        return "No similar code found."
    return json.dumps(results, default=str)


# ── Tool sets for different agent roles ────────────────────────────────

GRAPH_READ_TOOLS = [query_graph, get_spec_context, list_specs, list_requirements]
FILE_TOOLS = [write_file, read_file]
OPENSPEC_TOOLS = [write_openspec, read_openspec]
CODEGEN_TOOLS = [*GRAPH_READ_TOOLS, *FILE_TOOLS, *OPENSPEC_TOOLS]
TESTGEN_TOOLS = [*GRAPH_READ_TOOLS, *FILE_TOOLS]
EVAL_TOOLS = [evaluate_spec, evaluate_tests]
EVAL_HISTORY_TOOLS = [query_eval_history, query_run_history]
VECTOR_SEARCH_TOOLS = [search_similar_specs, search_similar_code]

MEMORY_READ_TOOLS = [recall_memories, search_memory]
MEMORY_WRITE_PATTERN = [record_pattern]
MEMORY_WRITE_MISTAKE = [record_mistake, record_solution]
MEMORY_WRITE_STRATEGY = [record_strategy]
