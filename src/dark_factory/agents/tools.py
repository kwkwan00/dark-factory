"""LangChain tools that expose the knowledge graph and file system to agents."""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
from pathlib import Path

import structlog

log = structlog.get_logger()
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

# Progress broker for global fan-out to any subscribers (Run tab bridge,
# Agent Logs tab stream). Accessed from worker threads during swarm execution.
#
# H6: Python's GIL makes these module-global reference reads atomic in CPython.
# On free-threaded (PEP 703) builds we'd want explicit synchronisation, but
# the hot path is performance-critical and locks would be excessive. The single
# writer is set_progress_broker(), called once at app startup and once at
# shutdown — no contention in practice.
_progress_broker: Any = None  # ProgressBroker | None

# Metrics recorder (Postgres-backed). Installed once at app startup by the
# lifespan; when ``None``, metric writes are skipped silently. The recorder
# itself is non-blocking — see :class:`MetricsRecorder`.
_metrics_recorder: Any = None  # MetricsRecorder | None


def set_progress_broker(broker: Any) -> None:
    """Install the global progress broker (installed once at app startup).

    Events published here are fanned out to all subscribers. Pass ``None`` to clear."""
    global _progress_broker
    _progress_broker = broker


def set_metrics_recorder(recorder: Any) -> None:
    """Install the global metrics recorder. Pass ``None`` to clear."""
    global _metrics_recorder
    _metrics_recorder = recorder


def emit_progress(event: str, **data: Any) -> None:
    """Emit a progress event to the global broker AND the metrics recorder.

    Never raises — broker/recorder failures are logged but swallowed so a
    downstream metrics outage can never take the pipeline down.
    """
    broker = _progress_broker
    if broker is not None:
        try:
            broker.publish({"event": event, **data})
        except Exception:  # pragma: no cover — defensive
            # L5 fix: include stack trace for diagnosis
            log.exception("progress_broker_publish_failed", progress_event=event)

    recorder = _metrics_recorder
    if recorder is not None:
        try:
            recorder.record_progress_event({"event": event, **data})
        except Exception:  # pragma: no cover — defensive
            log.exception("metrics_recorder_failed", progress_event=event)


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


# H6: in-flight deep-agent tracking. When a worker thread spawns a
# Claude Agent SDK invocation via ``_run_deep_agent``, we increment
# this counter at the start and decrement on exit. If the worker
# thread crashes unexpectedly, the orchestrator's crash handler can
# inspect the counter to know whether the crash left an orphaned
# subprocess — the actual cleanup is handled by the SDK's own async
# context managers running on BackgroundLoop, but the counter gives
# visibility into when orphans MAY have occurred.


def _increment_inflight_deep_agents() -> int:
    current = getattr(_thread_local, "inflight_deep_agents", 0)
    new = current + 1
    _thread_local.inflight_deep_agents = new
    return new


def _decrement_inflight_deep_agents() -> int:
    current = getattr(_thread_local, "inflight_deep_agents", 0)
    new = max(0, current - 1)
    _thread_local.inflight_deep_agents = new
    return new


def get_inflight_deep_agent_count() -> int:
    """Return the number of deep-agent calls currently running in this thread.

    Used by the orchestrator's ThreadPoolExecutor crash handler to
    detect potentially orphaned subprocess state after a worker
    exception. The SDK's own async cleanup still runs — this is a
    diagnostic signal, not a hard cleanup handle.
    """
    return getattr(_thread_local, "inflight_deep_agents", 0)


def set_current_run_id(run_id: str) -> None:
    global _current_run_id
    _current_run_id = run_id


def get_current_run_id() -> str:
    """Return the run_id set by set_current_run_id (empty string if unset)."""
    return _current_run_id


def set_eval_config(config: Any) -> None:
    global _eval_config
    _eval_config = config


def add_recalled_memory_ids(ids: list[str]) -> None:
    """Replace (not append) the recalled-memory IDs for this thread.

    C5 fix: each ``recall_memories`` call resets the list so feedback from
    a subsequent ``evaluate_*`` only applies to the MOST RECENT recall.
    Previously, IDs accumulated across specs in a thread, causing one
    spec's eval to boost/demote memories recalled for an unrelated spec.
    """
    _thread_local.recalled_memory_ids = list(ids)


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
    """Run a read-only Cypher query against the Neo4j knowledge graph and return results as JSON.

    H3 fix: uses ``session.execute_read`` which routes through a server-enforced
    READ transaction. Unlike ``default_access_mode=READ_ACCESS`` (which only
    affects routing in clusters), this guarantees the LLM-supplied Cypher
    cannot mutate the graph.
    """
    if _graph_repo is None:
        return "Error: graph repository not initialised."

    def _read(tx, query: str):
        return [dict(r) for r in tx.run(query)]

    try:
        with _graph_repo.client.session() as session:
            records = session.execute_read(_read, cypher)
        return json.dumps(records, default=str)
    except Exception as exc:
        # Surface a structured error rather than letting the agent see a stacktrace
        return json.dumps({"error": str(exc)})


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
    """Write content to a file in the output directory. Returns the absolute path written.

    M17 fix: writes are namespaced under ``<output_dir>/<feature_name>/``
    when a feature is active, so parallel features can't clobber each
    other's files (e.g. two features both writing ``utils.py``).
    """
    base = (_output_dir or Path("./output")).resolve()
    feature = get_current_feature()
    if feature:
        base = (base / feature).resolve()
    out = (base / file_path).resolve()
    if not out.is_relative_to(base):
        return f"Error: path '{file_path}' escapes the output directory."
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    _metric_artifact_write(
        file_path=file_path,
        feature=feature,
        bytes_written=len(content.encode("utf-8")),
    )
    return str(out)


@tool
def read_file(file_path: str) -> str:
    """Read a file from the output directory."""
    base = (_output_dir or Path("./output")).resolve()
    feature = get_current_feature()
    if feature:
        base = (base / feature).resolve()
    target = (base / file_path).resolve()
    if not target.is_relative_to(base):
        return f"Error: path '{file_path}' escapes the output directory."
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
    root = (_openspec_root or Path("./openspec")).resolve()
    spec_file = (root / "specs" / capability / "spec.md").resolve()
    if not spec_file.is_relative_to(root):
        return f"Error: capability '{capability}' contains an invalid path."
    if not spec_file.exists():
        return f"Error: no spec found at {spec_file}"
    return spec_file.read_text()


# ── Deep agent helper ─────────────────────────────────────────────────


# M16: hard cap on a single deep-agent invocation. A hung Claude SDK
# subprocess can otherwise block a worker thread indefinitely.
DEEP_AGENT_TIMEOUT_SECONDS = float(os.getenv("DEEP_AGENT_TIMEOUT_SECONDS", "600"))

# Bound the per-invocation stderr buffer so a runaway Claude Agent SDK
# subprocess can't exhaust the worker's memory by logging to stderr in a
# tight loop. 200 lines / 16 KiB is enough to capture a Node.js
# traceback + a handful of preceding warnings — anything beyond that
# is almost certainly noise we don't need to keep.
DEEP_AGENT_STDERR_MAX_LINES = 200
DEEP_AGENT_STDERR_MAX_BYTES = 16 * 1024

# When truthy, pass ``debug-to-stderr`` through ``extra_args`` to the
# Claude Agent SDK. The underlying Node CLI becomes verbose about its
# startup / transport / message protocol state and writes everything
# to its own stderr. Our ``_stderr_cb`` then captures those lines and
# dumps them via ``log.error(deep_agent_failed, stderr_tail=…)`` when
# the subprocess crashes — giving us actual diagnostic info instead
# of the SDK's "Check stderr output for details" placeholder. Disabled
# by default (noisy) and toggled via ``DEEP_AGENT_DEBUG_STDERR=1``.
DEEP_AGENT_DEBUG_STDERR = os.getenv("DEEP_AGENT_DEBUG_STDERR", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _run_deep_agent(
    prompt: str,
    allowed_tools: list[str],
    max_turns: int = 15,
    timeout_seconds: float | None = None,
) -> str:
    """Shared helper: spawn a Claude Agent SDK subagent and return its output.

    Uses :class:`BackgroundLoop` (a singleton daemon-thread loop) to run the
    coroutine, so subprocess cleanup callbacks always have a valid loop to
    land on. This eliminates "Event loop is closed" errors that previously
    occurred when ``asyncio.run`` closed a fresh loop while the Claude SDK
    subprocess was still cleaning up.

    M16 fix: enforces a timeout so a hung subprocess doesn't block a
    worker thread forever. Pass ``timeout_seconds`` to override the
    per-invocation budget (used by the reconciliation stage which needs
    a much longer ceiling than the per-spec codegen calls). When
    ``None``, falls back to ``DEEP_AGENT_TIMEOUT_SECONDS`` (default
    600s, override via env var).

    **Stderr capture.** The Claude Agent SDK only pipes the subprocess
    stderr when ``ClaudeAgentOptions.stderr`` is set. Without it, when
    the underlying ``claude`` CLI / Node.js process crashes, the SDK
    raises ``CLIConnectionError("Command failed with exit code 1...
    Check stderr output for details")`` — and the "details" don't exist
    anywhere we can read them. We pass a callback that buffers every
    stderr line into a per-invocation list (capped via
    ``DEEP_AGENT_STDERR_MAX_LINES`` / ``DEEP_AGENT_STDERR_MAX_BYTES``)
    and dumps the buffer at ``log.error`` + records an incident when
    the run raises. Successful runs ignore the buffer.
    """
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    from dark_factory.agents.background_loop import BackgroundLoop

    effective_timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else DEEP_AGENT_TIMEOUT_SECONDS
    )
    log.info(
        "deep_agent_spawning",
        tools=allowed_tools,
        max_turns=max_turns,
        prompt_len=len(prompt),
        timeout_seconds=effective_timeout,
    )
    cwd = str(_output_dir or Path("./output"))

    # Per-invocation stderr buffer + bookkeeping. The callback fires
    # from the SDK's stderr-reader task on the BackgroundLoop, which
    # is a different thread from the orchestrator worker that called
    # us. ``list.append`` is thread-safe under the GIL so we don't
    # need a lock for this small write pattern.
    stderr_lines: list[str] = []
    stderr_bytes: list[int] = [0]  # boxed counter for closure mutation
    stderr_truncated: list[bool] = [False]

    def _stderr_cb(line: str) -> None:
        if stderr_truncated[0]:
            return
        if (
            len(stderr_lines) >= DEEP_AGENT_STDERR_MAX_LINES
            or stderr_bytes[0] + len(line) > DEEP_AGENT_STDERR_MAX_BYTES
        ):
            stderr_lines.append("… (stderr truncated)")
            stderr_truncated[0] = True
            return
        stderr_lines.append(line)
        stderr_bytes[0] += len(line)

    # ``extra_args`` gets forwarded to the Node CLI as command-line
    # flags. Passing ``debug-to-stderr`` with value ``None`` flips the
    # CLI into verbose mode; the SDK's ``subprocess_cli.py`` still
    # routes subprocess stderr through our callback (``options.stderr``
    # takes precedence over the backward-compat ``debug_stderr`` file
    # sink), so the extra chatter lands in ``_stderr_cb`` without
    # bypassing our buffer or size caps.
    extra_args: dict[str, str | None] = {}
    if DEEP_AGENT_DEBUG_STDERR:
        extra_args["debug-to-stderr"] = None

    opts = ClaudeAgentOptions(
        cwd=cwd,
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        max_turns=max_turns,
        stderr=_stderr_cb,
        extra_args=extra_args,
    )

    async def _run() -> str:
        texts: list[str] = []
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, ResultMessage) and message.result:
                texts.append(message.result)
        return "\n".join(texts) if texts else "Analysis completed."

    # H6: track this call as in-flight on the calling thread so the
    # orchestrator's ThreadPoolExecutor crash handler can detect
    # orphaned subprocess state after a worker exception. The actual
    # subprocess cleanup still runs inside the SDK's own async
    # context managers; this counter is the diagnostic signal, not
    # the cleanup handle.
    _increment_inflight_deep_agents()
    try:
        try:
            return BackgroundLoop.get().run(_run(), timeout=effective_timeout)
        finally:
            _decrement_inflight_deep_agents()
    except concurrent.futures.TimeoutError:
        log.error(
            "deep_agent_timeout",
            tools=allowed_tools,
            timeout=effective_timeout,
            stderr_lines=len(stderr_lines),
            stderr_tail=stderr_lines[-20:],
        )
        try:
            from dark_factory.metrics.helpers import record_incident
            from dark_factory.metrics.prometheus import observe_deep_agent_timeout

            observe_deep_agent_timeout()
            record_incident(
                category="subprocess",
                severity="error",
                message=f"Deep agent timed out after {effective_timeout:.0f}s",
                phase="deep_agent",
                feature=get_current_feature() or None,
                stack="\n".join(stderr_lines[-50:]) or None,
            )
        except Exception:  # pragma: no cover — defensive
            pass
        return f"Error: deep agent timed out after {effective_timeout:.0f}s."
    except Exception as exc:
        # The Claude Agent SDK raises ``CLIConnectionError`` (or other
        # subclasses) when the subprocess exits non-zero or the JSONL
        # stream protocol breaks. Without the stderr buffer above this
        # error reaches the swarm with the SDK's generic placeholder
        # ("Check stderr output for details") — useless to debug. Now
        # we dump the captured stderr into the log AND attach it to an
        # incident row so the operator can read it from the Run Detail
        # popup's incidents table.
        log.error(
            "deep_agent_failed",
            tools=allowed_tools,
            error=str(exc),
            stderr_lines=len(stderr_lines),
            # Tail rather than head so the actual crash message —
            # which is always the LAST thing on stderr — is visible.
            stderr_tail=stderr_lines[-30:],
        )
        try:
            from dark_factory.metrics.helpers import record_incident

            record_incident(
                category="subprocess",
                severity="error",
                message=f"Deep agent crashed: {exc}",
                phase="deep_agent",
                feature=get_current_feature() or None,
                stack="\n".join(stderr_lines[-50:]) or None,
            )
        except Exception:  # pragma: no cover — defensive
            pass
        # Re-raise so callers that handle subprocess failures with
        # their own try/except (reconciliation, doc_extraction,
        # e2e_validation) still see the exception. The ``@tool``
        # wrappers that use ``_safe_tool_deep_agent`` below convert
        # this into a soft error string so a transient SDK crash in a
        # best-effort analyzer does not cascade into
        # feature_swarm_failed for the whole feature.
        raise


def _safe_tool_deep_agent(
    prompt: str,
    allowed_tools: list[str],
    max_turns: int = 15,
    timeout_seconds: float | None = None,
) -> str:
    """Thin wrapper around :func:`_run_deep_agent` for ``@tool`` callers.

    The bare helper re-raises on subprocess failure, which used to
    propagate all the way out of the swarm and mark the entire
    feature as ``feature_swarm_failed`` — even for best-effort
    analyzers like ``deep_risk_assessment``. That cascade is wrong:
    a transient Claude Agent SDK crash in an advisory tool should
    degrade the analysis, not kill the feature.

    This wrapper catches any exception from ``_run_deep_agent`` and
    returns a structured error string that the calling LangGraph
    agent can reason about. The underlying helper still logs the
    failure and records an incident, so observability is unchanged.
    ``BaseException`` (cancellation, ``KeyboardInterrupt``,
    ``SystemExit``) is *not* caught — those must still unwind.
    """
    try:
        return _run_deep_agent(
            prompt,
            allowed_tools,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        # The raise site in ``_run_deep_agent`` has already logged
        # the crash with structured context and recorded an
        # incident, so we only need to hand the agent a readable
        # error it can react to. Include the tool list so the
        # LangGraph agent has enough context to decide whether to
        # retry with different inputs or skip to the next step.
        return (
            f"Error: deep agent (tools={allowed_tools}) failed: {exc}. "
            f"This call was best-effort; continue with the rest of the "
            f"workflow using any partial results you already have."
        )


# ── Coder deep agent ─────────────────────────────────────────────────


@tool
def claude_agent_codegen(spec_context: str, instructions: str) -> str:
    """Delegate code generation to the Claude Agent SDK which has built-in
    file editing, shell, and search capabilities. Provide the spec context
    and any specific instructions (e.g. fix feedback). Returns the agent's
    output describing what was generated."""
    prompt = (
        f"Generate production-quality code based on this specification:\n\n"
        f"{spec_context}\n\n"
        f"Instructions: {instructions}\n\n"
        f"Write all files to the current working directory. Use Read/Write/Edit "
        f"tools to create well-structured code."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Write", "Edit", "Glob", "Grep", "Bash"], max_turns=25)


# ── Planner deep agents ──────────────────────────────────────────────


@tool
def deep_dependency_analysis(spec_ids_json: str, feature_name: str, relevant_memories: str = "") -> str:
    """Spawn a deep agent to analyze the full dependency tree for a set of specs.
    Pass relevant_memories from recall_memories to give the subagent context about
    past dependency issues."""
    context_block = f"\n\nRelevant memories from past runs:\n{relevant_memories}" if relevant_memories else ""
    prompt = (
        f"Analyze the dependency tree for feature '{feature_name}'.\n"
        f"Spec IDs: {spec_ids_json}\n\n"
        f"Trace all transitive DEPENDS_ON relationships. Look for:\n"
        f"- Circular dependency risks\n"
        f"- Missing dependencies (specs that reference unknown IDs)\n"
        f"- Optimal execution order based on the dependency graph\n"
        f"{context_block}\n"
        f"Return a structured dependency report."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Glob", "Grep"], max_turns=10)


@tool
def deep_risk_assessment(feature_name: str, spec_context: str, relevant_memories: str = "", eval_history: str = "") -> str:
    """Spawn a deep agent to assess risks for a feature. Pass relevant_memories
    from recall_memories and eval_history from query_eval_history to give the
    subagent full context about past failures and score trends."""
    context_block = ""
    if relevant_memories:
        context_block += f"\n\nRelevant memories from past runs:\n{relevant_memories}"
    if eval_history:
        context_block += f"\n\nEval history for this feature:\n{eval_history}"
    prompt = (
        f"Assess risks for feature '{feature_name}'.\n\n"
        f"Spec context:\n{spec_context}\n\n"
        f"Check for patterns that historically caused issues:\n"
        f"- Complex dependency chains\n"
        f"- Specs touching security-sensitive areas\n"
        f"- Features similar to past failures\n"
        f"{context_block}\n"
        f"Rate overall risk as low/medium/high with detailed reasoning."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Glob", "Grep"], max_turns=10)


# ── Reviewer deep agents ─────────────────────────────────────────────


@tool
def deep_security_review(code_content: str, spec_context: str, relevant_memories: str = "") -> str:
    """Spawn a deep agent for focused security analysis. Pass relevant_memories
    from recall_memories (especially past security mistakes) for richer context."""
    context_block = f"\n\nPast security-related memories:\n{relevant_memories}" if relevant_memories else ""
    prompt = (
        f"Perform a thorough security review of this code:\n\n"
        f"```\n{code_content}\n```\n\n"
        f"Spec context:\n{spec_context}\n\n"
        f"Check for:\n"
        f"- SQL/command/XSS injection vulnerabilities\n"
        f"- Authentication and authorization gaps\n"
        f"- Sensitive data exposure\n"
        f"- Input validation gaps\n"
        f"- OWASP Top 10 risks\n"
        f"{context_block}\n"
        f"Return findings with severity ratings and fix recommendations."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Glob", "Grep"], max_turns=15)


@tool
def deep_performance_review(code_content: str, spec_context: str, relevant_memories: str = "") -> str:
    """Spawn a deep agent for focused performance analysis. Pass relevant_memories
    from recall_memories (especially past performance mistakes) for richer context."""
    context_block = f"\n\nPast performance-related memories:\n{relevant_memories}" if relevant_memories else ""
    prompt = (
        f"Analyze this code for performance issues:\n\n"
        f"```\n{code_content}\n```\n\n"
        f"Spec context:\n{spec_context}\n\n"
        f"Check for:\n"
        f"- Poor algorithmic complexity (O(n^2) or worse where avoidable)\n"
        f"- Resource leaks (unclosed connections, file handles)\n"
        f"- Unnecessary memory allocations\n"
        f"- Blocking I/O in async code paths\n"
        f"- Missing caching opportunities\n"
        f"{context_block}\n"
        f"Return findings with impact ratings and optimization suggestions."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Glob", "Grep"], max_turns=15)


@tool
def deep_spec_compliance_review(code_content: str, acceptance_criteria_json: str, eval_history: str = "") -> str:
    """Spawn a deep agent that checks code against each acceptance criterion.
    Pass eval_history from query_eval_history to show which criteria failed before."""
    context_block = f"\n\nPast eval results for this spec:\n{eval_history}" if eval_history else ""
    prompt = (
        f"Verify this code against each acceptance criterion.\n\n"
        f"Code:\n```\n{code_content}\n```\n\n"
        f"Acceptance criteria (JSON array):\n{acceptance_criteria_json}\n\n"
        f"For EACH criterion:\n"
        f"1. State the criterion\n"
        f"2. Find the code that implements it (quote the relevant lines)\n"
        f"3. Rate: PASS or FAIL\n"
        f"4. If FAIL, explain what's missing or incorrect\n"
        f"{context_block}\n"
        f"Return a structured checklist."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Glob", "Grep"], max_turns=15)


# ── Tester deep agents ───────────────────────────────────────────────


@tool
def deep_unit_test_gen(code_content: str, spec_context: str, relevant_memories: str = "") -> str:
    """Spawn a deep agent for unit test generation. Pass relevant_memories from
    recall_memories to inform the subagent about past testing patterns and pitfalls."""
    context_block = f"\n\nRelevant testing memories:\n{relevant_memories}" if relevant_memories else ""
    prompt = (
        f"Generate comprehensive unit tests for this code:\n\n"
        f"```\n{code_content}\n```\n\n"
        f"Spec context:\n{spec_context}\n\n"
        f"Requirements:\n"
        f"- Test each function/method independently\n"
        f"- Mock all external dependencies\n"
        f"- Cover happy paths, error paths, and boundary conditions\n"
        f"- Use descriptive test names that explain the scenario\n"
        f"- Write the test file to disk using Write tool\n"
        f"{context_block}\n"
        f"Return the complete test code."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Write", "Edit", "Glob", "Grep"], max_turns=20)


@tool
def deep_integration_test_gen(code_content: str, spec_context: str, relevant_memories: str = "") -> str:
    """Spawn a deep agent for integration test generation. Pass relevant_memories
    from recall_memories to inform about past integration issues."""
    context_block = f"\n\nRelevant integration memories:\n{relevant_memories}" if relevant_memories else ""
    prompt = (
        f"Generate integration tests for this code:\n\n"
        f"```\n{code_content}\n```\n\n"
        f"Spec context:\n{spec_context}\n\n"
        f"Requirements:\n"
        f"- Test cross-module interactions\n"
        f"- Test API endpoints end-to-end if applicable\n"
        f"- Test database interactions if applicable\n"
        f"- Verify behavior matches the spec's WHEN/THEN scenarios\n"
        f"- Write the test file to disk using Write tool\n"
        f"{context_block}\n"
        f"Return the complete test code."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Write", "Edit", "Glob", "Grep"], max_turns=20)


@tool
def deep_edge_case_test_gen(code_content: str, spec_context: str, past_mistakes_json: str = "[]", relevant_memories: str = "") -> str:
    """Spawn a deep agent for adversarial tests. Pass past_mistakes_json from
    search_memory(type='mistake') and relevant_memories from recall_memories."""
    context_block = f"\n\nAdditional testing memories:\n{relevant_memories}" if relevant_memories else ""
    prompt = (
        f"Generate edge case and adversarial tests for this code:\n\n"
        f"```\n{code_content}\n```\n\n"
        f"Spec context:\n{spec_context}\n\n"
        f"Past mistakes to guard against:\n{past_mistakes_json}\n\n"
        f"Requirements:\n"
        f"- Each past mistake should inform at least one test\n"
        f"- Test boundary conditions (empty inputs, max values, nulls)\n"
        f"- Test error handling paths\n"
        f"- Test concurrent access if applicable\n"
        f"- Write the test file to disk using Write tool\n"
        f"{context_block}\n"
        f"Return the complete test code."
    )
    return _safe_tool_deep_agent(prompt, ["Read", "Write", "Edit", "Glob", "Grep"], max_turns=20)


# ── Procedural memory tools ───────────────────────────────────────────

_MEMORY_DISABLED_MSG = "Memory system is disabled."


@tool
def recall_memories(feature_name: str, spec_id: str = "") -> str:
    """Retrieve relevant procedural memories using hybrid semantic + keyword search.
    Call this before starting work on a feature or spec."""
    import time as _time

    from dark_factory.metrics.helpers import record_memory_operation as _rmo

    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG

    started = _time.time()

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

    latency = _time.time() - started
    _rmo(
        operation="recall",
        count=len(memories),
        source_feature=feature_name,
        latency_seconds=latency,
    )

    # Tier A observability: count every recall call, whether it
    # hit or not, so the Memory metrics dashboard can track whether
    # the graph is dead weight or actively contributing.
    try:
        from dark_factory.metrics.prometheus import observe_memory_recall

        observe_memory_recall(memory_type="all", hit=bool(memories))
    except Exception:  # pragma: no cover — defensive
        pass

    if not memories:
        return "No relevant memories found."
    # Track recalled memory IDs for eval feedback loop
    ids = [m.get("id", "") for m in memories if m.get("id")]
    if ids:
        add_recalled_memory_ids(ids)
        # Tier A: bump times_recalled on the memory nodes so the
        # "top-10 most-used memories" dashboard has real data.
        try:
            _memory_repo.increment_recall_counts(ids)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("increment_recall_counts_failed", error=str(exc))
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


# B3 fix: shared message returned when a record_* tool is invoked
# before set_current_run_id() has run. Previously the record would
# succeed but the memory node would be tagged with ``run_id=""``,
# becoming unlinked from any Run and polluting recall results.
# Refuse the write and log a warning so the condition is observable.
_MEMORY_NO_RUN_ID_MSG = (
    "Error: cannot record memory outside an active pipeline run. "
    "The ``_current_run_id`` module global is empty, which usually "
    "means the orchestrator hasn't started this run yet or the "
    "cleanup path already cleared it. Refusing the write to avoid "
    "producing orphaned memory nodes."
)


def _memory_run_id_or_error() -> str | None:
    """Return the current run id or log + return ``None``.

    Shared guard for the record_* tools. Returning ``None`` lets
    callers distinguish "no run id → short-circuit with error
    message" from "have a run id, proceed". Logs at WARN so a single
    occurrence is visible in the structured log stream.
    """
    if not _current_run_id:
        log.warning(
            "memory_write_without_run_id",
            feature=get_current_feature() or None,
        )
        return None
    return _current_run_id


@tool
def record_pattern(description: str, context: str) -> str:
    """Record a reusable coding pattern learned during this run.
    Example: 'When generating auth modules, always include rate limiting'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    run_id = _memory_run_id_or_error()
    if run_id is None:
        return _MEMORY_NO_RUN_ID_MSG
    node_id = _memory_repo.record_pattern(
        description=description, context=context,
        source_feature=get_current_feature(), agent="coder",
        run_id=run_id,
    )
    return f"Pattern recorded: {node_id}"


@tool
def record_mistake(description: str, error_type: str, trigger_context: str) -> str:
    """Record a mistake found during review or testing.
    Example: 'Using sync I/O in async handler caused test failures'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    run_id = _memory_run_id_or_error()
    if run_id is None:
        return _MEMORY_NO_RUN_ID_MSG
    node_id = _memory_repo.record_mistake(
        description=description, error_type=error_type,
        trigger_context=trigger_context,
        source_feature=get_current_feature(), agent="reviewer",
        run_id=run_id,
    )
    return f"Mistake recorded: {node_id}"


@tool
def record_solution(description: str, mistake_id: str = "", code_snippet: str = "") -> str:
    """Record a solution. Optionally link to a mistake_id it resolves."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    run_id = _memory_run_id_or_error()
    if run_id is None:
        return _MEMORY_NO_RUN_ID_MSG
    node_id = _memory_repo.record_solution(
        description=description, mistake_id=mistake_id,
        code_snippet=code_snippet,
        source_feature=get_current_feature(), agent="reviewer",
        run_id=run_id,
    )
    return f"Solution recorded: {node_id}"


@tool
def record_strategy(description: str, applicability: str) -> str:
    """Record a planning/execution strategy.
    Example: 'For specs with >3 dependencies, query all dep contexts first'."""
    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG
    run_id = _memory_run_id_or_error()
    if run_id is None:
        return _MEMORY_NO_RUN_ID_MSG
    node_id = _memory_repo.record_strategy(
        description=description, applicability=applicability,
        source_feature=get_current_feature(), agent="planner",
        run_id=run_id,
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


@tool
def recall_episodes(
    keywords: str,
    feature_name: str = "",
    outcome: str = "any",
    limit: int = 5,
) -> str:
    """Recall past feature-trajectory episodes via hybrid semantic + keyword search.

    Call this at the START of working on a feature, BEFORE picking
    a strategy. An "episode" is the autobiographical record of a
    past run's attempt at a feature — it carries the narrative of
    what strategy was picked, what went wrong, what fixed it, and
    what the final outcome was.

    Parameters:
      - keywords: free-text query matched against past episode
        summaries (e.g. "auth JWT CSRF").
      - feature_name: optional exact feature filter. Leave empty
        for a cross-feature search.
      - outcome: 'success', 'partial', 'failed', or 'any'. Use
        'success' when you want to bias toward strategies that
        worked.
      - limit: max episodes to return (default 5).

    Use the returned episodes to bias your next decision: if a past
    run succeeded with approach X, try X first unless the current
    spec has meaningfully changed.
    """
    import time as _time

    if _memory_repo is None:
        return _MEMORY_DISABLED_MSG

    started = _time.time()

    # Neo4j keyword match — always available
    try:
        neo4j_results = _memory_repo.search_episodes_keyword(
            keywords=keywords,
            feature=feature_name or None,
            outcome=outcome if outcome and outcome != "any" else None,
            limit=20,
        )
    except Exception as exc:
        log.warning("recall_episodes_neo4j_failed", error=str(exc))
        neo4j_results = []

    # Qdrant semantic search — best-effort
    vector_results: list[dict] = []
    if _vector_repo is not None:
        try:
            query_text = keywords
            if feature_name:
                query_text = f"{feature_name} {keywords}".strip()
            vector_results = _vector_repo.search_episodes(
                query_text=query_text or feature_name or "",
                feature=feature_name or None,
                outcome=outcome if outcome and outcome != "any" else None,
                limit=20,
            )
        except Exception as exc:
            log.warning("recall_episodes_vector_failed", error=str(exc))

    # Hybrid RRF merge
    if vector_results:
        from dark_factory.vector.merge import hybrid_merge

        episodes = hybrid_merge(neo4j_results, vector_results, limit=limit)
    else:
        episodes = neo4j_results[:limit]

    latency = _time.time() - started
    log.info(
        "recall_episodes",
        keywords=keywords[:60],
        feature=feature_name,
        outcome=outcome,
        count=len(episodes),
        latency_seconds=round(latency, 3),
    )

    if not episodes:
        return "No past episodes found. This may be the first attempt at this feature."

    # Track recalled episode ids so a future feedback loop can
    # boost/demote episode relevance the same way we do for the
    # four semantic memory types.
    ids = [ep.get("id", "") for ep in episodes if ep.get("id")]
    if ids:
        add_recalled_memory_ids(ids)

    # Compact representation for LLM consumption. Expand the JSON
    # fields that the repository stores as strings so the Planner
    # sees structured data rather than raw JSON strings.
    compact: list[dict] = []
    for ep in episodes:
        item: dict = {
            "id": ep.get("id", ""),
            "feature": ep.get("feature", ""),
            "outcome": ep.get("outcome", ""),
            "summary": ep.get("summary", ""),
            "turns_used": ep.get("turns_used", 0),
            "duration_seconds": ep.get("duration_seconds", 0.0),
            "run_id": ep.get("run_id", ""),
        }
        for raw_key, out_key in (
            ("key_events_json", "key_events"),
            ("tool_calls_json", "tool_calls"),
            ("eval_scores_json", "final_eval_scores"),
        ):
            raw = ep.get(raw_key)
            if isinstance(raw, str) and raw.strip():
                try:
                    item[out_key] = json.loads(raw)
                except Exception:
                    pass
        compact.append(item)
    return json.dumps(compact, default=str)


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
EPISODE_READ_TOOLS = [recall_episodes]
VECTOR_SEARCH_TOOLS = [search_similar_specs, search_similar_code]

MEMORY_READ_TOOLS = [recall_memories, search_memory]
MEMORY_WRITE_PATTERN = [record_pattern]
MEMORY_WRITE_MISTAKE = [record_mistake, record_solution]
MEMORY_WRITE_STRATEGY = [record_strategy]

DEEP_PLANNER_TOOLS = [deep_dependency_analysis, deep_risk_assessment]
DEEP_REVIEWER_TOOLS = [deep_security_review, deep_performance_review, deep_spec_compliance_review]
DEEP_TESTER_TOOLS = [deep_unit_test_gen, deep_integration_test_gen, deep_edge_case_test_gen]


# ── Metric helpers ─────────────────────────────────────────────────────


# Map common extensions → canonical language tags for artifact_writes.language.
_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
}


def _detect_language(file_path: str) -> str | None:
    from pathlib import Path as _P

    suffix = _P(file_path).suffix.lower()
    return _LANGUAGE_MAP.get(suffix)


def _is_test_path(file_path: str) -> bool:
    from pathlib import Path as _P

    p = _P(file_path)
    name = p.name.lower()
    if name.startswith("test_") or name.endswith("_test.py") or name.endswith(".test.ts"):
        return True
    return any(part.lower() in {"test", "tests", "__tests__"} for part in p.parts)


def _metric_artifact_write(
    *,
    file_path: str,
    feature: str,
    bytes_written: int,
) -> None:
    """Best-effort write to the metrics recorder. Never raises."""
    try:
        from dark_factory.metrics.helpers import record_artifact_write

        record_artifact_write(
            file_path=file_path,
            feature=feature or None,
            language=_detect_language(file_path),
            bytes_written=bytes_written,
            is_test=_is_test_path(file_path),
        )
    except Exception:  # pragma: no cover — defensive
        pass
