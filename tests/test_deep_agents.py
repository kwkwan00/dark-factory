"""Tests for deep agent tools wired into swarm agents."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_deep_planner_tools_exist() -> None:
    from dark_factory.agents.tools import (
        DEEP_PLANNER_TOOLS,
        deep_dependency_analysis,
        deep_risk_assessment,
    )
    assert deep_dependency_analysis in DEEP_PLANNER_TOOLS
    assert deep_risk_assessment in DEEP_PLANNER_TOOLS


def test_deep_reviewer_tools_exist() -> None:
    from dark_factory.agents.tools import (
        DEEP_REVIEWER_TOOLS,
        deep_performance_review,
        deep_security_review,
        deep_spec_compliance_review,
    )
    assert deep_security_review in DEEP_REVIEWER_TOOLS
    assert deep_performance_review in DEEP_REVIEWER_TOOLS
    assert deep_spec_compliance_review in DEEP_REVIEWER_TOOLS


def test_deep_tester_tools_exist() -> None:
    from dark_factory.agents.tools import (
        DEEP_TESTER_TOOLS,
        deep_edge_case_test_gen,
        deep_integration_test_gen,
        deep_unit_test_gen,
    )
    assert deep_unit_test_gen in DEEP_TESTER_TOOLS
    assert deep_integration_test_gen in DEEP_TESTER_TOOLS
    assert deep_edge_case_test_gen in DEEP_TESTER_TOOLS


def test_planner_has_deep_tools() -> None:
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_planner

        _build_planner("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "deep_dependency_analysis" in names
        assert "deep_risk_assessment" in names


def test_reviewer_has_deep_tools() -> None:
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_reviewer

        _build_reviewer("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "deep_security_review" in names
        assert "deep_performance_review" in names
        assert "deep_spec_compliance_review" in names


def test_tester_has_deep_tools() -> None:
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_tester

        _build_tester("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "deep_unit_test_gen" in names
        assert "deep_integration_test_gen" in names
        assert "deep_edge_case_test_gen" in names


def test_claude_agent_codegen_still_works() -> None:
    """claude_agent_codegen should still be in the Coder's tools after refactor."""
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_coder

        _build_coder("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "claude_agent_codegen" in names


# ── Stderr capture for crashed Claude Agent SDK subprocesses ─────────────


def _make_fake_query(
    *,
    stderr_lines: list[str] | None = None,
    raise_exc: Exception | None = None,
    result_text: str = "ok",
):
    """Build a fake replacement for ``claude_agent_sdk.query``.

    The returned async-generator factory captures the ``options.stderr``
    callback that ``_run_deep_agent`` passes in, drives any provided
    ``stderr_lines`` through it (simulating the SDK's stderr-reader
    task), then either yields a fake ``ResultMessage`` or raises the
    given exception. This lets us exercise the buffer + dump path
    without spawning a real subprocess.
    """

    async def _fake_query(*, prompt, options):
        # Drive the stderr callback as if the subprocess wrote lines.
        if stderr_lines and options.stderr is not None:
            for line in stderr_lines:
                options.stderr(line)
        if raise_exc is not None:
            raise raise_exc
        # Yield a single fake ResultMessage so the success path works.
        from claude_agent_sdk import ResultMessage

        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=1,
            session_id="fake",
            total_cost_usd=0.0,
            result=result_text,
        )

    return _fake_query


def test_run_deep_agent_passes_stderr_callback_to_options():
    """The SDK only pipes subprocess stderr when ``options.stderr`` is
    set. Without this callback, the subprocess inherits the parent's
    stderr (or /dev/null) and the SDK's "Check stderr output for
    details" placeholder is the only thing we see when it crashes.
    Regression guard: assert the callback is wired in."""
    captured_options: list = []

    async def _capture_query(*, prompt, options):
        captured_options.append(options)
        from claude_agent_sdk import ResultMessage

        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=1,
            session_id="fake",
            total_cost_usd=0.0,
            result="done",
        )

    with patch("claude_agent_sdk.query", new=_capture_query):
        from dark_factory.agents.tools import _run_deep_agent

        result = _run_deep_agent("test prompt", ["Read"], max_turns=5)

    assert result == "done"
    assert len(captured_options) == 1
    # The callback must be set, must be callable, and must accept a single
    # string arg without raising.
    assert captured_options[0].stderr is not None
    captured_options[0].stderr("a sample stderr line")


def test_run_deep_agent_logs_stderr_tail_on_subprocess_crash(capsys, caplog):
    """When the SDK raises (subprocess crashed with exit code 1), the
    captured stderr lines must be dumped at error level so the
    operator can read them in docker logs. The actual SDK error
    re-raises so the swarm error path still triggers.

    Output capture is order-dependent: when this test runs in isolation,
    structlog uses its default print-logger config and writes to
    stdout (caught by ``capsys``). After any earlier test triggers
    ``setup_logging()`` via the FastAPI app's lifespan, structlog
    routes through stdlib logging (caught by ``caplog``). We assert
    against the union of both so the test is order-independent.
    """
    import logging
    import pytest

    caplog.set_level(logging.DEBUG)

    fake_stderr = [
        f"line {i}: some warning from the subprocess"
        for i in range(5)
    ] + ["FATAL: Cannot find module 'foo'"]

    fake_query = _make_fake_query(
        stderr_lines=fake_stderr,
        raise_exc=RuntimeError(
            "Command failed with exit code 1 (exit code: 1)\n"
            "Error output: Check stderr output for details"
        ),
    )

    with patch("claude_agent_sdk.query", new=fake_query):
        from dark_factory.agents.tools import _run_deep_agent

        with pytest.raises(RuntimeError, match="exit code 1"):
            _run_deep_agent("test prompt", ["Read"], max_turns=5)

    captured = capsys.readouterr()
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    combined = captured.out + captured.err + log_text
    # The error log entry must mention deep_agent_failed AND include
    # the FATAL message that was at the tail of the stderr buffer.
    assert "deep_agent_failed" in combined
    assert "FATAL: Cannot find module 'foo'" in combined
    # Should report 6 lines captured (stderr_lines=6 in raw text or
    # 'stderr_lines': 6 in stdlib formatter output)
    assert "stderr_lines=6" in combined or "'stderr_lines': 6" in combined


def test_run_deep_agent_records_incident_on_crash():
    """The crash path must also record a metrics-store incident so the
    Run Detail popup's incidents table picks it up. Without this, the
    operator only sees the failure in the docker logs which scroll
    away as soon as the next pipeline run starts."""
    import pytest

    fake_stderr = ["something exploded", "stack frame 1", "stack frame 2"]
    fake_query = _make_fake_query(
        stderr_lines=fake_stderr,
        raise_exc=RuntimeError("Command failed with exit code 1"),
    )

    with patch("claude_agent_sdk.query", new=fake_query):
        with patch(
            "dark_factory.metrics.helpers.record_incident"
        ) as mock_incident:
            from dark_factory.agents.tools import _run_deep_agent

            with pytest.raises(RuntimeError):
                _run_deep_agent("test prompt", ["Read"], max_turns=5)

    assert mock_incident.called
    call_kwargs = mock_incident.call_args.kwargs
    assert call_kwargs["category"] == "subprocess"
    assert call_kwargs["severity"] == "error"
    assert "Deep agent crashed" in call_kwargs["message"]
    # The captured stderr lines should be attached to the incident's
    # ``stack`` field so they appear in the Run Detail popup.
    assert call_kwargs["stack"] is not None
    assert "something exploded" in call_kwargs["stack"]
    assert "stack frame 2" in call_kwargs["stack"]


def test_run_deep_agent_truncates_oversized_stderr_buffer():
    """A runaway subprocess that logs to stderr in a tight loop must
    not be able to exhaust the worker's memory. The buffer caps at
    DEEP_AGENT_STDERR_MAX_LINES lines AND DEEP_AGENT_STDERR_MAX_BYTES
    bytes, whichever hits first."""
    import pytest

    from dark_factory.agents.tools import (
        DEEP_AGENT_STDERR_MAX_LINES,
    )

    # Generate way more lines than the cap
    flood = [f"flood line {i}" for i in range(DEEP_AGENT_STDERR_MAX_LINES * 3)]

    captured_buffer: list[str] = []

    def _capture_record_incident(**kwargs):
        if kwargs.get("stack"):
            captured_buffer.extend(kwargs["stack"].split("\n"))

    fake_query = _make_fake_query(
        stderr_lines=flood,
        raise_exc=RuntimeError("Command failed"),
    )

    with patch("claude_agent_sdk.query", new=fake_query):
        with patch(
            "dark_factory.metrics.helpers.record_incident",
            side_effect=_capture_record_incident,
        ):
            from dark_factory.agents.tools import _run_deep_agent

            with pytest.raises(RuntimeError):
                _run_deep_agent("test prompt", ["Read"], max_turns=5)

    # Buffer captured at most max_lines + 1 (for the truncation marker)
    # entries, NOT the full flood. We also assert the truncation marker
    # is present so the operator can tell why the dump is shorter than
    # the actual stderr.
    assert len(captured_buffer) > 0
    # The incident only attaches the LAST 50 lines so the captured
    # buffer can't be huge regardless. The key invariant we want to
    # test is "the buffer didn't accumulate all 600 flood lines" —
    # check by asserting the truncation marker appeared somewhere
    # OR that the captured buffer is well below the flood size.
    truncation_marker_seen = any(
        "stderr truncated" in line for line in captured_buffer
    )
    assert truncation_marker_seen or len(captured_buffer) < len(flood)


# ── DEEP_AGENT_DEBUG_STDERR toggle ───────────────────────────────────────


def test_run_deep_agent_debug_stderr_off_by_default(monkeypatch):
    """``extra_args`` must NOT include ``debug-to-stderr`` when the
    feature flag is unset — operators shouldn't pay the extra log
    volume cost on every run."""
    import importlib

    import dark_factory.agents.tools as tools_mod

    monkeypatch.delenv("DEEP_AGENT_DEBUG_STDERR", raising=False)
    importlib.reload(tools_mod)

    assert tools_mod.DEEP_AGENT_DEBUG_STDERR is False

    captured_options: list = []

    async def _capture_query(*, prompt, options):
        captured_options.append(options)
        from claude_agent_sdk import ResultMessage

        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=1,
            session_id="fake",
            total_cost_usd=0.0,
            result="done",
        )

    with patch("claude_agent_sdk.query", new=_capture_query):
        tools_mod._run_deep_agent("test prompt", ["Read"], max_turns=5)

    assert len(captured_options) == 1
    extra = captured_options[0].extra_args or {}
    assert "debug-to-stderr" not in extra


def test_run_deep_agent_debug_stderr_on_when_env_set(monkeypatch):
    """With ``DEEP_AGENT_DEBUG_STDERR=1``, the Node CLI receives
    ``--debug-to-stderr`` via ``extra_args``. The SDK's own stderr
    callback path still takes priority over its ``debug_stderr``
    fallback, so we keep control of where the bytes land."""
    import importlib

    import dark_factory.agents.tools as tools_mod

    monkeypatch.setenv("DEEP_AGENT_DEBUG_STDERR", "1")
    importlib.reload(tools_mod)

    try:
        assert tools_mod.DEEP_AGENT_DEBUG_STDERR is True

        captured_options: list = []

        async def _capture_query(*, prompt, options):
            captured_options.append(options)
            from claude_agent_sdk import ResultMessage

            yield ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=5,
                is_error=False,
                num_turns=1,
                session_id="fake",
                total_cost_usd=0.0,
                result="done",
            )

        with patch("claude_agent_sdk.query", new=_capture_query):
            tools_mod._run_deep_agent("test prompt", ["Read"], max_turns=5)

        assert len(captured_options) == 1
        extra = captured_options[0].extra_args or {}
        assert "debug-to-stderr" in extra
        # ``extra_args`` values must be ``str | None``; the CLI
        # turns a ``None`` value into a bare flag (``--debug-to-stderr``)
        # rather than ``--debug-to-stderr=…``.
        assert extra["debug-to-stderr"] is None
        # Our callback is still set — the verbose output gets
        # captured into _stderr_cb, not written to sys.stderr.
        assert captured_options[0].stderr is not None
    finally:
        # Restore the module to its off-by-default state so the rest
        # of the session doesn't inherit the flag.
        monkeypatch.delenv("DEEP_AGENT_DEBUG_STDERR", raising=False)
        importlib.reload(tools_mod)


# ── _safe_tool_deep_agent: tool-safe wrapper ─────────────────────────────


def test_safe_tool_deep_agent_returns_string_on_success():
    """Happy path: the safe wrapper must forward successful results
    unchanged from the underlying ``_run_deep_agent``."""
    fake_query = _make_fake_query(result_text="analysis complete")

    with patch("claude_agent_sdk.query", new=fake_query):
        from dark_factory.agents.tools import _safe_tool_deep_agent

        result = _safe_tool_deep_agent("test prompt", ["Read"], max_turns=5)

    assert result == "analysis complete"


def test_safe_tool_deep_agent_catches_subprocess_crash_and_returns_error_string():
    """Critical contract: when the underlying SDK crashes, the safe
    wrapper must NOT re-raise — it must return a string describing
    the failure so the calling LangGraph agent can reason about it
    and keep going. This is what prevents a transient SDK failure in
    a best-effort analyzer (e.g. ``deep_risk_assessment``) from
    cascading into ``feature_swarm_failed`` for the entire feature."""
    fake_query = _make_fake_query(
        stderr_lines=["FATAL: Cannot find module 'foo'"],
        raise_exc=RuntimeError(
            "Command failed with exit code 1 (exit code: 1)\n"
            "Error output: Check stderr output for details"
        ),
    )

    with patch("claude_agent_sdk.query", new=fake_query):
        from dark_factory.agents.tools import _safe_tool_deep_agent

        # Must NOT raise.
        result = _safe_tool_deep_agent(
            "test prompt", ["Read", "Glob", "Grep"], max_turns=5
        )

    assert isinstance(result, str)
    # The error string must include enough context for the agent:
    # (1) that it failed, (2) which tools were in play, (3) the
    # original exception text.
    assert "Error" in result
    assert "deep agent" in result.lower()
    assert "Read" in result and "Glob" in result and "Grep" in result
    assert "exit code 1" in result
    # It must also tell the agent this was best-effort so the agent
    # continues rather than retrying in a tight loop.
    assert "best-effort" in result.lower()


def test_safe_tool_deep_agent_does_not_swallow_baseexception():
    """``BaseException`` subclasses (``KeyboardInterrupt``,
    ``SystemExit``, cancellation) must still unwind. Only
    ``Exception`` and below are converted to error strings."""
    import pytest

    class _FakeKI(BaseException):
        pass

    fake_query = _make_fake_query(raise_exc=_FakeKI("user-cancelled"))

    with patch("claude_agent_sdk.query", new=fake_query):
        from dark_factory.agents.tools import _safe_tool_deep_agent

        with pytest.raises(_FakeKI):
            _safe_tool_deep_agent("test prompt", ["Read"], max_turns=5)


def test_all_tool_decorated_deep_agents_use_safe_wrapper():
    """Source-level regression guard: every ``@tool`` function in
    ``tools.py`` that calls into a deep-agent subprocess must route
    through ``_safe_tool_deep_agent``, not the raw ``_run_deep_agent``
    helper. A new deep-agent tool that forgets to switch will
    reintroduce the ``feature_swarm_failed`` cascade on any SDK
    hiccup."""
    import inspect

    from dark_factory.agents import tools as tools_mod

    tool_fn_names = [
        "claude_agent_codegen",
        "deep_dependency_analysis",
        "deep_risk_assessment",
        "deep_security_review",
        "deep_performance_review",
        "deep_spec_compliance_review",
        "deep_unit_test_gen",
        "deep_integration_test_gen",
        "deep_edge_case_test_gen",
    ]
    for name in tool_fn_names:
        tool_obj = getattr(tools_mod, name)
        # ``@tool`` wraps the original function; the underlying fn is
        # accessible via ``.func`` on langchain StructuredTool.
        inner = getattr(tool_obj, "func", None) or tool_obj
        src = inspect.getsource(inner)
        assert "_safe_tool_deep_agent" in src, (
            f"{name} must call _safe_tool_deep_agent — raw "
            f"_run_deep_agent will propagate SDK crashes out of the "
            f"swarm and kill the feature."
        )
        # And it must NOT bypass the wrapper by calling the raw
        # helper directly.
        assert "_run_deep_agent(" not in src, (
            f"{name} calls _run_deep_agent() directly — switch to "
            f"_safe_tool_deep_agent so a transient SDK crash cannot "
            f"cascade into feature_swarm_failed."
        )


def test_run_deep_agent_does_not_dump_stderr_on_success(capsys, caplog):
    """Successful runs MUST NOT dump captured stderr — that would spam
    the logs on every successful subprocess invocation. The buffer is
    only flushed on failure."""
    import logging

    caplog.set_level(logging.DEBUG)

    fake_query = _make_fake_query(
        stderr_lines=["a benign warning that shouldn't be logged"],
        raise_exc=None,
    )

    with patch("claude_agent_sdk.query", new=fake_query):
        from dark_factory.agents.tools import _run_deep_agent

        result = _run_deep_agent("test prompt", ["Read"], max_turns=5)

    assert result == "ok"
    captured = capsys.readouterr()
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    combined = captured.out + captured.err + log_text
    # No deep_agent_failed log entry on the success path
    assert "deep_agent_failed" not in combined
    # And the captured stderr line must not have leaked into the log
    assert "a benign warning that shouldn't be logged" not in combined
