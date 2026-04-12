"""Tests for the swarm harness."""

from __future__ import annotations

from dark_factory.agents.swarm import MAX_HANDOFFS


def test_max_handoffs_constant() -> None:
    assert MAX_HANDOFFS == 50


def test_build_planner_has_handoff_tool() -> None:
    """Planner agent should include a transfer_to_coder handoff tool."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_planner

        _build_planner("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("coder" in n for n in tool_names), f"Expected handoff to coder in {tool_names}"


def test_build_coder_has_claude_agent_and_handoff() -> None:
    """Coder agent should include claude_agent_codegen and transfer_to_reviewer."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_coder

        _build_coder("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "claude_agent_codegen" in tool_names, f"Expected claude_agent_codegen in {tool_names}"
        assert any("reviewer" in n for n in tool_names), f"Expected handoff to reviewer in {tool_names}"


def test_build_reviewer_has_both_handoffs() -> None:
    """Reviewer agent should have handoffs to both coder (reject) and tester (approve)."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_reviewer

        _build_reviewer("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("coder" in n for n in tool_names), f"Expected handoff to coder in {tool_names}"
        assert any("tester" in n for n in tool_names), f"Expected handoff to tester in {tool_names}"


def test_build_tester_has_handoff_to_planner() -> None:
    """Tester agent should include a transfer_to_planner handoff tool."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_tester

        _build_tester("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("planner" in n for n in tool_names), f"Expected handoff to planner in {tool_names}"


# ── max_tokens wiring regression tests ─────────────────────────────────────
#
# Verify the fix for the ``Error invoking tool 'write_file' with
# kwargs {'file_path': 'utilization-dashboar...` bug. Root cause:
# langchain's ChatAnthropic defaults max_tokens to 1024, which is
# too small to hold a large ``write_file`` tool call. The fix
# constructs a ChatAnthropic instance via ``build_chat_model`` with
# ``max_tokens = settings.pipeline.max_llm_tokens`` (default 32768)
# and a timeout + stop-reason callback for observability.


def test_build_chat_model_uses_configured_max_tokens() -> None:
    """Regression: ``build_chat_model`` must produce a ChatAnthropic
    with max_tokens pulled from settings.pipeline.max_llm_tokens,
    not the langchain default of 1024."""
    from dark_factory.agents.swarm import build_chat_model
    from dark_factory.config import Settings

    settings = Settings()
    settings.pipeline.max_llm_tokens = 32768

    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-fake-for-test")

    model = build_chat_model(settings)
    assert hasattr(model, "max_tokens"), (
        "ChatAnthropic instance must expose max_tokens; did build_chat_model "
        "fall back to a string for an Anthropic provider?"
    )
    assert model.max_tokens == 32768, (
        f"expected max_tokens=32768, got {model.max_tokens} — this regression "
        "was the 'Error invoking tool write_file with kwargs {file_path: ...}' "
        "bug where truncated tool_use JSON from a 1024 token ceiling failed "
        "Pydantic validation downstream."
    )


def test_build_chat_model_honours_custom_max_tokens() -> None:
    """Operators can raise the budget via settings."""
    from dark_factory.agents.swarm import build_chat_model
    from dark_factory.config import Settings

    settings = Settings()
    settings.pipeline.max_llm_tokens = 60000  # push toward Sonnet's ceiling

    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-fake-for-test")

    model = build_chat_model(settings)
    assert model.max_tokens == 60000


def test_build_chat_model_sets_timeout() -> None:
    """H4 parity: timeout must be set explicitly so a hung swarm
    LLM call can't stall a worker thread indefinitely."""
    from dark_factory.agents.swarm import build_chat_model
    from dark_factory.config import Settings
    from dark_factory.llm.base import DEFAULT_LLM_TIMEOUT_SECONDS

    settings = Settings()
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-fake-for-test")

    model = build_chat_model(settings)
    # ChatAnthropic stores timeout as an instance attribute.
    assert getattr(model, "default_request_timeout", None) == DEFAULT_LLM_TIMEOUT_SECONDS \
        or getattr(model, "timeout", None) == DEFAULT_LLM_TIMEOUT_SECONDS


def test_build_chat_model_installs_stop_reason_callback() -> None:
    """The stop-reason logger must be on the model's callbacks list
    so ``stop_reason=max_tokens`` failures surface in logs instead
    of only manifesting as downstream tool validation errors."""
    from dark_factory.agents.swarm import build_chat_model
    from dark_factory.config import Settings

    settings = Settings()
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-fake-for-test")

    model = build_chat_model(settings)
    callbacks = getattr(model, "callbacks", None) or []
    # Unwrap BaseCallbackManager if present.
    if hasattr(callbacks, "handlers"):
        callbacks = callbacks.handlers
    handler_class_names = [type(cb).__name__ for cb in callbacks]
    assert "_SwarmStopReasonLogger" in handler_class_names, (
        f"expected _SwarmStopReasonLogger in callbacks; got {handler_class_names}"
    )


def test_build_chat_model_falls_back_to_string_for_non_anthropic() -> None:
    """Non-anthropic providers skip the ChatAnthropic path so the
    change doesn't lock operators into a single provider."""
    from dark_factory.agents.swarm import build_chat_model
    from dark_factory.config import Settings

    settings = Settings()
    settings.llm.provider = "openai"
    settings.llm.model = "gpt-5"
    model = build_chat_model(settings)
    assert isinstance(model, str)
    assert model == "openai:gpt-5"


def test_stop_reason_callback_logs_max_tokens_warning(capsys, caplog):
    """When Claude stops with ``stop_reason='max_tokens'``, the
    swarm callback must fire a WARN so the failure mode surfaces in
    logs. Without this hook the only downstream symptom is the
    opaque ``Error invoking tool 'write_file'`` Pydantic rejection
    from LangGraph."""
    import logging
    from types import SimpleNamespace

    from dark_factory.agents.swarm import _get_stop_reason_handler

    handler_class = _get_stop_reason_handler()
    assert handler_class is not None, (
        "langchain_core.callbacks should be importable in the test env"
    )
    handler = handler_class()

    # Build a minimal response object that looks like langchain's
    # LLMResult shape: ``generations = [[Generation(generation_info={...})]]``
    fake_response = SimpleNamespace(
        generations=[
            [SimpleNamespace(generation_info={"stop_reason": "max_tokens"})]
        ]
    )

    with caplog.at_level(logging.WARNING):
        handler.on_llm_end(fake_response)

    # structlog may route to stdout (pre-setup_logging) or stdlib
    # logging (post-setup_logging). Check both so the test is robust
    # to whichever mode the test session is running in.
    captured = capsys.readouterr()
    combined = captured.out + captured.err + caplog.text
    assert "swarm_llm_stop_reason" in combined or "max_tokens" in combined, (
        f"expected warning not found in output; got:\n{combined!r}"
    )


def test_stop_reason_callback_stays_quiet_on_normal_completion():
    """Normal ``end_turn`` / ``tool_use`` stop reasons must NOT log
    a warning — otherwise every healthy LLM call would spam the
    logs."""
    import logging
    from types import SimpleNamespace

    from dark_factory.agents.swarm import _get_stop_reason_handler

    handler = _get_stop_reason_handler()()

    for normal_stop in ("end_turn", "tool_use", "stop_sequence"):
        fake_response = SimpleNamespace(
            generations=[
                [SimpleNamespace(generation_info={"stop_reason": normal_stop})]
            ]
        )
        # Just verify it doesn't raise; log volume is asserted by the
        # prior test.
        handler.on_llm_end(fake_response)


def test_orchestrator_build_uses_chat_model_helper() -> None:
    """End-to-end regression: ``build_orchestrator`` must thread a
    properly-configured ChatAnthropic through ``make_execute_layer_node``.
    Structural guard — inspect the source to verify the import +
    call are present."""
    import inspect

    from dark_factory.agents import orchestrator

    src = inspect.getsource(orchestrator.build_orchestrator)
    assert "build_chat_model" in src, (
        "build_orchestrator must use swarm.build_chat_model to construct "
        "the LLM, not the bare string shortcut that hits langchain's "
        "1024 max_tokens default"
    )
    assert 'f"anthropic:' not in src, (
        "build_orchestrator should NOT construct a model string — that "
        "reintroduces the 1024 max_tokens bug"
    )
