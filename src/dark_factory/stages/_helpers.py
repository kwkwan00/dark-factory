"""Shared helpers for pipeline stages that invoke the Claude Agent SDK."""

from __future__ import annotations

from pathlib import Path

from dark_factory.agents.cancellation import PipelineCancelled


def run_deep_agent_in_dir(
    output_dir: Path,
    prompt: str,
    allowed_tools: list[str],
    timeout_seconds: float,
    max_turns: int = 15,
) -> str:
    """Run ``_run_deep_agent`` with ``_output_dir`` temporarily set to *output_dir*.

    Swaps the module-level ``_output_dir`` in :mod:`dark_factory.agents.tools`
    so the deep-agent subprocess's ``cwd`` resolves to the given directory,
    then restores the previous value in a ``finally`` block. Properly
    re-raises :class:`PipelineCancelled` so cooperative cancellation
    propagates through best-effort stages.

    Returns the agent's final result text.
    """
    from dark_factory.agents import tools as _tools_mod

    previous_output = _tools_mod._output_dir
    _tools_mod._output_dir = output_dir

    try:
        return _tools_mod._run_deep_agent(
            prompt=prompt,
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
        )
    except PipelineCancelled:
        raise
    except Exception:
        raise
    finally:
        _tools_mod._output_dir = previous_output
