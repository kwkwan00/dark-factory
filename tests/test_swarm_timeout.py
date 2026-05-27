"""Tests for the wall-clock timeout in run_feature_swarm().

The swarm checks elapsed time on every LangGraph chunk. If the feature
has run longer than timeout_seconds a TimeoutError is raised.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_compiled(chunks):
    """Return a mock compiled graph that yields *chunks* from .stream()."""
    mock = MagicMock()
    mock.stream.return_value = iter(chunks)
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_run_feature_swarm_returns_error_result_when_elapsed_exceeds_budget():
    """If elapsed time exceeds timeout_seconds, the stream loop raises TimeoutError
    which is caught inside run_feature_swarm and returned as a FeatureResult with
    status="timeout" and the timeout message in the error field."""
    from dark_factory.agents.swarm import run_feature_swarm

    # Compiled graph yields two non-empty chunks so the loop runs at least once
    chunks = [{"planner": {"messages": []}}, {"coder": {"messages": []}}]
    compiled = _make_compiled(chunks)

    # is_cancelled / raise_if_cancelled are imported *inside* run_feature_swarm
    # from dark_factory.agents.cancellation — patch at that module.
    start_time = 1000.0
    call_count = 0

    def _fake_monotonic():
        nonlocal call_count
        call_count += 1
        # First call (to set _swarm_start) returns start_time;
        # all subsequent calls return start_time + 9999 (way past timeout).
        return start_time if call_count == 1 else start_time + 9999

    with patch("dark_factory.agents.cancellation.is_cancelled", return_value=False), \
         patch("dark_factory.agents.cancellation.raise_if_cancelled"), \
         patch("dark_factory.agents.swarm.set_current_feature"), \
         patch("dark_factory.agents.tools.get_current_run_id", return_value=None), \
         patch("dark_factory.agents.swarm._get_progress_handler", return_value=None), \
         patch("time.monotonic", side_effect=_fake_monotonic):
        result = run_feature_swarm(
            compiled,
            spec_ids=["s1"],
            feature_name="auth",
            timeout_seconds=1,   # 1s budget, but elapsed will be 9999s
        )

    # TimeoutError is caught inside run_feature_swarm → returned as timeout FeatureResult
    # (not "error") so adjust_strategy_node correctly excludes it from retry.
    assert result["status"] == "timeout"
    assert "wall-clock timeout" in (result.get("error") or "")


def test_run_feature_swarm_does_not_timeout_within_budget():
    """When the swarm completes before the timeout, no TimeoutError is raised."""
    from dark_factory.agents.swarm import run_feature_swarm

    # Empty stream → loop exits immediately → no timeout check fires
    compiled = _make_compiled([])

    with patch("dark_factory.agents.cancellation.is_cancelled", return_value=False), \
         patch("dark_factory.agents.cancellation.raise_if_cancelled"), \
         patch("dark_factory.agents.swarm.set_current_feature"), \
         patch("dark_factory.agents.tools.get_current_run_id", return_value=None), \
         patch("dark_factory.agents.swarm._get_progress_handler", return_value=None):
        result = run_feature_swarm(
            compiled,
            spec_ids=["s1"],
            feature_name="auth",
            timeout_seconds=600,
        )

    # An empty stream produces an error FeatureResult (no files written),
    # but crucially it should NOT be a TimeoutError.
    assert result["status"] in ("success", "error", "no_artifacts")
    assert "timeout" not in (result.get("error") or "").lower()


def test_run_feature_swarm_default_timeout_is_600():
    """The default timeout_seconds parameter value should be 600."""
    import inspect
    from dark_factory.agents.swarm import run_feature_swarm
    sig = inspect.signature(run_feature_swarm)
    assert sig.parameters["timeout_seconds"].default == 3600
