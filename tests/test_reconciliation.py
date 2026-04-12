"""Tests for the Phase 5 reconciliation stage.

Covers:
- Skip guards (disabled / no features / all-errored / missing output dir)
- Happy path: _run_deep_agent called with expected prompt + tools
- Status classification from agent output (clean / partial / error)
- Report file detection
- Exception tolerance (best-effort policy: never raise)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dark_factory.stages.reconciliation import (
    ReconciliationResult,
    ReconciliationStage,
)


def _feature(name: str = "auth", status: str = "success") -> dict:
    """Minimal FeatureResult-shaped dict the stage knows how to read."""
    return {"feature": name, "status": status}


# ── Skip guards ────────────────────────────────────────────────────────────
#
# The reconciliation phase always runs when the feature swarms
# complete — there's no enable toggle. Only the degenerate "nothing
# to reconcile" cases below cause the stage to short-circuit without
# invoking the deep agent.


def test_reconciliation_skips_when_no_features(tmp_path: Path):
    stage = ReconciliationStage()
    result = stage.run(
        run_id="run-test",
        output_dir=tmp_path,
        feature_results=[],
    )
    assert result.status == "skipped"
    assert "no features" in result.summary.lower()


def test_reconciliation_skips_when_all_features_errored(tmp_path: Path):
    """If every feature swarm errored, there's no useful output to
    reconcile. Skip rather than waste a deep-agent invocation on an
    empty or garbage directory."""
    stage = ReconciliationStage()
    result = stage.run(
        run_id="run-test",
        output_dir=tmp_path,
        feature_results=[
            _feature("a", "error"),
            _feature("b", "error"),
            _feature("c", "skipped"),
        ],
    )
    assert result.status == "skipped"
    assert "errored" in result.summary.lower()


def test_reconciliation_skips_when_output_dir_missing(tmp_path: Path):
    """A missing output dir means the orchestrator didn't actually
    write anything — skip gracefully instead of crashing when
    _run_deep_agent tries to cd into a non-existent path."""
    stage = ReconciliationStage()
    missing_dir = tmp_path / "does-not-exist"
    result = stage.run(
        run_id="run-test",
        output_dir=missing_dir,
        feature_results=[_feature()],
    )
    assert result.status == "skipped"
    assert "missing" in result.summary.lower()


# ── Happy path ─────────────────────────────────────────────────────────────


def test_reconciliation_calls_deep_agent_with_expected_tools(tmp_path: Path):
    """The stage must invoke ``_run_deep_agent`` with the full Read/
    Write/Edit/Glob/Grep/Bash tool set and the configured max_turns."""
    # Create the output dir so the stage doesn't skip it
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "main.py").write_text("print('hi')\n")

    captured_kwargs: list[dict] = []

    def _fake_deep_agent(
        prompt: str,
        allowed_tools: list[str],
        max_turns: int = 15,
        timeout_seconds: float | None = None,
    ):
        captured_kwargs.append(
            {
                "prompt": prompt,
                "allowed_tools": allowed_tools,
                "max_turns": max_turns,
                "timeout_seconds": timeout_seconds,
            }
        )
        return "Overall status: clean\nAll good."

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=_fake_deep_agent,
    ):
        stage = ReconciliationStage(max_turns=77, timeout_seconds=999)
        result = stage.run(
            run_id="run-xyz",
            output_dir=tmp_path,
            feature_results=[_feature("auth", "success")],
        )

    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    # Tools cover read + write + shell — full scope
    assert set(kwargs["allowed_tools"]) == {
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Bash",
    }
    # max_turns respects the stage config
    assert kwargs["max_turns"] == 77
    # timeout_seconds is plumbed through from the stage config — a
    # regression guard against the C1 bug where the stage stored the
    # value but never passed it to _run_deep_agent, leaving a 100-turn
    # reconciliation pass bound to the 600s default instead of 1800s.
    assert kwargs["timeout_seconds"] == 999.0
    # Prompt includes the run id and the feature list
    assert "run-xyz" in kwargs["prompt"]
    assert "auth" in kwargs["prompt"]
    assert result.status == "clean"


def test_reconciliation_classifies_clean_status_from_output(tmp_path: Path):
    (tmp_path / "f1").mkdir()

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="... final summary ... Overall status: clean",
    ):
        stage = ReconciliationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
        )

    assert result.status == "clean"
    assert "clean" in result.summary.lower()


def test_reconciliation_classifies_partial_from_ambiguous_output(
    tmp_path: Path,
):
    """When the agent output doesn't have a clear ``Overall status``
    line, default to ``partial`` — it's the safest conservative
    classification (the operator can read the agent output to decide
    if it's actually fine)."""
    (tmp_path / "f1").mkdir()

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="I did some stuff but forgot to write a status line.",
    ):
        stage = ReconciliationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
        )

    assert result.status == "partial"


def test_reconciliation_classifies_broken_as_partial(tmp_path: Path):
    """An ``Overall status: broken`` self-reported outcome maps to
    ``partial`` on our side — the run still produced SOMETHING, just
    with significant remaining issues."""
    (tmp_path / "f1").mkdir()

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="Lots of issues. Overall status: broken",
    ):
        stage = ReconciliationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
        )

    assert result.status == "partial"
    assert "remaining issues" in result.summary.lower()


def test_reconciliation_detects_report_file(tmp_path: Path):
    """When the agent writes RECONCILIATION_REPORT.md to the output
    dir, the stage picks it up and surfaces the relative path. Also
    reads the report file to classify status when the agent_output
    itself didn't include a status line."""
    (tmp_path / "f1").mkdir()
    (tmp_path / "RECONCILIATION_REPORT.md").write_text(
        "# Report\n\n## Overall status: clean\n\nEverything works.\n"
    )

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="wrote the report",  # no status line in agent_output
    ):
        stage = ReconciliationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
        )

    assert result.report_path == "RECONCILIATION_REPORT.md"
    # Status classification fell back to reading the report file.
    assert result.status == "clean"


# ── Error tolerance ───────────────────────────────────────────────────────


def test_reconciliation_returns_error_status_when_deep_agent_crashes(
    tmp_path: Path,
):
    """A crash in _run_deep_agent must NOT raise out of the stage —
    the pipeline treats reconciliation as best-effort and only a
    clean return keeps the run going. The stage surfaces the error
    in the ReconciliationResult.summary so the Run Detail popup can
    show it."""
    (tmp_path / "f1").mkdir()

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=RuntimeError("SDK exploded"),
    ):
        stage = ReconciliationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
        )

    # Crashes are swallowed into an error-status result
    assert result.status == "error"
    assert "exploded" in result.summary.lower()
    assert result.agent_output == ""


def test_reconciliation_agent_output_is_truncated_to_8kb(tmp_path: Path):
    """Very long agent outputs are capped at 8 KiB so they fit in the
    metrics-store incident row without ballooning Postgres. The stage
    silently clips — the operator gets the start of the output."""
    (tmp_path / "f1").mkdir()

    huge = "x" * 20000  # 20 KiB
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value=huge,
    ):
        stage = ReconciliationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
        )

    assert len(result.agent_output) <= 8000


def test_reconciliation_result_model_roundtrips_to_dict():
    """``model_dump()`` must produce a JSON-safe dict that the
    ag_ui_bridge can attach to the StateSnapshot payload. Regression
    guard against accidentally adding non-serialisable fields."""
    import json as _json

    result = ReconciliationResult(
        status="clean",
        summary="ok",
        agent_output="done",
        report_path="RECONCILIATION_REPORT.md",
        duration_seconds=42.5,
    )
    payload = result.model_dump()
    # Must be JSON-serialisable
    serialised = _json.dumps(payload)
    loaded = _json.loads(serialised)
    assert loaded["status"] == "clean"
    assert loaded["duration_seconds"] == pytest.approx(42.5)
