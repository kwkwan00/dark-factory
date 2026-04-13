"""Tests for the Phase 6 E2E validation stage.

Covers:
- Skip guards (reconciliation errored / no features / missing output dir)
- Happy path: _run_deep_agent called with expected prompt + tools
- Browser matrix plumbing (default all 3 / custom subset / invalid filtered)
- Status classification from agent output + E2E_REPORT.md
- Report file + HTML report + screenshot discovery
- Test count parsing from report text
- Exception tolerance (best-effort policy: never raise)
- Output truncation to 8 KiB
- Model round-trip to dict for StateSnapshot serialisation
"""

from __future__ import annotations



from pathlib import Path
from unittest.mock import patch

import pytest

from dark_factory.stages.e2e_validation import (
    E2EValidationResult,
    E2EValidationStage,
    _parse_browsers_run,
    _parse_test_counts,
)


def _feature(name: str = "auth", status: str = "success") -> dict:
    return {"feature": name, "status": status}


# ── Skip guards ────────────────────────────────────────────────────────────


def test_e2e_skips_when_reconciliation_errored(tmp_path: Path):
    """If Phase 5 errored, there's no point starting a browser against
    code that couldn't even be polished. Skip cleanly."""
    (tmp_path / "auth").mkdir()
    stage = E2EValidationStage()
    result = stage.run(
        run_id="run-x",
        output_dir=tmp_path,
        feature_results=[_feature()],
        reconciliation_status="error",
    )
    assert result.status == "skipped"
    assert "reconciliation errored" in result.summary.lower()


def test_e2e_skips_when_no_features(tmp_path: Path):
    stage = E2EValidationStage()
    result = stage.run(
        run_id="run-x",
        output_dir=tmp_path,
        feature_results=[],
        reconciliation_status="clean",
    )
    assert result.status == "skipped"
    assert "no features" in result.summary.lower()


def test_e2e_skips_when_output_dir_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    stage = E2EValidationStage()
    result = stage.run(
        run_id="run-x",
        output_dir=missing,
        feature_results=[_feature()],
        reconciliation_status="clean",
    )
    assert result.status == "skipped"
    assert "missing" in result.summary.lower()


# ── Happy path ─────────────────────────────────────────────────────────────


def test_e2e_calls_deep_agent_with_expected_tools_and_browsers(tmp_path: Path):
    """The stage must invoke ``_run_deep_agent`` with the full Read/
    Write/Edit/Glob/Grep/Bash tool set, the configured max_turns and
    timeout, and a prompt that mentions every browser in the matrix."""
    (tmp_path / "app").mkdir()
    captured: list[dict] = []

    def _fake_deep_agent(
        prompt: str,
        allowed_tools: list[str],
        max_turns: int = 15,
        timeout_seconds: float | None = None,
    ) -> str:
        captured.append(
            {
                "prompt": prompt,
                "allowed_tools": allowed_tools,
                "max_turns": max_turns,
                "timeout_seconds": timeout_seconds,
            }
        )
        return "Overall status: pass\n12 passed, 0 failed, 12 total\nbrowsers: chromium firefox webkit"

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=_fake_deep_agent,
    ):
        stage = E2EValidationStage(max_turns=99, timeout_seconds=2000)
        result = stage.run(
            run_id="run-abc",
            output_dir=tmp_path,
            feature_results=[_feature("dashboard", "success")],
            reconciliation_status="clean",
        )

    assert len(captured) == 1
    call = captured[0]
    assert set(call["allowed_tools"]) == {
        "Read", "Write", "Edit", "Glob", "Grep", "Bash"
    }
    assert call["max_turns"] == 99
    assert call["timeout_seconds"] == 2000.0
    # Prompt includes the run id, feature list, all three browsers
    assert "run-abc" in call["prompt"]
    assert "dashboard" in call["prompt"]
    assert "chromium" in call["prompt"]
    assert "firefox" in call["prompt"]
    assert "webkit" in call["prompt"]
    assert result.status == "pass"
    assert result.tests_passed == 12
    assert result.tests_failed == 0
    assert result.tests_total == 12


def test_e2e_respects_custom_browser_subset(tmp_path: Path):
    """An operator who only cares about chromium should see a prompt
    that lists only chromium and a config block that only includes
    the chromium project."""
    (tmp_path / "app").mkdir()
    captured: list[str] = []

    def _fake(prompt, allowed_tools, max_turns=15, timeout_seconds=None):
        captured.append(prompt)
        return "Overall status: pass\n3 passed"

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=_fake,
    ):
        stage = E2EValidationStage(browsers=["chromium"])
        stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )

    prompt = captured[0]
    assert "chromium" in prompt
    # The playwright.config projects block should only list chromium
    assert "name: 'chromium'" in prompt
    assert "name: 'firefox'" not in prompt
    assert "name: 'webkit'" not in prompt


def test_e2e_filters_invalid_browsers_from_matrix(tmp_path: Path):
    """Anything outside {chromium, firefox, webkit} is silently
    dropped during stage init rather than crashing the pipeline on
    a typo."""
    stage = E2EValidationStage(browsers=["chromium", "safari", "edge"])
    # "safari" and "edge" are dropped; chromium survives
    assert stage.browsers == ["chromium"]


def test_e2e_empty_browser_list_falls_back_to_default(tmp_path: Path):
    """A fully-invalid list should not leave the stage with zero
    browsers — that would produce a malformed Playwright config."""
    stage = E2EValidationStage(browsers=["safari", "edge"])
    assert stage.browsers == ["chromium", "firefox", "webkit"]


# ── Status classification ──────────────────────────────────────────────────


def test_e2e_classifies_pass_from_output(tmp_path: Path):
    (tmp_path / "app").mkdir()
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="... Overall status: pass",
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert result.status == "pass"


def test_e2e_classifies_broken_as_partial(tmp_path: Path):
    """Like reconciliation, an agent-reported ``broken`` status maps
    to ``partial`` on our side — the run still produced output."""
    (tmp_path / "app").mkdir()
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="lots of failures. Overall status: broken",
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert result.status == "partial"


def test_e2e_classifies_ambiguous_output_as_partial(tmp_path: Path):
    """No clear status line → default to partial as the safe
    conservative classification."""
    (tmp_path / "app").mkdir()
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="I ran some stuff but forgot to write a status line.",
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert result.status == "partial"


def test_e2e_classifies_skipped_when_not_a_web_app(tmp_path: Path):
    """If the agent decides the run produced a pure backend CLI, it
    writes ``Overall status: skipped`` and the stage reports the
    same."""
    (tmp_path / "app").mkdir()
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="No web app detected. Overall status: skipped",
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert result.status == "skipped"


def test_e2e_falls_back_to_report_file_for_status(tmp_path: Path):
    """When the agent's final text doesn't include a status line but
    it wrote one into E2E_REPORT.md, the stage reads the report."""
    (tmp_path / "app").mkdir()
    (tmp_path / "E2E_REPORT.md").write_text(
        "# E2E\n\n## Overall status: pass\n\n14 passed, 0 failed\n"
    )
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="wrote the report",  # no status line in output
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert result.status == "pass"
    assert result.report_path == "E2E_REPORT.md"
    assert result.tests_passed == 14


# ── Artifact discovery ────────────────────────────────────────────────────


def test_e2e_detects_html_report_and_screenshots(tmp_path: Path):
    (tmp_path / "app").mkdir()
    html_dir = tmp_path / "e2e_artifacts" / "html-report"
    html_dir.mkdir(parents=True)
    (html_dir / "index.html").write_text("<html/>")

    shot_dir = tmp_path / "e2e_artifacts" / "test-failure"
    shot_dir.mkdir(parents=True)
    (shot_dir / "trace.png").write_bytes(b"PNG\x00")
    (shot_dir / "failure.png").write_bytes(b"PNG\x00")

    (tmp_path / "E2E_REPORT.md").write_text("Overall status: partial\n")

    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value="ok",
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )

    assert result.html_report_path == "e2e_artifacts/html-report"
    assert sorted(result.screenshots) == [
        "e2e_artifacts/test-failure/failure.png",
        "e2e_artifacts/test-failure/trace.png",
    ]


# ── Report parsing helpers ─────────────────────────────────────────────────


def test_parse_test_counts_from_playwright_summary():
    text = "Running 15 tests using 3 workers\n\n12 passed, 3 failed, 15 total (5.2s)"
    assert _parse_test_counts(text) == (15, 12, 3)


def test_parse_test_counts_handles_passing_wording():
    assert _parse_test_counts("5 passing, 0 failing") == (5, 5, 0)


def test_parse_test_counts_handles_empty_text():
    assert _parse_test_counts("") == (0, 0, 0)


def test_parse_browsers_run_filters_to_requested_matrix():
    text = "chromium: pass\nwebkit: pass\nopera: ignored"
    assert _parse_browsers_run(text, ["chromium", "firefox", "webkit"]) == [
        "chromium",
        "webkit",
    ]


def test_parse_browsers_run_falls_back_to_default_when_none_found():
    """If the report doesn't mention any browser by name, we
    conservatively assume the whole requested matrix was attempted."""
    assert _parse_browsers_run("nothing interesting", ["chromium"]) == ["chromium"]


# ── Error tolerance ───────────────────────────────────────────────────────


def test_e2e_returns_error_when_deep_agent_crashes(tmp_path: Path):
    """A crash in _run_deep_agent must NOT raise out of the stage —
    E2E is best-effort. The error is surfaced via the result's
    status + summary so the Run Detail popup can show it."""
    (tmp_path / "app").mkdir()
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        side_effect=RuntimeError("browser missing"),
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert result.status == "error"
    assert "browser missing" in result.summary.lower()
    assert result.agent_output == ""


def test_e2e_agent_output_is_truncated_to_8kb(tmp_path: Path):
    (tmp_path / "app").mkdir()
    huge = "x" * 20000  # 20 KiB
    with patch(
        "dark_factory.agents.tools._run_deep_agent",
        return_value=huge,
    ):
        stage = E2EValidationStage()
        result = stage.run(
            run_id="r1",
            output_dir=tmp_path,
            feature_results=[_feature()],
            reconciliation_status="clean",
        )
    assert len(result.agent_output) <= 8000


def test_e2e_restores_output_dir_after_call(tmp_path: Path):
    """Matches the reconciliation stage's contract: module-global
    ``_output_dir`` is swapped to the run output for the call and
    restored to its previous value on exit (success or crash)."""
    from dark_factory.agents import tools as _tools_mod

    (tmp_path / "app").mkdir()
    previous = Path("/some/other/dir")
    _tools_mod._output_dir = previous
    try:
        with patch(
            "dark_factory.agents.tools._run_deep_agent",
            return_value="Overall status: pass",
        ):
            stage = E2EValidationStage()
            stage.run(
                run_id="r1",
                output_dir=tmp_path,
                feature_results=[_feature()],
                reconciliation_status="clean",
            )
        assert _tools_mod._output_dir == previous

        # And on crash
        with patch(
            "dark_factory.agents.tools._run_deep_agent",
            side_effect=RuntimeError("boom"),
        ):
            stage = E2EValidationStage()
            stage.run(
                run_id="r1",
                output_dir=tmp_path,
                feature_results=[_feature()],
                reconciliation_status="clean",
            )
        assert _tools_mod._output_dir == previous
    finally:
        _tools_mod._output_dir = None


# ── Model round-trip ──────────────────────────────────────────────────────


def test_e2e_validation_result_model_roundtrips_to_dict():
    """``model_dump()`` must produce a JSON-safe dict that the
    ag_ui_bridge can attach to the StateSnapshot payload."""
    import json as _json

    result = E2EValidationResult(
        status="pass",
        summary="all good",
        tests_total=10,
        tests_passed=10,
        tests_failed=0,
        browsers_run=["chromium", "firefox", "webkit"],
        agent_output="ran tests",
        report_path="E2E_REPORT.md",
        html_report_path="e2e_artifacts/html-report",
        screenshots=[],
        duration_seconds=45.0,
    )
    payload = result.model_dump()
    serialised = _json.dumps(payload)
    loaded = _json.loads(serialised)
    assert loaded["status"] == "pass"
    assert loaded["tests_passed"] == 10
    assert loaded["browsers_run"] == ["chromium", "firefox", "webkit"]
    assert loaded["duration_seconds"] == pytest.approx(45.0)
