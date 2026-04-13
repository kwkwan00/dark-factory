"""Stage 5: Reconciliation — post-swarm cross-feature polishing pass.

Runs AFTER every feature swarm has completed. The goal is to catch
problems that per-feature swarms can't see because they each operate
in isolation:

- **Broken cross-feature imports**: the dashboard feature tries to
  call a backend endpoint the auth feature named differently.
- **Inconsistent API shapes**: frontend fetches ``/api/users`` but
  the backend wrote ``/api/user``.
- **Missing glue**: no top-level ``main.py`` / ``package.json`` /
  ``requirements.txt`` tying the features together.
- **Runtime failures**: syntax errors or missing deps that the
  reviewer/tester caught per-feature but across the whole tree the
  pieces don't actually run.

Implementation notes:

- This is a **single extended Claude Agent SDK invocation**, not a
  new LangGraph swarm. The SDK already handles the multi-turn
  review → fix → validate → iterate loop we need — wrapping it in a
  planner/coder/reviewer/tester handoff dance would just add
  overhead. The stage is isolated enough to graduate to a true swarm
  later if we need more structured observability.
- CWD is set to the run's output directory, so all file reads/writes
  and Bash executions are scoped to generated code for this run only.
  The SDK's subprocess cannot escape the output dir (the ``cwd``
  argument is baked into ``ClaudeAgentOptions``).
- Failures are **best-effort**: if reconciliation errors, crashes, or
  times out, we log a warning and return a ``ReconciliationResult``
  with ``status="error"`` — the pipeline continues and the feature
  output is still delivered as-is. Reconciliation is polishing, not
  gatekeeping.
- The stage writes a ``RECONCILIATION_REPORT.md`` at the root of the
  run's output directory summarising what it did, so the operator
  can read it from the Run Detail popup's Output screen.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel

from dark_factory.agents.cancellation import PipelineCancelled

log = structlog.get_logger()


# The prompt is long but deliberately structured as a 6-step checklist
# so the SDK's planning model can emit a clear TODO list at turn 0 and
# then tick items off as it progresses. Loose prose prompts produce
# much worse agent behaviour in our testing.
from dark_factory.prompts import get_prompt

_RECONCILIATION_PROMPT_TEMPLATE = get_prompt("reconciliation", "system")


class ReconciliationResult(BaseModel):
    """Outcome of a reconciliation pass."""

    status: Literal["clean", "partial", "error", "skipped"]
    """``clean`` = agent reported everything builds & tests pass.
    ``partial`` = agent finished but some validation still failed.
    ``error`` = agent crashed, timed out, or couldn't start.
    ``skipped`` = stage was disabled or nothing to reconcile."""

    summary: str
    """One-line human-readable summary shown in the Run Detail popup."""

    agent_output: str
    """The deep agent's final result text. Stored in the metrics row so
    the Run Detail popup's Incidents table can surface it."""

    report_path: str | None
    """Path to RECONCILIATION_REPORT.md relative to the run's output
    directory, or None if the agent didn't produce one."""

    duration_seconds: float


class ReconciliationStage:
    """Runs the reconciliation deep agent on a completed run's output.

    Not a ``Stage`` subclass because it doesn't fit the
    ``PipelineContext → PipelineContext`` contract — it operates on a
    filesystem path and a list of feature results, not on the pipeline
    context. Kept intentionally small: all the heavy lifting happens
    inside the Claude Agent SDK invocation.
    """

    name = "reconciliation"

    def __init__(
        self,
        *,
        max_turns: int = 100,
        timeout_seconds: int = 1800,
    ) -> None:
        self.max_turns = max(1, max_turns)
        self.timeout_seconds = max(60, timeout_seconds)

    def run(
        self,
        *,
        run_id: str,
        output_dir: Path,
        feature_results: list[dict[str, Any]],
    ) -> ReconciliationResult:
        """Execute the reconciliation pass.

        :param run_id: the pipeline run id (for logging / metrics only)
        :param output_dir: absolute path to the run's output root
        :param feature_results: ``completed_features`` list from the
            orchestrator. Each entry has at least ``feature`` + ``status``.
        """
        from dark_factory.agents.tools import emit_progress

        started = time.time()

        # ── Skip guards ─────────────────────────────────────────────
        # Only the degenerate "nothing to reconcile" cases — the
        # phase always runs when feature swarms produced output.
        if not feature_results:
            log.info(
                "reconciliation_skipped",
                reason="no_features",
                run_id=run_id,
            )
            return ReconciliationResult(
                status="skipped",
                summary="No features to reconcile",
                agent_output="",
                report_path=None,
                duration_seconds=0.0,
            )

        # If every feature errored, there's no output to reconcile —
        # the run dir is empty or full of garbage.
        if all(
            (r.get("status") or "").lower() in ("error", "skipped")
            for r in feature_results
        ):
            log.info(
                "reconciliation_skipped",
                reason="all_features_errored",
                run_id=run_id,
            )
            return ReconciliationResult(
                status="skipped",
                summary="All features errored — nothing to reconcile",
                agent_output="",
                report_path=None,
                duration_seconds=0.0,
            )

        if not output_dir.exists() or not output_dir.is_dir():
            log.warning(
                "reconciliation_skipped",
                reason="output_dir_missing",
                run_id=run_id,
                output_dir=str(output_dir),
            )
            return ReconciliationResult(
                status="skipped",
                summary=f"Output dir {output_dir} missing",
                agent_output="",
                report_path=None,
                duration_seconds=0.0,
            )

        # ── Build the prompt ────────────────────────────────────────
        feature_names = [r.get("feature", "?") for r in feature_results]
        statuses = [
            f"{r.get('feature', '?')}={(r.get('status') or '?')}"
            for r in feature_results
        ]
        prompt = _RECONCILIATION_PROMPT_TEMPLATE.format(
            run_id=run_id,
            feature_count=len(feature_results),
            feature_list=", ".join(feature_names) or "(none)",
            feature_status_summary=", ".join(statuses),
        )

        # ── Announce the phase ──────────────────────────────────────
        emit_progress(
            "reconciliation_started",
            run_id=run_id,
            feature_count=len(feature_results),
            max_turns=self.max_turns,
            timeout_seconds=self.timeout_seconds,
        )
        log.info(
            "reconciliation_starting",
            run_id=run_id,
            output_dir=str(output_dir),
            feature_count=len(feature_results),
            max_turns=self.max_turns,
        )

        # ── Run the deep agent ──────────────────────────────────────
        # We import lazily to keep reconciliation.py cheap to import
        # for test code that stubs out _run_deep_agent.
        from dark_factory.agents import tools as _tools_mod

        # Switch the module-level output dir so _run_deep_agent's
        # ``cwd`` resolves to the run's output root. Save + restore the
        # previous value so we don't leak this into other phases.
        previous_output = _tools_mod._output_dir
        _tools_mod._output_dir = output_dir

        agent_output = ""
        agent_error: Exception | None = None
        try:
            agent_output = _tools_mod._run_deep_agent(
                prompt=prompt,
                allowed_tools=[
                    "Read",
                    "Write",
                    "Edit",
                    "Glob",
                    "Grep",
                    "Bash",
                ],
                max_turns=self.max_turns,
                timeout_seconds=float(self.timeout_seconds),
            )
        except PipelineCancelled:
            # B2 fix: cancel signal must propagate out of best-effort
            # stages. Without this guard the broad ``except Exception``
            # below would swallow PipelineCancelled and the pipeline
            # would blithely proceed to Phase 6 after a user cancel
            # during Phase 5, making the Cancel button unreliable.
            _tools_mod._output_dir = previous_output
            raise
        except Exception as exc:
            agent_error = exc
            log.error(
                "reconciliation_failed",
                run_id=run_id,
                error=str(exc),
            )
        finally:
            _tools_mod._output_dir = previous_output

        duration = time.time() - started

        # ── Classify the outcome ────────────────────────────────────
        report_path: str | None = None
        report_abs = output_dir / "RECONCILIATION_REPORT.md"
        if report_abs.exists():
            report_path = "RECONCILIATION_REPORT.md"

        if agent_error is not None:
            status: Literal["clean", "partial", "error", "skipped"] = "error"
            summary = f"Reconciliation errored: {agent_error}"
        elif not agent_output:
            status = "error"
            summary = "Reconciliation produced no output"
        else:
            # Pattern-match the agent's final summary for a status
            # hint. The prompt asks for "Overall status: clean|partial|
            # broken" at the end of the report. We check both the
            # agent_output (final message) and the report file if it
            # exists — agents sometimes put the status in one but not
            # the other.
            haystack = agent_output
            if report_path is not None:
                try:
                    haystack += "\n" + report_abs.read_text(errors="replace")
                except Exception:  # pragma: no cover — defensive
                    pass
            lowered = haystack.lower()
            if (
                "overall status: clean" in lowered
                or "status: clean" in lowered
            ):
                status = "clean"
                summary = "Reconciliation clean — all validation passed"
            elif (
                "overall status: broken" in lowered
                or "status: broken" in lowered
            ):
                status = "partial"
                summary = (
                    "Reconciliation finished with significant remaining issues"
                )
            else:
                status = "partial"
                summary = "Reconciliation finished with some remaining issues"

        emit_progress(
            "reconciliation_completed",
            run_id=run_id,
            status=status,
            summary=summary,
            duration_seconds=duration,
            report_path=report_path,
        )
        log.info(
            "reconciliation_complete",
            run_id=run_id,
            status=status,
            duration_seconds=round(duration, 1),
            report_path=report_path,
        )

        return ReconciliationResult(
            status=status,
            summary=summary,
            agent_output=agent_output[:8000],  # cap for incident storage
            report_path=report_path,
            duration_seconds=duration,
        )
