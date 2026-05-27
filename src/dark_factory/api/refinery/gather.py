"""Data gathering for the refinery — collects run evidence and ingests documents."""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


def _derive_gaps_from_traceability(trace: dict) -> dict:
    """Extract gap-like data from the traceability matrix."""
    rows = trace.get("rows", [])
    specs_without_artifacts = []
    specs_failing_evals = []
    unimplemented = []
    for row in rows:
        status = row.get("overall_status", "")
        req = row.get("requirement", {})
        if status == "no_specs":
            unimplemented.append({"id": req.get("id"), "title": req.get("title")})
        for spec in row.get("specs", []):
            if not spec.get("files"):
                specs_without_artifacts.append({"id": spec.get("id"), "title": spec.get("title")})
            if spec.get("all_passed") is False:
                specs_failing_evals.append({"id": spec.get("id"), "title": spec.get("title"), "eval_scores": spec.get("eval_scores", {})})
    return {
        "unimplemented_requirements": unimplemented,
        "specs_without_artifacts": specs_without_artifacts,
        "specs_failing_evals": specs_failing_evals,
    }


def gather_run_context(request: Any, run_id: str) -> dict[str, Any]:
    """Collect all available evidence from a historical run."""
    from dark_factory.api.routes_dashboard import get_traceability

    context: dict[str, Any] = {"run_id": run_id}

    try:
        context["traceability"] = get_traceability(request, run_id=run_id)
    except Exception as exc:
        log.warning("refinery_traceability_failed", error=str(exc))
        context["traceability"] = None

    if context["traceability"] is not None:
        context["gaps"] = _derive_gaps_from_traceability(context["traceability"])
    else:
        context["gaps"] = None

    memory_repo = getattr(request.app.state, "memory_repo", None)
    if memory_repo:
        try:
            context["episodes"] = memory_repo.get_episodes_for_run(run_id=run_id)
        except Exception as exc:
            log.debug("refinery_episodes_failed", error=str(exc))
            context["episodes"] = []

        try:
            context["learnings"] = memory_repo.get_run_learnings(run_id)
        except Exception as exc:
            log.debug("refinery_learnings_failed", error=str(exc))
            context["learnings"] = []

        try:
            all_evals = memory_repo.list_evals_by_run(run_limit=1)
            context["evals"] = [r for r in all_evals if r.get("run_id") == run_id]
        except Exception as exc:
            log.debug("refinery_evals_failed", error=str(exc))
            context["evals"] = []

        try:
            context["run"] = memory_repo.get_run(run_id)
        except Exception as exc:
            log.debug("refinery_run_failed", error=str(exc))
            context["run"] = None
    else:
        context["episodes"] = []
        context["learnings"] = []
        context["evals"] = []
        context["run"] = None

    return context


def ingest_requirements(input_path: str, settings: Any) -> list[dict]:
    """Run IngestStage standalone to extract requirements from uploaded files."""
    from dark_factory.models.domain import PipelineContext
    from dark_factory.stages.ingest import IngestStage
    from dark_factory.ui.helpers import build_llm

    ctx = PipelineContext(input_path=input_path)
    llm = None
    try:
        llm = build_llm(settings)
    except Exception:
        pass
    stage = IngestStage(llm=llm)
    result = stage.run(ctx)
    return [r.model_dump() for r in result.requirements]
