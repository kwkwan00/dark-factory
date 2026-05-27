"""REST dashboard endpoints — gap finder, history, memory, eval, settings, health, watcher."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from dark_factory.config import Settings

import structlog
from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dark_factory.api.refinery import RequirementPatchRequest

log = structlog.get_logger()

router = APIRouter()


def _get_metrics_repo(request: Request):
    """Build a ``RefineryMetricsRepository`` from app state, or return
    ``None`` when Postgres is unwired. Used by every endpoint that
    writes refinery telemetry — keeps the "metrics_client → repo or
    None" prelude in one place."""

    metrics_client = getattr(request.app.state, "metrics_client", None)
    if metrics_client is None:
        return None
    try:
        from dark_factory.metrics.refinery_repository import (
            RefineryMetricsRepository,
        )
        return RefineryMetricsRepository(metrics_client)
    except Exception:  # pragma: no cover — defensive
        return None


def _get_storage(request: Request):
    """Return the storage backend from app state, or create one."""
    storage = getattr(request.app.state, "storage", None)
    if storage is not None:
        return storage
    from dark_factory.storage.backend import get_storage
    return get_storage()


# ── Graph Gap Finder (run-scoped) ─────────────────────────────────────────────


@router.get("/graph/gaps/{run_id}")
def get_run_gaps(
    request: Request,
    run_id: str = Path(..., pattern=r"^[A-Za-z0-9_\-]{1,128}$"),
):
    """Return actionable gaps for a specific pipeline run.

    Derives gap analysis from the traceability matrix (which already
    joins Neo4j specs with Postgres artifacts/evals for the run) plus
    additional Neo4j queries for structural and episodic gaps.

    Categories:

    1. **specs_without_artifacts** — Specs in this run's scope that
       produced no output files.
    2. **specs_failing_evals** — Specs whose evals failed in this run.
    3. **unimplemented_requirements** — Requirements whose specs have
       no artifacts in this run (nothing was built for them).
    4. **broken_dependencies** — Specs in this run whose DEPENDS_ON
       targets don't exist in the graph.
    5. **capability_islands** — Specs in this run sharing a capability
       but with no dependency path between them.
    6. **missing_episodes** — Features in this run that have no
       Episode node.
    """
    neo4j_client = request.app.state.neo4j_client

    # ── Reuse the traceability endpoint to get the full matrix ────
    trace_resp = get_traceability(request, run_id=run_id)
    trace_rows = (
        trace_resp.get("rows", []) if isinstance(trace_resp, dict)
        else []
    )

    # ── Extract gaps from the traceability matrix ────────────────
    specs_without_artifacts: list[dict] = []
    specs_failing_evals: list[dict] = []
    unimplemented_reqs: list[dict] = []
    run_spec_ids: list[str] = []

    for row in trace_rows:
        req = row.get("requirement", {})
        specs = row.get("specs", [])
        overall = row.get("overall_status", "no_specs")

        # Requirements with no specs at all, or no artifacts for any spec
        if overall == "no_specs":
            unimplemented_reqs.append({
                "id": req.get("id"),
                "title": req.get("title"),
                "priority": req.get("priority"),
                "reason": "no implementing specs",
            })
        elif overall == "no_evals":
            # Has specs with files but no evals — still a gap
            has_any_files = any(s.get("files") for s in specs)
            if not has_any_files:
                unimplemented_reqs.append({
                    "id": req.get("id"),
                    "title": req.get("title"),
                    "priority": req.get("priority"),
                    "reason": "no artifacts produced",
                })

        for spec in specs:
            sid = spec.get("id", "")
            if sid:
                run_spec_ids.append(sid)

            has_files = bool(spec.get("files"))
            all_passed = spec.get("all_passed")

            # Specs with no output files in this run
            if not has_files:
                specs_without_artifacts.append({
                    "id": sid,
                    "title": spec.get("title"),
                    "capability": spec.get("capability"),
                })

            # Specs with failing evals in this run
            if all_passed is False:
                eval_scores = spec.get("eval_scores", {})
                specs_failing_evals.append({
                    "id": sid,
                    "title": spec.get("title"),
                    "capability": spec.get("capability"),
                    "eval_scores": eval_scores,
                })

    # ── Neo4j: structural gaps among this run's specs ────────────
    broken_deps: list[dict] = []
    capability_islands: list[dict] = []
    missing_episodes: list[dict] = []

    try:
        with neo4j_client.session() as session:
            # Broken dependencies among this run's specs
            if run_spec_ids:
                broken_rows = session.run(
                    """
                    MATCH (s:Spec)-[:DEPENDS_ON]->(d:Spec)
                    WHERE s.id IN $sids
                      AND NOT d.id IN $sids
                      AND NOT EXISTS { MATCH (existing:Spec {id: d.id}) }
                    RETURN s.id AS spec_id, s.title AS spec_title,
                           d.id AS missing_dep_id
                    """,
                    sids=run_spec_ids,
                )
                for r in broken_rows:
                    broken_deps.append({
                        "spec_id": r["spec_id"],
                        "spec_title": r["spec_title"],
                        "missing_dep_id": r["missing_dep_id"],
                    })

            # Capability islands among this run's specs
            if run_spec_ids:
                cap_rows = session.run(
                    """
                    MATCH (s:Spec)
                    WHERE s.id IN $sids
                      AND s.capability IS NOT NULL AND s.capability <> ''
                    WITH s.capability AS cap, collect(s.id) AS spec_ids
                    WHERE size(spec_ids) > 1
                    RETURN cap, spec_ids
                    """,
                    sids=run_spec_ids,
                )
                cap_groups = [(r["cap"], list(r["spec_ids"])) for r in cap_rows]

                all_cap_spec_ids = []
                for _cap, sids in cap_groups:
                    all_cap_spec_ids.extend(sids)

                adjacency: dict[str, set[str]] = {}
                if all_cap_spec_ids:
                    adj_rows = session.run(
                        """
                        MATCH (a:Spec)-[:DEPENDS_ON]-(b:Spec)
                        WHERE a.id IN $sids AND b.id IN $sids
                        RETURN DISTINCT a.id AS src, b.id AS dst
                        """,
                        sids=all_cap_spec_ids,
                    )
                    for ar in adj_rows:
                        adjacency.setdefault(ar["src"], set()).add(ar["dst"])
                        adjacency.setdefault(ar["dst"], set()).add(ar["src"])

                for cap, sids in cap_groups:
                    sid_set = set(sids)
                    visited: set[str] = set()
                    queue = [sids[0]]
                    visited.add(sids[0])
                    while queue:
                        current = queue.pop(0)
                        for neighbor in adjacency.get(current, ()):
                            if neighbor in sid_set and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    disconnected = [s for s in sids if s not in visited]
                    if disconnected:
                        capability_islands.append({
                            "capability": cap,
                            "total_specs": len(sids),
                            "disconnected_specs": disconnected,
                        })

            # Missing episodes for THIS run
            ep_rows = session.run(
                """
                MATCH (r:Run {id: $run_id})
                OPTIONAL MATCH (r)<-[:PRODUCED_IN]-(ep:Episode)
                WITH r, collect(ep.feature) AS ep_features
                UNWIND CASE WHEN size(ep_features) > 0
                       THEN ep_features ELSE [null] END AS ef
                WITH r, collect(ef) AS ep_features_flat
                MATCH (s:Spec)
                WHERE s.id IN $sids
                  AND s.capability IS NOT NULL AND s.capability <> ''
                WITH r, ep_features_flat,
                     s.capability AS feature, collect(DISTINCT s.id) AS spec_ids
                WHERE NOT feature IN ep_features_flat
                RETURN feature, spec_ids
                """,
                run_id=run_id,
                sids=run_spec_ids,
            )
            for r in ep_rows:
                missing_episodes.append({
                    "feature": r["feature"],
                    "spec_count": len(r["spec_ids"]),
                })
    except Exception as exc:
        log.debug("run_gaps_neo4j_failed", error=str(exc), run_id=run_id)

    # ── Totals ───────────────────────────────────────────────────
    req_count = len({
        row.get("requirement", {}).get("id")
        for row in trace_rows
        if row.get("requirement", {}).get("id")
    })

    return {
        "run_id": run_id,
        "specs_without_artifacts": specs_without_artifacts,
        "specs_failing_evals": specs_failing_evals,
        "unimplemented_requirements": unimplemented_reqs,
        "broken_dependencies": broken_deps,
        "capability_islands": capability_islands,
        "missing_episodes": missing_episodes,
        "totals": {
            "requirements": req_count,
            "specs": len(run_spec_ids),
        },
    }


# ── Requirements Refinery ──────────────────────────────────────────────────────


class _DirectRequirement(BaseModel):
    """Phase 6 direct-input mode — user types one requirement directly."""

    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field("", max_length=20000)
    priority: str = Field("medium", pattern=r"^(low|medium|high|critical)$")
    tags: list[str] = Field(default_factory=list, max_length=32)


class _RefineryRequest(BaseModel):
    run_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_\-]{1,128}$")
    input_path: str | None = None
    # Third input mode: a single directly-submitted requirement. When
    # set, the refinery runs the full debate on exactly one item and
    # skips Phase-3 cross-req review (n=1).
    direct: _DirectRequirement | None = None


@router.post("/refinery")
async def run_refinery(request: Request, body: _RefineryRequest):
    """SSE stream that runs the requirements refinery.

    Three input modes, any one of which must be set:

    - ``run_id`` — refine from a historical run's evidence
    - ``input_path`` — refine from uploaded requirement documents
    - ``direct`` — refine a single user-typed requirement
    """
    from dark_factory.api.refinery import run_refinery_stream

    direct_payload = body.direct.model_dump() if body.direct else None

    async def generator():
        async for event in run_refinery_stream(
            request, body.run_id, body.input_path, direct=direct_payload,
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/refinery/resume/{refinery_run_id}")
async def resume_refinery(request: Request, refinery_run_id: str = Path(...)):
    """SSE stream that resumes an interrupted refinery run.

    Looks up the saved input snapshot in the resume registry, replays
    every cached debate completion, and re-runs only the requirements
    that didn't complete on the original run. Phase 3 + 4 always run
    fresh on the merged set so the cross-req review reflects the
    completed state.

    Returns 404 when the run is unknown to the registry (Postgres
    disabled or row purged); returns 409 when the run is already
    completed (the caller should ``GET /api/refinery/{result_id}``
    instead).
    """

    from dark_factory.api.refinery import run_refinery_stream
    from dark_factory.api.refinery.resume import ResumeRegistry

    metrics_client = getattr(request.app.state, "metrics_client", None)
    registry = ResumeRegistry(metrics_client)
    saved = registry.load_run(refinery_run_id)
    if saved is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Run {refinery_run_id} unknown to the resume registry. "
                "Postgres may be disabled, or the run row may have been purged."
            ),
        )
    if saved.get("status") == "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run {refinery_run_id} is already completed. "
                f"GET /api/refinery/{refinery_run_id} to load its saved result."
            ),
        )

    # Atomically transition the run to ``in_progress``. The claim is
    # single-winner under concurrent double-clicks (row lock + the
    # ``started_at = NOW()`` reset on success); a stale ``in_progress``
    # row (worker hard-killed before its finally could flip status)
    # becomes reclaimable after the staleness window elapses.
    if not registry.try_claim_in_progress(refinery_run_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run {refinery_run_id} is in progress on another worker "
                "and is still within the freshness window. Wait for it "
                "to finish (or for the staleness window to elapse) "
                "before retrying."
            ),
        )

    async def generator():
        async for event in run_refinery_stream(
            request, run_id=None, input_path=None, direct=None,
            resume_run_id=refinery_run_id,
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class _MemoryFeedbackRequest(BaseModel):
    """Operator decision on a suggested memory.

    Drives the active-learning loop: telemetry to Postgres for cross-run
    aggregation, relevance-score adjustment in Neo4j to re-rank future
    retrieval, and (on a reasoned dismissal) a Conflict memory tagged
    ``user_override`` so the next debate sees the prior judgement.
    """

    decision: str = Field(..., pattern=r"^(accepted|dismissed|edited)$")
    memory_kind: str = Field(
        ...,
        pattern=r"^(pattern|mistake|solution|strategy|"
                 r"decision|incident|constraint|conflict)$",
    )
    refinery_run_id: str | None = None
    source_role: str | None = Field(None, max_length=64)
    reason: str | None = Field(None, max_length=2000)


@router.post("/refinery/memory/{memory_id}/feedback")
def submit_memory_feedback(
    request: Request,
    body: _MemoryFeedbackRequest,
    memory_id: str = Path(..., pattern=r"^[A-Za-z0-9_\-]{1,128}$"),
):
    """Record one operator feedback decision on a suggested memory.

    Best-effort across three sinks (Postgres telemetry, Neo4j relevance,
    optional Conflict emission). Each sink fails closed and returns a
    flag in the response so the UI can surface partial success without
    blocking the user action.
    """

    from dark_factory.api.refinery.feedback import record_memory_feedback

    metrics_repo = _get_metrics_repo(request)
    memory_repo = getattr(request.app.state, "memory_repo", None)

    return record_memory_feedback(
        memory_id=memory_id,
        memory_kind=body.memory_kind,
        decision=body.decision,
        metrics_repo=metrics_repo,
        memory_repo=memory_repo,
        refinery_run_id=body.refinery_run_id,
        source_role=body.source_role,
        reason=body.reason,
    )


class _RequirementDecisionRequest(BaseModel):
    """Operator decision on a refined requirement.

    Drives the Judge confidence-calibration loop: ``accepted`` vs.
    ``dismissed`` joined against the Judge's predicted overall score
    tells us whether the Judge is over- or under-confident on average,
    and the next run nudges the convergence threshold accordingly.
    """

    decision: str = Field(..., pattern=r"^(accepted|dismissed|edited)$")
    judge_overall_score: float | None = Field(None, ge=0.0, le=1.0)
    convergence_status: str | None = Field(
        None, pattern=r"^(converged|short_circuited|aborted)$",
    )


@router.post("/refinery/{refinery_run_id}/requirement/{requirement_id}/decision")
def submit_requirement_decision(
    request: Request,
    body: _RequirementDecisionRequest,
    refinery_run_id: str = Path(..., pattern=r"^[A-Za-z0-9_\-]{1,128}$"),
    requirement_id: str = Path(..., pattern=r"^[A-Za-z0-9_\-]{1,128}$"),
):
    """Record one operator decision on a refined requirement.

    Best-effort write to ``refinery_requirement_decisions``. The
    aggregator reads this on every run to compute the next debate's
    threshold multiplier. Postgres outage logs and continues so the
    UI action never fails.
    """

    repo = _get_metrics_repo(request)
    if repo is None:
        return {"ok": True, "telemetry_recorded": False, "reason": "no_metrics_client"}

    try:
        repo.record_requirement_decision(
            refinery_run_id=refinery_run_id,
            requirement_id=requirement_id,
            decision=body.decision,
            judge_overall_score=body.judge_overall_score,
            convergence_status=body.convergence_status,
        )
        return {"ok": True, "telemetry_recorded": True}
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("submit_requirement_decision_failed", error=str(exc))
        return {"ok": True, "telemetry_recorded": False, "reason": str(exc)}


@router.post("/refinery/export")
def export_refinery_zip(body: dict):
    """Export refinery results as a ZIP containing the report + individual requirement files.

    Structure::

        refinery-{run_id}/
            REPORT.md                    — full refinery report
            requirements/
                {req-id}.md              — one file per requirement (if >1)
                {req-id}.md              — or a single requirements.md
    """
    import io
    import re
    import zipfile

    from dark_factory.api.refinery import (
        RefineryResponse,
        render_report_markdown,
        render_requirement_markdown,
    )

    try:
        response = RefineryResponse(**body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid refinery response: {exc}") from exc

    run_label = response.source_run_id or "upload"
    prefix = f"refinery-{run_label}"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Report
        zf.writestr(f"{prefix}/REPORT.md", render_report_markdown(response))

        for rr in response.refined_requirements:
            safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", rr.id)
            zf.writestr(
                f"{prefix}/requirements/{safe}.md",
                render_requirement_markdown(rr),
            )

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{prefix}.zip"',
        },
    )


@router.get("/refinery/history")
def refinery_history(limit: int = Query(default=20, ge=1, le=100)):
    """List historical refinery results, newest first."""
    from dark_factory.api.refinery import list_refinery_results

    return {"results": list_refinery_results(limit=limit)}


@router.get("/refinery/research/provider-stats")
def refinery_research_provider_stats(
    request: Request,
    window_days: int = Query(default=30, ge=1, le=365),
):
    """Per-provider research contribution stats over the last
    ``window_days``, plus the trust-weight adjustment the learning
    loop is currently applying.

    Reads from the resume registry's Postgres surface; returns an
    empty list + a message when Postgres is disabled.
    """

    repo = _get_metrics_repo(request)
    if repo is None:
        return {
            "providers": [],
            "tiers": [],
            "message": "Provider learning disabled (Postgres off)",
        }

    from dark_factory.api.refinery.research.learning import (
        load_adjusted_trust_weights,
    )

    settings = request.app.state.settings
    pipeline = settings.pipeline
    base_weights = {
        int(k): float(v)
        for k, v in (pipeline.refinery_research_tier_trust_weights or {}).items()
    }
    if not base_weights:
        # Plan defaults when no operator override is present.
        base_weights = {0: 1.0, 1: 0.95, 2: 0.90, 3: 0.80, 4: 0.60, 5: 0.30}

    try:
        per_provider = repo.compute_provider_stats(window_days=window_days)
    except Exception as exc:
        log.warning(
            "refinery_research_provider_stats_failed", error=str(exc),
        )
        return {"providers": [], "tiers": [], "message": f"query failed: {exc}"}

    _, tier_adjustments = load_adjusted_trust_weights(
        metrics_client=request.app.state.metrics_client,
        base_weights=base_weights,
        window_days=window_days,
        provider_stats=per_provider,
    )

    return {
        "window_days": window_days,
        "providers": per_provider,
        "tiers": [
            {
                "tier": adj.tier,
                "base_weight": adj.base_weight,
                "adjusted_weight": adj.adjusted_weight,
                "contribution_rate": adj.contribution_rate,
                "calls": adj.calls,
                "propagated": adj.propagated,
                "reason": adj.reason,
            }
            for adj in tier_adjustments.values()
        ],
    }


@router.get("/refinery/resumable")
def refinery_resumable(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
):
    """List refinery runs that didn't complete cleanly and can be
    resumed via ``POST /api/refinery/resume/{refinery_run_id}``.

    Reads from the resume registry (Postgres). Returns an empty list
    when Postgres is disabled or no rows exist. ``cancelled`` runs
    (operator-dismissed via ``DELETE /api/refinery/resumable/{id}``)
    are filtered out — they're tombstoned for telemetry but should
    not surface in the resume UI."""

    metrics_client = getattr(request.app.state, "metrics_client", None)
    if metrics_client is None:
        return {"results": [], "message": "Resume registry disabled (Postgres off)"}
    try:
        with metrics_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT refinery_run_id, source_mode, source_run_id,
                           requirements_count, started_at, status
                      FROM refinery_runs
                     WHERE status NOT IN ('completed', 'cancelled')
                     ORDER BY started_at DESC
                     LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
    except Exception as exc:
        log.warning("refinery_resumable_query_failed", error=str(exc))
        return {"results": [], "message": f"query failed: {exc}"}

    return {
        "results": [
            {
                "refinery_run_id": r["refinery_run_id"],
                "source_mode": r["source_mode"],
                "source_run_id": r["source_run_id"],
                "requirements_count": r["requirements_count"],
                "started_at": (
                    r["started_at"].isoformat() if r["started_at"] else None
                ),
                "status": r["status"],
            }
            for r in rows
        ],
    }


@router.delete("/refinery/resumable/{refinery_run_id}")
def dismiss_resumable_refinery(
    request: Request,
    refinery_run_id: str = Path(..., pattern=r"^refinery-[A-Za-z0-9_\-]+$"),
):
    """Tombstone a resumable refinery run so it stops appearing in the
    resume list. The Postgres row is preserved (status → ``cancelled``)
    so forensic telemetry stays intact; only the UI surface is cleared.

    Returns 404 when the run is unknown to the registry, 409 when the
    run is already completed (completed runs have results worth
    keeping — load them via ``GET /api/refinery/{refinery_run_id}``
    instead of dismissing). Returns the new status on success.
    """

    from dark_factory.api.refinery.resume import ResumeRegistry

    metrics_client = getattr(request.app.state, "metrics_client", None)
    if metrics_client is None:
        raise HTTPException(
            status_code=503,
            detail="Resume registry disabled (Postgres off)",
        )

    registry = ResumeRegistry(metrics_client)
    saved = registry.load_run(refinery_run_id)
    if saved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run {refinery_run_id} unknown to the resume registry.",
        )
    if saved.get("status") == "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run {refinery_run_id} is completed. Use "
                f"DELETE /api/refinery/{refinery_run_id} to remove the "
                "saved result, or leave it alone — completed runs do "
                "not appear in the resumable list."
            ),
        )

    registry.mark_status(refinery_run_id, "cancelled")
    return {"refinery_run_id": refinery_run_id, "status": "cancelled"}


@router.get("/refinery/{result_id}")
def load_refinery(result_id: str = Path(..., pattern=r"^refinery-[A-Za-z0-9_\-]+$")):
    """Load a saved refinery result."""
    from dark_factory.api.refinery import load_refinery_result

    result = load_refinery_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Refinery result {result_id} not found")
    return result.model_dump()


@router.delete("/refinery/{result_id}")
def delete_refinery(result_id: str = Path(..., pattern=r"^refinery-[A-Za-z0-9_\-]+$")):
    """Delete a saved refinery result from storage."""
    from dark_factory.api.refinery import delete_refinery_result

    if not delete_refinery_result(result_id):
        raise HTTPException(status_code=404, detail=f"Refinery result {result_id} not found")
    return {"deleted": result_id}


# ── Memory dedup check ────────────────────────────────────────────────────────


class _MemoryDedupRequest(BaseModel):
    type: str = Field(
        ...,
        pattern=(
            # Phase 9: extended pattern accepts refinery-institutional kinds
            # in addition to legacy swarm types. The ``kind`` field is
            # preferred going forward; ``type`` is kept as the legacy name
            # for backwards compatibility with existing frontends.
            r"^(pattern|mistake|solution|strategy|"
            r"decision|incident|constraint|conflict)$"
        ),
    )
    description: str = Field(..., min_length=1)
    context: str = ""
    kind: str | None = Field(
        None,
        pattern=r"^(decision|incident|pattern|constraint|conflict)$",
    )
    """Phase 9: optional refinery memory kind. When set, dedup is
    kind-scoped — a Decision and a Pattern with identical text are
    not considered duplicates."""


@router.post("/memory/check-duplicate")
def check_memory_duplicate(request: Request, body: _MemoryDedupRequest):
    """Check if a memory is a semantic duplicate of an existing one.

    Read-only — does NOT write or boost anything. Uses the same
    ``MemoryDedupHelper.find_existing_match()`` that the write path
    uses, but without calling ``boost_relevance``.

    Phase 9: honours the optional ``kind`` field for kind-scoped dedup.
    Legacy callers (only passing ``type``) keep today's behaviour
    exactly — the endpoint falls back to the ``type`` field and the
    underlying dedup helper when ``kind`` is absent.
    """
    memory_repo = request.app.state.memory_repo
    no_match = {"is_duplicate": False, "existing_id": None, "existing_description": None, "similarity": None}
    if memory_repo is None:
        return no_match

    # Phase 9 kind-scoped dedup. The kind field, when present, takes
    # precedence over the legacy type for scope — but we still thread
    # the type through for backward compat with the underlying helper.
    memory_type = body.kind or body.type
    match = memory_repo.check_duplicate(
        memory_type=memory_type,
        description=body.description,
        context=body.context,
    )
    if match is None:
        return no_match

    return {
        "is_duplicate": True,
        "existing_id": match.get("id"),
        "existing_description": match.get("description", ""),
        "similarity": round(match.get("score", 0), 3),
    }


# ── Requirements CRUD ─────────────────────────────────────────────────────────


class _SuggestedMemoryApplyPayload(BaseModel):
    """Phase 9 write-back payload — one suggested memory to save
    atomically with the requirement PATCH."""

    kind: str = Field(
        ...,
        pattern=r"^(decision|incident|pattern|constraint|conflict|hypothesis|anti_pattern)$",
    )
    summary: str = Field(..., min_length=1, max_length=500)
    body: str = Field("", max_length=20000)
    context: str = Field("", max_length=2000)
    source_requirement_id: str = Field("", max_length=128)
    source_role: str = Field("", max_length=64)
    rationale: str = Field("", max_length=4000)
    provenance_refinery_run_id: str = Field("", max_length=128)
    # Kind-specific optional fields
    decision_alternatives: list[str] | None = Field(None, max_length=20)
    incident_severity: str | None = Field(None, pattern=r"^(sev1|sev2|sev3)$")
    constraint_domain: str | None = Field(None, pattern=r"^(system|business)$")
    conflict_parties: list[str] | None = Field(None, max_length=20)
    conflict_resolution: str | None = None
    # HYPOTHESIS-specific
    hypothesis_verification_query: str | None = Field(None, max_length=2000)
    hypothesis_status: str | None = Field(None, pattern=r"^(open|verified|refuted)$")
    # ANTI_PATTERN-specific
    anti_pattern_alternative: str | None = Field(None, max_length=2000)
    anti_pattern_harm: str | None = Field(None, max_length=2000)


class _RequirementPatchWithMemories(RequirementPatchRequest):
    """Phase 9 extension — the operator's ``Apply to Graph`` can
    atomically write selected memories alongside the requirement
    update. Legacy callers that don't supply ``apply_memories`` get
    exactly today's behaviour."""

    apply_memories: list[_SuggestedMemoryApplyPayload] = Field(
        default_factory=list,
    )


@router.patch("/graph/requirements/{req_id}")
def patch_requirement(
    request: Request,
    req_id: str = Path(..., pattern=r"^[A-Za-z0-9_\-]{1,128}$"),
    patch: _RequirementPatchWithMemories = ...,
):
    """Update a requirement node (partial update, preserves ID).

    Write order:

    1. The requirement is upserted in its own transaction
       (``repo.upsert_requirement``). It either succeeds or 4xx is
       returned to the user; the user's primary action is never
       silently dropped.
    2. When ``apply_memories`` is non-empty, the listed memories
       commit together via ``record_refinery_memories_atomic`` — all
       of them or none — independent of step 1. Qdrant upserts run
       after the Neo4j commit; on Qdrant failure we best-effort
       delete the just-created Neo4j nodes so the stores don't
       diverge. A memory-write failure is logged + counted but never
       fails the requirement update from step 1.

    Atomicity guarantee is therefore "memories atomic with each
    other" — *not* "memories atomic with the requirement". Operators
    re-applying memories after a transient memory-write failure get
    the second-attempt benefit of dedup short-circuiting any
    duplicates from a partial first attempt.
    """
    from dark_factory.graph.repository import GraphRepository
    from dark_factory.models.domain import Priority, Requirement
    from dark_factory.metrics.prometheus import (
        observe_refinery_memory_audit,
        observe_refinery_memory_suggested,
        observe_refinery_memory_write_back_failure,
    )

    neo4j_client = request.app.state.neo4j_client
    repo = GraphRepository(neo4j_client)
    existing = repo.get_requirement(req_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Requirement {req_id} not found")

    # Merge patch fields onto existing
    merged = Requirement(
        id=existing.id,
        title=patch.title if patch.title is not None else existing.title,
        description=patch.description if patch.description is not None else existing.description,
        source_file=existing.source_file,
        priority=Priority(patch.priority) if patch.priority is not None else existing.priority,
        tags=patch.tags if patch.tags is not None else existing.tags,
    )
    repo.upsert_requirement(merged)

    # Atomic memory write-back via ``record_refinery_memories_atomic``:
    # the new memories all commit together (one Neo4j transaction) or
    # all roll back; Qdrant failures trigger a compensating delete.
    # Legacy kinds (pattern / incident via :Mistake) flow through the
    # existing per-memory path because their record methods aren't
    # covered by the refinery-atomic helper.
    saved_memory_ids: list[str] = []
    if patch.apply_memories:
        memory_repo = getattr(request.app.state, "memory_repo", None)
        if memory_repo is not None:
            atomic_specs: list[dict] = []
            legacy_mems: list[_SuggestedMemoryApplyPayload] = []
            for mem in patch.apply_memories:
                spec = _payload_to_atomic_spec(mem)
                if spec is not None:
                    atomic_specs.append(spec)
                else:
                    legacy_mems.append(mem)

            if atomic_specs:
                try:
                    ids = memory_repo.record_refinery_memories_atomic(atomic_specs)
                    saved_memory_ids.extend(ids)
                    for mem in patch.apply_memories:
                        if _payload_to_atomic_spec(mem) is not None:
                            observe_refinery_memory_audit(outcome="saved")
                            observe_refinery_memory_suggested(
                                kind=mem.kind,
                                source_role=mem.source_role or "unknown",
                            )
                except Exception:
                    observe_refinery_memory_write_back_failure()

            for mem in legacy_mems:
                try:
                    new_id = _apply_suggested_memory(memory_repo, mem)
                    if new_id:
                        saved_memory_ids.append(new_id)
                        observe_refinery_memory_audit(outcome="saved")
                        observe_refinery_memory_suggested(
                            kind=mem.kind,
                            source_role=mem.source_role or "unknown",
                        )
                except Exception:
                    observe_refinery_memory_write_back_failure()

    return {
        **merged.model_dump(),
        "saved_memory_ids": saved_memory_ids,
    }


def _payload_to_atomic_spec(
    mem: "_SuggestedMemoryApplyPayload",
) -> dict | None:
    """Translate an Apply payload into a ``record_refinery_memories_atomic``
    spec. Returns ``None`` for kinds not supported by the atomic helper
    (legacy ``pattern`` / ``incident`` flow through per-memory writes)."""

    if mem.kind == "decision":
        return {
            "memory_type": "decision",
            "summary": mem.summary, "body": mem.body,
            "source_role": mem.source_role,
            "source_requirement_id": mem.source_requirement_id,
            "rationale": mem.rationale,
            "provenance_refinery_run_id": mem.provenance_refinery_run_id,
            "kind_props": {
                "context": mem.context,
                "decision_alternatives": mem.decision_alternatives or [],
            },
        }
    if mem.kind == "constraint":
        return {
            "memory_type": "constraint",
            "summary": mem.summary, "body": mem.body,
            "source_role": mem.source_role,
            "source_requirement_id": mem.source_requirement_id,
            "rationale": mem.rationale,
            "provenance_refinery_run_id": mem.provenance_refinery_run_id,
            "kind_props": {
                "constraint_domain": mem.constraint_domain or "system",
                "applicability": "",
            },
        }
    if mem.kind == "conflict":
        return {
            "memory_type": "conflict",
            "summary": mem.summary, "body": mem.body,
            "source_role": "judge",
            "source_requirement_id": mem.source_requirement_id,
            "rationale": mem.rationale,
            "provenance_refinery_run_id": mem.provenance_refinery_run_id,
            "kind_props": {
                "conflict_parties": mem.conflict_parties or [],
                "conflict_resolution": mem.conflict_resolution or "",
                "cause": "disagreement",
            },
        }
    if mem.kind == "hypothesis":
        return {
            "memory_type": "hypothesis",
            "summary": mem.summary, "body": mem.body,
            "source_role": mem.source_role,
            "source_requirement_id": mem.source_requirement_id,
            "rationale": mem.rationale,
            "provenance_refinery_run_id": mem.provenance_refinery_run_id,
            "kind_props": {
                "verification_query": mem.hypothesis_verification_query or "",
                "status": mem.hypothesis_status or "open",
            },
        }
    if mem.kind == "anti_pattern":
        return {
            "memory_type": "anti_pattern",
            "summary": mem.summary, "body": mem.body,
            "source_role": mem.source_role,
            "source_requirement_id": mem.source_requirement_id,
            "rationale": mem.rationale,
            "provenance_refinery_run_id": mem.provenance_refinery_run_id,
            "kind_props": {
                "alternative": mem.anti_pattern_alternative or "",
                "harm": mem.anti_pattern_harm or "",
                "applicability": "",
            },
        }
    # Legacy kinds (pattern / incident) use per-memory path.
    return None


def _apply_suggested_memory(
    memory_repo,
    mem: _SuggestedMemoryApplyPayload,
) -> str | None:
    """Route a SuggestedMemory apply payload to the right MemoryRepository
    method based on ``kind``. Returns the new memory id."""

    if mem.kind == "decision":
        return memory_repo.record_decision(
            summary=mem.summary, body=mem.body, context=mem.context,
            source_requirement_id=mem.source_requirement_id,
            source_role=mem.source_role,
            decision_alternatives=mem.decision_alternatives,
            rationale=mem.rationale,
            provenance_refinery_run_id=mem.provenance_refinery_run_id,
        )
    if mem.kind == "constraint":
        return memory_repo.record_constraint(
            summary=mem.summary, body=mem.body,
            constraint_domain=mem.constraint_domain or "system",
            source_requirement_id=mem.source_requirement_id,
            source_role=mem.source_role, rationale=mem.rationale,
            provenance_refinery_run_id=mem.provenance_refinery_run_id,
        )
    if mem.kind == "conflict":
        return memory_repo.record_conflict(
            summary=mem.summary, body=mem.body,
            conflict_parties=mem.conflict_parties,
            conflict_resolution=mem.conflict_resolution,
            source_requirement_id=mem.source_requirement_id,
            rationale=mem.rationale,
            provenance_refinery_run_id=mem.provenance_refinery_run_id,
        )
    if mem.kind == "hypothesis":
        return memory_repo.record_hypothesis(
            summary=mem.summary, body=mem.body,
            verification_query=mem.hypothesis_verification_query or "",
            status=mem.hypothesis_status or "open",
            source_requirement_id=mem.source_requirement_id,
            source_role=mem.source_role,
            rationale=mem.rationale,
            provenance_refinery_run_id=mem.provenance_refinery_run_id,
        )
    if mem.kind == "anti_pattern":
        return memory_repo.record_anti_pattern(
            summary=mem.summary, body=mem.body,
            alternative=mem.anti_pattern_alternative or "",
            harm=mem.anti_pattern_harm or "",
            source_requirement_id=mem.source_requirement_id,
            source_role=mem.source_role,
            rationale=mem.rationale,
            provenance_refinery_run_id=mem.provenance_refinery_run_id,
        )
    if mem.kind == "incident":
        # Incident → :Mistake label with severity annotation on the
        # description. Uses the existing record_mistake path.
        return memory_repo.record_mistake(
            description=f"[{mem.incident_severity or 'sev3'}] {mem.summary}",
            error_type=f"incident:{mem.incident_severity or 'sev3'}",
            trigger_context=mem.body,
            source_feature=mem.source_requirement_id or "refinery",
            agent=mem.source_role or "refinery",
        )
    if mem.kind == "pattern":
        return memory_repo.record_pattern(
            description=mem.summary,
            context=mem.body,
            source_feature=mem.source_requirement_id or "refinery",
            agent=mem.source_role or "refinery",
        )
    return None


@router.get("/graph/requirements/export")
def export_requirements(request: Request):
    """Export all requirements as a downloadable JSON file."""
    from dark_factory.graph.repository import GraphRepository

    neo4j_client = request.app.state.neo4j_client
    repo = GraphRepository(neo4j_client)
    reqs = repo.get_all_requirements()
    payload = [r.model_dump() for r in reqs]
    return payload


# ── Run History ────────────────────────────────────────────────────────────────


@router.get("/history")
def get_history(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    """Return recent pipeline run history."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"runs": [], "message": "Memory system is disabled"}
    runs = memory_repo.get_run_history(limit=limit)
    return {"runs": runs}


@router.delete("/history/{run_id}")
def delete_run(request: Request, run_id: str = Path(...)):
    """Delete a run and all its linked data (episodes, evals, memories, files).

    Cannot delete a run that is currently in progress.
    """
    # Block deletion of the currently-running pipeline
    run_lock = getattr(request.app.state, "run_lock", None)
    if run_lock is not None and run_lock.locked():
        from dark_factory.agents.tools import get_current_run_id

        if get_current_run_id() == run_id:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete a run that is currently in progress.",
            )

    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        raise HTTPException(status_code=503, detail="Memory system is disabled")

    counts = memory_repo.delete_run(run_id=run_id)

    # Delete files from storage
    try:
        from dark_factory.storage.backend import RunStorage

        rs = RunStorage(_get_storage(request), run_id)
        rs.delete_run()
        counts["storage"] = 1
    except Exception:
        counts["storage"] = 0

    # Delete from Postgres metrics store
    try:
        rec = getattr(request.app.state, "metrics_recorder", None)
        if rec is not None and hasattr(rec, "delete_run"):
            rec.delete_run(run_id=run_id)
            counts["postgres"] = 1
    except Exception:
        counts["postgres"] = 0

    return {"deleted": run_id, "counts": counts}


# ── Memory Search ──────────────────────────────────────────────────────────────


@router.get("/memory/list")
def list_memory(
    request: Request,
    memory_type: Literal["all", "pattern", "mistake", "solution", "strategy"] = Query(
        default="all", alias="type"
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Browse all procedural memories ordered by relevance.

    Unlike ``/memory/search``, this endpoint requires no keywords — it
    returns the most relevant N memories so the dashboard can show what's
    available without forcing the user to guess search terms.
    """
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"results": [], "message": "Memory system is disabled", "total": 0}
    results = memory_repo.list_memories(memory_type=memory_type, limit=limit)
    return {"results": results, "total": len(results), "type": memory_type}


@router.get("/memory/search")
def search_memory(
    request: Request,
    keywords: str = Query(..., min_length=1),
    # H5/H7 fix: renamed from 'type' (shadows builtin), constrained to valid values
    memory_type: Literal["all", "pattern", "mistake", "solution", "strategy"] = Query(
        default="all", alias="type"
    ),
):
    """Search procedural memory by keywords."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"results": [], "message": "Memory system is disabled"}
    results = []
    search_map = {
        "pattern": memory_repo.search_patterns,
        "mistake": memory_repo.search_mistakes,
        "solution": memory_repo.search_solutions,
        "strategy": memory_repo.get_strategies,
    }
    targets = (
        list(search_map.items())
        if memory_type == "all"
        else [(memory_type, search_map[memory_type])]
    )
    for mtype, fn in targets:
        found = fn(keywords=keywords)
        results.extend([{"type": mtype, **m} for m in found])
    return {"results": results, "keywords": keywords}


@router.delete("/memory/{memory_id}")
def delete_memory(request: Request, memory_id: str = Path(...)):
    """Delete a single memory node and its Qdrant vector."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        raise HTTPException(status_code=503, detail="Memory system is disabled")

    # Detect the label from the ID prefix
    label = memory_repo._detect_label(memory_id)
    if label is None:
        raise HTTPException(status_code=400, detail=f"Unknown memory ID format: {memory_id}")
    if label == "Episode":
        raise HTTPException(status_code=400, detail="Use DELETE /api/history/{run_id} to delete episodes")

    deleted = memory_repo.delete_memory_node(memory_id)

    return {"deleted": memory_id, "count": deleted}


class MemoryImportItem(BaseModel):
    type: str
    description: str
    context: str = ""
    trigger_context: str = ""
    error_type: str = ""
    applicability: str = ""
    code_snippet: str = ""
    mistake_id: str = ""
    source_feature: str = ""
    agent: str = ""
    run_id: str = "import"


class MemoryImportRequest(BaseModel):
    memories: list[MemoryImportItem]


@router.post("/memory/import")
def import_memories(request: Request, body: MemoryImportRequest):
    """Bulk-import memories from a JSON export.

    Expects ``{"memories": [...]}``.  Each entry must have at least
    ``type`` and ``description``.  Returns counts of imported / skipped.
    """
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        raise HTTPException(status_code=503, detail="Memory system is disabled")

    items = body.memories
    imported = 0
    skipped = 0
    for item in items:
        mtype = item.type.lower()
        if not item.description:
            skipped += 1
            continue
        try:
            common = dict(
                description=item.description,
                source_feature=item.source_feature,
                agent=item.agent,
                run_id=item.run_id,
            )
            if mtype == "pattern":
                memory_repo.record_pattern(context=item.context, **common)
            elif mtype == "mistake":
                memory_repo.record_mistake(
                    error_type=item.error_type,
                    trigger_context=item.trigger_context,
                    **common,
                )
            elif mtype == "solution":
                memory_repo.record_solution(
                    mistake_id=item.mistake_id,
                    code_snippet=item.code_snippet,
                    **common,
                )
            elif mtype == "strategy":
                memory_repo.record_strategy(
                    applicability=item.applicability,
                    **common,
                )
            else:
                skipped += 1
                continue
            imported += 1
        except Exception as exc:
            log.warning("memory_import_item_failed", type=mtype, error=str(exc))
            skipped += 1

    return {"imported": imported, "skipped": skipped, "total": len(items)}


# ── Traceability Matrix ───────────────────────────────────────────────────────


@router.get("/traceability/{run_id}")
def get_traceability(
    request: Request,
    run_id: str = Path(...),
):
    """Return a requirements → specs → files → tests → eval scores matrix.

    Joins Neo4j (requirement→spec graph) with Postgres (artifacts,
    eval scores) for a single run to show full traceability.
    """
    neo4j_client = request.app.state.neo4j_client
    metrics_client = getattr(request.app.state, "metrics_client", None)

    # Neo4j: requirements and their implementing specs
    rows: list[dict] = []
    try:
        with neo4j_client.session() as session:
            result = session.run(
                """
                MATCH (r:Requirement)
                OPTIONAL MATCH (s:Spec)-[:IMPLEMENTS]->(r)
                RETURN r.id AS req_id, r.title AS req_title,
                       r.priority AS req_priority,
                       collect(DISTINCT {id: s.id, title: s.title,
                               capability: s.capability}) AS specs
                ORDER BY r.priority, r.id
                """
            )
            for record in result:
                specs_raw = record["specs"]
                # Filter out null specs (from OPTIONAL MATCH)
                specs_list = [
                    s for s in (specs_raw or [])
                    if isinstance(s, dict) and s.get("id") is not None
                ]
                rows.append({
                    "requirement": {
                        "id": record["req_id"],
                        "title": record["req_title"],
                        "priority": record["req_priority"],
                    },
                    "specs": specs_list,
                })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc

    # Postgres: artifacts and eval scores per spec for this run.
    # Artifacts are matched by spec_id when available, otherwise by
    # feature name (capability) since the write_file tool records
    # feature but not spec_id.
    spec_artifacts: dict[str, list[dict]] = {}
    feature_artifacts: dict[str, list[dict]] = {}
    spec_evals: dict[str, list[dict]] = {}

    if metrics_client is not None:
        try:
            with metrics_client.connection() as conn:
                with conn.cursor() as cur:
                    # Artifacts: fetch ALL for this run (some have spec_id, some have feature only)
                    cur.execute(
                        """
                        SELECT spec_id, feature, file_path, is_test, bytes_written
                        FROM artifact_writes
                        WHERE run_id = %s
                        ORDER BY file_path
                        """,
                        (run_id,),
                    )
                    for row in cur.fetchall():
                        entry = {
                            "file_path": row["file_path"],
                            "is_test": row["is_test"],
                            "bytes": row["bytes_written"],
                        }
                        sid = row.get("spec_id") or ""
                        feat = row.get("feature") or ""
                        fpath = row.get("file_path") or ""
                        if sid:
                            spec_artifacts.setdefault(sid, []).append(entry)
                        elif feat:
                            feature_artifacts.setdefault(feat, []).append(entry)
                        elif "/" in fpath:
                            # Infer feature from path prefix (e.g.
                            # "utilization-dashboard/main.py" → "utilization-dashboard")
                            prefix = fpath.split("/")[0]
                            feature_artifacts.setdefault(prefix, []).append(entry)

                    # Eval scores: try this run first, fall back to latest across all runs
                    cur.execute(
                        """
                        SELECT DISTINCT ON (spec_id, metric_name)
                               spec_id, metric_name, score, passed
                        FROM eval_metrics
                        WHERE run_id = %s AND spec_id IS NOT NULL
                              AND spec_id <> ''
                        ORDER BY spec_id, metric_name, timestamp DESC
                        """,
                        (run_id,),
                    )
                    eval_rows = cur.fetchall()
                    if not eval_rows:
                        # No evals for this run — use most recent across all runs
                        cur.execute(
                            """
                            SELECT DISTINCT ON (spec_id, metric_name)
                                   spec_id, metric_name, score, passed
                            FROM eval_metrics
                            WHERE spec_id IS NOT NULL AND spec_id <> ''
                            ORDER BY spec_id, metric_name, timestamp DESC
                            """
                        )
                        eval_rows = cur.fetchall()
                    for row in eval_rows:
                        spec_evals.setdefault(row["spec_id"], []).append({
                            "metric": row["metric_name"],
                            "score": row["score"],
                            "passed": row["passed"],
                        })
        except Exception as exc:
            log.warning("traceability_postgres_failed", error=str(exc))

    # Build a RunStorage for presigned URL generation
    rs = None
    try:
        from dark_factory.storage.backend import RunStorage
        rs = RunStorage(_get_storage(request), run_id)
    except Exception:
        pass

    def _file_entry(file_path: str) -> dict:
        """Build a file entry with an optional presigned S3 URL and
        a link to the file preview API."""
        entry: dict = {
            "path": file_path,
            "preview_url": f"/api/runs/{run_id}/file?path={file_path}",
        }
        if rs is not None:
            url = rs.presign_output(file_path)
            if url:
                entry["s3_url"] = url
        return entry

    # Assemble matrix
    matrix = []
    for row in rows:
        spec_entries = []
        for spec in row["specs"]:
            sid = spec["id"]
            cap = spec.get("capability") or ""
            # Try spec_id match first, then feature/capability match
            arts = spec_artifacts.get(sid, [])
            if not arts and cap:
                arts = feature_artifacts.get(cap, [])
            evals = spec_evals.get(sid, [])
            code_files = [_file_entry(a["file_path"]) for a in arts if not a.get("is_test")]
            test_files = [_file_entry(a["file_path"]) for a in arts if a.get("is_test")]
            all_passed = all(e["passed"] for e in evals) if evals else None
            spec_entries.append({
                **spec,
                "files": code_files,
                "test_files": test_files,
                "eval_scores": {e["metric"]: e["score"] for e in evals},
                "all_passed": all_passed,
            })
        overall = (
            "no_specs" if not spec_entries
            else "pass" if all(s.get("all_passed") for s in spec_entries)
            else "fail" if any(s.get("all_passed") is False for s in spec_entries)
            else "no_evals"
        )
        matrix.append({
            "requirement": row["requirement"],
            "specs": spec_entries,
            "overall_status": overall,
        })

    return {"run_id": run_id, "rows": matrix}


# ── Graph Topology ────────────────────────────────────────────────────────────


@router.get("/graph/topology/{run_id}")
def get_graph_topology(
    request: Request,
    run_id: str = Path(...),
):
    """Return nodes and edges for the run's dependency graph.

    Scoped to a run via its traceability data — only includes
    requirements and specs that are part of this run. Nodes are
    annotated with status (pass/fail/no_evals/no_artifacts) from
    the traceability matrix.
    """
    # Reuse the traceability endpoint to get the full matrix
    trace_resp = get_traceability(request, run_id=run_id)
    trace_rows = trace_resp.get("rows", []) if isinstance(trace_resp, dict) else trace_resp.body if hasattr(trace_resp, "body") else []
    # Handle both dict and JSONResponse
    if hasattr(trace_resp, "body"):
        import json as _json
        trace_rows = _json.loads(trace_resp.body).get("rows", [])
    elif isinstance(trace_resp, dict):
        trace_rows = trace_resp.get("rows", [])

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()

    for row in trace_rows:
        req = row.get("requirement", {})
        req_id = req.get("id", "")
        if not req_id or req_id in seen_ids:
            continue
        seen_ids.add(req_id)
        nodes.append({
            "id": req_id,
            "type": "requirement",
            "label": req.get("title") or req_id,
            "priority": req.get("priority"),
            "status": row.get("overall_status", "no_specs"),
        })

        for spec in row.get("specs", []):
            sid = spec.get("id", "")
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)

            # Determine spec status from traceability data
            has_files = bool(spec.get("files"))
            has_evals = bool(spec.get("eval_scores"))
            all_passed = spec.get("all_passed")
            spec_status = (
                "pass" if all_passed is True
                else "fail" if all_passed is False
                else "no_evals" if has_files and not has_evals
                else "no_artifacts" if not has_files
                else "unknown"
            )

            nodes.append({
                "id": sid,
                "type": "spec",
                "label": spec.get("title") or sid,
                "capability": spec.get("capability"),
                "status": spec_status,
                "file_count": len(spec.get("files", [])),
                "test_count": len(spec.get("test_files", [])),
            })

            # IMPLEMENTS edge
            edges.append({
                "id": f"impl-{sid}-{req_id}",
                "source": sid,
                "target": req_id,
                "type": "IMPLEMENTS",
            })

    # Add DEPENDS_ON edges from Neo4j (only between specs in this run)
    neo4j_client = request.app.state.neo4j_client
    spec_ids_in_run = [n["id"] for n in nodes if n["type"] == "spec"]
    if spec_ids_in_run:
        try:
            with neo4j_client.session() as session:
                for record in session.run(
                    """
                    MATCH (s:Spec)-[:DEPENDS_ON]->(d:Spec)
                    WHERE s.id IN $sids AND d.id IN $sids
                    RETURN s.id AS source, d.id AS target
                    """,
                    sids=spec_ids_in_run,
                ):
                    edges.append({
                        "id": f"dep-{record['source']}-{record['target']}",
                        "source": record["source"],
                        "target": record["target"],
                        "type": "DEPENDS_ON",
                    })
        except Exception:
            pass  # best-effort — graph renders without dep edges

    return {"run_id": run_id, "nodes": nodes, "edges": edges}


# ── Eval Scores ────────────────────────────────────────────────────────────────


@router.get("/eval")
def list_evals(
    request: Request,
    run_limit: int = Query(default=20, ge=1, le=100),
    run_id: str | None = Query(
        default=None,
        pattern=r"^[A-Za-z0-9_\-]{1,128}$",
        description=(
            "When set, filter the response to a single run by id. "
            "Used by the Run Detail popup's Evaluations screen so it "
            "doesn't have to fetch every run's evals just to display one."
        ),
    ),
):
    """Browse all eval results grouped by pipeline run.

    Returns a hierarchical structure of run → spec → attempts. With
    no ``run_id`` filter, returns the most recent ``run_limit`` runs.
    With a ``run_id``, returns just that one run (still wrapped in a
    ``runs: [...]`` array for response-shape consistency) — or an
    empty array if the run isn't in memory.
    """
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"runs": [], "message": "Memory system is disabled"}
    runs = memory_repo.list_evals_by_run(run_limit=run_limit)
    if run_id is not None:
        runs = [r for r in runs if r.get("run_id") == run_id]
    return {"runs": runs}


@router.get("/eval/{spec_id}")
def get_eval_history(
    request: Request,
    # C3 fix: regex constraint to prevent injection
    spec_id: str = Path(pattern=r"^[a-zA-Z0-9_. -]+$"),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Return eval score history for a spec."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"history": [], "message": "Memory system is disabled"}
    history = memory_repo.get_eval_history(spec_id=spec_id, limit=limit)
    return {"spec_id": spec_id, "history": history}


# ── Tunable Pipeline Settings ─────────────────────────────────────────────────


class PipelineSettingsResponse(BaseModel):
    """Tunable settings exposed by the Settings tab."""

    max_parallel_features: int
    max_parallel_specs: int
    max_spec_handoffs: int
    max_codegen_handoffs: int
    spec_eval_threshold: float
    enable_spec_decomposition: bool
    max_specs_per_requirement: int
    reuse_existing_specs: bool
    max_reconciliation_turns: int
    reconciliation_timeout_seconds: int
    requirement_dedup_threshold: float
    enable_e2e_validation: bool
    e2e_timeout_seconds: int
    e2e_browsers: list[str]
    enable_episodic_memory: bool
    memory_dedup_threshold: float

    # Editable model selections
    llm_model: str
    planning_model: str
    eval_model: str
    refinery_model: str
    refinery_reasoning_effort: str

    # Read-only metadata
    output_dir: str


class PipelineSettingsUpdate(BaseModel):
    """Partial update — only fields present in the request body are applied."""

    max_parallel_features: int | None = Field(default=None, ge=1, le=8)
    max_parallel_specs: int | None = Field(default=None, ge=1, le=8)
    max_spec_handoffs: int | None = Field(default=None, ge=1, le=10)
    max_codegen_handoffs: int | None = Field(default=None, ge=5, le=100)
    spec_eval_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_spec_decomposition: bool | None = None
    max_specs_per_requirement: int | None = Field(default=None, ge=1, le=32)
    reuse_existing_specs: bool | None = None
    max_reconciliation_turns: int | None = Field(default=None, ge=1, le=500)
    reconciliation_timeout_seconds: int | None = Field(
        default=None, ge=60, le=7200
    )
    requirement_dedup_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    enable_e2e_validation: bool | None = None
    e2e_timeout_seconds: int | None = Field(default=None, ge=60, le=7200)
    e2e_browsers: list[str] | None = Field(default=None, min_length=1)
    enable_episodic_memory: bool | None = None
    memory_dedup_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # Model name changes are validated as non-empty strings. We don't
    # enum-restrict them here — the frontend offers a curated dropdown
    # but operators can still pipe in any string via the API / a custom
    # override.
    llm_model: str | None = Field(default=None, min_length=1, max_length=128)
    planning_model: str | None = Field(default=None, min_length=1, max_length=128)
    eval_model: str | None = Field(default=None, min_length=1, max_length=128)
    refinery_model: str | None = Field(default=None, min_length=1, max_length=128)
    refinery_reasoning_effort: str | None = Field(
        default=None, pattern=r"^(low|medium|high|xhigh)$"
    )


class _SettingsPayloadDict(TypedDict):
    """L3 fix: typed shape for the settings GET/PATCH responses.

    Mirrors :class:`PipelineSettingsResponse` field-for-field so
    the dict returned by :func:`_settings_payload` cannot drift
    silently from the Pydantic schema the REST clients consume.
    Any new field added to ``PipelineSettingsResponse`` should
    also be added here; a TypeError will fire if they diverge.
    """

    max_parallel_features: int
    max_parallel_specs: int
    max_spec_handoffs: int
    max_codegen_handoffs: int
    spec_eval_threshold: float
    enable_spec_decomposition: bool
    max_specs_per_requirement: int
    reuse_existing_specs: bool
    max_reconciliation_turns: int
    reconciliation_timeout_seconds: int
    requirement_dedup_threshold: float
    enable_e2e_validation: bool
    e2e_timeout_seconds: int
    e2e_browsers: list[str]
    enable_episodic_memory: bool
    memory_dedup_threshold: float
    output_dir: str
    llm_model: str
    planning_model: str
    eval_model: str
    refinery_model: str
    refinery_reasoning_effort: str


def _settings_payload(settings: "Settings") -> _SettingsPayloadDict:
    # L11 fix: single source of truth for the eval model. Read via the
    # metrics-module getter so runtime Settings-tab changes are reflected
    # immediately (the old ``from ... import EVAL_MODEL`` pattern captured
    # the module global at import time and stayed stale across PATCHes).
    from dark_factory.evaluation.metrics import get_eval_model

    p = settings.pipeline
    return {
        "max_parallel_features": p.max_parallel_features,
        "max_parallel_specs": p.max_parallel_specs,
        "max_spec_handoffs": p.max_spec_handoffs,
        "max_codegen_handoffs": p.max_codegen_handoffs,
        "spec_eval_threshold": p.spec_eval_threshold,
        "enable_spec_decomposition": p.enable_spec_decomposition,
        "max_specs_per_requirement": p.max_specs_per_requirement,
        "reuse_existing_specs": p.reuse_existing_specs,
        "max_reconciliation_turns": p.max_reconciliation_turns,
        "reconciliation_timeout_seconds": p.reconciliation_timeout_seconds,
        "requirement_dedup_threshold": p.requirement_dedup_threshold,
        "enable_e2e_validation": p.enable_e2e_validation,
        "e2e_timeout_seconds": p.e2e_timeout_seconds,
        "e2e_browsers": list(p.e2e_browsers),
        "enable_episodic_memory": p.enable_episodic_memory,
        "memory_dedup_threshold": p.memory_dedup_threshold,
        "output_dir": p.output_dir,
        "llm_model": settings.llm.model,
        "planning_model": settings.model_routing.resolve("planner", settings.llm.model),
        "eval_model": get_eval_model(),
        "refinery_model": p.refinery_model,
        "refinery_reasoning_effort": p.refinery_reasoning_effort,
    }


@router.get("/settings", response_model=PipelineSettingsResponse)
def get_pipeline_settings(request: Request):
    """Return the current tunable pipeline settings."""
    return _settings_payload(request.app.state.settings)


@router.patch("/settings", response_model=PipelineSettingsResponse)
def update_pipeline_settings(request: Request, body: PipelineSettingsUpdate):
    """Update tunable pipeline settings at runtime.

    Only fields present in the request body are applied. Changes take effect
    on the **next** pipeline run — currently-running pipelines are not interrupted.

    Returns the new full settings payload after applying changes.
    """
    settings_obj = request.app.state.settings
    pipeline = settings_obj.pipeline
    updates: dict[str, object] = {}

    # Data-driven loop for simple pipeline fields (no side effects).
    _PIPELINE_FIELDS = [
        "max_parallel_features",
        "max_parallel_specs",
        "max_spec_handoffs",
        "max_codegen_handoffs",
        "spec_eval_threshold",
        "enable_spec_decomposition",
        "max_specs_per_requirement",
        "reuse_existing_specs",
        "max_reconciliation_turns",
        "reconciliation_timeout_seconds",
        "requirement_dedup_threshold",
        "enable_e2e_validation",
        "e2e_timeout_seconds",
        "enable_episodic_memory",
    ]

    try:
        for field in _PIPELINE_FIELDS:
            val = getattr(body, field, None)
            if val is not None:
                setattr(pipeline, field, val)
                updates[field] = val

        # Fields with side effects handled individually below.

        if body.e2e_browsers is not None:
            # Validate against the allowed engine set so an operator
            # can't persist a malformed browser list through the UI
            # that would then crash the next run's Playwright config.
            allowed = {"chromium", "firefox", "webkit"}
            cleaned = [
                b.strip().lower() for b in body.e2e_browsers if b and b.strip()
            ]
            invalid = [b for b in cleaned if b not in allowed]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid e2e_browsers entries: {invalid}. "
                        f"Allowed: {sorted(allowed)}"
                    ),
                )
            if not cleaned:
                raise HTTPException(
                    status_code=400,
                    detail="e2e_browsers must contain at least one browser",
                )
            pipeline.e2e_browsers = cleaned
            updates["e2e_browsers"] = cleaned

        if body.memory_dedup_threshold is not None:
            pipeline.memory_dedup_threshold = body.memory_dedup_threshold
            updates["memory_dedup_threshold"] = body.memory_dedup_threshold
            # Live-update the installed MemoryRepository so the new
            # threshold takes effect immediately rather than on next
            # app start. Tolerate a missing repo (memory disabled).
            repo = getattr(request.app.state, "memory_repo", None)
            if repo is not None:
                try:
                    repo.set_dedup_threshold(body.memory_dedup_threshold)
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning("memory_dedup_threshold_live_update_failed", error=str(exc))

        if body.llm_model is not None:
            settings_obj.llm.model = body.llm_model.strip()
            updates["llm_model"] = settings_obj.llm.model

        if body.planning_model is not None:
            model_name = body.planning_model.strip()
            routing = settings_obj.model_routing
            routing.planner = model_name
            routing.reviewer = model_name
            routing.spec = model_name
            updates["planning_model"] = model_name

        if body.eval_model is not None:
            new_eval = body.eval_model.strip()
            settings_obj.evaluation.eval_model = new_eval
            # Propagate to the metrics module global so the NEXT pipeline
            # run's GEval builders pick up the new model. Already-built
            # metrics in flight are unaffected (we explicitly document
            # that changes take effect on the next run).
            from dark_factory.evaluation.metrics import set_eval_model

            set_eval_model(new_eval)
            updates["eval_model"] = new_eval

        if body.refinery_model is not None:
            pipeline.refinery_model = body.refinery_model.strip()
            updates["refinery_model"] = pipeline.refinery_model

        if body.refinery_reasoning_effort is not None:
            pipeline.refinery_reasoning_effort = body.refinery_reasoning_effort.strip()
            updates["refinery_reasoning_effort"] = pipeline.refinery_reasoning_effort
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info("pipeline_settings_updated", **updates)
    return _settings_payload(request.app.state.settings)


# ── Health ─────────────────────────────────────────────────────────────────────


@router.get("/health")
def get_health(request: Request):
    """Return health status of all external services."""
    from dark_factory.ui.health import check_all

    settings = request.app.state.settings
    # M14 fix: reuse the shared neo4j_client / metrics_client instead of
    # opening new ones per request.
    raw = check_all(
        settings,
        neo4j_client=request.app.state.neo4j_client,
        metrics_client=getattr(request.app.state, "metrics_client", None),
    )
    return {
        service: {"ok": ok, "message": message}
        for service, (ok, message) in raw.items()
    }


# ── File Watcher ───────────────────────────────────────────────────────────────


@router.post("/watch/start")
def watch_start(request: Request):
    """Start the file watcher."""
    from dark_factory.ui.watcher import FileWatcher

    settings = request.app.state.settings
    watcher = request.app.state.watcher
    if watcher and watcher.is_running:
        return {"status": "already_running", "paths": watcher.paths}

    watcher = FileWatcher(
        paths=settings.watch.paths,
        debounce_seconds=settings.watch.debounce_seconds,
    )
    watcher.start()
    request.app.state.watcher = watcher
    log.info("watcher_started_via_api", paths=settings.watch.paths)
    return {"status": "started", "paths": settings.watch.paths}


@router.post("/watch/stop")
def watch_stop(request: Request):
    """Stop the file watcher."""
    watcher = request.app.state.watcher
    if watcher and watcher.is_running:
        watcher.stop()
        request.app.state.watcher = None
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/watch/status")
def watch_status(request: Request):
    """Return current watcher status."""
    watcher = request.app.state.watcher
    if watcher and watcher.is_running:
        last = watcher.last_event
        return {
            "running": True,
            "paths": watcher.paths,
            "last_event": (
                {"path": last.path, "type": last.event_type, "timestamp": last.timestamp}
                if last
                else None
            ),
        }
    return {"running": False}


@router.get("/watch/events")
async def watch_events(request: Request):
    """SSE stream of file watcher events with heartbeat and timeout."""
    max_duration = 300  # 5 minutes
    heartbeat_interval = 15  # seconds
    # M12 fix: poll_interval dropped from 1s → 0.1s so watcher events reach
    # the UI within ~100ms instead of ~1s. Heartbeat cadence is unchanged.
    poll_interval = 0.1

    async def generator():
        start = time.monotonic()
        last_heartbeat = start

        while True:
            now = time.monotonic()
            if now - start > max_duration:
                break
            if await request.is_disconnected():
                break

            watcher = request.app.state.watcher
            if watcher and watcher.is_running:
                events = watcher.drain_events()
                for ev in events:
                    payload = json.dumps(
                        {"path": ev.path, "type": ev.event_type, "timestamp": ev.timestamp}
                    )
                    yield f"data: {payload}\n\n"

            if now - last_heartbeat >= heartbeat_interval:
                yield ": keepalive\n\n"
                last_heartbeat = now

            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
