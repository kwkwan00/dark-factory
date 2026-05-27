"""SSE stream orchestrator for the refinery pipeline."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import structlog

from dark_factory.agents.tools import emit_progress
from dark_factory.api.refinery.gather import gather_run_context, ingest_requirements
from dark_factory.api.refinery.models import (
    ReconciliationReport,
    RefinedRequirement,
    RefineryResponse,
    SuggestedMemory,
)
from dark_factory.api.refinery.storage import generate_refinery_id, save_refinery_result

log = structlog.get_logger()


def _broadcast_to_agent_log(event: dict) -> None:
    """Fan a refinery SSE event out to the global progress broker so it
    appears on the Agent Log tab alongside swarm-pipeline events.

    Debate-graph events (those carrying a ``debate_event`` field) are
    named ``refinery_<debate_event>`` so each agent role gets its own
    filterable row in the agent log. Other refinery events (phase
    headers, completion counters) fall back to a generic
    ``refinery_phase`` name.
    """

    debate_event = event.get("debate_event")
    event_name = f"refinery_{debate_event}" if debate_event else "refinery_phase"
    # Drop fields the broker call would conflict with:
    #  - ``debate_event`` drives the event_name and is structural
    #  - ``event`` collides with ``emit_progress``'s first positional arg
    #    (the LLM helper stamps it in for the SSE-queue relay; the broker
    #    path doesn't need it since ``event_name`` is already explicit).
    payload = {
        k: v for k, v in event.items() if k not in ("debate_event", "event")
    }
    try:
        emit_progress(event_name, **payload)
    except Exception:  # pragma: no cover — defensive
        log.exception(
            "refinery_broker_broadcast_failed", broadcast_event=event_name,
        )


def _broadcast(event: dict) -> dict:
    """Broadcast to the agent log and return the event unchanged so
    ``yield _broadcast({...})`` does both in one line."""

    _broadcast_to_agent_log(event)
    return event


def _finish_bus(result_id: str) -> None:
    """Clear the cross-debate bus's per-run buffer. Idempotent and
    best-effort — long-lived processes call this on both success and
    crash/cancel paths to avoid leaking findings between runs."""

    try:
        from dark_factory.api.refinery.debate.cross_debate_bus import (
            get_global_bus,
        )

        get_global_bus().finish_run(result_id)
    except Exception:  # pragma: no cover — defensive
        pass


def _collect_risk_areas(errors: list[str], recon_report: Any) -> list[str]:
    """Merge per-requirement errors, cross-set coherence issues, the
    Judge's explicit ``risk_areas`` list, and any short-circuited
    debates surfaced as ``unresolved_debates`` into one list of risk
    descriptions for ``RefineryResponse.risk_areas``."""

    out: list[str] = list(errors)
    coherence = getattr(recon_report, "coherence_issues", None) or []
    out.extend(
        f"Coherence: {ci.get('issue', '')}"
        for ci in coherence
        if isinstance(ci, dict)
    )
    risk_areas = getattr(recon_report, "risk_areas", None) or []
    out.extend(r for r in risk_areas if isinstance(r, str) and r)
    unresolved = getattr(recon_report, "unresolved_debates", None) or []
    for ud in unresolved:
        if not isinstance(ud, dict):
            continue
        req_id = ud.get("requirement_id") or "?"
        reason = ud.get("reason") or "panel did not converge"
        out.append(f"Unresolved debate ({req_id}): {reason}")
    return out


# Concurrent per-requirement debate workers.
MAX_CONCURRENT_AGENTS = 5


def _get_refinery_config() -> tuple[str, str, int, int]:
    """Resolve refinery config once. Returns (model, effort, max_turns, timeout).

    Falls through to a fresh ``PipelineConfig()`` on settings-load failure
    so the field-level defaults stay the single source of truth — no
    hardcoded model IDs in the orchestrator.
    """
    from dark_factory.config import PipelineConfig, load_settings

    try:
        p = load_settings().pipeline
    except Exception:
        log.warning("refinery_settings_load_failed_using_defaults")
        p = PipelineConfig()
    return (
        p.refinery_model,
        p.refinery_reasoning_effort,
        p.refinery_max_turns,
        p.refinery_timeout_seconds,
    )


def _run_in_thread_with_progress(
    fn: Any,
    loop: asyncio.AbstractEventLoop,
    progress_queue: asyncio.Queue[dict | None],
) -> None:
    """Run ``fn`` in the current thread, then push a None sentinel.

    If ``fn`` raises, an error event is pushed before the sentinel so
    the SSE consumer can surface it to the client.
    """
    try:
        fn()
    except Exception as exc:
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            {"phase": "error", "message": f"Thread failed: {exc}"},
        )
    finally:
        loop.call_soon_threadsafe(progress_queue.put_nowait, None)


async def run_refinery_stream(
    request: Any,
    run_id: str | None,
    input_path: str | None,
    *,
    direct: dict | None = None,
    resume_run_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """SSE generator for the refinery pipeline.

    Phases:
    1. Gather data (traceability + episodes + memories OR document ingest
       OR a single directly-submitted requirement)
    2. Per-requirement refinement (concurrent deep agents)
    3. Reconciliation (single agent reviews the full set)
    4. Merge, persist, and yield done event

    Third input mode: when ``direct`` is supplied (``{"title",
    "description", "priority", "tags"}``), the refinery wraps that
    single requirement in a minimal run_context with
    ``source_mode="direct"`` and runs the debate on exactly one item.

    Resume: when ``resume_run_id`` is supplied, Phase 1 is replaced by
    a registry lookup that hydrates the original input (source mode +
    requirements list + run_context) from Postgres. Phase 2 then skips
    every requirement whose debate has a cached completion, replaying
    the cached result, and runs the rest from scratch. Phase 3 + 4 run
    fresh on the merged set so the cross-req review reflects the
    completed state.
    """
    from dark_factory.api.refinery.resume import ResumeRegistry
    from dark_factory.ui.helpers import get_settings

    settings = get_settings()
    model, reasoning_effort, max_turns, timeout_secs = _get_refinery_config()
    # Each per-requirement agent gets a reasonable turn budget
    per_req_turns = max(15, max_turns // 2)

    metrics_client = getattr(request.app.state, "metrics_client", None)
    resume_registry = ResumeRegistry(metrics_client)

    # Track whether the run reached the success path. The finally block
    # below uses this to mark crashed / cancelled runs as ``failed`` and
    # to clean up the cross-debate bus buffer that ``finish_run``
    # otherwise only clears on success. Without this guard, a browser
    # disconnect mid-Phase-2 leaves an in_progress Postgres row, a leaked
    # bus buffer (until 60-min TTL eviction), and worker threads still
    # billing LLM tokens for nobody.
    _completed_normally = False
    result_id: str | None = None

    try:
        requirements: list[dict] = []
        run_context: dict | None = None
        cached_completions: dict[str, dict] = {}
        if resume_run_id:
            # Resume path: hydrate state from Postgres and skip Phase 1.
            result_id = resume_run_id
            saved = resume_registry.load_run(resume_run_id)
            if saved is None:
                yield _broadcast({
                    "phase": "error",
                    "message": f"Resume failed: run {resume_run_id} unknown to "
                                f"the registry (Postgres disabled or row purged)",
                })
                return
            if saved.get("status") == "completed":
                yield _broadcast({
                    "phase": "error",
                    "message": f"Resume failed: run {resume_run_id} is already "
                                f"completed; load it via GET /api/refinery/{resume_run_id}",
                })
                return
            snapshot = saved.get("input_snapshot") or {}
            source_mode = saved.get("source_mode") or "run"
            run_id = snapshot.get("run_id") or saved.get("source_run_id")
            input_path = snapshot.get("input_path")
            direct = snapshot.get("direct")
            requirements = list(snapshot.get("requirements") or [])
            run_context = snapshot.get("run_context")
            cached_completions = resume_registry.load_completed(resume_run_id)
            yield _broadcast({
                "phase": "gathering",
                "step": "resuming",
                "requirement_count": len(requirements),
                "resumed_completed": len(cached_completions),
            })
        else:
            result_id = generate_refinery_id()
            if direct:
                source_mode = "direct"
            elif run_id:
                source_mode = "run"
            else:
                source_mode = "upload"

        # ── Phase 1: Gather data ─────────────────────────────────────
        # Skipped when resuming — the requirements list, run_context, and
        # source mode are already hydrated from the registry above.
        if resume_run_id:
            pass
        elif run_id:
            yield _broadcast({"phase": "gathering", "step": "traceability"})
            run_context = gather_run_context(request, run_id)

            yield _broadcast({"phase": "gathering", "step": "requirements"})
            trace = run_context.get("traceability")
            if trace and isinstance(trace, dict):
                for row in trace.get("rows", []):
                    req = row.get("requirement", {})
                    if req.get("id"):
                        requirements.append({
                            "id": req["id"],
                            "title": req.get("title") or "",
                            "description": req.get("description") or "",
                            "priority": req.get("priority") or "medium",
                            "tags": req.get("tags") or [],
                            "source_file": req.get("source_file") or "",
                        })
            seen: set[str] = set()
            deduped: list[dict] = []
            for r in requirements:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    deduped.append(r)
            requirements = deduped

            # Enrich from Neo4j if descriptions are missing
            if requirements and any(not r.get("description") for r in requirements):
                try:
                    from dark_factory.graph.repository import GraphRepository
                    repo = GraphRepository(request.app.state.neo4j_client)
                    all_reqs = repo.get_all_requirements()
                    req_map = {r.id: r for r in all_reqs}
                    for req_dict in requirements:
                        full = req_map.get(req_dict["id"])
                        if full:
                            req_dict["title"] = full.title
                            req_dict["description"] = full.description
                            req_dict["priority"] = full.priority.value
                            req_dict["tags"] = full.tags
                            req_dict["source_file"] = full.source_file
                except Exception as exc:
                    log.debug("refinery_neo4j_enrich_failed", error=str(exc))

        elif input_path:
            yield _broadcast({"phase": "gathering", "step": "ingesting documents"})
            try:
                requirements = ingest_requirements(input_path, settings)
            except Exception as exc:
                yield _broadcast({"phase": "error", "message": f"Failed to ingest documents: {exc}"})
                return
        elif direct:
            # Direct-input mode — one user-typed requirement.
            yield _broadcast({"phase": "gathering", "step": "direct_input"})
            import uuid as _uuid
            req_id = f"req-direct-{_uuid.uuid4().hex[:8]}"
            requirements = [{
                "id": req_id,
                "title": direct.get("title", "") or "",
                "description": direct.get("description", "") or "",
                "priority": direct.get("priority") or "medium",
                "tags": list(direct.get("tags") or []),
                "source_file": "",
            }]
            # No historical run — minimal run_context with source_mode so
            # downstream consumers (role-filter policy audit, memory producers)
            # can tell it was a direct submission.
            run_context = {"source_mode": "direct", "run_id": None}
        else:
            yield _broadcast({
                "phase": "error",
                "message": "One of run_id, input_path, or direct is required",
            })
            return

        if not requirements:
            yield _broadcast({"phase": "error", "message": "No requirements found to refine"})
            return

        yield _broadcast({"phase": "gathering", "step": "complete", "requirement_count": len(requirements)})

        # Register the run on the resume registry. On a fresh run this
        # creates the row + persists the input snapshot. On resume the
        # row already exists; ``start`` is idempotent on the run id.
        if not resume_run_id:
            resume_registry.start(
                refinery_run_id=result_id,
                source_mode=source_mode,
                source_run_id=run_id,
                input_snapshot={
                    "source_mode": source_mode,
                    "run_id": run_id,
                    "input_path": input_path,
                    "direct": direct,
                    "requirements": requirements,
                    "run_context": run_context,
                },
            )

        # ── Phase 2: Per-requirement refinement ───────────────────────
        pending = [
            r for r in requirements
            if r.get("id") not in cached_completions
        ]
        yield _broadcast({
            "phase": "refining",
            "message": (
                f"Refining {len(pending)} requirements "
                f"({MAX_CONCURRENT_AGENTS} concurrent)"
                + (f" — {len(cached_completions)} replayed from cache" if cached_completions else "")
                + "..."
            ),
            "pending": len(pending),
            "cached": len(cached_completions),
        })

        with tempfile.TemporaryDirectory(prefix="refinery-") as tmpdir:
            (Path(tmpdir) / "INPUT_REQUIREMENTS.json").write_text(
                json.dumps(requirements, indent=2, default=str)
            )

            progress_queue: asyncio.Queue[dict | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _on_progress(event: dict) -> None:
                # Fan the event out to the Agent Log tab's global broker
                # whenever the debate graph emits a per-agent event. Non-
                # debate events (generic "refining" messages) use a single
                # catch-all name so operators can still filter them.
                _broadcast_to_agent_log(event)
                loop.call_soon_threadsafe(progress_queue.put_nowait, event)

            all_refined: list[RefinedRequirement] = []
            all_memories: list[SuggestedMemory] = []
            errors: list[str] = []
            completed_count = 0
            total_reqs = len(requirements)
            start = time.monotonic()

            # Replay cached completions first so the SSE consumer sees
            # consistent progress events even when a resumed run has zero
            # requirements left to actually refine.
            for req in requirements if cached_completions else ():
                cached = cached_completions.get(req.get("id"))
                if cached is None:
                    continue
                try:
                    refined = RefinedRequirement.model_validate(cached["refined"])
                    all_refined.append(refined)
                    for mem in cached.get("suggested_memories") or []:
                        try:
                            all_memories.append(SuggestedMemory.model_validate(mem))
                        except Exception:
                            continue
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning(
                        "refinery_resume_replay_failed",
                        req_id=req.get("id"), error=str(exc),
                    )
                    continue
                completed_count += 1
                _on_progress({
                    "phase": "refining",
                    "turn": completed_count,
                    "max_turns": total_reqs,
                    "message": f"Replayed {completed_count}/{total_reqs} — {req.get('id', '?')} (cached)",
                    "from_cache": True,
                })

            # Each requirement is refined by the LangGraph adversarial-panel
            # debate. The runner returns ``(RefinedRequirement, memories,
            # output)`` so the thread-pool + progress machinery doesn't
            # need to know about the panel internals.
            from dark_factory.api.refinery.debate.runner import run_phase2_generator
            from dark_factory.api.refinery.research.learning import (
                load_adjusted_trust_weights,
            )
            from dark_factory.api.refinery.judge_calibration import load_judge_calibration
            from dark_factory.api.refinery.role_weighting import (
                flatten_role_dim_weights,
                flatten_role_weights,
                load_role_dimension_weights,
                load_role_weights,
            )
            from dark_factory.api.refinery.archetype_rounds import (
                effective_max_rounds_for,
                load_archetype_rounds,
            )
            from dark_factory.api.refinery.rule_severity import load_rule_severities
            from dark_factory.api.refinery.roles._shared.llm import (
                install_progress_callback,
            )

            # Provider learning: compute the adjusted per-tier trust
            # weights once at run start so every concurrent debate uses
            # the same snapshot. Falls back to the configured base
            # weights when Postgres is unavailable.
            _base_weights = {
                int(k): float(v)
                for k, v in (settings.pipeline.refinery_research_tier_trust_weights or {}).items()
            } or {0: 1.0, 1: 0.95, 2: 0.90, 3: 0.80, 4: 0.60, 5: 0.30}
            _adjusted_weights, _ = load_adjusted_trust_weights(
                metrics_client=metrics_client,
                base_weights=_base_weights,
                window_days=30,
            )

            # Adaptive role weighting: compute one snapshot of per-role
            # multipliers from the trailing window of disposition rows. Empty
            # dict means identity (×1.0 for every role) — early-life systems
            # and Postgres outages both land here. The map is stamped into
            # every debate's evidence_bag so JudgeRole.defend can render it
            # into the synthesis prompt.
            _role_weights = load_role_weights(
                metrics_client=metrics_client,
                window_days=30,
                severity="blocker",
            )

            _role_dim_weights = load_role_dimension_weights(
                metrics_client=metrics_client,
                window_days=30,
                severity="blocker",
            )
            _flat_role_dim_weights = flatten_role_dim_weights(_role_dim_weights)

            # Judge confidence calibration: one snapshot of the overall-
            # threshold multiplier from the trailing window of operator
            # apply/dismiss decisions. 1.0 = no signal yet → identity; the
            # composition pipeline reads this off the evidence bag and
            # scales its effective threshold by it.
            _judge_calibration = load_judge_calibration(
                metrics_client=metrics_client,
                window_days=30,
                score_split=settings.pipeline.refinery_judge_overall_threshold,
            )

            _base_max_rounds = int(settings.pipeline.refinery_debate_max_rounds)
            _archetype_rounds = load_archetype_rounds(
                metrics_client=metrics_client,
                base_max_rounds=_base_max_rounds,
                window_days=30,
            )

            _rule_severities = load_rule_severities(
                metrics_client=metrics_client,
                window_days=30,
            )
            _flat_rule_severities = _rule_severities.to_flat()

            # Disposition sink — fed by the synthesize node every time a
            # rebuttal is produced. Best-effort write to Postgres; the run
            # never fails because the sink is unhappy.
            _disposition_repo = None
            if metrics_client is not None:
                try:
                    from dark_factory.metrics.refinery_repository import (
                        RefineryMetricsRepository,
                    )
                    _disposition_repo = RefineryMetricsRepository(metrics_client)
                except Exception:  # pragma: no cover — defensive
                    _disposition_repo = None

            def _disposition_sink(rows: list[dict]) -> None:
                if _disposition_repo is None or not rows:
                    return
                try:
                    _disposition_repo.record_critique_dispositions(rows)
                except Exception:  # pragma: no cover — defensive
                    pass

            def _research_agent_factory():
                """Build a per-debate ResearchAgent stamped with the
                run's learning-adjusted tier trust weights. Each worker
                thread gets its own instance so tier budgets don't leak
                between concurrent debates."""

                from dark_factory.api.refinery.research import (
                    build_research_agent_from_config,
                )

                return build_research_agent_from_config(
                    settings.pipeline, trust_weights=_adjusted_weights,
                )

            # Stamp role weights + judge-calibration multiplier onto the
            # run_context so JudgeRole.defend / score read them off the
            # evidence bag. Flat shapes — formatters don't need to know
            # about RoleWeight or JudgeCalibration.
            _flat_role_weights = flatten_role_weights(_role_weights)
            # Pre-render the calibration block once per run via the Judge
            # role's verb — invariant across rounds of every debate in
            # the run, and the orchestrator stays out of the role's
            # prompt module (Parnas). One RoleRegistry is built per run
            # and reused across all later role lookups (Phase-3 review,
            # Phase-5 planning).
            from dark_factory.api.refinery.roles.registry import RoleRegistry
            _role_registry = RoleRegistry(settings.pipeline)
            _calibration_block = _role_registry.get("judge").prepare_calibration(
                role_weights=_flat_role_weights or None,
                role_dim_weights=_flat_role_dim_weights or None,
            )

            if (
                _flat_role_weights
                or _flat_role_dim_weights
                or _judge_calibration.multiplier != 1.0
                or _flat_rule_severities
            ):
                run_context = dict(run_context or {})
                if _flat_role_weights:
                    run_context["role_weights"] = _flat_role_weights
                if _flat_role_dim_weights:
                    run_context["role_dim_weights"] = _flat_role_dim_weights
                if _calibration_block:
                    run_context["calibration_block"] = _calibration_block
                if _judge_calibration.multiplier != 1.0:
                    run_context["judge_threshold_multiplier"] = (
                        _judge_calibration.multiplier
                    )
                if _flat_rule_severities:
                    run_context["rule_severity_overrides"] = (
                        _flat_rule_severities
                    )

            _source_mode = source_mode or "unknown"

            def _run_one(req: dict) -> tuple[RefinedRequirement | None, list[SuggestedMemory], str]:
                # Install the on_progress callback into a thread-local slot
                # so the shared LLM helper inside each role can echo
                # ``refinery_llm_*`` events back into the refinery SSE
                # stream alongside the broker. Worker thread + thread-local
                # → no cross-talk between concurrent requirements.
                effective_max_rounds = effective_max_rounds_for(
                    priority=req.get("priority"),
                    source_mode=_source_mode,
                    tags=list(req.get("tags") or []),
                    archetypes=_archetype_rounds,
                    base_max_rounds=_base_max_rounds,
                )
                with install_progress_callback(_on_progress):
                    return run_phase2_generator(
                        req=req,
                        all_requirements=requirements,
                        run_context=run_context,
                        tmpdir=tmpdir,
                        max_turns=per_req_turns,
                        timeout_seconds=float(timeout_secs),
                        model=model,
                        reasoning_effort=reasoning_effort,
                        on_progress=_on_progress,
                        config=settings.pipeline,
                        refinery_run_id=result_id,
                        research_agent_factory=_research_agent_factory,
                        disposition_sink=_disposition_sink,
                        max_rounds_override=effective_max_rounds,
                    )

            def _run_all() -> None:
                nonlocal completed_count
                with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_AGENTS) as executor:
                    futures = {executor.submit(_run_one, req): req for req in pending}
                    for future in as_completed(futures):
                        req = futures[future]
                        req_id = req.get("id", "?")
                        try:
                            refined, memories, _output = future.result()
                            if refined:
                                all_refined.append(refined)
                                # Cache the completion so a subsequent
                                # resume can replay it without re-running.
                                try:
                                    resume_registry.cache_debate(
                                        refinery_run_id=result_id,
                                        requirement_id=req_id,
                                        convergence_status=getattr(
                                            refined, "convergence_status", None,
                                        ) or "converged",
                                        refined_payload=refined.model_dump(mode="json"),
                                        suggested_memories=[
                                            m.model_dump(mode="json") for m in memories
                                        ],
                                    )
                                except Exception:  # pragma: no cover — defensive
                                    log.exception(
                                        "refinery_resume_cache_failed",
                                        req_id=req_id,
                                    )
                            all_memories.extend(memories)
                        except Exception as exc:
                            errors.append(f"{req_id}: {exc}")
                            log.warning("refinery_req_failed", req_id=req_id, error=str(exc))

                        completed_count += 1
                        _on_progress({
                            "phase": "refining",
                            "turn": completed_count,
                            "max_turns": total_reqs,
                            "message": f"Completed {completed_count}/{total_reqs} — {req_id}",
                        })

            pool_thread = threading.Thread(
                target=_run_in_thread_with_progress,
                args=(_run_all, loop, progress_queue),
                daemon=True,
            )
            pool_thread.start()

            while True:
                event = await progress_queue.get()
                if event is None:
                    break
                yield event

            pool_thread.join(timeout=10)

            # ── Phase 3: Cross-requirement review (Judge.review_set) ───
            # The LLM-backed cross-set review runs through JudgeRole.review_set
            # which formats a CrossReviewReport prompt, single-shots through
            # the shared LLM helper, and falls through to a no-op stub when
            # no LLM is wired. The structural patcher then applies duplicate /
            # coherence / relationship / priority / spec-overlap mutations
            # back onto the refined set.
            from dark_factory.api.refinery.contracts import (
                CrossReviewReport,
                Draft,
            )
            from dark_factory.api.refinery.judge import apply_cross_review_report
            from dark_factory.api.refinery.roles.registry import RoleRegistry
            # ``install_progress_callback`` already imported above for Phase 2.

            recon_report: CrossReviewReport | ReconciliationReport = CrossReviewReport()
            if len(all_refined) > 1:
                yield _broadcast({
                    "phase": "reconciling",
                    "message": f"Reviewing {len(all_refined)} requirements as a set...",
                })

                recon_queue: asyncio.Queue[dict | None] = asyncio.Queue()
                recon_result: dict = {"refined": all_refined, "report": recon_report}

                def _on_recon_progress(event: dict) -> None:
                    _broadcast_to_agent_log(event)
                    loop.call_soon_threadsafe(recon_queue.put_nowait, event)

                def _run_recon() -> None:
                    # Convert each RefinedRequirement → Draft so the Judge's
                    # signature is satisfied. We only need the fields the
                    # review_set prompt serialises (id / title / description
                    # / priority / tags / suggested_specs / relationships).
                    drafts: list[Draft] = []
                    for r in all_refined:
                        drafts.append(Draft(
                            requirement_id=r.id,
                            title=r.title,
                            description=r.description,
                            priority=r.priority,
                            tags=list(r.tags),
                            suggested_specs=[],
                            relationships=[],
                            produced_by="reviewer",
                            iteration=0,
                        ))
                    run_context_dict = run_context or {
                        "run_id": run_id, "source_mode": source_mode,
                    }
                    _on_recon_progress({
                        "phase": "reconciling",
                        "message": f"Judge reviewing {len(drafts)} refined requirements as a set...",
                    })

                    # Multi-pass set-level Judge: a bounded critique-and-
                    # ratify loop around the Judge.review_set call. With
                    # ``refinery_set_review_max_rounds=1`` (the default)
                    # this collapses to a single-shot call equivalent to
                    # the prior behaviour; ramping up is opt-in.
                    from dark_factory.api.refinery.set_review import (
                        run_set_review,
                    )
                    with install_progress_callback(_on_recon_progress):
                        try:
                            report = run_set_review(
                                refined_set=drafts,
                                run_context=run_context_dict,
                                registry=_role_registry,
                                max_rounds=int(
                                    getattr(
                                        settings.pipeline,
                                        "refinery_set_review_max_rounds",
                                        1,
                                    ),
                                ),
                                base_model=model,
                                reasoning_effort=reasoning_effort,
                                on_progress=_on_recon_progress,
                            )
                        except Exception as exc:
                            log.warning(
                                "refinery_set_review_failed", error=str(exc),
                            )
                            report = CrossReviewReport(
                                summary=f"Cross-req review failed: {exc}",
                            )

                    # Patch the refined set in place.
                    patched = apply_cross_review_report(all_refined, report)

                    _on_recon_progress({
                        "phase": "reconciling",
                        "message": (
                            f"Review complete — {len(report.duplicate_pairs)} duplicates, "
                            f"{len(report.coherence_issues)} coherence issues, "
                            f"{len(report.relationship_fixes)} relationship fixes, "
                            f"{len(report.priority_changes)} priority changes."
                        ),
                    })

                    recon_result["refined"] = patched
                    recon_result["report"] = report

                recon_thread = threading.Thread(
                    target=_run_in_thread_with_progress,
                    args=(_run_recon, loop, recon_queue),
                    daemon=True,
                )
                recon_thread.start()

                while True:
                    event = await recon_queue.get()
                    if event is None:
                        break
                    yield event

                recon_thread.join(timeout=10)
                all_refined = recon_result["refined"]
                recon_report = recon_result["report"]

            duration = time.monotonic() - start

            # ── Phase 4: Merge, persist, yield ────────────────────────
            modified = [r for r in all_refined if r.changes]
            unchanged = [r for r in all_refined if not r.changes]

            recon_summary = ""
            if recon_report.summary:
                recon_summary = f" Reconciliation: {recon_report.summary}"

            pass_summaries = [
                f"Per-requirement analysis with {per_req_turns} turns/agent, "
                f"{MAX_CONCURRENT_AGENTS} concurrent.",
            ]
            if recon_report.summary:
                recon_stats = (
                    f"{len(recon_report.duplicate_pairs)} duplicates, "
                    f"{len(recon_report.coherence_issues)} coherence issues, "
                    f"{len(recon_report.relationship_fixes)} relationship fixes, "
                    f"{len(recon_report.priority_changes)} priority changes."
                )
                pass_summaries.append(f"Reconciliation: {recon_stats} {recon_report.summary}")

            response = RefineryResponse(
                summary=(
                    f"Refined {len(requirements)} requirements using {MAX_CONCURRENT_AGENTS} "
                    f"concurrent agents. {len(modified)} modified, {len(unchanged)} unchanged."
                    + (f" {len(errors)} errors." if errors else "")
                    + recon_summary
                ),
                pass_summaries=pass_summaries,
                refined_requirements=all_refined,
                suggested_memories=all_memories,
                new_relationships_count=sum(len(r.relationships) for r in all_refined),
                requirements_modified_count=len(modified),
                requirements_unchanged_count=len(unchanged),
                source_run_id=run_id,
                methodology=(
                    f"Each requirement was refined by a six-seat adversarial "
                    f"panel (model: {model}, effort: {reasoning_effort}); up to "
                    f"{MAX_CONCURRENT_AGENTS} debates ran concurrently. Once "
                    f"every per-requirement debate finished, the Judge ran a "
                    f"single cross-set review over the merged refined set, "
                    f"checking for duplicates, coherence issues, relationship "
                    f"gaps, priority inversions, spec overlaps, and "
                    f"set-level dimension scores. Unresolved debates flow "
                    f"through to the response's risk areas."
                ),
                evidence_summary=(
                    f"Processed {len(requirements)} requirements in {duration:.0f}s."
                    + (f" Errors: {'; '.join(errors)}" if errors else "")
                ),
                risk_areas=_collect_risk_areas(errors, recon_report),
            )

            save_refinery_result(result_id, response, duration, source_mode)

            # Best-effort: write each requirement's Debate Episode to the
            # Episode memory store (Neo4j + Qdrant) so future debates can
            # recall how the panel resolved similar requirements.
            try:
                from dark_factory.api.refinery.episode import (
                    DebateEpisode,
                    write_debate_episode_to_memory,
                )

                memory_repo = getattr(request.app.state, "memory_repo", None)
                vector_repo = getattr(request.app.state, "vector_repo", None)

                if memory_repo is not None or vector_repo is not None:
                    written = 0
                    for r in all_refined:
                        ep_dict = (r.debate or {}).get("episode") if isinstance(r.debate, dict) else None
                        if not isinstance(ep_dict, dict):
                            continue
                        try:
                            ep = DebateEpisode.model_validate(ep_dict)
                        except Exception:
                            continue
                        if write_debate_episode_to_memory(
                            ep, memory_repo=memory_repo, vector_repo=vector_repo,
                        ):
                            written += 1
                    if written:
                        log.info(
                            "refinery_episodes_persisted_to_memory",
                            result_id=result_id, count=written,
                        )
            except Exception as exc:  # pragma: no cover — best-effort
                log.warning(
                    "refinery_episode_memory_persist_failed",
                    result_id=result_id, error=str(exc),
                )

            log.info(
                "refinery_completed",
                result_id=result_id,
                total=len(requirements),
                modified=len(modified),
                unchanged=len(unchanged),
                errors=len(errors),
                duration=round(duration, 1),
            )

            # ── Phase 5: Goal generation ──────────────────────────────
            # The Planner reads the completed refinement and proposes
            # additional requirements that look load-bearing but absent.
            # Operators approve / dismiss individually; nothing auto-runs.
            # Failures are non-fatal — the run still completes.
            if len(all_refined) >= 1:
                try:
                    from dark_factory.api.refinery.contracts import (
                        Draft as _Draft,
                        CrossReviewReport as _CrossReviewReport,
                    )

                    yield _broadcast({
                        "phase": "planning",
                        "message": "Looking for adjacent requirements that may be missing...",
                    })

                    if _role_registry.has("planner"):
                        drafts_for_planner = [
                            _Draft(
                                requirement_id=r.id,
                                title=r.title,
                                description=r.description,
                                priority=r.priority,
                                tags=list(r.tags),
                                suggested_specs=[],
                                relationships=[],
                                produced_by="planner-input",
                                iteration=0,
                            )
                            for r in all_refined
                        ]
                        cross_review_for_planner = (
                            recon_report
                            if isinstance(recon_report, _CrossReviewReport)
                            else _CrossReviewReport()
                        )
                        planner = _role_registry.get("planner")
                        proposals = planner.plan(
                            refined_set=drafts_for_planner,
                            cross_review=cross_review_for_planner,
                            run_context=run_context,
                        )
                        yield _broadcast({
                            "phase": "planning",
                            "step": "complete",
                            "proposals_count": len(proposals.proposals),
                            "summary": proposals.summary,
                        })
                        yield _broadcast({
                            "phase": "proposed_additions",
                            "result_id": result_id,
                            "data": proposals.model_dump(),
                        })
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning("refinery_planner_phase_failed", error=str(exc))

            # Broadcast a compact "done" signal to the agent log; the full
            # response payload stays on the refinery SSE stream where the UI
            # needs it to render the results view.
            _broadcast_to_agent_log({
                "phase": "done",
                "result_id": result_id,
                "duration_seconds": round(duration, 1),
                "total": len(requirements),
                "modified": len(modified),
                "unchanged": len(unchanged),
                "errors": len(errors),
            })
            # Mark the run completed on the resume registry so a subsequent
            # POST /api/refinery/resume/{result_id} short-circuits with a
            # 'already completed' error instead of re-running anything.
            resume_registry.mark_status(result_id, "completed")
            _finish_bus(result_id)
            yield {
                "phase": "done",
                "result_id": result_id,
                "data": response.model_dump(),
                "duration_seconds": round(duration, 1),
            }
            _completed_normally = True
    finally:
        # Cleanup for crashed / cancelled runs: mark the run as
        # failed and clear the cross-debate bus buffer that
        # finish_run otherwise only clears on success. Idempotent —
        # mark_status on a 'completed' row is a no-op via the
        # UPDATE guard. Without this finally, a browser disconnect
        # mid-Phase-2 leaves an in_progress Postgres row + a leaked
        # bus buffer (until 60-min TTL eviction).
        if not _completed_normally and result_id:
            try:
                resume_registry.mark_status(result_id, "failed")
            except Exception:  # pragma: no cover — defensive
                pass
            _finish_bus(result_id)
