"""Bridge between pipeline execution and the AG-UI Server-Sent Events stream."""

from __future__ import annotations

import asyncio
import json
import os as _os
import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import structlog
from ag_ui.core import (
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder

from dark_factory.config import Settings

log = structlog.get_logger()


class _ApiKeyOverride:
    """Per-run API key override that redacts under introspection.

    The function parameters carrying the keys are plain ``str`` (they
    have to be — the request body is JSON). Storing them as plain
    strings in the generator frame is risky: any logging path that
    captures locals via ``exc_info=True`` or
    ``traceback.format_exception`` can leak the value.

    This helper wraps each secret in :class:`pydantic.SecretStr` so
    ``repr()`` returns ``SecretStr('**********')`` instead of the raw
    value, and ``str()`` on the wrapper still redacts. The raw value
    is only dereferenced inside :meth:`apply` / :meth:`restore`, where
    it lives for a single statement before being discarded.

    Saved previous values are also wrapped so the restore path never
    materialises them as plain strings in a traceback-visible frame.
    """

    __slots__ = (
        "_anthropic",
        "_openai",
        "_prev_anthropic",
        "_prev_openai",
        "_anthropic_was_set",
        "_openai_was_set",
    )

    def __init__(
        self,
        *,
        anthropic: str | None,
        openai: str | None,
    ) -> None:
        from pydantic import SecretStr

        # Capture non-empty overrides as SecretStr wrappers. Empty
        # strings are treated as "no override" — matches the caller's
        # existing contract.
        self._anthropic = SecretStr(anthropic) if anthropic else None
        self._openai = SecretStr(openai) if openai else None
        self._prev_anthropic: Any = None
        self._prev_openai: Any = None
        self._anthropic_was_set = False
        self._openai_was_set = False

    def apply(self) -> None:
        """Install the overrides into the process environment.

        Captures the previous env values as ``SecretStr`` wrappers
        so a later traceback walk over this object never sees plain
        text, even for the previously-installed keys.
        """
        from pydantic import SecretStr

        if self._anthropic is not None:
            self._anthropic_was_set = True
            prev = _os.environ.get("ANTHROPIC_API_KEY")
            self._prev_anthropic = SecretStr(prev) if prev else None
            _os.environ["ANTHROPIC_API_KEY"] = self._anthropic.get_secret_value()
        if self._openai is not None:
            self._openai_was_set = True
            prev = _os.environ.get("OPENAI_API_KEY")
            self._prev_openai = SecretStr(prev) if prev else None
            _os.environ["OPENAI_API_KEY"] = self._openai.get_secret_value()

    def restore(self) -> None:
        """Restore the previous environment variables.

        Always safe to call multiple times — uses the ``_was_set``
        guards to skip the restore for overrides that were never
        installed.
        """
        if self._anthropic_was_set:
            if self._prev_anthropic is None:
                _os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                _os.environ["ANTHROPIC_API_KEY"] = self._prev_anthropic.get_secret_value()
        if self._openai_was_set:
            if self._prev_openai is None:
                _os.environ.pop("OPENAI_API_KEY", None)
            else:
                _os.environ["OPENAI_API_KEY"] = self._prev_openai.get_secret_value()

    def __repr__(self) -> str:
        # Explicit repr so even if a traceback formatter prints the
        # object, the secret values don't leak.
        return (
            "<_ApiKeyOverride anthropic=***redacted*** "
            "openai=***redacted***>"
        )


def _text_events(encoder: EventEncoder, text: str) -> list[str]:
    """Return AG-UI-encoded SSE strings for a single assistant progress message."""
    msg_id = str(uuid4())
    return [
        encoder.encode(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=msg_id,
                role="assistant",
            )
        ),
        encoder.encode(
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=msg_id,
                delta=text,
            )
        ),
        encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=msg_id,
            )
        ),
    ]


def _reflect_on_reconciliation(
    result: object,
    attempt: int,
    max_attempts: int,
) -> dict | None:
    """Call the reflection LLM to decide whether to retry reconciliation."""
    try:
        import json as _json

        from dark_factory.agents.tools import _resolve_deep_model
        from dark_factory.llm.anthropic import AnthropicClient
        from dark_factory.prompts import get_prompt

        status = getattr(result, "status", "unknown")
        summary = getattr(result, "summary", "")
        agent_output = getattr(result, "agent_output", "")

        prompt = get_prompt("reconciliation_reflection", "user").format(
            status=status,
            summary=summary,
            attempt=attempt,
            max_attempts=max_attempts,
            agent_output=agent_output[-4000:] if agent_output else "",
        )
        system = get_prompt("reconciliation_reflection", "system")

        model = _resolve_deep_model("deep_analysis")
        client = AnthropicClient(model=model) if model else AnthropicClient()
        raw = client.complete(prompt, system=system, timeout_seconds=30)

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return _json.loads(text)
    except Exception as exc:
        log.warning("reconciliation_reflection_failed", error=str(exc))
        return None


def _translate_progress(
    encoder: EventEncoder,
    progress: dict,
    feature_step_ids: dict[str, str],
    last_agent_per_feature: dict[str, str],
):
    """Translate a swarm progress event dict into AG-UI SSE chunks.

    H3 fix: this is a **synchronous** generator — it performs no awaits
    and returning a sync generator is cheaper and clearer. Callers can
    iterate with a regular ``for``.

    Emits:
    - ``feature_started`` → nested ``StepStartedEvent`` for the feature
    - ``feature_completed`` / ``feature_skipped`` → ``StepFinishedEvent`` + text summary
    - ``agent_active`` → text message naming the active agent (deduped)
    - ``layer_started`` / ``layer_completed`` → text messages
    """
    event = progress.get("event", "")
    unknown = "(unknown)"  # L2 fix: better placeholder

    if event == "layer_started":
        layer = progress.get("layer", unknown)
        total = progress.get("total_layers", unknown)
        features = progress.get("features", [])
        text = f"Layer {layer}/{total}: running {len(features)} feature(s): {', '.join(features) or '—'}"
        yield from _text_events(encoder, text)

    elif event == "layer_completed":
        yield from _text_events(encoder, f"Layer {progress.get('layer', unknown)} complete")

    elif event == "feature_started":
        feature = progress.get("feature", unknown)
        sid = str(uuid4())
        feature_step_ids[feature] = sid
        yield encoder.encode(
            StepStartedEvent(
                type=EventType.STEP_STARTED,
                step_name=f"Feature: {feature}",
                step_id=sid,
            )
        )
        yield from _text_events(
            encoder,
            f"Starting feature `{feature}` ({progress.get('spec_count', 0)} spec(s))",
        )

    elif event == "feature_completed":
        feature = progress.get("feature", unknown)
        status = progress.get("status", unknown)
        icon = {"success": "✓", "error": "✕", "skipped": "⏭"}.get(status, "•")
        artifacts = progress.get("artifacts", 0)
        tests = progress.get("tests", 0)
        summary = f"{icon} Feature `{feature}` {status} — {artifacts} artifact(s), {tests} test(s)"
        if progress.get("error"):
            summary += f" — {progress['error']}"
        yield from _text_events(encoder, summary)
        sid = feature_step_ids.pop(feature, None)
        if sid:
            yield encoder.encode(
                StepFinishedEvent(
                    type=EventType.STEP_FINISHED,
                    step_name=f"Feature: {feature}",
                    step_id=sid,
                )
            )
        last_agent_per_feature.pop(feature, None)

    elif event == "feature_skipped":
        feature = progress.get("feature", unknown)
        reason = progress.get("reason", "")
        yield from _text_events(encoder, f"⏭ Feature `{feature}` skipped — {reason}")

    elif event == "agent_active":
        feature = progress.get("feature", unknown)
        agent = progress.get("agent", unknown)
        # Dedupe: only emit when the active agent changes for this feature
        if last_agent_per_feature.get(feature) != agent:
            last_agent_per_feature[feature] = agent
            messages = progress.get("messages", 0)
            yield from _text_events(
                encoder, f"  `{feature}` → **{agent}** (handoff #{messages})"
            )

    elif event == "agent_decision":
        feature = progress.get("feature", unknown)
        agent = progress.get("agent", unknown)
        text = progress.get("text", "")
        # Render decision text on its own line; truncate to keep step output compact
        snippet = (text[:200] + "…") if len(text) > 200 else text
        yield from _text_events(
            encoder,
            f"    💭 `{feature}` **{agent}**: {snippet}",
        )

    elif event == "agent_handoff":
        feature = progress.get("feature", unknown)
        from_agent = progress.get("from_agent", unknown)
        to_agent = progress.get("to_agent", unknown)
        yield from _text_events(
            encoder,
            f"    ↪ `{feature}` handoff: **{from_agent}** → **{to_agent}**",
        )

    elif event == "tool_call":
        feature = progress.get("feature", unknown)
        agent = progress.get("agent", unknown)
        tool = progress.get("tool", unknown)
        args_preview = progress.get("args_preview", "")
        suffix = f" `{args_preview}`" if args_preview else ""
        yield from _text_events(
            encoder,
            f"    🔧 `{feature}` {agent} → tool `{tool}`{suffix}",
        )

    elif event == "tool_result":
        feature = progress.get("feature", unknown)
        tool = progress.get("tool", unknown)
        preview = progress.get("result_preview", "")
        # Truncate preview to keep messages compact
        snippet = (preview[:80] + "…") if len(preview) > 80 else preview
        yield from _text_events(
            encoder,
            f"    ← `{feature}` {tool} → {snippet}" if snippet else f"    ← `{feature}` {tool} done",
        )

    elif event == "spec_plan_started":
        title = progress.get("requirement_title", unknown)
        yield from _text_events(
            encoder, f"  🧩 Planning sub-specs for `{title}`…"
        )

    elif event == "spec_plan_completed":
        title = progress.get("requirement_title", unknown)
        n = progress.get("sub_spec_count", 0)
        titles = progress.get("titles", []) or []
        preview = ", ".join(titles[:5]) + ("…" if len(titles) > 5 else "")
        suffix = f": {preview}" if preview else ""
        yield from _text_events(
            encoder,
            f"  ✓ Planned {n} sub-spec(s) for `{title}`{suffix}",
        )

    elif event == "spec_plan_failed":
        title = progress.get("requirement_title") or progress.get("requirement_id", unknown)
        err = progress.get("error", "")
        yield from _text_events(
            encoder,
            f"  ⚠ Spec planning failed for `{title}` — {err} (falling back to single spec)",
        )

    elif event == "spec_plan_resolved":
        req_id = progress.get("requirement_id", unknown)
        resolved = progress.get("resolved", 0)
        unresolved = progress.get("unresolved", 0)
        tail = f" ({unresolved} unresolved)" if unresolved else ""
        yield from _text_events(
            encoder,
            f"  🔗 Resolved {resolved} sub-spec dep(s) for `{req_id}`{tail}",
        )

    elif event == "pipeline_cancelled":
        reason = progress.get("reason", "user_requested")
        yield from _text_events(
            encoder, f"⛔ Pipeline cancelled ({reason})"
        )

    elif event == "spec_gen_layer_started":
        total = progress.get("total", 0)
        parallel = progress.get("parallel", 1)
        planned = progress.get("planned_sub_specs", total)
        decomposition = bool(progress.get("decomposition_enabled", False))
        if decomposition and planned != total:
            text = (
                f"Generating {planned} sub-spec(s) from {total} requirement(s) "
                f"with {parallel} worker(s) in parallel"
            )
        else:
            text = (
                f"Generating {total} spec(s) with {parallel} worker(s) in parallel"
            )
        yield from _text_events(encoder, text)

    elif event == "spec_gen_started":
        title = progress.get("requirement_title", unknown)
        idx = progress.get("index", 0) + 1
        total = progress.get("total", 0)
        yield from _text_events(
            encoder, f"  [{idx}/{total}] Generating spec for `{title}`..."
        )

    elif event == "eval_rubric":
        req_title = progress.get("requirement_title", unknown)
        attempt = progress.get("attempt", "?")
        max_h = progress.get("max_handoffs", "?")
        avg = progress.get("avg_score", 0.0)
        threshold = progress.get("threshold", 0.0)
        metrics = progress.get("metrics", [])

        header = (
            f"    📊 `{req_title}` eval rubric (attempt {attempt}/{max_h}) — "
            f"avg {avg:.2f} (threshold {threshold:.2f})"
        )
        yield from _text_events(encoder, header)

        for m in metrics:
            name = m.get("name", "?")
            score = m.get("score", 0.0)
            passed = m.get("passed", False)
            reason = m.get("reason", "")
            marker = "✓" if passed else "✕"
            line = f"        {marker} {name}: {score:.2f}"
            if reason:
                snippet = (reason[:120] + "…") if len(reason) > 120 else reason
                line += f" — {snippet}"
            yield from _text_events(encoder, line)

    elif event == "spec_handoff":
        req_title = progress.get("requirement_title", unknown)
        attempt = progress.get("attempt", "?")
        max_h = progress.get("max_handoffs", "?")
        score = progress.get("score", 0.0)
        threshold = progress.get("threshold", 0.0)
        role = progress.get("role", "?")
        marker = "✓" if isinstance(score, (int, float)) and score >= threshold else "↻"
        yield from _text_events(
            encoder,
            f"    {marker} `{req_title}` handoff {attempt}/{max_h} ({role}) — score {score:.2f} (threshold {threshold:.2f})",
        )

    elif event == "spec_gen_completed":
        title = progress.get("spec_title", unknown)
        spec_id = progress.get("spec_id", unknown)
        final_score = progress.get("final_score", 0.0)
        attempts = progress.get("attempts", 1)
        yield from _text_events(
            encoder,
            f"  ✓ Spec generated: `{title}` (`{spec_id}`) — final score {final_score:.2f} after {attempts} attempt(s)",
        )

    elif event == "spec_gen_failed":
        req_id = progress.get("requirement_id", unknown)
        err = progress.get("error", "")
        yield from _text_events(
            encoder, f"  ✕ Spec generation failed for `{req_id}` — {err}"
        )

    elif event == "spec_gen_layer_completed":
        total = progress.get("total", 0)
        failed = progress.get("failed", 0)
        msg = f"Spec generation complete: {total} spec(s)"
        if failed:
            msg += f", {failed} failed"
        yield from _text_events(encoder, msg)


async def run_pipeline_stream(
    settings: Settings,
    requirements_path: str,
    thread_id: str,
    run_id: str,
    accept: str | None = None,
    memory_repo: object | None = None,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> AsyncIterator[str]:
    """Async generator that runs the full pipeline and yields AG-UI SSE strings.

    Each pipeline phase emits StepStarted/StepFinished events with progress
    TextMessage events in between. The final result is emitted as a StateSnapshot.

    If ``memory_repo`` is provided, a Run node is created in Neo4j memory at
    the start of the pipeline (so it appears in Run History immediately) and
    is updated as phases progress. The orchestrator will reuse this run_id
    rather than creating a duplicate Run node.

    ``anthropic_api_key`` / ``openai_api_key`` are optional per-run overrides.
    When provided, they are installed as ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
    environment variables for the duration of this run and restored to the
    previous values on completion. Safe because the route serialises runs
    via ``app.state.run_lock``.
    """
    from dark_factory.agents.cancellation import (
        PipelineCancelled,
        is_cancelled,
        raise_if_cancelled,
        reset_cancel,
    )

    # Reset the cancellation flag at the top of every run so a cancel
    # requested mid-previous-run can't leak forward. Concurrent runs are
    # blocked by ``app.state.run_lock``, so no race.
    reset_cancel()

    # ── Install optional per-run API key overrides (process env) ────────────
    #
    # The function parameters ``anthropic_api_key`` / ``openai_api_key``
    # arrive as plain ``str`` (they have to — the HTTP request body is
    # just JSON). Storing them in plain-string locals for the life of
    # this generator is risky because any exception path that logs
    # ``exc_info=True`` or ``traceback.format_exception(exc)`` can
    # capture the frame's locals and leak the key value into logs.
    #
    # H1 hardening: wrap the overrides + saved previous values in the
    # ``_ApiKeyOverride`` helper below, which stores each secret as a
    # ``SecretStr`` so ``repr()`` redacts it and traceback formatters
    # never see the raw value. The function parameters are deleted
    # immediately after the helper captures them so the raw strings
    # don't linger in the generator frame.
    _overrides = _ApiKeyOverride(
        anthropic=anthropic_api_key,
        openai=openai_api_key,
    )
    # Clear the raw function parameters now that the helper owns them.
    # This ensures a traceback walk over this frame cannot see the
    # plain-text keys — only the SecretStr wrappers inside the helper.
    anthropic_api_key = None  # type: ignore[assignment]
    openai_api_key = None  # type: ignore[assignment]
    _overrides.apply()

    def _restore_api_keys() -> None:
        _overrides.restore()

    encoder = EventEncoder(accept=accept)

    yield encoder.encode(
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id,
        )
    )

    neo4j_client = None
    pipeline_start_time = time.time()

    # Generate a run ID unconditionally so storage sync always works.
    # If memory is enabled, create_run returns a Neo4j-backed ID;
    # otherwise fall back to a timestamp-based ID.
    from datetime import datetime

    pipeline_run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4]}"

    from dark_factory.agents.tools import set_current_run_id

    set_current_run_id(pipeline_run_id)

    # ── Create the Run History entry IMMEDIATELY ─────────────────────────────
    # This creates a Neo4j Run node in 'running' status before Phase 1 even
    # starts, so the entry is visible in the Run History tab right away.
    if memory_repo is not None and settings.memory.enabled:
        try:
            pipeline_run_id = await asyncio.to_thread(
                lambda: memory_repo.create_run(spec_count=0, feature_count=0)
            )
            set_current_run_id(pipeline_run_id)
            log.info("run_history_entry_created_early", run_id=pipeline_run_id)
        except Exception as exc:
            log.warning("create_run_early_failed", error=str(exc))
            # Keep the fallback run ID — don't reset to ""

    # Create the RunStorage for durable persistence (local or S3).
    # pipeline_run_id is always set (either from Neo4j or timestamp fallback).
    from dark_factory.storage.backend import RunStorage, get_storage

    run_storage: RunStorage | None = None
    try:
        run_storage = RunStorage(
            get_storage(local_root=Path(settings.pipeline.output_dir)),
            pipeline_run_id,
        )
    except Exception as exc:
        log.warning("run_storage_init_failed", run_id=pipeline_run_id, error=str(exc))

    # Stamp the run start into the Postgres metrics store (if enabled) and
    # push the run_id into the recorder so every subsequent progress event
    # is automatically scoped to this run. Also bump the Prometheus
    # pipeline_runs_total counter and running_pipelines gauge.
    from dark_factory.agents import tools as _tools_mod_early

    try:
        from dark_factory.metrics.prometheus import observe_pipeline_run_start

        observe_pipeline_run_start(run_id=pipeline_run_id or None)
    except Exception:  # pragma: no cover — defensive
        pass

    metrics_recorder = _tools_mod_early._metrics_recorder
    if metrics_recorder is not None and pipeline_run_id:
        metrics_recorder.set_run_id(pipeline_run_id)
        metrics_recorder.record_pipeline_run_start(
            run_id=pipeline_run_id,
            spec_count=0,
            feature_count=0,
            metadata={
                "requirements_path": requirements_path,
                "thread_id": thread_id,
                "ag_ui_run_id": run_id,
            },
        )

    try:
        from dark_factory.graph.client import Neo4jClient
        from dark_factory.graph.repository import GraphRepository
        from dark_factory.models.domain import PipelineContext
        from dark_factory.stages.graph import GraphStage
        from dark_factory.stages.ingest import IngestStage
        from dark_factory.stages.spec import SpecStage
        from dark_factory.ui.helpers import build_llm

        routing = settings.model_routing
        default_model = settings.llm.model
        ingest_model = routing.resolve("ingest", default_model)
        spec_model = routing.resolve("spec", default_model)
        # Build per-stage LLM clients (same instance if models match).
        llm_ingest = build_llm(settings, model_override=ingest_model)
        llm_spec = build_llm(settings, model_override=spec_model) if spec_model != ingest_model else llm_ingest
        llm = llm_ingest  # default for other uses
        neo4j_client = Neo4jClient(settings.neo4j)
        repo = GraphRepository(neo4j_client)
        ctx = PipelineContext(input_path=requirements_path)

        # Clear the broker history at the START of the run so the Agent Logs
        # tab gets a fresh slate, and so spec_gen_* / agent_active events
        # emitted by Phase 2 (Spec Generation) and Phase 4 (Swarm) all
        # accumulate in history together for late subscribers.
        from dark_factory.agents import tools as _tools_mod

        if _tools_mod._progress_broker is not None:
            _tools_mod._progress_broker.clear_history()

        # ── Phase 1: Ingest ───────────────────────────────────────────────────
        raise_if_cancelled()
        step_id = str(uuid4())
        yield encoder.encode(
            StepStartedEvent(type=EventType.STEP_STARTED, step_name="Ingest", step_id=step_id)
        )
        for ev in _text_events(encoder, f"Ingesting requirements from `{requirements_path}`..."):
            yield ev

        # Build the embedding callable for the in-ingest semantic dedup
        # pass. We instantiate a fresh EmbeddingService per run so that
        # any per-run OPENAI_API_KEY override (installed above) is
        # picked up by the OpenAI client. Failures during construction
        # (missing key, import error, ...) fall back to ``None`` which
        # makes the IngestStage skip dedup — logged as a warning but
        # never fatal, so a transient outage can't block the pipeline.
        ingest_embed_fn = None
        try:
            from dark_factory.vector.embeddings import EmbeddingService

            _embedder = EmbeddingService(
                model=settings.qdrant.embedding_model,
            )
            ingest_embed_fn = _embedder.embed_batch
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "ingest_dedup_embedder_unavailable",
                error=str(exc),
            )

        # Pass the LLM so large documents get split into granular
        # requirements, and the embedding function so the stage
        # semantically dedupes cross-document near-duplicates before
        # the Spec stage runs. Both are optional — the stage handles
        # ``None`` for either gracefully.
        ingest_stage = IngestStage(
            llm=llm_ingest,
            embed_fn=ingest_embed_fn,
            dedup_threshold=settings.pipeline.requirement_dedup_threshold,
        )
        ctx = await asyncio.to_thread(ingest_stage.run, ctx)
        raise_if_cancelled()
        log.info("ingest_done", requirements=len(ctx.requirements))

        # Surface dedup results in the Agent Logs stream so the
        # operator can see exactly which requirements were merged. A
        # big cross-document dedup (e.g. 8 meeting notes + a Word doc
        # collapsing from 40 → 24 requirements) shows up here as a
        # line per merged cluster and a total count.
        # Defensive isinstance check so MagicMock'd IngestStage mocks in
        # the bridge unit tests don't accidentally pass the truthiness
        # check below — a bare ``is not None`` would hit a MagicMock
        # and crash on ``dropped_count > 0``.
        from dark_factory.stages.dedup import DedupeResult as _DedupeResult

        dedup = ingest_stage.last_dedup_result
        if isinstance(dedup, _DedupeResult) and dedup.dropped_count > 0:
            from dark_factory.agents.tools import emit_progress as _emit_dedup

            _emit_dedup(
                "requirements_deduped",
                input_count=dedup.dropped_count + len(ctx.requirements),
                output_count=len(ctx.requirements),
                dropped=dedup.dropped_count,
                groups=len(dedup.groups),
                threshold=dedup.threshold,
            )
            for ev in _text_events(
                encoder,
                f"Semantic dedup: collapsed {dedup.dropped_count} duplicate "
                f"requirement(s) across {len(dedup.groups)} cluster(s) at "
                f"threshold {dedup.threshold:.2f}",
            ):
                yield ev
            # Compact preview of what was merged — cap at 5 groups so
            # a giant dedup doesn't flood the logs.
            for g in dedup.groups[:5]:
                preview = ", ".join(g.merged_titles[:3])
                if len(g.merged_titles) > 3:
                    preview += f", +{len(g.merged_titles) - 3} more"
                for ev in _text_events(
                    encoder,
                    f"  • kept `{g.canonical_title}` (merged: {preview})",
                ):
                    yield ev
            if len(dedup.groups) > 5:
                for ev in _text_events(
                    encoder,
                    f"  … and {len(dedup.groups) - 5} more cluster(s) — see logs",
                ):
                    yield ev

        for ev in _text_events(encoder, f"Ingested {len(ctx.requirements)} requirements"):
            yield ev
        yield encoder.encode(
            StepFinishedEvent(type=EventType.STEP_FINISHED, step_name="Ingest", step_id=step_id)
        )

        # Sync uploaded input files to durable storage
        if run_storage is not None:
            try:
                input_path = Path(requirements_path)
                if input_path.is_dir():
                    await asyncio.to_thread(
                        run_storage.sync_input_from_local, input_path
                    )
                elif input_path.is_file():
                    await asyncio.to_thread(
                        run_storage.upload_input_from_local,
                        input_path,
                        input_path.name,
                    )
            except Exception as exc:
                log.warning("storage_input_sync_failed", error=str(exc))

        # Sync parsed requirements to durable storage
        if run_storage is not None and ctx.requirements:
            try:
                def _sync_requirements():
                    for req in ctx.requirements:
                        run_storage.write_requirement(
                            f"{req.id}.json", req.model_dump_json(indent=2)
                        )
                await asyncio.to_thread(_sync_requirements)
            except Exception as exc:
                log.warning("storage_requirements_sync_failed", error=str(exc))

        # ── Phase 2: Spec Generation ──────────────────────────────────────────
        step_id = str(uuid4())
        yield encoder.encode(
            StepStartedEvent(
                type=EventType.STEP_STARTED, step_name="Spec Generation", step_id=step_id
            )
        )
        for ev in _text_events(
            encoder, f"Generating specs from {len(ctx.requirements)} requirements..."
        ):
            yield ev

        spec_stage = SpecStage(
            llm=llm_spec,
            max_parallel=settings.pipeline.max_parallel_specs,
            max_handoffs=settings.pipeline.max_spec_handoffs,
            eval_threshold=settings.pipeline.spec_eval_threshold,
            enable_decomposition=settings.pipeline.enable_spec_decomposition,
            max_specs_per_requirement=settings.pipeline.max_specs_per_requirement,
            graph_repo=repo,
            reuse_existing_specs=settings.pipeline.reuse_existing_specs,
        )
        ctx = await asyncio.to_thread(spec_stage.run, ctx)
        raise_if_cancelled()
        log.info("spec_gen_done", specs=len(ctx.specs))

        # Bump the spec_count on the Run History entry now that we know it
        if pipeline_run_id and memory_repo is not None:
            try:
                await asyncio.to_thread(
                    lambda: memory_repo.update_run_counts(
                        run_id=pipeline_run_id, spec_count=len(ctx.specs)
                    )
                )
            except Exception as exc:
                log.warning("update_run_counts_failed", error=str(exc))

        for ev in _text_events(encoder, f"Generated {len(ctx.specs)} specs"):
            yield ev

        # Sync generated specs to durable storage
        if run_storage is not None and ctx.specs:
            try:
                def _sync_specs():
                    for spec in ctx.specs:
                        run_storage.write_spec(
                            f"{spec.id}.json", spec.model_dump_json(indent=2)
                        )
                await asyncio.to_thread(_sync_specs)
            except Exception as exc:
                log.warning("storage_specs_sync_failed", error=str(exc))

        yield encoder.encode(
            StepFinishedEvent(
                type=EventType.STEP_FINISHED, step_name="Spec Generation", step_id=step_id
            )
        )

        # ── Phase 3: Knowledge Graph ──────────────────────────────────────────
        step_id = str(uuid4())
        yield encoder.encode(
            StepStartedEvent(
                type=EventType.STEP_STARTED, step_name="Knowledge Graph", step_id=step_id
            )
        )
        for ev in _text_events(encoder, "Populating knowledge graph..."):
            yield ev

        ctx = await asyncio.to_thread(GraphStage(repo=repo).run, ctx)
        raise_if_cancelled()
        log.info("graph_done")

        for ev in _text_events(encoder, "Knowledge graph updated"):
            yield ev
        yield encoder.encode(
            StepFinishedEvent(
                type=EventType.STEP_FINISHED, step_name="Knowledge Graph", step_id=step_id
            )
        )

        # ── Phase 4: Swarm Orchestrator ───────────────────────────────────────
        raise_if_cancelled()
        spec_ids = [s.id for s in ctx.specs]
        step_id = str(uuid4())
        yield encoder.encode(
            StepStartedEvent(
                type=EventType.STEP_STARTED, step_name="Swarm Orchestrator", step_id=step_id
            )
        )
        for ev in _text_events(encoder, f"Running swarm across {len(spec_ids)} features..."):
            yield ev

        from dark_factory.agents.orchestrator import run_orchestrator
        from dark_factory.agents import tools as _tools

        # Subscribe to the global progress broker for this run's events.
        # C1: concurrent runs are rejected at the route level; this bridge
        # only sees events while `app.state.run_lock` is held, so event
        # history from a previous run has already finished.
        broker = _tools._progress_broker
        if broker is None:
            # Fall back gracefully — still runs the pipeline, just without
            # the nested feature sub-steps.
            result = await asyncio.to_thread(run_orchestrator, settings, spec_ids=spec_ids)
        else:
            # H2 fix: bounded queue with drop-oldest policy (broker handles
            # overflow internally). Subscribe BEFORE launching the orchestrator
            # so we see every event it publishes from the first tick.
            # NOTE: history was already cleared at the start of the run, so
            # the Agent Logs tab still has Phase 2 (Spec Generation) events.
            progress_queue = broker.subscribe(include_history=False)

            # Track nested step IDs per feature so we can emit matching STEP_FINISHED
            feature_step_ids: dict[str, str] = {}
            # Dedupe agent_active events so we only emit one TEXT per (feature, agent)
            last_agent_per_feature: dict[str, str] = {}

            # Launch the orchestrator in a worker thread
            orch_task = asyncio.create_task(
                asyncio.to_thread(run_orchestrator, settings, spec_ids=spec_ids)
            )

            try:
                while True:
                    get_task = asyncio.create_task(progress_queue.get())
                    done, _pending = await asyncio.wait(
                        {orch_task, get_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if get_task in done:
                        progress = get_task.result()
                        for chunk in _translate_progress(
                            encoder, progress, feature_step_ids, last_agent_per_feature
                        ):
                            yield chunk
                    else:
                        # orch_task is done — cancel the get and break out
                        get_task.cancel()
                        try:
                            await get_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        break

                # H1 fix: flush any `call_soon_threadsafe` callbacks that the
                # swarm worker threads scheduled just before returning. Without
                # this, late `feature_completed` / `layer_completed` events can
                # be dropped from the Run tab.
                for _ in range(3):
                    await asyncio.sleep(0)

                # Drain any events still in the queue after the orchestrator finishes
                while not progress_queue.empty():
                    progress = progress_queue.get_nowait()
                    for chunk in _translate_progress(
                        encoder, progress, feature_step_ids, last_agent_per_feature
                    ):
                        yield chunk
            finally:
                broker.unsubscribe(progress_queue)

            result = await orch_task

        log.info("swarm_done", features=len(result.get("completed_features", [])))

        completed = result.get("completed_features", [])
        pass_rate = result.get("pass_rate", 0)
        summary = (
            f"Completed {len(completed)} features | Pass rate: {pass_rate:.0%} | "
            f"Artifacts: {len(result.get('all_artifacts', []))} | "
            f"Tests: {len(result.get('all_tests', []))}"
        )
        for ev in _text_events(encoder, summary):
            yield ev

        yield encoder.encode(
            StepFinishedEvent(
                type=EventType.STEP_FINISHED, step_name="Swarm Orchestrator", step_id=step_id
            )
        )

        # Sync swarm-generated output to durable storage
        if run_storage is not None and pipeline_run_id:
            try:
                swarm_output_dir = Path(settings.pipeline.output_dir) / pipeline_run_id
                count = await asyncio.to_thread(
                    run_storage.sync_output_from_local, swarm_output_dir
                )
                log.info("storage_swarm_sync_done", run_id=pipeline_run_id, files=count)
            except Exception as exc:
                log.warning("storage_swarm_sync_failed", run_id=pipeline_run_id, error=str(exc))

        # ── Phase 5: Reconciliation ──────────────────────────────────────────
        # Best-effort cross-feature polishing pass — always runs after
        # the feature swarms complete. A single extended Claude Agent
        # SDK invocation in the run's output directory reviews all
        # generated code, fixes cross-feature issues, runs validation,
        # and writes a RECONCILIATION_REPORT.md at the root. The stage
        # itself has skip guards for the degenerate cases (no features
        # / all features errored / output dir missing). Reconciliation
        # failures do NOT fail the pipeline — the feature output is
        # still delivered as-is.
        reconciliation_result: dict[str, Any] | None = None
        if pipeline_run_id:
            from dark_factory.stages.reconciliation import ReconciliationStage

            recon_step_id = str(uuid4())
            yield encoder.encode(
                StepStartedEvent(
                    type=EventType.STEP_STARTED,
                    step_name="Reconciliation",
                    step_id=recon_step_id,
                )
            )
            for ev in _text_events(
                encoder,
                "Reviewing the full run output, fixing cross-feature issues, "
                "and validating the generated code…",
            ):
                yield ev

            recon_output_dir = Path(settings.pipeline.output_dir) / pipeline_run_id
            max_recon_retries = settings.pipeline.max_reconciliation_retries

            for recon_attempt in range(1 + max_recon_retries):
                try:
                    extra_turns = 0
                    if recon_attempt > 0:
                        # Retry with more turns based on reflection recommendation
                        extra_turns = getattr(recon_stage, "_extra_turns", 20)
                        for ev in _text_events(
                            encoder,
                            f"Retrying reconciliation (attempt {recon_attempt + 1})…",
                        ):
                            yield ev

                    recon_stage = ReconciliationStage(
                        max_turns=settings.pipeline.max_reconciliation_turns + extra_turns,
                        timeout_seconds=settings.pipeline.reconciliation_timeout_seconds,
                    )
                    recon_result_obj = await asyncio.to_thread(
                        recon_stage.run,
                        run_id=pipeline_run_id,
                        output_dir=recon_output_dir,
                        feature_results=completed,
                    )
                    reconciliation_result = recon_result_obj.model_dump()

                    for ev in _text_events(
                        encoder,
                        f"Reconciliation {recon_result_obj.status}: "
                        f"{recon_result_obj.summary}",
                    ):
                        yield ev

                    # Prometheus: reconciliation_runs_total{status} counter
                    try:
                        from dark_factory.metrics.prometheus import (
                            observe_reconciliation_run,
                        )

                        observe_reconciliation_run(
                            status=recon_result_obj.status,
                            duration_seconds=recon_result_obj.duration_seconds,
                        )
                    except Exception:  # pragma: no cover — defensive
                        pass

                    # If clean or last attempt, accept the result.
                    if recon_result_obj.status == "clean" or recon_attempt >= max_recon_retries:
                        # Record an incident for partial / error outcomes.
                        if recon_result_obj.status in ("partial", "error"):
                            try:
                                from dark_factory.metrics.helpers import record_incident

                                record_incident(
                                    category="subprocess",
                                    severity=(
                                        "warning"
                                        if recon_result_obj.status == "partial"
                                        else "error"
                                    ),
                                    message=recon_result_obj.summary,
                                    phase="reconciliation",
                                    feature=None,
                                    stack=recon_result_obj.agent_output[-2000:]
                                    or None,
                                )
                            except Exception:  # pragma: no cover — defensive
                                pass
                        break

                    # Not clean and retries remain — run reflection to decide.
                    try:
                        recon_reflection = await asyncio.to_thread(
                            _reflect_on_reconciliation,
                            recon_result_obj,
                            recon_attempt + 1,
                            1 + max_recon_retries,
                        )
                        if recon_reflection and recon_reflection.get("should_retry"):
                            extra = recon_reflection.get("extra_turns", 20)
                            recon_stage._extra_turns = extra  # type: ignore[attr-defined]
                            for ev in _text_events(
                                encoder,
                                f"Reflection: {recon_reflection.get('diagnosis', '')} — retrying with +{extra} turns.",
                            ):
                                yield ev
                            continue
                        else:
                            # Reflection declined retry — accept result.
                            break
                    except Exception:
                        break

                except Exception as exc:
                    # Reconciliation is best-effort. Log, emit, don't fail.
                    log.warning(
                        "reconciliation_stage_crashed",
                        run_id=pipeline_run_id,
                        error=str(exc),
                    )
                    for ev in _text_events(
                        encoder,
                        f"Reconciliation crashed: {exc}. Continuing with "
                        f"feature output as-is.",
                    ):
                        yield ev
                    break

            # Sync reconciliation output to durable storage
            if run_storage is not None:
                try:
                    await asyncio.to_thread(
                        run_storage.sync_output_from_local, recon_output_dir
                    )
                except Exception as exc:
                    log.warning("storage_recon_sync_failed", error=str(exc))

            yield encoder.encode(
                StepFinishedEvent(
                    type=EventType.STEP_FINISHED,
                    step_name="Reconciliation",
                    step_id=recon_step_id,
                )
            )

        # ── Phase 6: End-to-End Validation ───────────────────────────────────
        # Playwright-powered cross-browser smoke testing of the
        # reconciled code. Runs AFTER reconciliation completes
        # cleanly, only if the feature is enabled. A single
        # extended Claude Agent SDK invocation detects whether the
        # run produced a web app, installs Playwright if needed,
        # writes smoke tests from the specs, runs them across
        # chromium / firefox / webkit, and writes E2E_REPORT.md.
        # Best-effort — E2E failures do NOT fail the pipeline.
        #
        # Skip conditions:
        #   - feature flag disabled
        #   - reconciliation did not run (e.g., no completed features)
        #   - reconciliation status != "clean" (partial, error, skipped)
        # Running browser tests against known-broken or
        # never-reconciled code is wasted turn budget, so we gate
        # E2E strictly on a clean reconciliation.
        e2e_validation_result: dict[str, Any] | None = None
        recon_status_for_e2e = (
            reconciliation_result.get("status") if reconciliation_result else None
        )
        e2e_enabled = bool(settings.pipeline.enable_e2e_validation)
        e2e_should_run = (
            pipeline_run_id is not None
            and e2e_enabled
            and recon_status_for_e2e == "clean"
        )
        if (
            e2e_enabled
            and pipeline_run_id is not None
            and not e2e_should_run
        ):
            if recon_status_for_e2e is None:
                skip_reason = (
                    "E2E validation skipped: reconciliation did not run."
                )
            else:
                skip_reason = (
                    f"E2E validation skipped: reconciliation status "
                    f"'{recon_status_for_e2e}' is not clean. Browser tests "
                    f"require a clean reconciliation pass to be meaningful."
                )
            for ev in _text_events(encoder, skip_reason):
                yield ev
        if e2e_should_run:
            from dark_factory.stages.e2e_validation import E2EValidationStage

            e2e_step_id = str(uuid4())
            yield encoder.encode(
                StepStartedEvent(
                    type=EventType.STEP_STARTED,
                    step_name="E2E Validation",
                    step_id=e2e_step_id,
                )
            )
            for ev in _text_events(
                encoder,
                f"Running Playwright smoke tests across "
                f"{', '.join(settings.pipeline.e2e_browsers)}…",
            ):
                yield ev

            try:
                e2e_stage = E2EValidationStage(
                    max_turns=settings.pipeline.max_e2e_turns,
                    timeout_seconds=settings.pipeline.e2e_timeout_seconds,
                    browsers=list(settings.pipeline.e2e_browsers),
                )
                e2e_output_dir = Path(settings.pipeline.output_dir) / pipeline_run_id
                e2e_result_obj = await asyncio.to_thread(
                    e2e_stage.run,
                    run_id=pipeline_run_id,
                    output_dir=e2e_output_dir,
                    feature_results=completed,
                    reconciliation_status=recon_status_for_e2e or "clean",
                )
                e2e_validation_result = e2e_result_obj.model_dump()

                for ev in _text_events(
                    encoder,
                    f"E2E {e2e_result_obj.status}: {e2e_result_obj.summary}",
                ):
                    yield ev
                if e2e_result_obj.tests_total:
                    for ev in _text_events(
                        encoder,
                        f"  {e2e_result_obj.tests_passed}/{e2e_result_obj.tests_total} "
                        f"tests passed across {len(e2e_result_obj.browsers_run)} browser(s)",
                    ):
                        yield ev

                # Prometheus: phase outcome + per-browser test counts.
                try:
                    from dark_factory.metrics.prometheus import (
                        observe_e2e_validation_run,
                    )

                    observe_e2e_validation_run(
                        status=e2e_result_obj.status,
                        duration_seconds=e2e_result_obj.duration_seconds,
                        tests_passed=e2e_result_obj.tests_passed,
                        tests_failed=e2e_result_obj.tests_failed,
                        browsers_run=e2e_result_obj.browsers_run,
                    )
                except Exception:  # pragma: no cover — defensive
                    pass

                # Record an incident for partial / error outcomes so
                # the Run Detail popup surfaces what happened.
                if e2e_result_obj.status in ("partial", "error"):
                    try:
                        from dark_factory.metrics.helpers import record_incident

                        record_incident(
                            category="e2e",
                            severity=(
                                "warning"
                                if e2e_result_obj.status == "partial"
                                else "error"
                            ),
                            message=e2e_result_obj.summary,
                            phase="e2e_validation",
                            feature=None,
                            stack=e2e_result_obj.agent_output[-2000:] or None,
                        )
                    except Exception:  # pragma: no cover — defensive
                        pass
            except Exception as exc:
                # E2E is best-effort. Log, emit, don't fail.
                log.warning(
                    "e2e_stage_crashed",
                    run_id=pipeline_run_id,
                    error=str(exc),
                )
                for ev in _text_events(
                    encoder,
                    f"E2E validation crashed: {exc}. Continuing with "
                    f"reconciled output as-is.",
                ):
                    yield ev

            # Sync E2E output (report, screenshots, html-report) to storage
            if run_storage is not None:
                try:
                    await asyncio.to_thread(
                        run_storage.sync_output_from_local, e2e_output_dir
                    )
                except Exception as exc:
                    log.warning("storage_e2e_sync_failed", error=str(exc))

            yield encoder.encode(
                StepFinishedEvent(
                    type=EventType.STEP_FINISHED,
                    step_name="E2E Validation",
                    step_id=e2e_step_id,
                )
            )

        # Emit state snapshot with serialised result
        safe_result = json.loads(json.dumps(result, default=str))
        if reconciliation_result is not None:
            safe_result["reconciliation"] = reconciliation_result
        if e2e_validation_result is not None:
            safe_result["e2e_validation"] = e2e_validation_result
        yield encoder.encode(
            StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=safe_result)
        )

        # Metrics: stamp the successful run lifecycle end in Postgres +
        # Prometheus.
        succeeded = sum(1 for r in completed if r.get("status") == "success")
        total = len(completed)
        status = "success" if succeeded == total and total > 0 else "partial"
        duration = time.time() - pipeline_start_time

        try:
            from dark_factory.metrics.prometheus import observe_pipeline_run_end

            observe_pipeline_run_end(status=status, duration_seconds=duration)
        except Exception:  # pragma: no cover — defensive
            pass

        if metrics_recorder is not None and pipeline_run_id:
            try:
                metrics_recorder.record_pipeline_run_end(
                    run_id=pipeline_run_id,
                    status=status,
                    pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
                    duration_seconds=duration,
                )
            except Exception as exc:
                log.warning("metrics_run_end_record_failed", error=str(exc))

        # Neo4j run history: flip the Run node from "running" to its
        # terminal status. Without this the ``/api/history`` endpoint
        # (which queries the Neo4j memory_repo) shows the run as
        # "running" forever even though Postgres knows it ended —
        # producing zombie entries on every successful-but-not-fully-
        # passing pipeline run. The cancellation + exception paths
        # already call ``mark_run_failed``; this is the missing
        # success-side equivalent.
        if pipeline_run_id and memory_repo is not None:
            try:
                await asyncio.to_thread(
                    lambda: memory_repo.complete_run(
                        run_id=pipeline_run_id,
                        status=status,
                        pass_rate=float(pass_rate)
                        if isinstance(pass_rate, (int, float))
                        else 0.0,
                        mean_eval_scores=result.get("mean_eval_scores", {}) or {},
                        worst_features=result.get("worst_features", []) or [],
                        duration_seconds=duration,
                    )
                )
            except Exception as exc:
                log.warning("complete_run_failed", error=str(exc))

        yield encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_id,
                run_id=run_id,
            )
        )

    except PipelineCancelled:
        # Kill-switch path: user clicked Cancel via POST /api/agent/cancel.
        # Emit a clean cancellation end state instead of a critical incident.
        log.warning("pipeline_cancelled_by_user", run_id=pipeline_run_id)
        duration = time.time() - pipeline_start_time

        # Progress event so the Agent Logs tab shows the cancel
        try:
            from dark_factory.agents.tools import emit_progress as _emit_cancel

            _emit_cancel(
                "pipeline_cancelled",
                run_id=pipeline_run_id or None,
                reason="user_requested",
            )
        except Exception:  # pragma: no cover — defensive
            pass

        # Run history: mark as cancelled rather than failed
        if pipeline_run_id and memory_repo is not None:
            try:
                await asyncio.to_thread(
                    lambda: memory_repo.mark_run_failed(
                        run_id=pipeline_run_id,
                        error="Cancelled by user",
                    )
                )
            except Exception as inner:
                log.warning("mark_run_failed_failed", error=str(inner))

        # Prometheus: pipeline_runs_total{status="cancelled"}
        try:
            from dark_factory.metrics.prometheus import observe_pipeline_run_end

            observe_pipeline_run_end(status="cancelled", duration_seconds=duration)
        except Exception:  # pragma: no cover — defensive
            pass

        # Postgres metrics store: dedicated "cancelled" status
        if metrics_recorder is not None and pipeline_run_id:
            try:
                metrics_recorder.record_pipeline_run_end(
                    run_id=pipeline_run_id,
                    status="cancelled",
                    duration_seconds=duration,
                    error="Cancelled by user",
                )
            except Exception as inner:
                log.warning("metrics_run_end_record_failed", error=str(inner))

        # Tell the SSE client the run ended cleanly-cancelled
        for ev in _text_events(encoder, "⛔ Pipeline cancelled by user"):
            yield ev
        yield encoder.encode(
            RunErrorEvent(
                type=EventType.RUN_ERROR,
                message="Pipeline cancelled by user",
            )
        )

    except Exception as exc:
        log.error("pipeline_stream_error", error=str(exc))
        # Mark the Run History entry as failed so it doesn't stay 'running' forever
        if pipeline_run_id and memory_repo is not None:
            try:
                await asyncio.to_thread(
                    lambda: memory_repo.mark_run_failed(
                        run_id=pipeline_run_id, error=str(exc)
                    )
                )
            except Exception as inner:
                log.warning("mark_run_failed_failed", error=str(inner))
        # Metrics: record the failed run lifecycle end in Postgres + Prometheus
        duration = time.time() - pipeline_start_time

        try:
            from dark_factory.metrics.prometheus import observe_pipeline_run_end

            observe_pipeline_run_end(status="error", duration_seconds=duration)
        except Exception:  # pragma: no cover — defensive
            pass

        if metrics_recorder is not None and pipeline_run_id:
            try:
                metrics_recorder.record_pipeline_run_end(
                    run_id=pipeline_run_id,
                    status="error",
                    duration_seconds=duration,
                    error=str(exc),
                )
            except Exception as inner:
                log.warning("metrics_run_end_record_failed", error=str(inner))
        try:
            import traceback as _tb

            from dark_factory.metrics.helpers import record_incident as _rec_inc

            _rec_inc(
                category="pipeline",
                severity="critical",
                message=str(exc)[:500],
                stack=_tb.format_exc()[:4000],
                phase="pipeline",
                run_id=pipeline_run_id or None,
            )
        except Exception as inner:  # pragma: no cover — defensive
            log.warning("incident_record_failed", error=str(inner))
        yield encoder.encode(
            RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc))
        )
    finally:
        # Each cleanup step is isolated so a failure in one (e.g. Neo4j
        # close raises a connection error, env restore hits an OSError)
        # cannot prevent the others from running. The cancel flag reset
        # is done FIRST — it's the cheapest and most critical step, and
        # its failure would poison the next run.
        try:
            if is_cancelled():
                log.info("pipeline_cancel_cleared_on_exit")
            reset_cancel()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("reset_cancel_failed_on_exit", error=str(exc))

        # Clear module-global run id / feature so the next pipeline
        # run doesn't inherit stale state from this one. Without this,
        # a subsequent run that doesn't call set_current_run_id early
        # (e.g. a crash before the Neo4j Run node is created) would
        # have its progress events and incident rows tagged with the
        # PREVIOUS run's id — producing impossible-to-debug telemetry.
        try:
            from dark_factory.agents.tools import (
                set_current_feature,
                set_current_run_id,
            )

            set_current_run_id("")
            set_current_feature("")
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("clear_agent_globals_failed_on_exit", error=str(exc))

        try:
            _restore_api_keys()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("restore_api_keys_failed_on_exit", error=str(exc))

        try:
            if metrics_recorder is not None:
                metrics_recorder.set_run_id(None)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("metrics_recorder_clear_run_failed", error=str(exc))

        try:
            if neo4j_client is not None:
                neo4j_client.close()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("neo4j_close_failed_on_exit", error=str(exc))
