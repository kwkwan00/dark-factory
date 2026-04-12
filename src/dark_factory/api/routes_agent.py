"""AG-UI agent endpoint — streams pipeline execution as SSE events."""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator

from dark_factory.api.ag_ui_bridge import run_pipeline_stream

log = structlog.get_logger()

router = APIRouter()

# C2 fix: restrict requirements_path to prevent path traversal
_CWD = pathlib.Path.cwd().resolve()


class RunRequest(BaseModel):
    # CRITICAL security fix: ``hide_input_in_errors=True`` tells Pydantic to
    # omit the raw ``input_value`` from validation error payloads. Without
    # this, FastAPI's default 422 response includes the rejected value in
    # ``detail[*].input`` — which for ``anthropic_api_key`` / ``openai_api_key``
    # fields would leak the raw secret into the frontend error message and
    # subsequently into browser console logs via ``describeFailure()``.
    model_config = ConfigDict(hide_input_in_errors=True)

    requirements_path: str = "./openspec"
    # Optional per-run API key overrides. When supplied, these replace the
    # server's ANTHROPIC_API_KEY / OPENAI_API_KEY env vars for the duration
    # of this single pipeline run (restored on completion). Safe because
    # the app serialises runs via ``run_lock`` — no concurrent env races.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    @field_validator("requirements_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        resolved = pathlib.Path(v).resolve()
        if not resolved.is_relative_to(_CWD):
            raise ValueError(
                f"Path must be within the working directory ({_CWD})"
            )
        return v

    @field_validator("anthropic_api_key", "openai_api_key")
    @classmethod
    def validate_key(cls, v: str | None) -> str | None:
        # Empty string → None so the frontend can clear a field without
        # accidentally clearing the server-side default.
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Cheap guardrail against accidentally pasting giant blobs —
        # real keys are ≤ 200 chars for both providers.
        if len(v) > 512:
            raise ValueError("API key too long (max 512 chars)")
        # Reject obvious control characters / whitespace that could
        # break subsequent env var comparisons.
        if any(c.isspace() for c in v):
            raise ValueError("API key must not contain whitespace")
        return v


@router.post("/agent/run")
async def agent_run(request: Request, body: RunRequest):
    """Stream pipeline execution as AG-UI Server-Sent Events.

    The client should set ``Accept: text/event-stream`` to receive the SSE stream.
    Each event is a JSON-encoded AG-UI event on a ``data:`` line.

    Only one pipeline run may execute at a time (C1 fix): concurrent requests
    receive a 409 Conflict. This avoids races on the module-global
    ``_current_run_id`` / ``_current_feature`` state in ``agents.tools``.
    """
    run_lock: asyncio.Lock = request.app.state.run_lock
    if run_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already in progress. Please wait for it to finish.",
        )

    settings = request.app.state.settings
    memory_repo = getattr(request.app.state, "memory_repo", None)
    thread_id = str(uuid4())
    run_id = str(uuid4())
    accept = request.headers.get("accept")

    log.info(
        "agent_run_started",
        requirements_path=body.requirements_path,
        thread_id=thread_id,
        run_id=run_id,
        anthropic_key_override=bool(body.anthropic_api_key),
        openai_key_override=bool(body.openai_api_key),
    )

    async def event_stream():
        # Acquire the lock inside the generator so it's held for the full
        # duration of the SSE stream, not just the route handler.
        async with run_lock:
            async for chunk in run_pipeline_stream(
                settings=settings,
                memory_repo=memory_repo,
                requirements_path=body.requirements_path,
                thread_id=thread_id,
                run_id=run_id,
                accept=accept,
                anthropic_api_key=body.anthropic_api_key,
                openai_api_key=body.openai_api_key,
            ):
                yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent/cancel")
async def agent_cancel(request: Request):
    """Kill-switch for an in-flight pipeline run.

    Sets a process-wide cancellation flag that hot paths in the spec,
    swarm, and orchestrator phases check at safe points. Cancellation
    is **cooperative** — already-in-flight LLM calls and deep-agent
    subprocesses are allowed to finish their current step rather than
    being hard-killed mid-generation. Typical stop-to-halt latency is
    sub-second for the spec phase, up to the length of a single LLM
    call for the swarm codegen phase, and up to the deep-agent timeout
    (default 600s) for an in-flight Claude SDK subprocess.

    Idempotent: calling twice while a run is still halting is a no-op.
    Returns ``{"cancelled": false, "reason": "no active run"}`` with
    HTTP 200 when no run is active, so the frontend can safely fire
    cancel on button click without worrying about race conditions.
    """
    from dark_factory.agents.cancellation import is_cancelled, request_cancel

    run_lock: asyncio.Lock = request.app.state.run_lock
    if not run_lock.locked():
        return {"cancelled": False, "reason": "no active run"}

    already = is_cancelled()
    request_cancel()
    log.warning(
        "agent_cancel_requested",
        already=already,
        remote=str(request.client.host) if request.client else "?",
    )
    return {"cancelled": True, "already_pending": already}


@router.get("/agent/events")
async def agent_events(request: Request):
    """SSE stream of swarm progress events, for the Agent Logs tab.

    Subscribes to the app-wide :class:`ProgressBroker` so every client sees
    events from any pipeline run happening on the server. New connections
    receive the most recent history immediately, then stream live events.
    """
    # M1 fix: guard missing broker
    broker = getattr(request.app.state, "progress_broker", None)
    if broker is None:
        raise HTTPException(status_code=503, detail="Progress broker not initialised")

    heartbeat_interval = 15  # seconds
    max_duration = 24 * 60 * 60  # 24 hours
    poll_timeout = 0.25  # M13: 250ms — faster disconnect detection

    async def generator():
        # C4 fix: subscribe INSIDE the generator so subscribe + unsubscribe
        # always run on the same loop/task that iterates the queue. This
        # also defers work until the response actually starts streaming.
        queue = broker.subscribe(include_history=True)
        start = time.monotonic()
        last_heartbeat = start
        try:
            # Immediately emit a connected marker
            yield f"data: {json.dumps({'event': 'log_connected', 'timestamp': time.time()})}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                now = time.monotonic()
                if now - start > max_duration:
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=poll_timeout)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    pass

                if time.monotonic() - last_heartbeat >= heartbeat_interval:
                    yield ": keepalive\n\n"
                    last_heartbeat = time.monotonic()
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
