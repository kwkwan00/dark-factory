"""Cooperative pipeline cancellation.

A single module-level :class:`threading.Event` acts as the kill-switch.
The frontend's "Cancel" button posts to ``/api/agent/cancel`` which calls
:func:`request_cancel`. Hot paths throughout the pipeline call
:func:`is_cancelled` at safe points and raise :class:`PipelineCancelled`
when the flag is set, unwinding the worker stack cleanly.

Why a ``threading.Event`` and not an asyncio primitive:

- The orchestrator runs inside ``asyncio.to_thread(run_orchestrator, ...)``
  on a worker thread. The swarm spins a ``ThreadPoolExecutor`` inside that
  worker. Spec generation uses another ``ThreadPoolExecutor``. All of
  these are plain sync Python code from asyncio's perspective.
- ``threading.Event`` is thread-safe, wait-free for the common "is it set"
  read, and works identically in sync and async code. No loop binding.

The lifecycle is managed by :mod:`ag_ui_bridge`: it calls
:func:`reset_cancel` at the top of every ``run_pipeline_stream`` so a
cancel from a previous run can't bleed into the next. Concurrent runs
are already blocked by ``app.state.run_lock``.
"""

from __future__ import annotations

import threading

import structlog

log = structlog.get_logger()


class PipelineCancelled(Exception):
    """Raised at cooperative cancellation checkpoints when a cancel has
    been requested. Caught at phase boundaries in ``ag_ui_bridge`` so the
    run ends with a clean ``cancelled`` status rather than a generic error.
    """


_cancel_event = threading.Event()


def request_cancel() -> None:
    """Mark the currently-running pipeline for cancellation.

    Idempotent. Called from the ``/api/agent/cancel`` route handler.
    """
    if _cancel_event.is_set():
        log.debug("pipeline_cancel_already_set")
        return
    _cancel_event.set()
    log.warning("pipeline_cancel_requested")


def reset_cancel() -> None:
    """Clear the cancel flag. Called by ``run_pipeline_stream`` at the
    start of a new run so residual state from a prior cancelled run
    doesn't leak forward.
    """
    _cancel_event.clear()


def is_cancelled() -> bool:
    """Return True if :func:`request_cancel` has been called since the
    most recent :func:`reset_cancel`.

    Cheap enough to call on every iteration of hot loops — a single
    atomic event read under the GIL.
    """
    return _cancel_event.is_set()


def raise_if_cancelled() -> None:
    """Raise :class:`PipelineCancelled` if a cancel has been requested.

    Placed at safe points (start of a phase, before dispatching a worker,
    between swarm stream chunks, between spec refinement attempts) so the
    worker stack can unwind without stranding resources.
    """
    if _cancel_event.is_set():
        raise PipelineCancelled("pipeline cancelled by user")
