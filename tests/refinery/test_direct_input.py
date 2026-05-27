"""Direct-input mode tests (the third input mode on POST /api/refinery).

User submits one typed requirement directly. The refinery wraps it in
a minimal run_context with source_mode="direct" and runs the full
debate on exactly one item. Cross-req review auto-skips at n=1.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dark_factory.api.refinery.stream import run_refinery_stream

from tests.refinery.conftest import patched_product_propose


class _StubRequest:
    class app:
        class state:
            neo4j_client = None
            memory_repo = None


def _drive(fake=None, direct=None, patch_save=True):
    """Run the SSE generator to completion and return the event list."""

    async def _run():
        events: list[dict] = []
        patchers = []
        if fake is not None:
            patchers.append(patched_product_propose(fake))
        if patch_save:
            patchers.append(patch("dark_factory.api.refinery.stream.save_refinery_result"))
        # Enter all context managers manually so we can nest at runtime.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            async for event in run_refinery_stream(
                _StubRequest(), run_id=None, input_path=None, direct=direct,
            ):
                events.append(event)
        return events

    return asyncio.run(_run())


def test_direct_input_routes_through_stream_with_single_requirement(fake_refined):
    events = _drive(
        fake=fake_refined,
        direct={
            "title": "Session timeout policy",
            "description": "System must enforce a 30-minute inactivity timeout.",
            "priority": "high",
            "tags": ["auth", "security"],
        },
    )
    # Gathering phase emits a direct_input step.
    gather_steps = [e for e in events if e.get("phase") == "gathering"]
    assert any(e.get("step") == "direct_input" for e in gather_steps), gather_steps

    # Refining ran on exactly one requirement.
    refining = [e for e in events if e.get("phase") == "refining"]
    assert any(
        "1 requirements" in (e.get("message") or "")
        or "1/1" in (e.get("message") or "")
        for e in refining
    ), refining

    # Cross-req reconciliation is SKIPPED for a single-requirement debate.
    assert not any(e.get("phase") == "reconciling" for e in events)

    # Terminal done event present.
    assert len([e for e in events if e.get("phase") == "done"]) == 1


def test_direct_input_rejects_when_all_three_sources_missing():
    """All three input modes must be mutually exclusive AND at least one
    must be set. Calling with none yields an error event."""

    events = _drive(direct=None, patch_save=False)
    err = next(e for e in events if e.get("phase") == "error")
    assert "run_id" in err["message"] or "direct" in err["message"]
