"""Tests for the cross-debate coordination bus."""

from __future__ import annotations

import threading

import pytest

from dark_factory.api.refinery.debate.cross_debate_bus import (
    CrossDebateBus,
    CrossDebateFinding,
    get_global_bus,
)


def test_bus_isolates_runs():
    """Findings posted to one run never leak to another."""

    bus = CrossDebateBus()
    bus.post(
        refinery_run_id="run-A", requirement_id="r-1",
        kind="critic_blocker", body={"finding": "X"},
    )
    bus.post(
        refinery_run_id="run-B", requirement_id="r-1",
        kind="critic_blocker", body={"finding": "Y"},
    )
    a = bus.read_for(refinery_run_id="run-A", requirement_id="r-2")
    b = bus.read_for(refinery_run_id="run-B", requirement_id="r-2")
    assert len(a) == 1
    assert a[0].body["finding"] == "X"
    assert len(b) == 1
    assert b[0].body["finding"] == "Y"


def test_read_for_excludes_caller_own_posts():
    """A debate doesn't read its own findings back — that would
    just rehash its previous round's reasoning."""

    bus = CrossDebateBus()
    bus.post(
        refinery_run_id="run-A", requirement_id="r-1",
        kind="critic_blocker", body={"finding": "self"},
    )
    bus.post(
        refinery_run_id="run-A", requirement_id="r-2",
        kind="critic_blocker", body={"finding": "sibling"},
    )
    out = bus.read_for(refinery_run_id="run-A", requirement_id="r-1")
    assert len(out) == 1
    assert out[0].body["finding"] == "sibling"


def test_read_filters_by_kind():
    bus = CrossDebateBus()
    bus.post(
        refinery_run_id="run-A", requirement_id="r-1",
        kind="critic_blocker", body={"x": 1},
    )
    bus.post(
        refinery_run_id="run-A", requirement_id="r-1",
        kind="judge_tradeoff", body={"x": 2},
    )
    out = bus.read_for(
        refinery_run_id="run-A", requirement_id="r-2",
        kinds=["critic_blocker"],
    )
    assert len(out) == 1
    assert out[0].kind == "critic_blocker"


def test_finish_run_clears_buffer():
    bus = CrossDebateBus()
    bus.post(
        refinery_run_id="run-A", requirement_id="r-1",
        kind="critic_blocker", body={},
    )
    assert bus.stats("run-A")["total"] == 1
    bus.finish_run("run-A")
    assert bus.stats("run-A")["total"] == 0


def test_bus_is_thread_safe_under_concurrent_posts():
    """Concurrent posts from many threads never lose data."""

    bus = CrossDebateBus(max_findings_per_run=10000)
    posts_per_thread = 50
    threads = []

    def _worker(tid: int):
        for i in range(posts_per_thread):
            bus.post(
                refinery_run_id="run-A",
                requirement_id=f"r-{tid}",
                kind="critic_blocker",
                body={"i": i},
            )

    for tid in range(8):
        t = threading.Thread(target=_worker, args=(tid,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # 8 threads × 50 posts each = 400 total findings.
    assert bus.stats("run-A")["total"] == 8 * posts_per_thread


def test_bounded_buffer_drops_oldest():
    """When the per-run cap is hit, the oldest entries get dropped
    so memory stays bounded."""

    bus = CrossDebateBus(max_findings_per_run=5)
    for i in range(10):
        bus.post(
            refinery_run_id="run-A", requirement_id="r-1",
            kind="critic_blocker", body={"i": i},
        )
    out = bus.read_for(
        refinery_run_id="run-A", requirement_id="r-2", limit=20,
    )
    # Most recent 5; oldest 5 dropped.
    assert len(out) == 5
    indices = sorted(f.body["i"] for f in out)
    assert indices == [5, 6, 7, 8, 9]


def test_global_bus_is_a_singleton():
    a = get_global_bus()
    b = get_global_bus()
    assert a is b
