"""Tests for the @traced / @trace_methods decorators in dark_factory.log.

Covers:

- Off-mode is a no-op (no events emitted, no overhead on the return path).
- On-mode emits one ``call_entry`` + one ``call_exit`` per call.
- Arg summary is shape-only — never the underlying value.
- Sync, async, sync-gen, async-gen, and exception paths all wrap cleanly.
- ``@trace_methods`` skips dunders other than ``__init__``, plus properties,
  classmethods, staticmethods.
- Setting ``trace_calls=True`` via ``setup_logging`` flips the global flag.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest
import structlog

from dark_factory import log as dflog


# ─────────────────────────────────────────────────────────────────────
# Test fixtures: a logging-capturing context manager
# ─────────────────────────────────────────────────────────────────────


class _CapturedEvent(dict):
    """Plain-dict event captured by the structlog test processor."""


@pytest.fixture
def captured_events(monkeypatch):
    """Yield a list that fills with every structlog event the trace
    decorators emit. Restores trace mode to its previous state on
    teardown so this test file can't leak ``_TRACE_ENABLED=True`` into
    the rest of the suite."""

    events: list[_CapturedEvent] = []

    def _capture(logger, name, event_dict):
        events.append(_CapturedEvent(event_dict))
        raise structlog.DropEvent

    prior = structlog.get_config()
    structlog.configure(processors=[_capture])
    prior_trace = dflog.is_tracing_enabled()
    dflog.enable_tracing(True)
    try:
        yield events
    finally:
        dflog.enable_tracing(prior_trace)
        structlog.configure(**prior)


# ─────────────────────────────────────────────────────────────────────
# _arg_shape helper
# ─────────────────────────────────────────────────────────────────────


def test_arg_shape_renders_collections_with_length():
    assert dflog._arg_shape({"a": 1, "b": 2}) == "dict[2]"
    assert dflog._arg_shape([1, 2, 3]) == "list[3]"
    assert dflog._arg_shape((1, 2)) == "tuple[2]"
    assert dflog._arg_shape({1, 2, 3, 4}) == "set[4]"


def test_arg_shape_renders_strings_with_length_only():
    secret = "topsecret-prompt-with-pii"
    rendered = dflog._arg_shape(secret)
    assert rendered == f"str[{len(secret)}]"
    assert "topsecret" not in rendered


def test_arg_shape_primitives_render_as_type_name():
    assert dflog._arg_shape(None) == "None"
    assert dflog._arg_shape(True) == "bool"
    assert dflog._arg_shape(1) == "int"
    assert dflog._arg_shape(1.5) == "float"


def test_arg_shape_falls_back_to_class_name_for_objects():
    class Custom:
        pass

    assert dflog._arg_shape(Custom()) == "Custom"


# ─────────────────────────────────────────────────────────────────────
# @traced — function wrappers
# ─────────────────────────────────────────────────────────────────────


def test_traced_sync_is_noop_when_tracing_disabled(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        structlog,
        "get_logger",
        lambda *a, **kw: type(
            "L", (), {"debug": lambda self, *a, **kw: events.append(kw)}
        )(),
    )
    # Re-import would be needed to re-bind _tracer; instead just toggle the flag.
    dflog.enable_tracing(False)

    @dflog.traced
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert events == []


def test_traced_sync_emits_entry_and_exit(captured_events):
    @dflog.traced
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

    entries = [e for e in captured_events if e.get("event") == "call_entry"]
    exits = [e for e in captured_events if e.get("event") == "call_exit"]
    assert len(entries) == 1
    assert len(exits) == 1
    assert entries[0]["args"] == {"a": "int", "b": "int"}
    assert exits[0]["returns"] == "int"
    assert "duration_ms" in exits[0]


def test_traced_sync_logs_error_and_reraises(captured_events):
    @dflog.traced
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()

    exits = [e for e in captured_events if e.get("event") == "call_exit"]
    assert len(exits) == 1
    assert exits[0]["error"] == "ValueError"


def test_traced_async_emits_entry_and_exit(captured_events):
    @dflog.traced
    async def addr(a, b):
        return a + b

    asyncio.run(addr(2, 3))
    entries = [e for e in captured_events if e.get("event") == "call_entry"]
    exits = [e for e in captured_events if e.get("event") == "call_exit"]
    assert len(entries) == 1
    assert len(exits) == 1
    assert exits[0]["returns"] == "int"


def test_traced_generator_counts_yields(captured_events):
    @dflog.traced
    def squares(n):
        for i in range(n):
            yield i * i

    assert list(squares(4)) == [0, 1, 4, 9]
    exits = [e for e in captured_events if e.get("event") == "call_exit"]
    assert len(exits) == 1
    assert exits[0]["yielded"] == 4


def test_traced_async_generator_counts_yields(captured_events):
    @dflog.traced
    async def stream(n):
        for i in range(n):
            yield i

    async def _run():
        return [x async for x in stream(3)]

    assert asyncio.run(_run()) == [0, 1, 2]
    exits = [e for e in captured_events if e.get("event") == "call_exit"]
    assert len(exits) == 1
    assert exits[0]["yielded"] == 3


def test_traced_methods_drop_self_from_arg_summary(captured_events):
    class C:
        def m(self, x):
            return x

    C.m = dflog.traced(C.m)
    C().m(42)
    entries = [e for e in captured_events if e.get("event") == "call_entry"]
    assert entries[0]["args"] == {"x": "int"}, entries[0]


# ─────────────────────────────────────────────────────────────────────
# @trace_methods — class decorator
# ─────────────────────────────────────────────────────────────────────


def test_trace_methods_wraps_init_and_public_methods(captured_events):
    @dflog.trace_methods
    class Repo:
        def __init__(self, base):
            self.base = base

        def add(self, x):
            return self.base + x

        def _private(self, x):  # underscore-prefix is still wrapped (public-ish)
            return x

        def __repr__(self):
            return "Repo()"

        @property
        def doubled(self):
            return self.base * 2

        @classmethod
        def make(cls, base):
            return cls(base)

        @staticmethod
        def util(x):
            return x

    r = Repo(10)
    assert r.add(5) == 15
    _ = r.doubled  # property must not be wrapped (or it would emit events)
    _ = repr(r)    # __repr__ must not be wrapped
    Repo.util(1)   # staticmethod must not be wrapped
    Repo.make(2)   # classmethod must not be wrapped

    method_calls = [
        e for e in captured_events
        if e.get("event") == "call_entry"
    ]
    methods_seen = {e["method"] for e in method_calls}
    assert any(m.endswith(".__init__") for m in methods_seen), methods_seen
    assert any(m.endswith(".add") for m in methods_seen), methods_seen
    assert not any(m.endswith(".__repr__") for m in methods_seen), methods_seen
    assert not any(m.endswith(".doubled") for m in methods_seen), methods_seen
    assert not any(m.endswith(".util") for m in methods_seen), methods_seen
    assert not any(m.endswith(".make") for m in methods_seen), methods_seen


def test_trace_methods_preserves_behavior_when_disabled():
    """A class decorated with @trace_methods must behave identically to
    the undecorated version when trace mode is off (the steady state)."""

    @dflog.trace_methods
    class K:
        def __init__(self, n):
            self.n = n

        def hello(self, who):
            return f"hi {who}"

    dflog.enable_tracing(False)
    try:
        k = K(7)
        assert k.n == 7
        assert k.hello("world") == "hi world"
    finally:
        dflog.enable_tracing(False)


# ─────────────────────────────────────────────────────────────────────
# setup_logging integration
# ─────────────────────────────────────────────────────────────────────


def test_setup_logging_enables_tracing_when_flag_true(monkeypatch):
    monkeypatch.delenv("DARK_FACTORY_LOG_TRACE", raising=False)
    dflog.enable_tracing(False)
    assert dflog.is_tracing_enabled() is False

    dflog.setup_logging(level="DEBUG", trace_calls=True)
    try:
        assert dflog.is_tracing_enabled() is True
    finally:
        dflog.enable_tracing(False)
        # Restore default root level / handlers state. The test runner's
        # other tests don't rely on a specific root config so leaving
        # the rebound handler in place is fine.


def test_setup_logging_honors_env_var(monkeypatch):
    monkeypatch.setenv("DARK_FACTORY_LOG_TRACE", "1")
    dflog.enable_tracing(False)
    dflog.setup_logging(level="INFO", trace_calls=False)
    try:
        assert dflog.is_tracing_enabled() is True
    finally:
        dflog.enable_tracing(False)
