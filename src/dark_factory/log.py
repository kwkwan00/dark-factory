"""Structured logging setup + optional call-tracing decorators.

Two modes share this module:

* **Event-based (default).** Modules call ``structlog.get_logger()``
  and emit named events with structured kwargs — the established
  pattern across the codebase.
* **Trace mode (opt-in).** When ``LoggingConfig.trace_calls`` is true
  *or* ``DARK_FACTORY_LOG_TRACE=1`` is set, classes decorated with
  :func:`trace_methods` emit ``call_entry`` / ``call_exit`` events at
  DEBUG level for every public method, including duration and the
  *shape* (type + length) of arguments. No argument values are ever
  logged — matches the CLAUDE.md rule that payloads carry semantics,
  not content.

The decorators are always installed; the per-call cost when trace
mode is off is a single bool check, so leaving them in place in
production is free.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar

import structlog

_F = TypeVar("_F", bound=Callable[..., Any])


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Set once at import from the env var so trace mode is live even
# before ``setup_logging`` runs (relevant for early-import side
# effects). ``enable_tracing`` lets ``setup_logging`` flip it on
# from config.
_TRACE_ENABLED: bool = _env_truthy("DARK_FACTORY_LOG_TRACE")

_tracer = structlog.get_logger("dark_factory.trace")


def enable_tracing(enabled: bool = True) -> None:
    """Globally toggle call tracing. Called by :func:`setup_logging`."""
    global _TRACE_ENABLED
    _TRACE_ENABLED = bool(enabled)


def is_tracing_enabled() -> bool:
    return _TRACE_ENABLED


def setup_logging(
    level: str = "INFO",
    fmt: str = "console",
    trace_calls: bool = False,
) -> None:
    """Configure structlog with console or JSON rendering.

    Pass ``trace_calls=True`` (or set ``DARK_FACTORY_LOG_TRACE=1``) to
    turn on the ``@trace_methods`` entry/exit decorators. Trace events
    emit at DEBUG, so the caller must also set ``level="DEBUG"`` to
    actually see them.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )

    # ``foreign_pre_chain`` applies the same shared processors
    # (timestamp, level, context vars) to log records emitted by
    # *stdlib* ``logging.getLogger`` calls — for example
    # ``claude_agent_sdk._internal.query.logger.error("Fatal error in
    # message reader: …")``. Without it, foreign records render as
    # bare message text with no timestamp / level prefix, which is
    # why the Claude Agent SDK's error logs used to appear unprefixed
    # in docker output.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # The Claude Agent SDK uses ``logging.getLogger(__name__)`` with
    # no explicit level, so it inherits root. Pin it to WARNING so
    # the chatty ``DEBUG``/``INFO`` chatter from its internal
    # subprocess transport doesn't flood the logs, while still
    # surfacing the ``error`` messages we care about (``Fatal error
    # in message reader``, transport close failures, etc.) through
    # structlog with proper timestamps + level tags.
    logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)

    if trace_calls or _env_truthy("DARK_FACTORY_LOG_TRACE"):
        enable_tracing(True)


# ─────────────────────────────────────────────────────────────────────
# Call tracing — @traced / @trace_methods
# ─────────────────────────────────────────────────────────────────────


_PRIMITIVE = (bool, int, float, complex, type(None))


def _arg_shape(value: Any) -> str:
    """Render a value's *shape* (type + length) without leaking content.

    Returns short strings like ``dict[3]``, ``str[42]``, ``list[5]``,
    ``RefinedRequirement`` (class name for non-collection objects).
    Never returns the value's ``repr()`` or any embedded text — the
    no-content-in-logs invariant assumed by the rest of the codebase
    extends to trace events too.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):  # bool subclasses int; check first
        return "bool"
    if isinstance(value, _PRIMITIVE):
        return type(value).__name__
    if isinstance(value, (str, bytes, bytearray)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, Mapping):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, Sequence):
        try:
            return f"{type(value).__name__}[{len(value)}]"
        except TypeError:
            return type(value).__name__
    return type(value).__name__


def _build_arg_summary(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    is_method: bool,
) -> dict[str, str]:
    """Pair argument names with their shape descriptions, dropping self/cls."""
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters)
    except (ValueError, TypeError):
        params = []

    if is_method and params and params[0] in ("self", "cls"):
        params = params[1:]
        args = args[1:]

    summary: dict[str, str] = {}
    for i, value in enumerate(args):
        name = params[i] if i < len(params) else f"arg{i}"
        summary[name] = _arg_shape(value)
    for name, value in kwargs.items():
        summary[name] = _arg_shape(value)
    return summary


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def traced(func: _F) -> _F:
    """Wrap a function/method to emit entry/exit events when trace mode is on.

    Handles sync, async, sync-generator, and async-generator callables.
    When trace mode is off the wrapper short-circuits to a direct call,
    so the steady-state cost is one bool check per invocation.
    """
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "<func>"))
    is_method = "." in qualname

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _TRACE_ENABLED:
                return await func(*args, **kwargs)
            _tracer.debug(
                "call_entry",
                method=qualname,
                args=_build_arg_summary(func, args, kwargs, is_method),
            )
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except BaseException as e:
                _tracer.debug(
                    "call_exit",
                    method=qualname,
                    duration_ms=_ms(t0),
                    error=type(e).__name__,
                )
                raise
            _tracer.debug(
                "call_exit",
                method=qualname,
                duration_ms=_ms(t0),
                returns=_arg_shape(result),
            )
            return result

        return _async_wrapper  # type: ignore[return-value]

    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def _async_gen_wrapper(*args: Any, **kwargs: Any):
            if not _TRACE_ENABLED:
                async for item in func(*args, **kwargs):
                    yield item
                return
            _tracer.debug(
                "call_entry",
                method=qualname,
                kind="async_gen",
                args=_build_arg_summary(func, args, kwargs, is_method),
            )
            t0 = time.perf_counter()
            count = 0
            try:
                async for item in func(*args, **kwargs):
                    count += 1
                    yield item
            except BaseException as e:
                _tracer.debug(
                    "call_exit",
                    method=qualname,
                    duration_ms=_ms(t0),
                    yielded=count,
                    error=type(e).__name__,
                )
                raise
            _tracer.debug(
                "call_exit",
                method=qualname,
                duration_ms=_ms(t0),
                yielded=count,
            )

        return _async_gen_wrapper  # type: ignore[return-value]

    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def _gen_wrapper(*args: Any, **kwargs: Any):
            if not _TRACE_ENABLED:
                yield from func(*args, **kwargs)
                return
            _tracer.debug(
                "call_entry",
                method=qualname,
                kind="generator",
                args=_build_arg_summary(func, args, kwargs, is_method),
            )
            t0 = time.perf_counter()
            count = 0
            try:
                for item in func(*args, **kwargs):
                    count += 1
                    yield item
            except BaseException as e:
                _tracer.debug(
                    "call_exit",
                    method=qualname,
                    duration_ms=_ms(t0),
                    yielded=count,
                    error=type(e).__name__,
                )
                raise
            _tracer.debug(
                "call_exit",
                method=qualname,
                duration_ms=_ms(t0),
                yielded=count,
            )

        return _gen_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _TRACE_ENABLED:
            return func(*args, **kwargs)
        _tracer.debug(
            "call_entry",
            method=qualname,
            args=_build_arg_summary(func, args, kwargs, is_method),
        )
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except BaseException as e:
            _tracer.debug(
                "call_exit",
                method=qualname,
                duration_ms=_ms(t0),
                error=type(e).__name__,
            )
            raise
        _tracer.debug(
            "call_exit",
            method=qualname,
            duration_ms=_ms(t0),
            returns=_arg_shape(result),
        )
        return result

    return _sync_wrapper  # type: ignore[return-value]


def trace_methods(cls: type) -> type:
    """Class decorator: wrap every public instance method with :func:`traced`.

    Wraps ``__init__`` and all non-dunder regular instance methods.
    Skips properties, classmethods, staticmethods, and any non-function
    attribute (descriptors, nested classes, class-level constants).
    Subclasses don't inherit the wrapping — decorate each concrete
    class explicitly.

    Use on behaviour-bearing classes (roles, repos, runners, stages,
    clients). Avoid on Pydantic / dataclass / Enum entities — there's
    no logic worth tracing and the wrappers would add overhead to
    constructor paths that get hit on every event.
    """
    for name, attr in list(cls.__dict__.items()):
        if isinstance(attr, (classmethod, staticmethod, property)):
            continue
        if name.startswith("__") and name != "__init__":
            continue
        if not inspect.isfunction(attr):
            continue
        setattr(cls, name, traced(attr))
    return cls
