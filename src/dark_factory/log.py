"""Structured logging setup using structlog."""

from __future__ import annotations

import logging

import structlog


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog with console or JSON rendering."""
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
