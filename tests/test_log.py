"""Tests for ``dark_factory.log.setup_logging``.

Regression coverage for the ``foreign_pre_chain`` fix: without it,
stdlib log records emitted by third-party libraries (notably
``claude_agent_sdk._internal.query`` when its subprocess crashes) used
to render as bare message text with no timestamp, level, or
contextvars — polluting docker logs with unprefixed lines like

    Fatal error in message reader: Command failed with exit code 1
    Error output: Check stderr output for details

This module asserts that foreign records now pick up the same
shared processors as structlog-native records.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def reconfigure_root_logging():
    """Snapshot + restore the root logger around a test.

    ``setup_logging`` clears and replaces the root handler, which
    leaks into every subsequent test in the session. This fixture
    captures the pre-test handler/level and restores them after the
    test body runs so the rest of the suite is unaffected.
    """
    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_level = root.level
    prior_claude_level = logging.getLogger("claude_agent_sdk").level
    try:
        yield
    finally:
        root.handlers.clear()
        for h in prior_handlers:
            root.addHandler(h)
        root.setLevel(prior_level)
        logging.getLogger("claude_agent_sdk").setLevel(prior_claude_level)


def test_setup_logging_formats_foreign_log_records_with_level_and_timestamp(
    reconfigure_root_logging, capsys
):
    """A plain-stdlib ``logger.error`` call (simulating the Claude
    Agent SDK's ``_read_messages`` error path) must render through
    structlog with a timestamp and ``[error]`` level tag, not as a
    raw unprefixed line."""
    from dark_factory.log import setup_logging

    setup_logging(level="INFO", fmt="console")

    # Simulate exactly what ``claude_agent_sdk._internal.query`` does:
    #   logger = logging.getLogger(__name__)
    #   logger.error(f"Fatal error in message reader: {e}")
    foreign_logger = logging.getLogger("claude_agent_sdk._internal.query")
    foreign_logger.error(
        "Fatal error in message reader: Command failed with exit code 1"
    )

    captured = capsys.readouterr()
    output = captured.out + captured.err
    # The error message itself must appear.
    assert "Fatal error in message reader" in output
    # And — the whole point of the fix — it must carry a level tag.
    # ``ConsoleRenderer`` writes ``[error    ]`` for error-level
    # records. Without ``foreign_pre_chain`` this tag was missing.
    assert "error" in output.lower()
    # A year prefix proves the TimeStamper processor ran on the
    # foreign record. We can't assert the exact timestamp, but the
    # ISO format always starts with the current century.
    assert "20" in output  # year prefix from iso timestamp


def test_setup_logging_pins_claude_agent_sdk_logger_to_warning(
    reconfigure_root_logging,
):
    """The Claude Agent SDK's internal transport emits chatty
    ``DEBUG``/``INFO`` records during normal operation (subprocess
    startup, stream open/close, each control message). Letting those
    through at the root level floods the logs. ``setup_logging`` must
    pin the ``claude_agent_sdk`` logger to ``WARNING`` so only the
    messages we actually care about — ``Fatal error in message reader``
    and friends — make it through."""
    from dark_factory.log import setup_logging

    setup_logging(level="DEBUG", fmt="console")

    sdk_logger = logging.getLogger("claude_agent_sdk")
    # WARNING = 30. Must be >= WARNING regardless of the root level.
    assert sdk_logger.level >= logging.WARNING


def test_setup_logging_still_renders_structlog_native_events(
    reconfigure_root_logging, capsys
):
    """Adding ``foreign_pre_chain`` must not break the native
    structlog path. Events logged via ``structlog.get_logger()`` must
    still render with their key=value fields intact."""
    import structlog

    from dark_factory.log import setup_logging

    setup_logging(level="INFO", fmt="console")

    log = structlog.get_logger(__name__)
    log.info("pipeline_started", run_id="run-123", features=2)

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "pipeline_started" in output
    assert "run-123" in output
    assert "features" in output
