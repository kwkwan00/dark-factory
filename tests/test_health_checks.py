"""Unit tests for the services health helpers in ``dark_factory.ui.health``.

Focuses on the Postgres + Prometheus checks that were added alongside
the existing Neo4j + Qdrant ones. The goal is to cover the branches the
UI health section cares about:
- disabled-in-settings → reported as OK with a neutral message
- enabled-but-no-client → reported as not OK (init failed)
- enabled-and-reachable → returns True
- enabled-and-broken → returns False with the exception message
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ── check_postgres ─────────────────────────────────────────────────────────


def test_check_postgres_disabled_returns_ok():
    """When ``settings.postgres.enabled=False`` the helper returns (True,
    'Disabled …') so the UI renders a neutral dot rather than an error."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_postgres

    settings = Settings()
    settings.postgres.enabled = False

    ok, msg = check_postgres(settings, client=None)
    assert ok is True
    assert "disabled" in msg.lower()


def test_check_postgres_enabled_without_client_returns_error():
    """Enabled in settings but no pool initialised → not OK. This is the
    state the app ends up in if the Postgres service was unreachable at
    boot."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_postgres

    settings = Settings()
    settings.postgres.enabled = True

    ok, msg = check_postgres(settings, client=None)
    assert ok is False
    assert "not initialised" in msg.lower()


def test_check_postgres_enabled_with_healthy_client_returns_ok():
    """Happy path: enabled + pool answers SELECT 1. The check uses a
    MagicMock as the client so we don't need a real Postgres."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_postgres

    settings = Settings()
    settings.postgres.enabled = True

    # Mock the client.connection() context manager chain:
    #   with client.connection() as conn:
    #       with conn.cursor() as cur:
    #           cur.execute("SELECT 1"); cur.fetchone()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (1,)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_client = MagicMock()
    mock_client.connection.return_value.__enter__.return_value = mock_conn

    ok, msg = check_postgres(settings, client=mock_client)
    assert ok is True
    assert "connected" in msg.lower()
    mock_cur.execute.assert_called_with("SELECT 1")


def test_check_postgres_enabled_with_broken_client_returns_error():
    """If the pool raises (e.g. Postgres container died) the helper
    swallows the exception and reports it in the message string."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_postgres

    settings = Settings()
    settings.postgres.enabled = True

    mock_client = MagicMock()
    mock_client.connection.side_effect = RuntimeError("pool exhausted")

    ok, msg = check_postgres(settings, client=mock_client)
    assert ok is False
    assert "pool exhausted" in msg


# ── check_prometheus ───────────────────────────────────────────────────────


def test_check_prometheus_disabled_returns_ok():
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_prometheus

    settings = Settings()
    settings.prometheus.enabled = False

    ok, msg = check_prometheus(settings)
    assert ok is True
    assert "disabled" in msg.lower()


def test_check_prometheus_ready_returns_ok():
    """Happy path: /-/ready responds 200."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_prometheus

    settings = Settings()
    settings.prometheus.enabled = True
    settings.prometheus.url = "http://prom.test:9090"

    class FakeResponse:
        status_code = 200

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.calls: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url: str) -> FakeResponse:
            self.calls.append(url)
            return FakeResponse()

    with patch("httpx.Client", FakeClient):
        ok, msg = check_prometheus(settings)

    assert ok is True
    assert "prom.test:9090" in msg


def test_check_prometheus_not_ready_returns_error():
    """Non-200 from /-/ready → not OK with the status code in the message."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_prometheus

    settings = Settings()
    settings.prometheus.enabled = True
    settings.prometheus.url = "http://prom.test:9090"

    class FakeResponse:
        status_code = 503

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    with patch("httpx.Client", FakeClient):
        ok, msg = check_prometheus(settings)

    assert ok is False
    assert "503" in msg


def test_check_prometheus_unreachable_returns_error():
    """httpx connection error → not OK with the exception message attached."""
    from dark_factory.config import Settings
    from dark_factory.ui.health import check_prometheus

    settings = Settings()
    settings.prometheus.enabled = True
    settings.prometheus.url = "http://prom.test:9090"

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    with patch("httpx.Client", FakeClient):
        ok, msg = check_prometheus(settings)

    assert ok is False
    assert "connection refused" in msg


# ── check_all wiring ───────────────────────────────────────────────────────


def test_check_all_includes_postgres_and_prometheus():
    """The aggregated check_all must surface all four services so the
    UI can render a dot per docker-compose data store."""
    from dark_factory.config import Settings
    from dark_factory.ui import health

    settings = Settings()
    settings.postgres.enabled = False
    settings.prometheus.enabled = False

    # Stub the individual checks so this test focuses on wiring, not
    # network calls. Neo4j + Qdrant branches are covered elsewhere.
    with (
        patch.object(health, "check_neo4j", return_value=(True, "ok")),
        patch.object(health, "check_qdrant", return_value=(True, "ok")),
    ):
        result = health.check_all(settings)

    assert set(result.keys()) == {"neo4j", "qdrant", "postgres", "prometheus"}
    assert result["postgres"][0] is True
    assert result["prometheus"][0] is True
