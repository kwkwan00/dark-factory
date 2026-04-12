"""Health checks for external services (Neo4j, Qdrant, Postgres, Prometheus)."""

from __future__ import annotations

import structlog

from dark_factory.config import Settings

log = structlog.get_logger()


def check_neo4j(settings: Settings, client: object | None = None) -> tuple[bool, str]:
    """Check Neo4j connectivity. Returns (ok, message).

    M14 fix: when called with an existing ``client`` (the shared one from
    ``app.state.neo4j_client``), reuse it instead of creating + closing a
    new driver per request — avoids handshake overhead under polling.
    """
    try:
        if client is not None:
            client.verify()  # type: ignore[attr-defined]
            return True, f"Connected to {settings.neo4j.uri}"

        from dark_factory.graph.client import Neo4jClient

        new_client = Neo4jClient(settings.neo4j)
        try:
            new_client.verify()
            log.debug("health_neo4j_ok", uri=settings.neo4j.uri)
            return True, f"Connected to {settings.neo4j.uri}"
        finally:
            new_client.close()
    except Exception as exc:
        log.warning("health_neo4j_failed", error=str(exc))
        return False, f"Neo4j error: {exc}"


def check_qdrant(settings: Settings) -> tuple[bool, str]:
    """Check Qdrant connectivity. Returns (ok, message)."""
    if not settings.qdrant.enabled:
        return True, "Disabled (using Neo4j fallback)"
    try:
        from dark_factory.vector.client import QdrantClientWrapper

        client = QdrantClientWrapper(settings.qdrant)
        try:
            if client.is_available():
                return True, f"Connected to {settings.qdrant.url}"
            else:
                return False, f"Qdrant unavailable at {settings.qdrant.url}"
        finally:
            client.close()
    except Exception as exc:
        return False, f"Qdrant error: {exc}"


def check_postgres(
    settings: Settings, client: object | None = None
) -> tuple[bool, str]:
    """Check Postgres (metrics store) connectivity. Returns (ok, message).

    When ``client`` is provided (``app.state.metrics_client``) we reuse
    its connection pool and run a trivial ``SELECT 1`` to verify the
    server is reachable — this is cheap and mirrors a real query path.

    When no client is available we don't dial a new pool here: opening
    one just for a health check is expensive and the client factory
    already reports pool startup failures at app boot. Instead we
    report the configured enabled flag so the UI can distinguish
    "disabled" from "enabled but unavailable".
    """
    if not settings.postgres.enabled:
        return True, "Disabled (metrics store off)"
    if client is None:
        return False, "Postgres enabled but client not initialised"
    try:
        with client.connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "Connected (metrics pool)"
    except Exception as exc:
        log.warning("health_postgres_failed", error=str(exc))
        return False, f"Postgres error: {exc}"


def check_prometheus(settings: Settings) -> tuple[bool, str]:
    """Check Prometheus server reachability via ``/-/ready``. Returns (ok, message).

    ``/-/ready`` is the canonical liveness endpoint exposed by every
    recent Prometheus build — it returns 200 once the TSDB is loaded and
    the scrape loop is running, regardless of whether the admin API is
    enabled. Short timeout so a wedged Prometheus can't block health
    polling.
    """
    cfg = getattr(settings, "prometheus", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return True, "Disabled"
    try:
        import httpx

        url = cfg.url.rstrip("/") + "/-/ready"
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(url)
        if resp.status_code == 200:
            return True, f"Ready at {cfg.url}"
        return False, f"Prometheus not ready (HTTP {resp.status_code})"
    except Exception as exc:
        log.warning("health_prometheus_failed", error=str(exc))
        return False, f"Prometheus error: {exc}"


def check_all(
    settings: Settings,
    neo4j_client: object | None = None,
    metrics_client: object | None = None,
) -> dict[str, tuple[bool, str]]:
    """Check all external services.

    Pass ``neo4j_client`` / ``metrics_client`` to reuse the shared
    drivers/pools from ``app.state`` instead of opening new ones per
    request.
    """
    return {
        "neo4j": check_neo4j(settings, client=neo4j_client),
        "qdrant": check_qdrant(settings),
        "postgres": check_postgres(settings, client=metrics_client),
        "prometheus": check_prometheus(settings),
    }
