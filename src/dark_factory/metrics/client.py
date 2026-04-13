"""Thin wrapper around a psycopg ConnectionPool."""

from __future__ import annotations

import structlog
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from dark_factory.config import PostgresConfig

log = structlog.get_logger()


class PostgresClient:
    """Connection pool facade for Postgres metric writes and reads.

    The pool is opened eagerly with ``open=True`` so connection failures
    surface at startup rather than on the first write. Callers should use
    ``with client.connection() as conn:`` and always rely on context
    managers for cleanup.
    """

    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        dsn = self._build_dsn(config)
        self._pool = ConnectionPool(
            dsn,
            min_size=config.pool_min_size,
            max_size=config.pool_max_size,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        self._pool.open(wait=True, timeout=10.0)

    @staticmethod
    def _build_dsn(config: PostgresConfig) -> str:
        """Return a DSN with the password from ``config.password`` applied.

        When ``config.password`` is non-empty it **always** wins — if
        the URL already embeds credentials, the password component is
        replaced so the operator only needs to set ``POSTGRES_PASSWORD``
        in one place.  When ``config.password`` is empty the URL is
        used verbatim (credentials and all).
        """
        raw = config.password.get_secret_value()
        if not raw:
            return config.url

        # Replace or inject the password in the URL.
        if "://" not in config.url:
            return config.url

        scheme, rest = config.url.split("://", 1)
        if "@" in rest:
            # URL has user(:pass)?@host — replace just the password.
            userinfo, hostpart = rest.split("@", 1)
            user = userinfo.split(":", 1)[0]
            return f"{scheme}://{user}:{raw}@{hostpart}"
        # No credentials in URL — prepend :<password>@
        return f"{scheme}://:{raw}@{rest}"

    def connection(self):
        """Context manager yielding a pooled connection."""
        return self._pool.connection()

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception as exc:  # pragma: no cover
            log.warning("postgres_pool_close_failed", error=str(exc))

    @property
    def closed(self) -> bool:
        return self._pool.closed

    def pool_stats(self) -> dict[str, int]:
        """L5 fix: return pool stats for the Prometheus sampler.

        Returns a dict with ``size`` (configured max), ``idle``
        (currently available connections), ``active`` (checked out
        in use), ``waiting`` (requesters blocked on pool
        exhaustion). Used by the BackgroundLoop sampler to emit
        pool-depth gauges every 10s so dashboards can alert on pool
        starvation before it causes pipeline stalls.

        All fields default to 0 if psycopg_pool doesn't expose the
        stat we expect — keeps this method stable across psycopg_pool
        versions.
        """
        try:
            stats = self._pool.get_stats()
        except Exception:
            return {"size": 0, "idle": 0, "active": 0, "waiting": 0}
        return {
            "size": int(stats.get("pool_size", 0) or 0),
            "idle": int(stats.get("pool_available", 0) or 0),
            "active": int(
                (stats.get("pool_size", 0) or 0)
                - (stats.get("pool_available", 0) or 0)
            ),
            "waiting": int(stats.get("requests_waiting", 0) or 0),
        }
