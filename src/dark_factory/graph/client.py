"""Neo4j driver wrapper."""

from __future__ import annotations

import structlog
from neo4j import GraphDatabase, Session

from dark_factory.config import Neo4jConfig

log = structlog.get_logger()


class Neo4jClient:
    """Manages the Neo4j driver lifecycle."""

    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        # L14: password is SecretStr — extract before passing to the driver
        password = config.password.get_secret_value() if hasattr(config.password, "get_secret_value") else config.password
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, password),
        )
        log.info("neo4j_client_created", uri=config.uri, database=config.database)

    def verify(self) -> None:
        """Verify connectivity to Neo4j."""
        self._driver.verify_connectivity()

    def session(self, **kwargs) -> Session:  # type: ignore[type-arg]
        """Return a new Neo4j session."""
        return self._driver.session(database=self.config.database, **kwargs)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
