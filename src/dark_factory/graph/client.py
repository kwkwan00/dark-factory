"""Neo4j driver wrapper."""

from __future__ import annotations

from neo4j import GraphDatabase

from dark_factory.config import Neo4jConfig


class Neo4jClient:
    """Manages the Neo4j driver lifecycle."""

    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    def verify(self) -> None:
        """Verify connectivity to Neo4j."""
        self._driver.verify_connectivity()

    def session(self, **kwargs):
        """Return a new Neo4j session."""
        return self._driver.session(database=self.config.database, **kwargs)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
