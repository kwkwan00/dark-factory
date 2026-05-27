"""Tests for timeout-related configuration fields added to Neo4jConfig and PipelineConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# ── Neo4jConfig timeout fields ────────────────────────────────────────────────


def test_neo4j_config_default_timeouts():
    from dark_factory.config import Neo4jConfig
    cfg = Neo4jConfig()
    assert cfg.connection_timeout == 30
    assert cfg.connection_acquisition_timeout == 60
    assert cfg.max_connection_pool_size == 50


def test_neo4j_config_custom_timeouts():
    from dark_factory.config import Neo4jConfig
    cfg = Neo4jConfig(
        connection_timeout=15,
        connection_acquisition_timeout=90,
        max_connection_pool_size=25,
    )
    assert cfg.connection_timeout == 15
    assert cfg.connection_acquisition_timeout == 90
    assert cfg.max_connection_pool_size == 25


def test_neo4j_config_connection_timeout_lower_bound():
    from dark_factory.config import Neo4jConfig
    with pytest.raises(ValidationError):
        Neo4jConfig(connection_timeout=4)  # ge=5


def test_neo4j_config_connection_timeout_upper_bound():
    from dark_factory.config import Neo4jConfig
    with pytest.raises(ValidationError):
        Neo4jConfig(connection_timeout=121)  # le=120


def test_neo4j_config_acquisition_timeout_lower_bound():
    from dark_factory.config import Neo4jConfig
    with pytest.raises(ValidationError):
        Neo4jConfig(connection_acquisition_timeout=9)  # ge=10


# ── PipelineConfig timeout fields ─────────────────────────────────────────────


def test_pipeline_config_default_stage_timeouts():
    from dark_factory.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.ingest_timeout_seconds == 300
    assert cfg.spec_timeout_seconds == 3600
    assert cfg.spec_recon_timeout_seconds == 1200
    assert cfg.graph_timeout_seconds == 120
    assert cfg.swarm_feature_timeout_seconds == 3600


def test_pipeline_config_custom_stage_timeouts():
    from dark_factory.config import PipelineConfig
    cfg = PipelineConfig(
        ingest_timeout_seconds=60,
        spec_timeout_seconds=300,
        spec_recon_timeout_seconds=120,
        graph_timeout_seconds=60,
        swarm_feature_timeout_seconds=1200,
    )
    assert cfg.ingest_timeout_seconds == 60
    assert cfg.spec_timeout_seconds == 300
    assert cfg.spec_recon_timeout_seconds == 120
    assert cfg.graph_timeout_seconds == 60
    assert cfg.swarm_feature_timeout_seconds == 1200


def test_pipeline_config_ingest_timeout_lower_bound():
    from dark_factory.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig(ingest_timeout_seconds=29)  # ge=30


def test_pipeline_config_spec_timeout_lower_bound():
    from dark_factory.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig(spec_timeout_seconds=59)  # ge=60


def test_pipeline_config_swarm_timeout_lower_bound():
    from dark_factory.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig(swarm_feature_timeout_seconds=59)  # ge=60


def test_pipeline_config_graph_timeout_upper_bound():
    from dark_factory.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig(graph_timeout_seconds=601)  # le=600


# ── Neo4jClient passes timeouts to driver ────────────────────────────────────


def test_neo4j_client_passes_connection_timeout_to_driver():
    """Neo4jClient must forward connection_timeout to GraphDatabase.driver()."""
    from unittest.mock import patch, MagicMock
    from dark_factory.config import Neo4jConfig
    from dark_factory.graph.client import Neo4jClient

    cfg = Neo4jConfig(
        uri="bolt://localhost:7687",
        connection_timeout=15,
        connection_acquisition_timeout=45,
        max_connection_pool_size=10,
    )

    with patch("dark_factory.graph.client.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value = MagicMock()
        Neo4jClient(cfg)

    call_kwargs = mock_driver.call_args.kwargs
    assert call_kwargs["connection_timeout"] == 15
    assert call_kwargs["connection_acquisition_timeout"] == 45
    assert call_kwargs["max_connection_pool_size"] == 10


def test_neo4j_client_passes_default_timeouts_to_driver():
    """Defaults (30s / 60s / 50 pool) are forwarded to the driver when not overridden."""
    from unittest.mock import patch, MagicMock
    from dark_factory.config import Neo4jConfig
    from dark_factory.graph.client import Neo4jClient

    cfg = Neo4jConfig(uri="bolt://localhost:7687")

    with patch("dark_factory.graph.client.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value = MagicMock()
        Neo4jClient(cfg)

    call_kwargs = mock_driver.call_args.kwargs
    assert call_kwargs["connection_timeout"] == 30
    assert call_kwargs["connection_acquisition_timeout"] == 60
    assert call_kwargs["max_connection_pool_size"] == 50
