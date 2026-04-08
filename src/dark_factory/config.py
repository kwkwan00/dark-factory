"""Configuration loading: config.toml defaults + environment variable overrides."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    database: str = "neo4j"
    user: str = Field(default="neo4j")
    password: str = Field(default="")


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"


class PipelineConfig(BaseModel):
    output_dir: str = "./output"
    max_parallel_features: int = 4


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"


class OpenSpecConfig(BaseModel):
    root_dir: str = "./openspec"


class MemoryConfig(BaseModel):
    database: str = "memory"
    enabled: bool = True


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str = ""
    collection_prefix: str = "dark_factory"
    embedding_model: str = "text-embedding-3-large"
    enabled: bool = True


class EvaluationConfig(BaseModel):
    base_threshold: float = 0.5
    adaptive: bool = True
    decay_factor: float = 0.95
    boost_delta: float = 0.1
    demote_delta: float = 0.05
    trend_window: int = 5
    threshold_min: float = 0.3
    threshold_max: float = 0.9
    strategy_threshold: float = 0.5


class Settings(BaseModel):
    neo4j: Neo4jConfig = Neo4jConfig()
    llm: LLMConfig = LLMConfig()
    pipeline: PipelineConfig = PipelineConfig()
    logging: LoggingConfig = LoggingConfig()
    openspec: OpenSpecConfig = OpenSpecConfig()
    memory: MemoryConfig = MemoryConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    qdrant: QdrantConfig = QdrantConfig()


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config.toml, then overlay environment variables."""
    data: dict = {}

    if config_path is None:
        config_path = Path("config.toml")

    if config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

    settings = Settings(**data)

    # Environment variable overrides for secrets
    if neo4j_user := os.getenv("NEO4J_USER"):
        settings.neo4j.user = neo4j_user
    if neo4j_password := os.getenv("NEO4J_PASSWORD"):
        settings.neo4j.password = neo4j_password

    if qdrant_url := os.getenv("QDRANT_URL"):
        settings.qdrant.url = qdrant_url
    if qdrant_api_key := os.getenv("QDRANT_API_KEY"):
        settings.qdrant.api_key = qdrant_api_key

    return settings
