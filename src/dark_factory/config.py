"""Configuration loading: config.toml defaults + environment variable overrides."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Neo4jConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    uri: str = "bolt://localhost:7687"
    database: str = "neo4j"
    user: str = Field(default="neo4j")
    # L14 fix: SecretStr so password isn't accidentally logged via model_dump
    password: SecretStr = Field(default=SecretStr(""))


class LLMConfig(BaseModel):
    # validate_assignment=True so Settings-tab PATCHes go through the
    # field validators instead of silently skipping them.
    model_config = ConfigDict(validate_assignment=True)

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"


class ModelRoutingConfig(BaseModel):
    """Per-role model overrides for multi-model routing.

    Any field left as ``None`` falls back to ``settings.llm.model``.
    This lets operators assign cheaper/faster models to simpler tasks
    and reserve the most capable model for complex ones.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Swarm agents
    planner: str | None = None
    coder: str | None = None
    reviewer: str | None = None
    tester: str | None = None

    # Deep agents (direct API calls)
    deep_analysis: str | None = None   # Category A: read-only analysis
    deep_codegen: str | None = None    # Category B: file-creating tools

    # Pipeline stages (non-swarm)
    spec: str | None = None
    ingest: str | None = None

    def resolve(self, role: str, fallback: str) -> str:
        """Return the model for *role*, falling back to *fallback*."""
        return getattr(self, role, None) or fallback


class PipelineConfig(BaseModel):
    # validate_assignment=True so runtime mutations from the Settings tab
    # go through Pydantic validators (range checks below).
    model_config = ConfigDict(validate_assignment=True)

    output_dir: str = "./output"
    max_parallel_features: int = Field(default=4, ge=1, le=8)
    max_parallel_specs: int = Field(default=4, ge=1, le=8)
    # Per-requirement spec refinement: each requirement runs up to N
    # generate→evaluate→refine handoffs, early-exiting when the average
    # eval score reaches the threshold.
    max_spec_handoffs: int = Field(default=5, ge=1, le=10)
    spec_eval_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Codegen swarm: max LangGraph handoffs allowed per feature swarm
    # before the run terminates. The adaptive strategy can lower this
    # mid-run via strategy_overrides.
    max_codegen_handoffs: int = Field(default=50, ge=5, le=100)
    # Spec decomposition: when enabled, each requirement is first planned
    # into multiple smaller sub-specs by an LLM planner before the
    # refinement loop. Each sub-spec then runs through the full
    # architect/critic loop independently. Produces more granular specs
    # that downstream swarm workers can implement in isolation.
    enable_spec_decomposition: bool = True
    max_specs_per_requirement: int = Field(default=12, ge=1, le=32)
    # Preflight skip: when True, the spec stage queries Neo4j for any
    # target spec ids that already exist and short-circuits the swarm
    # refinement loop for those — re-running the pipeline on an
    # unchanged requirements directory becomes a near-no-op instead of
    # re-spending the full LLM budget. Existing Spec objects are loaded
    # from Neo4j and passed through to downstream stages unchanged.
    # Turn off via the Settings tab or ``REUSE_EXISTING_SPECS=0`` for a
    # forced full regeneration.
    reuse_existing_specs: bool = True

    # Reconciliation phase: always runs AFTER every feature swarm
    # completes (assuming the swarms actually produced output — the
    # stage skips when every feature errored or the output dir is
    # empty). A single extended Claude Agent SDK invocation gets
    # ``cwd`` set to the run's output directory and runs a 6-step
    # pass (inventory → review → fix → validate → iterate → report)
    # using Read/Write/Edit/Glob/Grep/Bash tools. Goal is to catch
    # cross-feature issues that per-feature swarms can't see: broken
    # imports between features, inconsistent API shapes, missing
    # glue (main entry points, package manifests), unrunnable code.
    # Best-effort — a reconciliation failure does NOT fail the run.
    max_reconciliation_turns: int = Field(default=50, ge=1, le=500)
    reconciliation_timeout_seconds: int = Field(default=1800, ge=60, le=7200)

    # Self-healing: max retry attempts per layer / reconciliation pass.
    # Set to 0 to disable reflection-based retries entirely.
    max_layer_retries: int = Field(default=1, ge=0, le=3)
    max_reconciliation_retries: int = Field(default=1, ge=0, le=3)

    # Semantic dedup of requirements prior to spec generation. A real
    # requirements corpus assembled from multiple uploaded documents
    # (meeting notes + Word brief + spreadsheet) routinely contains
    # the same underlying requirement expressed multiple ways. This
    # threshold is the cosine similarity (text-embedding-3-large)
    # above which two requirements are considered duplicates and
    # collapsed into a single canonical entry before the Spec stage
    # runs. 0.90 is conservative — paraphrases typically land at
    # 0.92–0.97, distinct requirements well below 0.85. Raise to
    # tighten the clusters; lower to catch more paraphrases at the
    # risk of false-positive merges. Dedup is always on when the
    # embedding service is available; this field has no "enable"
    # toggle because the feature is a correctness guarantee, not an
    # optimisation.
    requirement_dedup_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0
    )

    # Phase 6: end-to-end validation via Playwright. Runs AFTER
    # reconciliation (only if reconciliation returned "clean" or
    # "partial" — an errored reconciliation skips E2E because the
    # code almost certainly won't even start). A single extended
    # Claude Agent SDK invocation with ``cwd`` set to the run's
    # output directory detects whether the generated code is a web
    # application, installs Playwright if needed, writes smoke
    # tests derived from the specs' acceptance criteria, starts
    # the server in the background, runs the tests across every
    # browser in ``e2e_browsers``, and writes ``E2E_REPORT.md``.
    # Best-effort — failures do NOT fail the run; the pipeline
    # still delivers the code and reconciliation report. The
    # Docker image bundles chromium + firefox + webkit binaries so
    # the default browser matrix ships fully functional.
    enable_e2e_validation: bool = True
    max_e2e_turns: int = Field(default=40, ge=1, le=500)
    e2e_timeout_seconds: int = Field(default=1200, ge=60, le=7200)
    e2e_browsers: list[str] = Field(
        default_factory=lambda: ["chromium", "firefox", "webkit"]
    )

    # Episodic memory (Stage 3): after every feature swarm completes,
    # synthesise a narrative Episode via a small LLM call, embed it
    # with text-embedding-3-large, and write it to Neo4j + Qdrant so
    # future Planners can recall past trajectories via
    # ``recall_episodes``. Turning this off saves one LLM call per
    # feature (~1k tokens) at the cost of losing the temporal
    # reasoning layer — agents fall back to the four semantic memory
    # types only.
    enable_episodic_memory: bool = True

    # Memory write-time deduplication threshold. Before creating a new
    # Pattern / Mistake / Solution / Strategy node, the repository
    # embeds the candidate text and searches for same-type same-
    # feature memories above this cosine similarity. Hits get boosted
    # instead of duplicated. Higher than ``requirement_dedup_threshold``
    # (0.90) because a false-positive memory merge is harder to untangle
    # than a false-positive requirement merge — memories feed every
    # agent decision. Set to 0.0 to disable dedup entirely (creates a
    # new node on every record_* call, matching pre-Tier-A behaviour).
    memory_dedup_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    # Max output tokens for every LLM call the swarm makes. LangChain's
    # default ChatAnthropic max_tokens is 1024 — far too small to hold
    # a ``write_file`` tool call for a real dashboard or multi-hundred-
    # line component. When the Coder agent hits that ceiling mid-
    # generation, Claude stops with ``stop_reason="max_tokens"``, the
    # tool_use JSON is truncated mid-string, Pydantic rejects the
    # malformed kwargs, and LangGraph surfaces the failure as
    # ``Error invoking tool 'write_file' with kwargs {...}``. We pin
    # the default to 32768 (matching our direct AnthropicClient) so
    # real file writes fit comfortably in a single call. Claude Sonnet
    # 4.6 supports up to 64000 output tokens; raise if you see
    # recurring truncation on massive files.
    max_llm_tokens: int = Field(default=32768, ge=1024, le=64000)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"


class OpenSpecConfig(BaseModel):
    root_dir: str = "./openspec"


class MemoryConfig(BaseModel):
    database: str = "memory"
    enabled: bool = True


class QdrantConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    url: str = "http://localhost:6333"
    # L9 fix: use SecretStr so API key is not exposed in logs/serialization
    api_key: SecretStr = SecretStr("")
    collection_prefix: str = "dark_factory"
    embedding_model: str = "text-embedding-3-large"
    enabled: bool = True


class WatchConfig(BaseModel):
    enabled: bool = False
    paths: list[str] = ["./openspec/specs"]
    debounce_seconds: int = 5
    auto_run: bool = True


class EvaluationConfig(BaseModel):
    # validate_assignment=True so Settings-tab PATCHes go through the
    # field validators instead of silently skipping them.
    model_config = ConfigDict(validate_assignment=True)

    # DeepEval judge model. Mirrors evaluation/metrics.py's ``_eval_model``
    # module global — ``load_settings`` pushes this value into the module
    # via ``set_eval_model`` so runtime Settings-tab changes propagate.
    # Default mirrors the env var fallback in metrics.py.
    eval_model: str = "gpt-5.4"
    base_threshold: float = 0.5
    adaptive: bool = True
    decay_factor: float = 0.95
    boost_delta: float = 0.1
    demote_delta: float = 0.05
    trend_window: int = 5
    threshold_min: float = 0.3
    threshold_max: float = 0.9
    strategy_threshold: float = 0.5


class PostgresConfig(BaseModel):
    """PostgreSQL for metrics/telemetry (swarm, eval, LLM calls, runs).

    Disabled by default so local `uv run pytest` doesn't require a running
    Postgres instance. In docker-compose the `dark-factory` service sets
    ``POSTGRES_ENABLED=true`` and provides ``POSTGRES_URL``.
    """

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = False
    url: str = "postgresql://darkfactory:darkfactory@localhost:5432/darkfactory_metrics"
    # Password can also be embedded in ``url``. When set separately, it
    # overrides the password component of ``url`` so secrets can be kept
    # out of config.toml.
    password: SecretStr = Field(default=SecretStr(""))
    pool_min_size: int = Field(default=1, ge=1, le=10)
    pool_max_size: int = Field(default=5, ge=1, le=50)
    # Drop progress events on the floor when the recorder queue overflows
    # rather than blocking the pipeline.
    recorder_queue_size: int = Field(default=2000, ge=100, le=50000)


class PrometheusConfig(BaseModel):
    """Prometheus server settings.

    The in-process prometheus_client collectors are always on (they're
    zero-cost in-memory counters). This config only controls the remote
    TSDB admin API the admin clear-all endpoint talks to — the URL is
    used to call ``/api/v1/admin/tsdb/delete_series`` to wipe stored
    time series.
    """

    model_config = ConfigDict(validate_assignment=True)

    # When false the admin clear-all flow skips the remote delete but
    # still resets in-process collectors.
    enabled: bool = True
    # Base URL of the Prometheus server. In docker-compose this resolves
    # to the ``prometheus`` service on the internal network.
    url: str = "http://prometheus:9090"


class Settings(BaseModel):
    neo4j: Neo4jConfig = Neo4jConfig()
    llm: LLMConfig = LLMConfig()
    model_routing: ModelRoutingConfig = ModelRoutingConfig()
    pipeline: PipelineConfig = PipelineConfig()
    logging: LoggingConfig = LoggingConfig()
    openspec: OpenSpecConfig = OpenSpecConfig()
    memory: MemoryConfig = MemoryConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    qdrant: QdrantConfig = QdrantConfig()
    watch: WatchConfig = WatchConfig()
    postgres: PostgresConfig = PostgresConfig()
    prometheus: PrometheusConfig = PrometheusConfig()


def _env_int(name: str) -> int | None:
    """Parse an int env var, returning None if unset or malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer value for %s: %r", name, raw)
        return None


def _env_float(name: str) -> float | None:
    """Parse a float env var, returning None if unset or malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-float value for %s: %r", name, raw)
        return None


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config.toml, then overlay environment variables."""
    data: dict = {}

    if config_path is None:
        config_path = Path("config.toml")

    if config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

    settings = Settings(**data)

    # M5 fix: validate_assignment=True on Neo4jConfig/QdrantConfig ensures
    # env var overrides go through Pydantic validation.
    if neo4j_uri := os.getenv("NEO4J_URI"):
        settings.neo4j.uri = neo4j_uri
    if neo4j_user := os.getenv("NEO4J_USER"):
        settings.neo4j.user = neo4j_user
    if neo4j_password := os.getenv("NEO4J_PASSWORD"):
        settings.neo4j.password = SecretStr(neo4j_password)
    else:
        # L8 fix: warn if no password is set
        if not settings.neo4j.password.get_secret_value():
            logger.warning("NEO4J_PASSWORD not set — connecting with empty password")

    if qdrant_url := os.getenv("QDRANT_URL"):
        settings.qdrant.url = qdrant_url
    if qdrant_api_key := os.getenv("QDRANT_API_KEY"):
        settings.qdrant.api_key = SecretStr(qdrant_api_key)

    # Postgres metrics store — disabled unless explicitly enabled
    if (pg_enabled := os.getenv("POSTGRES_ENABLED")) is not None:
        settings.postgres.enabled = pg_enabled.strip().lower() in (
            "1", "true", "yes", "on",
        )
    if pg_url := os.getenv("POSTGRES_URL"):
        settings.postgres.url = pg_url
    if pg_password := os.getenv("POSTGRES_PASSWORD"):
        settings.postgres.password = SecretStr(pg_password)

    # Prometheus admin endpoint — only used by the clear-all flow
    if (prom_enabled := os.getenv("PROMETHEUS_ENABLED")) is not None:
        settings.prometheus.enabled = prom_enabled.strip().lower() in (
            "1", "true", "yes", "on",
        )
    if prom_url := os.getenv("PROMETHEUS_URL"):
        settings.prometheus.url = prom_url

    # M14 fix: numeric PipelineConfig overrides from env vars. Makes the
    # container-friendly path (env-based config) on par with config.toml.
    # validate_assignment=True on PipelineConfig means each assignment
    # runs Pydantic's Field(ge=..., le=...) range checks — bad values
    # raise rather than silently clamping.
    if (val := _env_int("MAX_PARALLEL_FEATURES")) is not None:
        settings.pipeline.max_parallel_features = val
    if (val := _env_int("MAX_PARALLEL_SPECS")) is not None:
        settings.pipeline.max_parallel_specs = val
    if (val := _env_int("MAX_SPEC_HANDOFFS")) is not None:
        settings.pipeline.max_spec_handoffs = val
    if (val := _env_int("MAX_CODEGEN_HANDOFFS")) is not None:
        settings.pipeline.max_codegen_handoffs = val
    if (val := _env_float("SPEC_EVAL_THRESHOLD")) is not None:
        settings.pipeline.spec_eval_threshold = val
    if output_dir := os.getenv("OUTPUT_DIR"):
        settings.pipeline.output_dir = output_dir
    if (decomp_raw := os.getenv("ENABLE_SPEC_DECOMPOSITION")) is not None:
        settings.pipeline.enable_spec_decomposition = decomp_raw.strip().lower() in (
            "1", "true", "yes", "on",
        )
    if (val := _env_int("MAX_SPECS_PER_REQUIREMENT")) is not None:
        settings.pipeline.max_specs_per_requirement = val
    if (reuse_raw := os.getenv("REUSE_EXISTING_SPECS")) is not None:
        settings.pipeline.reuse_existing_specs = reuse_raw.strip().lower() in (
            "1", "true", "yes", "on",
        )

    # Reconciliation phase overrides
    if (val := _env_int("MAX_RECONCILIATION_TURNS")) is not None:
        settings.pipeline.max_reconciliation_turns = val
    if (val := _env_int("RECONCILIATION_TIMEOUT_SECONDS")) is not None:
        settings.pipeline.reconciliation_timeout_seconds = val
    if (val := _env_int("MAX_LAYER_RETRIES")) is not None:
        settings.pipeline.max_layer_retries = val
    if (val := _env_int("MAX_RECONCILIATION_RETRIES")) is not None:
        settings.pipeline.max_reconciliation_retries = val

    # Semantic requirement dedup threshold
    if (val := _env_float("REQUIREMENT_DEDUP_THRESHOLD")) is not None:
        settings.pipeline.requirement_dedup_threshold = val

    # Episodic memory toggle
    if (episodic_raw := os.getenv("ENABLE_EPISODIC_MEMORY")) is not None:
        settings.pipeline.enable_episodic_memory = episodic_raw.strip().lower() in (
            "1", "true", "yes", "on",
        )

    # Memory write-time dedup threshold
    if (val := _env_float("MEMORY_DEDUP_THRESHOLD")) is not None:
        settings.pipeline.memory_dedup_threshold = val

    # Max output tokens for swarm LLM calls (see PipelineConfig doc).
    if (val := _env_int("MAX_LLM_TOKENS")) is not None:
        settings.pipeline.max_llm_tokens = val

    # E2E validation (Phase 6) overrides
    if (e2e_enabled := os.getenv("ENABLE_E2E_VALIDATION")) is not None:
        settings.pipeline.enable_e2e_validation = e2e_enabled.strip().lower() in (
            "1", "true", "yes", "on",
        )
    if (val := _env_int("MAX_E2E_TURNS")) is not None:
        settings.pipeline.max_e2e_turns = val
    if (val := _env_int("E2E_TIMEOUT_SECONDS")) is not None:
        settings.pipeline.e2e_timeout_seconds = val
    if e2e_browsers_raw := os.getenv("E2E_BROWSERS"):
        # Comma-separated list → validated list. Anything outside
        # {chromium, firefox, webkit} is dropped with a warning
        # rather than crashing startup, because an operator typo
        # should not take down the whole container.
        allowed = {"chromium", "firefox", "webkit"}
        parsed = [b.strip().lower() for b in e2e_browsers_raw.split(",") if b.strip()]
        valid = [b for b in parsed if b in allowed]
        invalid = [b for b in parsed if b not in allowed]
        if invalid:
            logger.warning(
                "Ignoring unknown E2E_BROWSERS entries: %s (allowed: %s)",
                invalid,
                sorted(allowed),
            )
        if valid:
            settings.pipeline.e2e_browsers = valid

    # LLM model overrides — main swarm model + DeepEval judge model.
    # Both are also editable from the Settings tab at runtime.
    if llm_model := os.getenv("ANTHROPIC_MODEL"):
        settings.llm.model = llm_model
    if eval_model_env := os.getenv("EVAL_MODEL"):
        settings.evaluation.eval_model = eval_model_env

    # Per-role model routing overrides (env vars take precedence over config.toml).
    _routing_env_map = {
        "MODEL_PLANNER": "planner",
        "MODEL_CODER": "coder",
        "MODEL_REVIEWER": "reviewer",
        "MODEL_TESTER": "tester",
        "MODEL_DEEP_ANALYSIS": "deep_analysis",
        "MODEL_DEEP_CODEGEN": "deep_codegen",
        "MODEL_SPEC": "spec",
        "MODEL_INGEST": "ingest",
    }
    for env_var, field in _routing_env_map.items():
        if val := os.getenv(env_var):
            setattr(settings.model_routing, field, val)

    # Push the resolved eval model name into the metrics module global
    # so the DeepEval builders pick it up. Importing here (not at module
    # top) avoids a circular import between config → evaluation.metrics.
    try:
        from dark_factory.evaluation.metrics import set_eval_model

        set_eval_model(settings.evaluation.eval_model)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("eval_model_propagation_failed: %s", exc)

    # L3 fix: log after env var overrides are applied
    logger.info(
        "Settings loaded from %s (neo4j=%s pipeline=%s)",
        config_path,
        settings.neo4j.uri,
        {
            "max_parallel_features": settings.pipeline.max_parallel_features,
            "max_parallel_specs": settings.pipeline.max_parallel_specs,
            "max_spec_handoffs": settings.pipeline.max_spec_handoffs,
            "max_codegen_handoffs": settings.pipeline.max_codegen_handoffs,
            "spec_eval_threshold": settings.pipeline.spec_eval_threshold,
            "enable_spec_decomposition": settings.pipeline.enable_spec_decomposition,
            "max_specs_per_requirement": settings.pipeline.max_specs_per_requirement,
        },
    )

    return settings
