"""Configuration loading: config.toml defaults + environment variable overrides."""

from __future__ import annotations

import functools
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
    # Connection-level timeouts (seconds). connection_timeout is the TCP
    # handshake / TLS deadline; connection_acquisition_timeout is how long
    # to wait for a slot in the driver's connection pool.
    connection_timeout: int = Field(default=30, ge=5, le=120)
    connection_acquisition_timeout: int = Field(default=60, ge=10, le=300)
    max_connection_pool_size: int = Field(default=50, ge=5, le=200)


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
    planner: str | None = "claude-opus-4-6"
    coder: str | None = None
    reviewer: str | None = "claude-opus-4-6"
    tester: str | None = None

    # Deep agents (direct API calls)
    deep_analysis: str | None = None   # Category A: read-only analysis
    deep_codegen: str | None = None    # Category B: file-creating tools

    # Pipeline stages (non-swarm)
    spec: str | None = "claude-opus-4-6"
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
    # Post-generation reconciliation: validate requirement coverage,
    # fix cross-spec dependencies, detect cycles, and strip phantom
    # references before committing to the knowledge graph.
    enable_spec_reconciliation: bool = True
    max_specs_per_requirement: int = Field(default=12, ge=1, le=32)
    # Preflight skip: when True, the spec stage queries Neo4j for any
    # target spec ids that already exist and short-circuits the swarm
    # refinement loop for those — re-running the pipeline on an
    # unchanged requirements directory becomes a near-no-op instead of
    # re-spending the full LLM budget. Existing Spec objects are loaded
    # from Neo4j and passed through to downstream stages unchanged.
    # Turn off via the Settings tab or ``REUSE_EXISTING_SPECS=0`` for a
    # forced full regeneration.
    reuse_existing_specs: bool = False

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

    # Requirements Refinery: model, reasoning effort, turn budget, and
    # timeout for the deep agent that performs multi-pass analysis.
    refinery_model: str = "gpt-5.4"
    refinery_reasoning_effort: str = "xhigh"
    refinery_max_turns: int = Field(default=40, ge=5, le=100)
    refinery_timeout_seconds: int = Field(default=900, ge=60, le=3600)

    # ────────────────────────────────────────────────────────────
    # Adversarial Refinery — Parnas modules + LangGraph debate.
    # The per-requirement debate at ``refinery/debate/graph.py`` is
    # the only refinery path; the legacy single-agent flow has been
    # removed.
    # ────────────────────────────────────────────────────────────

    # Debate structure
    refinery_debate_max_rounds: int = Field(default=3, ge=1, le=8)
    refinery_debate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    refinery_research_cap: int = Field(default=1, ge=0, le=3)
    refinery_escalation_cap: int = Field(default=1, ge=0, le=3)
    refinery_model_strong: str | None = None  # escalation-tier model; falls back to base_model

    # Role overrides
    refinery_roles_enabled: list[str] = Field(
        default_factory=lambda: [
            "product", "engineering", "security",
            "operations", "cost", "judge",
        ]
    )
    refinery_role_models: dict[str, str] = Field(default_factory=dict)
    refinery_role_reasoning_effort: dict[str, str] = Field(default_factory=dict)
    refinery_role_filter_overrides: dict[str, dict] = Field(default_factory=dict)

    # Context / retrieval
    refinery_context_vector_limit: int = Field(default=12, ge=1, le=50)
    refinery_context_bm25_limit: int = Field(default=10, ge=1, le=50)
    refinery_context_graph_weight: float = Field(default=1.5, ge=0.5, le=5.0)
    refinery_context_cache_size: int = Field(default=256, ge=16, le=4096)

    # Judge — semantic (DeepEval)
    refinery_judge_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "clarity": 0.7, "testability": 0.7, "feasibility": 0.7,
            "completeness": 0.7, "risk_coverage": 0.7,
            "reversibility": 0.7,
        }
    )
    refinery_judge_overall_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    refinery_judge_aggregation: str = "min"  # "min" | "mean" | "weighted"
    refinery_judge_timeout_seconds: int = Field(default=120, ge=10, le=600)
    refinery_judge_fallback_enabled: bool = True

    # Multi-pass set-level Judge — caps how many critique-and-ratify
    # rounds the cross-set review runs. Default 1 = single-shot
    # (equivalent to the original cross-set review behaviour); ramping
    # up is opt-in once operators have confidence in the loop.
    refinery_set_review_max_rounds: int = Field(default=1, ge=1, le=5)

    # Judge — rules gate (fused with LLM into one combined pipeline)
    refinery_rules_enabled: bool = True
    refinery_rules_disabled: list[str] = Field(default_factory=list)
    refinery_rules_extra_modules: list[str] = Field(default_factory=list)
    refinery_rules_inject_as_critique: bool = True
    refinery_rules_dimension_overrides: dict[str, str] = Field(default_factory=dict)
    refinery_rules_short_circuit_llm: bool = False
    refinery_rules_short_circuit_threshold: int = Field(default=3, ge=1, le=20)
    refinery_rules_penalty_blocker_cap: float = Field(default=0.5, ge=0.0, le=1.0)
    refinery_rules_penalty_warning_delta: float = Field(default=0.1, ge=0.0, le=1.0)

    # Research agent (layered sourcing)
    refinery_research_enabled_tiers: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5]
    )
    refinery_research_tier_budgets: dict[int, int] = Field(
        default_factory=lambda: {0: 8, 1: 6, 2: 4, 3: 4, 4: 2, 5: 6}
    )
    refinery_research_internal_sufficient_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0
    )
    refinery_research_editor_min_confidence: float = Field(
        default=0.55, ge=0.0, le=1.0
    )
    refinery_research_tier_trust_weights: dict[int, float] = Field(
        default_factory=lambda: {0: 1.0, 1: 0.95, 2: 0.90, 3: 0.80, 4: 0.60, 5: 0.30}
    )
    refinery_research_official_url_allowlist: list[str] = Field(
        default_factory=lambda: [
            "docs.aws.amazon.com", "learn.microsoft.com",
            "cloud.google.com", "docs.anthropic.com",
            "platform.openai.com", "fastapi.tiangolo.com",
            "qdrant.tech", "neo4j.com",
        ]
    )
    refinery_research_max_sources_per_call: int = Field(default=20, ge=1, le=100)

    # Institutional memory
    refinery_memory_kinds_enabled: list[str] = Field(
        default_factory=lambda: [
            "decision", "incident", "pattern", "constraint", "conflict",
        ]
    )
    refinery_conflict_emit_threshold: float = Field(default=0.4, ge=0.0, le=5.0)
    refinery_decision_emit_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    refinery_pattern_emit_generality: float = Field(default=0.6, ge=0.0, le=1.0)
    refinery_memory_dedup_by_kind: bool = True
    refinery_auto_save_on_apply: bool = True

    # Observability
    refinery_disagreement_escalate_threshold: float = Field(
        default=0.5, ge=0.0, le=5.0
    )
    refinery_langsmith_enabled: bool | None = None  # env-driven when None
    refinery_postgres_forensics_enabled: bool = True
    refinery_prometheus_metrics_enabled: bool = True

    # Swarm memory recall scoping — Phase 9 sneaky-touchpoint mitigation.
    # Today the swarm's ``MemoryRepository.search_memories`` returns all
    # memory kinds; once the refinery starts writing Decision/Constraint/
    # Conflict kinds to the same Qdrant collection, swarm agents would
    # see them unless explicitly scoped. Default preserves today's
    # behaviour by scoping to the 4 legacy kinds. Flip to include the
    # new kinds once validated.
    swarm_memory_kinds_enabled: list[str] = Field(
        default_factory=lambda: ["pattern", "mistake", "solution", "strategy"]
    )

    # Wall-clock timeouts (seconds) for each pipeline stage. These are
    # enforced via asyncio.wait_for() in ag_ui_bridge and via elapsed-time
    # checks in the LangGraph stream loop so a hung LLM call or blocked
    # DB write cannot stall the pipeline indefinitely.
    ingest_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    spec_timeout_seconds: int = Field(default=3600, ge=60, le=7200)
    spec_recon_timeout_seconds: int = Field(default=1200, ge=30, le=1800)
    graph_timeout_seconds: int = Field(default=120, ge=10, le=600)
    # Per-feature swarm wall-clock timeout. Checked on every LangGraph
    # chunk so it fires even if the LLM is mid-call.
    swarm_feature_timeout_seconds: int = Field(default=3600, ge=60, le=7200)

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
    # When true, classes decorated with ``@trace_methods`` emit
    # ``call_entry`` / ``call_exit`` DEBUG events for every public
    # method (with arg *shape*, never values). Off by default;
    # ``DARK_FACTORY_LOG_TRACE=1`` is the env-var shorthand.
    trace_calls: bool = False


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
    # Days since last boost/demote before decay applies. Memories that
    # received feedback within this window keep their score — only
    # stale memories decay.
    decay_grace_days: int = 7
    boost_delta: float = 0.1
    demote_delta: float = 0.1
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


def _env_bool(name: str) -> bool | None:
    """Parse a boolean env var, returning None if unset."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
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
    if (pg_enabled := _env_bool("POSTGRES_ENABLED")) is not None:
        settings.postgres.enabled = pg_enabled
    if pg_url := os.getenv("POSTGRES_URL"):
        settings.postgres.url = pg_url
    if pg_password := os.getenv("POSTGRES_PASSWORD"):
        settings.postgres.password = SecretStr(pg_password)

    # Prometheus admin endpoint — only used by the clear-all flow
    if (prom_enabled := _env_bool("PROMETHEUS_ENABLED")) is not None:
        settings.prometheus.enabled = prom_enabled
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
    if (decomp_val := _env_bool("ENABLE_SPEC_DECOMPOSITION")) is not None:
        settings.pipeline.enable_spec_decomposition = decomp_val
    if (recon_val := _env_bool("ENABLE_SPEC_RECONCILIATION")) is not None:
        settings.pipeline.enable_spec_reconciliation = recon_val
    if (val := _env_int("MAX_SPECS_PER_REQUIREMENT")) is not None:
        settings.pipeline.max_specs_per_requirement = val
    if (reuse_val := _env_bool("REUSE_EXISTING_SPECS")) is not None:
        settings.pipeline.reuse_existing_specs = reuse_val

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
    if (episodic_val := _env_bool("ENABLE_EPISODIC_MEMORY")) is not None:
        settings.pipeline.enable_episodic_memory = episodic_val

    # Memory write-time dedup threshold
    if (val := _env_float("MEMORY_DEDUP_THRESHOLD")) is not None:
        settings.pipeline.memory_dedup_threshold = val

    # Max output tokens for swarm LLM calls (see PipelineConfig doc).
    if (val := _env_int("MAX_LLM_TOKENS")) is not None:
        settings.pipeline.max_llm_tokens = val

    # E2E validation (Phase 6) overrides
    if (e2e_val := _env_bool("ENABLE_E2E_VALIDATION")) is not None:
        settings.pipeline.enable_e2e_validation = e2e_val
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


def reload_settings(config_path: Path | None = None) -> Settings:
    """Clear the settings cache and reload from disk + env vars."""
    load_settings.cache_clear()
    return load_settings(config_path)
