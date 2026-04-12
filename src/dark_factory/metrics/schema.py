"""Postgres schema for the metrics store.

Core tables (one row per event):

- ``pipeline_runs``          — lifecycle row per pipeline run
- ``progress_events``        — raw append-only audit log of every broker event
- ``eval_metrics``           — per-metric DeepEval rubric rows
- ``llm_calls``              — per-LLM-call telemetry (latency, tokens, cost)
- ``swarm_feature_events``   — per-feature lifecycle rows
- ``tool_calls``             — per-tool invocation with latency + success
- ``agent_stats``            — per-(run, feature, agent) rollup written at feature completion
- ``decomposition_stats``    — per-requirement planner outcome
- ``memory_operations``      — procedural memory create/recall/boost/demote/decay
- ``incidents``              — structured errors/timeouts/worker crashes
- ``artifact_writes``        — per-file output written by the swarm
- ``background_loop_samples``— periodic snapshot of the background event loop

Views (derived aggregates):

- ``v_cost_per_run``
- ``v_cost_per_phase``
- ``v_runs_per_day``
- ``v_pass_rate_per_metric``
- ``v_attempts_per_requirement``

All DDL uses ``IF NOT EXISTS`` / ``CREATE OR REPLACE`` so :func:`ensure_schema`
is idempotent and safe to run on every startup.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()


_DDL = """
-- ── pipeline_runs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    spec_count INTEGER NOT NULL DEFAULT 0,
    feature_count INTEGER NOT NULL DEFAULT 0,
    pass_rate DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at ON pipeline_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs (status);

-- ── progress_events ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS progress_events (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    event TEXT NOT NULL,
    feature TEXT,
    agent TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_progress_events_run_id ON progress_events (run_id);
CREATE INDEX IF NOT EXISTS idx_progress_events_event ON progress_events (event);
CREATE INDEX IF NOT EXISTS idx_progress_events_timestamp ON progress_events (timestamp DESC);

-- ── eval_metrics ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    requirement_id TEXT,
    spec_id TEXT,
    eval_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    passed BOOLEAN NOT NULL,
    threshold DOUBLE PRECISION,
    attempt INTEGER,
    role TEXT,
    reason TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_metrics_run_id ON eval_metrics (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_spec_id ON eval_metrics (spec_id);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_metric_name ON eval_metrics (metric_name);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_timestamp ON eval_metrics (timestamp DESC);

-- ── llm_calls ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_calls (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    client TEXT NOT NULL,
    model TEXT NOT NULL,
    phase TEXT,
    prompt_chars INTEGER,
    completion_chars INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_input_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    system_prompt_chars INTEGER,
    max_tokens_requested INTEGER,
    temperature DOUBLE PRECISION,
    latency_seconds DOUBLE PRECISION,
    time_to_first_token_seconds DOUBLE PRECISION,
    queue_wait_seconds DOUBLE PRECISION,
    retry_count INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    http_status INTEGER,
    rate_limited BOOLEAN NOT NULL DEFAULT FALSE,
    cost_usd DOUBLE PRECISION,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Backfill new columns on pre-existing deployments.
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS cache_read_input_tokens INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS cache_creation_input_tokens INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS system_prompt_chars INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS max_tokens_requested INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS temperature DOUBLE PRECISION;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS time_to_first_token_seconds DOUBLE PRECISION;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS queue_wait_seconds DOUBLE PRECISION;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS http_status INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS rate_limited BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls (run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_phase ON llm_calls (phase);
CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp ON llm_calls (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls (model);

-- ── swarm_feature_events ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS swarm_feature_events (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    feature TEXT NOT NULL,
    event TEXT NOT NULL,
    status TEXT,
    artifact_count INTEGER,
    test_count INTEGER,
    handoff_count INTEGER,
    layer INTEGER,
    error TEXT,
    duration_seconds DOUBLE PRECISION,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    agent_transitions INTEGER,
    unique_agents_visited INTEGER,
    planner_calls INTEGER,
    coder_calls INTEGER,
    reviewer_calls INTEGER,
    tester_calls INTEGER,
    tool_call_count INTEGER,
    tool_failure_count INTEGER,
    deep_agent_invocations INTEGER,
    deep_agent_timeout_count INTEGER,
    subprocess_spawn_count INTEGER,
    worker_crash_count INTEGER,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS agent_transitions INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS unique_agents_visited INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS planner_calls INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS coder_calls INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS reviewer_calls INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS tester_calls INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS tool_call_count INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS tool_failure_count INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS deep_agent_invocations INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS deep_agent_timeout_count INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS subprocess_spawn_count INTEGER;
ALTER TABLE swarm_feature_events ADD COLUMN IF NOT EXISTS worker_crash_count INTEGER;

CREATE INDEX IF NOT EXISTS idx_swarm_feature_events_run_id ON swarm_feature_events (run_id);
CREATE INDEX IF NOT EXISTS idx_swarm_feature_events_feature ON swarm_feature_events (feature);
CREATE INDEX IF NOT EXISTS idx_swarm_feature_events_event ON swarm_feature_events (event);
CREATE INDEX IF NOT EXISTS idx_swarm_feature_events_timestamp ON swarm_feature_events (timestamp DESC);

-- ── tool_calls ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tool_calls (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    feature TEXT,
    agent TEXT,
    tool TEXT NOT NULL,
    success BOOLEAN,
    latency_seconds DOUBLE PRECISION,
    args_chars INTEGER,
    result_chars INTEGER,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls (run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls (tool);
CREATE INDEX IF NOT EXISTS idx_tool_calls_feature ON tool_calls (feature);
CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp ON tool_calls (timestamp DESC);

-- ── agent_stats ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_stats (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    feature TEXT,
    agent TEXT NOT NULL,
    activations INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    decisions INTEGER NOT NULL DEFAULT 0,
    handoffs_in INTEGER NOT NULL DEFAULT 0,
    handoffs_out INTEGER NOT NULL DEFAULT 0,
    total_time_seconds DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_stats_run_id ON agent_stats (run_id);
CREATE INDEX IF NOT EXISTS idx_agent_stats_agent ON agent_stats (agent);
CREATE INDEX IF NOT EXISTS idx_agent_stats_timestamp ON agent_stats (timestamp DESC);

-- ── decomposition_stats ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decomposition_stats (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    requirement_id TEXT,
    requirement_title TEXT,
    planned_sub_specs_count INTEGER NOT NULL DEFAULT 0,
    fallback BOOLEAN NOT NULL DEFAULT FALSE,
    empty_result BOOLEAN NOT NULL DEFAULT FALSE,
    truncated BOOLEAN NOT NULL DEFAULT FALSE,
    depends_on_declared INTEGER NOT NULL DEFAULT 0,
    depends_on_resolved INTEGER NOT NULL DEFAULT 0,
    depends_on_unresolved INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decomposition_stats_run_id ON decomposition_stats (run_id);
CREATE INDEX IF NOT EXISTS idx_decomposition_stats_timestamp ON decomposition_stats (timestamp DESC);

-- ── memory_operations ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_operations (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    operation TEXT NOT NULL, -- 'create' | 'recall' | 'boost' | 'demote' | 'decay'
    memory_type TEXT,        -- 'pattern' | 'mistake' | 'solution' | 'strategy'
    memory_id TEXT,
    source_feature TEXT,
    count INTEGER,
    delta DOUBLE PRECISION,
    latency_seconds DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_operations_run_id ON memory_operations (run_id);
CREATE INDEX IF NOT EXISTS idx_memory_operations_operation ON memory_operations (operation);
CREATE INDEX IF NOT EXISTS idx_memory_operations_memory_type ON memory_operations (memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_operations_timestamp ON memory_operations (timestamp DESC);

-- ── incidents ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS incidents (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    category TEXT NOT NULL,  -- 'llm' | 'neo4j' | 'qdrant' | 'postgres' | 'subprocess' | 'pipeline' | 'memory' | 'vector' | 'tool' | 'other'
    severity TEXT NOT NULL,  -- 'info' | 'warning' | 'error' | 'critical'
    message TEXT NOT NULL,
    stack TEXT,
    phase TEXT,
    feature TEXT,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incidents_run_id ON incidents (run_id);
CREATE INDEX IF NOT EXISTS idx_incidents_category ON incidents (category);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents (severity);
CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents (timestamp DESC);

-- ── artifact_writes ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS artifact_writes (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    feature TEXT,
    spec_id TEXT,
    file_path TEXT NOT NULL,
    language TEXT,
    bytes_written INTEGER NOT NULL DEFAULT 0,
    is_test BOOLEAN NOT NULL DEFAULT FALSE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artifact_writes_run_id ON artifact_writes (run_id);
CREATE INDEX IF NOT EXISTS idx_artifact_writes_feature ON artifact_writes (feature);
CREATE INDEX IF NOT EXISTS idx_artifact_writes_language ON artifact_writes (language);
CREATE INDEX IF NOT EXISTS idx_artifact_writes_timestamp ON artifact_writes (timestamp DESC);

-- ── background_loop_samples ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS background_loop_samples (
    id BIGSERIAL PRIMARY KEY,
    active_task_count INTEGER NOT NULL DEFAULT 0,
    pending_task_count INTEGER NOT NULL DEFAULT 0,
    completed_task_count INTEGER NOT NULL DEFAULT 0,
    loop_restarts INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_background_loop_samples_timestamp ON background_loop_samples (timestamp DESC);

-- ── Views / rollups ──────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_cost_per_run AS
SELECT
    run_id,
    COUNT(*) AS call_count,
    COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
    COALESCE(SUM(input_tokens), 0) AS input_tokens,
    COALESCE(SUM(output_tokens), 0) AS output_tokens,
    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_tokens,
    COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_tokens
FROM llm_calls
WHERE run_id IS NOT NULL
GROUP BY run_id;

CREATE OR REPLACE VIEW v_cost_per_phase AS
SELECT
    COALESCE(phase, 'unknown') AS phase,
    COUNT(*) AS call_count,
    COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
    COALESCE(SUM(input_tokens), 0) AS input_tokens,
    COALESCE(SUM(output_tokens), 0) AS output_tokens,
    AVG(latency_seconds) AS avg_latency_seconds
FROM llm_calls
GROUP BY phase;

CREATE OR REPLACE VIEW v_runs_per_day AS
SELECT
    DATE(started_at) AS day,
    COUNT(*) AS runs,
    COUNT(*) FILTER (WHERE status = 'success') AS success_runs,
    COUNT(*) FILTER (WHERE status = 'partial') AS partial_runs,
    COUNT(*) FILTER (WHERE status = 'error') AS error_runs,
    AVG(pass_rate) FILTER (WHERE pass_rate IS NOT NULL) AS avg_pass_rate,
    AVG(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL) AS avg_duration_seconds
FROM pipeline_runs
GROUP BY DATE(started_at)
ORDER BY day DESC;

CREATE OR REPLACE VIEW v_pass_rate_per_metric AS
SELECT
    metric_name,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE passed) AS passed,
    CASE WHEN COUNT(*) > 0
         THEN COUNT(*) FILTER (WHERE passed)::DOUBLE PRECISION / COUNT(*)
         ELSE 0 END AS pass_rate,
    AVG(score) AS avg_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score
FROM eval_metrics
GROUP BY metric_name
ORDER BY metric_name;

CREATE OR REPLACE VIEW v_attempts_per_requirement AS
SELECT
    run_id,
    requirement_id,
    MAX(attempt) AS final_attempt,
    BOOL_OR(passed) AS ever_passed,
    MAX(attempt) FILTER (WHERE passed) AS first_pass_attempt
FROM eval_metrics
WHERE requirement_id IS NOT NULL
GROUP BY run_id, requirement_id;
"""


def ensure_schema(client) -> None:
    """Install the metrics schema. Idempotent — safe to call on every startup."""
    with client.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
    log.info("metrics_schema_ensured")
