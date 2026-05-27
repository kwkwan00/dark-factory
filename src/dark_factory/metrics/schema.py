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

-- ════════════════════════════════════════════════════════════════════════
-- Refinery v2 (adversarial debate) — forensic tables
-- ════════════════════════════════════════════════════════════════════════
-- All tables FK back to ``pipeline_runs`` via ``source_run_id`` where
-- applicable, with ON DELETE SET NULL so deleting a pipeline run does not
-- wipe refinery history. Rows land here in addition to the existing
-- ``llm_calls`` table (dual-write: ``refinery_llm_calls`` carries role +
-- requirement_id + round_number + tool_calls that ``llm_calls`` does not).

-- One row per refinery invocation.
CREATE TABLE IF NOT EXISTS refinery_runs (
    refinery_run_id        TEXT PRIMARY KEY,
    source_mode            TEXT NOT NULL,
    source_run_id          TEXT REFERENCES pipeline_runs(run_id) ON DELETE SET NULL,
    requirements_count     INTEGER NOT NULL DEFAULT 0,
    converged_count        INTEGER NOT NULL DEFAULT 0,
    short_circuited_count  INTEGER NOT NULL DEFAULT 0,
    aborted_count          INTEGER NOT NULL DEFAULT 0,
    started_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at               TIMESTAMPTZ,
    duration_seconds       DOUBLE PRECISION,
    total_cost_usd         DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_tokens_in        BIGINT NOT NULL DEFAULT 0,
    total_tokens_out       BIGINT NOT NULL DEFAULT 0,
    settings_snapshot      JSONB NOT NULL DEFAULT '{}'::jsonb,
    applied_to_graph       BOOLEAN NOT NULL DEFAULT FALSE,
    applied_at             TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_refinery_runs_started_at ON refinery_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_refinery_runs_source_run ON refinery_runs (source_run_id);
CREATE INDEX IF NOT EXISTS idx_refinery_runs_source_mode ON refinery_runs (source_mode);

-- ── Resume support (V1 — requirement-level) ────────────────────────────────
-- ``status`` tracks the run's lifecycle so the resume endpoint can find
-- runs that didn't complete cleanly. ``input_snapshot`` captures the
-- original input payload (run_id / input_path / direct + the gathered
-- requirements list) so a resumed run can skip Phase 1. Both columns
-- are added idempotently so existing deployments migrate in place.

ALTER TABLE refinery_runs
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'in_progress';

ALTER TABLE refinery_runs
    ADD COLUMN IF NOT EXISTS input_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_refinery_runs_status ON refinery_runs (status);

-- One row per per-requirement debate.
CREATE TABLE IF NOT EXISTS refinery_debates (
    id                     BIGSERIAL PRIMARY KEY,
    refinery_run_id        TEXT NOT NULL REFERENCES refinery_runs(refinery_run_id) ON DELETE CASCADE,
    requirement_id         TEXT NOT NULL,
    convergence_status     TEXT NOT NULL,
    final_overall_score    DOUBLE PRECISION,
    final_dimension_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    rounds_executed        INTEGER NOT NULL DEFAULT 0,
    escalation_level       INTEGER NOT NULL DEFAULT 0,
    research_calls_used    INTEGER NOT NULL DEFAULT 0,
    disagreement_score_max DOUBLE PRECISION,
    duration_seconds       DOUBLE PRECISION,
    cost_usd               DOUBLE PRECISION NOT NULL DEFAULT 0,
    termination_reason     TEXT,
    reconcile_invoked      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_refinery_debates_run ON refinery_debates (refinery_run_id);
CREATE INDEX IF NOT EXISTS idx_refinery_debates_status ON refinery_debates (convergence_status);
CREATE INDEX IF NOT EXISTS idx_refinery_debates_req ON refinery_debates (requirement_id);

-- Archetype columns feed the adaptive max_rounds loop. Bucketed by
-- (priority, source_mode-from-runs, primary_tag) so per-archetype
-- convergence behaviour can be aggregated without re-parsing payloads.
ALTER TABLE refinery_debates
    ADD COLUMN IF NOT EXISTS priority TEXT;
ALTER TABLE refinery_debates
    ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_refinery_debates_priority
    ON refinery_debates (priority);

-- ── Per-debate output cache (resume V1) ──────────────────────────────────
-- A separate table from the forensic ``refinery_debates`` (which can have
-- multiple rows per (run, req) for retried debates) — this one is the
-- canonical "this debate is done, here's its output" cache. Resumed runs
-- look up by (refinery_run_id, requirement_id) and replay completed
-- debates without re-invoking the panel.
CREATE TABLE IF NOT EXISTS refinery_debate_completions (
    refinery_run_id        TEXT NOT NULL REFERENCES refinery_runs(refinery_run_id) ON DELETE CASCADE,
    requirement_id         TEXT NOT NULL,
    convergence_status     TEXT NOT NULL,
    refined_payload        JSONB NOT NULL,
    suggested_memories     JSONB NOT NULL DEFAULT '[]'::jsonb,
    completed_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (refinery_run_id, requirement_id)
);

CREATE INDEX IF NOT EXISTS idx_refinery_debate_completions_run
    ON refinery_debate_completions (refinery_run_id);

-- One row per round of a debate.
CREATE TABLE IF NOT EXISTS refinery_debate_rounds (
    id                    BIGSERIAL PRIMARY KEY,
    debate_id             BIGINT NOT NULL REFERENCES refinery_debates(id) ON DELETE CASCADE,
    round_number          INTEGER NOT NULL,
    critic_count          INTEGER NOT NULL DEFAULT 0,
    critic_blockers_count INTEGER NOT NULL DEFAULT 0,
    critic_warnings_count INTEGER NOT NULL DEFAULT 0,
    rule_violations_count INTEGER NOT NULL DEFAULT 0,
    rule_warnings_count   INTEGER NOT NULL DEFAULT 0,
    judge_overall         DOUBLE PRECISION,
    judge_dimensions      JSONB NOT NULL DEFAULT '{}'::jsonb,
    disagreement_score    DOUBLE PRECISION,
    router_decision       TEXT NOT NULL,
    duration_seconds      DOUBLE PRECISION,
    UNIQUE (debate_id, round_number)
);

CREATE INDEX IF NOT EXISTS idx_refinery_rounds_debate ON refinery_debate_rounds (debate_id);

-- Extends llm_calls with refinery-specific columns. Dual-write contract:
-- every refinery LLM call also lands in ``llm_calls`` with
-- ``phase = 'refinery.{role}.{kind}'`` so existing cost dashboards auto-
-- include refinery without modification.
CREATE TABLE IF NOT EXISTS refinery_llm_calls (
    id                BIGSERIAL PRIMARY KEY,
    refinery_run_id   TEXT NOT NULL REFERENCES refinery_runs(refinery_run_id) ON DELETE CASCADE,
    requirement_id    TEXT,
    round_number      INTEGER,
    role              TEXT NOT NULL,
    kind              TEXT NOT NULL,
    model             TEXT NOT NULL,
    reasoning_effort  TEXT,
    tokens_in         INTEGER,
    tokens_out        INTEGER,
    cache_read_tokens INTEGER,
    latency_ms        INTEGER,
    cost_usd          DOUBLE PRECISION,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error             TEXT,
    tool_calls        JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_refinery_llm_run ON refinery_llm_calls (refinery_run_id);
CREATE INDEX IF NOT EXISTS idx_refinery_llm_role ON refinery_llm_calls (role);
CREATE INDEX IF NOT EXISTS idx_refinery_llm_model ON refinery_llm_calls (model);

-- One row per rule violation.
CREATE TABLE IF NOT EXISTS refinery_rule_violations (
    id                   BIGSERIAL PRIMARY KEY,
    debate_id            BIGINT NOT NULL REFERENCES refinery_debates(id) ON DELETE CASCADE,
    round_number         INTEGER NOT NULL,
    rule_id              TEXT NOT NULL,
    severity             TEXT NOT NULL,
    dimension            TEXT NOT NULL,
    finding              TEXT NOT NULL,
    suggested_fix        TEXT,
    injected_as_critique BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_refinery_rule_viol_rule ON refinery_rule_violations (rule_id);
CREATE INDEX IF NOT EXISTS idx_refinery_rule_viol_dim ON refinery_rule_violations (dimension);

-- Research provenance: one row per Source returned by a research call.
CREATE TABLE IF NOT EXISTS refinery_research_sources (
    id                 BIGSERIAL PRIMARY KEY,
    debate_id          BIGINT NOT NULL REFERENCES refinery_debates(id) ON DELETE CASCADE,
    round_number       INTEGER NOT NULL,
    tier               SMALLINT NOT NULL,
    provider           TEXT NOT NULL,
    url                TEXT,
    title              TEXT,
    fetched_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    propagated         BOOLEAN NOT NULL,
    insight_confidence DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_refinery_research_tier ON refinery_research_sources (tier);

-- ``compute_provider_stats`` aggregates over a time window
-- (``WHERE fetched_at >= NOW() - interval``) — backstop the scan
-- with an index so frequent operator dashboard hits don't degrade
-- as ``refinery_research_sources`` grows.
CREATE INDEX IF NOT EXISTS idx_refinery_research_fetched_at
    ON refinery_research_sources (fetched_at DESC);

-- Memory lifecycle — refinery-specific audit row per suggested memory.
CREATE TABLE IF NOT EXISTS refinery_memory_audits (
    id                          BIGSERIAL PRIMARY KEY,
    debate_id                   BIGINT REFERENCES refinery_debates(id) ON DELETE SET NULL,
    refinery_run_id             TEXT NOT NULL REFERENCES refinery_runs(refinery_run_id) ON DELETE CASCADE,
    suggested_memory_id         TEXT NOT NULL,
    kind                        TEXT NOT NULL,
    source_role                 TEXT NOT NULL,
    validation_status           TEXT NOT NULL,
    outcome                     TEXT NOT NULL,
    existing_memory_id          TEXT,
    similarity                  DOUBLE PRECISION,
    provenance_source_tier_mix  SMALLINT[] NOT NULL DEFAULT ARRAY[]::SMALLINT[],
    provenance_confidence       DOUBLE PRECISION,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_decision_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_refinery_mem_run ON refinery_memory_audits (refinery_run_id);
CREATE INDEX IF NOT EXISTS idx_refinery_mem_kind ON refinery_memory_audits (kind);
CREATE INDEX IF NOT EXISTS idx_refinery_mem_outcome ON refinery_memory_audits (outcome);


-- ── Active-learning: user feedback on suggested memories ─────────────────
-- One row per operator decision (save / dismiss / edit). Cross-run
-- aggregation feeds the role-weighting module and re-ranks future
-- retrieval (boost on save, demote on dismiss).
CREATE TABLE IF NOT EXISTS refinery_memory_feedback (
    id                  BIGSERIAL PRIMARY KEY,
    refinery_run_id     TEXT REFERENCES refinery_runs(refinery_run_id) ON DELETE SET NULL,
    memory_id           TEXT NOT NULL,
    memory_kind         TEXT NOT NULL,
    source_role         TEXT,
    decision            TEXT NOT NULL,           -- accepted | dismissed | edited
    reason              TEXT,
    decided_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_refinery_mem_fb_memory ON refinery_memory_feedback (memory_id);
CREATE INDEX IF NOT EXISTS idx_refinery_mem_fb_role ON refinery_memory_feedback (source_role);
CREATE INDEX IF NOT EXISTS idx_refinery_mem_fb_decided ON refinery_memory_feedback (decided_at DESC);


-- ── Adaptive role weighting: critique dispositions ───────────────────────
-- One row per (round, critique) showing how the Judge disposed of it
-- in its rebuttal — accepted vs. rejected vs. deferred. The aggregator
-- computes per-role acceptance rates and converts them into bounded
-- multipliers applied at synthesis time.
CREATE TABLE IF NOT EXISTS refinery_critique_dispositions (
    id                BIGSERIAL PRIMARY KEY,
    refinery_run_id   TEXT NOT NULL REFERENCES refinery_runs(refinery_run_id) ON DELETE CASCADE,
    requirement_id    TEXT NOT NULL,
    round_number      INTEGER NOT NULL,
    role              TEXT NOT NULL,
    severity          TEXT NOT NULL,             -- info | warning | blocker
    dimension         TEXT,
    action            TEXT NOT NULL,             -- accepted | rejected | deferred
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_refinery_crit_disp_role ON refinery_critique_dispositions (role);
CREATE INDEX IF NOT EXISTS idx_refinery_crit_disp_recorded ON refinery_critique_dispositions (recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_refinery_crit_disp_run ON refinery_critique_dispositions (refinery_run_id);


-- ── Judge confidence calibration: per-requirement operator decisions ─────
-- One row per operator decision (apply / dismiss / edit) on a refined
-- requirement. Joined against ``refinery_debates.final_overall_score``
-- by the calibration aggregator: high score + frequent dismiss →
-- Judge overconfident; low score + frequent apply → underconfident.
CREATE TABLE IF NOT EXISTS refinery_requirement_decisions (
    id                  BIGSERIAL PRIMARY KEY,
    refinery_run_id     TEXT NOT NULL REFERENCES refinery_runs(refinery_run_id) ON DELETE CASCADE,
    requirement_id      TEXT NOT NULL,
    decision            TEXT NOT NULL,             -- accepted | dismissed | edited
    judge_overall_score DOUBLE PRECISION,
    convergence_status  TEXT,
    decided_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_refinery_req_dec_run ON refinery_requirement_decisions (refinery_run_id);
CREATE INDEX IF NOT EXISTS idx_refinery_req_dec_decided ON refinery_requirement_decisions (decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_refinery_req_dec_decision ON refinery_requirement_decisions (decision);


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
