/** Base URL for the FastAPI backend. */
const BASE = import.meta.env.VITE_API_BASE ?? "";

/** AG-UI event types emitted by the backend. */
export type AGUIEventType =
  | "RUN_STARTED"
  | "RUN_FINISHED"
  | "RUN_ERROR"
  | "STEP_STARTED"
  | "STEP_FINISHED"
  | "TEXT_MESSAGE_START"
  | "TEXT_MESSAGE_CONTENT"
  | "TEXT_MESSAGE_END"
  | "STATE_SNAPSHOT";

export interface AGUIEvent {
  type: AGUIEventType;
  // AG-UI serialises as camelCase
  threadId?: string;
  runId?: string;
  stepName?: string;
  step_id?: string;
  messageId?: string;
  role?: string;
  delta?: string;
  snapshot?: Record<string, unknown>;
  message?: string;
}

/** Optional per-run API key overrides — never persisted. */
export interface AgentRunKeys {
  anthropicApiKey?: string;
  openaiApiKey?: string;
}

/** Stream AG-UI events from POST /api/agent/run. */
export async function* streamAgentRun(
  requirementsPath: string,
  signal?: AbortSignal,
  keys?: AgentRunKeys,
): AsyncGenerator<AGUIEvent> {
  const body: Record<string, unknown> = {
    requirements_path: requirementsPath,
  };
  // Only include the key fields when they're non-empty so the server's
  // existing env var defaults stay in place when the user hasn't typed
  // anything into the UI.
  if (keys?.anthropicApiKey && keys.anthropicApiKey.trim()) {
    body.anthropic_api_key = keys.anthropicApiKey.trim();
  }
  if (keys?.openaiApiKey && keys.openaiApiKey.trim()) {
    body.openai_api_key = keys.openaiApiKey.trim();
  }

  const response = await fetch(`${BASE}/api/agent/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    // Reuse the same shape as the REST helpers for consistency in logs.
    const detail = response.body
      ? await response.text().catch(() => "")
      : "(no response body)";
    const snippet = detail.length > 200 ? `${detail.slice(0, 200)}…` : detail;
    const suffix = snippet ? ` — ${snippet}` : "";
    throw new Error(
      `[POST /api/agent/run] ${response.status} ${response.statusText}${suffix}`,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6)) as AGUIEvent;
          yield event;
        } catch {
          // skip malformed lines
        }
      }
    }
  }

  // Flush the decoder's internal buffer. ``{ stream: true }`` preserves
  // any incomplete UTF-8 sequence at the end of a chunk across calls;
  // without a final non-streaming decode, a multi-byte character split
  // across the last two chunks (or the very last event of a fast stream)
  // would be silently dropped. Finish decoding + drain any remaining
  // buffered line that never got terminated by a newline.
  buffer += decoder.decode();
  if (buffer) {
    const finalLines = buffer.split("\n");
    for (const line of finalLines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6)) as AGUIEvent;
          yield event;
        } catch {
          // skip malformed trailing lines
        }
      }
    }
  }
}

// ── REST helpers ──────────────────────────────────────────────────────────────

/**
 * Build a readable error message from a failed fetch Response. Reads the body
 * as text so non-JSON error pages (HTML 502s from a proxy, plain 404s, etc.)
 * surface a useful message instead of crashing `.json()`.
 */
async function describeFailure(
  method: string,
  path: string,
  res: Response,
): Promise<string> {
  let detail = "";
  try {
    const text = await res.text();
    // Try to extract a `detail` field from FastAPI JSON errors; fall back
    // to the raw text truncated to keep the log line reasonable.
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === "string") {
        detail = parsed.detail;
      } else if (parsed.detail != null) {
        detail = JSON.stringify(parsed.detail);
      }
    } catch {
      detail = text;
    }
  } catch {
    /* ignore read errors — we'll fall back to status code only */
  }
  const trimmed = detail.length > 200 ? `${detail.slice(0, 200)}…` : detail;
  const suffix = trimmed ? ` — ${trimmed}` : "";
  return `[${method} ${path}] ${res.status} ${res.statusText}${suffix}`;
}

/**
 * Parse a Response body as JSON with a helpful error on parse failure.
 * The raw .json() call throws a vague `SyntaxError: Unexpected token <`
 * when the server returns HTML, which is unhelpful in logs.
 */
async function parseJsonOrThrow<T>(
  method: string,
  path: string,
  res: Response,
): Promise<T> {
  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    const snippet = text.length > 200 ? `${text.slice(0, 200)}…` : text;
    throw new Error(
      `[${method} ${path}] invalid JSON response: ${snippet || "(empty)"}`,
    );
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(await describeFailure("GET", path, res));
  return parseJsonOrThrow<T>("GET", path, res);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await describeFailure("POST", path, res));
  return parseJsonOrThrow<T>("POST", path, res);
}

// ── Typed API calls ───────────────────────────────────────────────────────────

/** Tunable pipeline settings shape (GET/PATCH /api/settings). */
export interface PipelineSettings {
  max_parallel_features: number;
  max_parallel_specs: number;
  max_spec_handoffs: number;
  max_codegen_handoffs: number;
  spec_eval_threshold: number;
  enable_spec_decomposition: boolean;
  reuse_existing_specs: boolean;
  max_specs_per_requirement: number;
  max_reconciliation_turns: number;
  reconciliation_timeout_seconds: number;
  requirement_dedup_threshold: number;
  enable_e2e_validation: boolean;
  max_e2e_turns: number;
  e2e_timeout_seconds: number;
  e2e_browsers: string[];
  enable_episodic_memory: boolean;
  memory_dedup_threshold: number;
  llm_model: string;
  eval_model: string;
  output_dir: string;
}

export interface PipelineSettingsUpdate {
  max_parallel_features?: number;
  max_parallel_specs?: number;
  max_spec_handoffs?: number;
  max_codegen_handoffs?: number;
  spec_eval_threshold?: number;
  enable_spec_decomposition?: boolean;
  reuse_existing_specs?: boolean;
  max_specs_per_requirement?: number;
  max_reconciliation_turns?: number;
  reconciliation_timeout_seconds?: number;
  requirement_dedup_threshold?: number;
  enable_e2e_validation?: boolean;
  max_e2e_turns?: number;
  e2e_timeout_seconds?: number;
  e2e_browsers?: string[];
  enable_episodic_memory?: boolean;
  memory_dedup_threshold?: number;
  llm_model?: string;
  eval_model?: string;
}

/** Eval browser data shapes (returned by GET /api/eval). */
export interface EvalMetric {
  name: string;
  score: number;
  passed: boolean;
  reason?: string;
}

export interface EvalAttempt {
  id: string;
  eval_type: string;
  overall_score: number;
  all_passed: boolean;
  timestamp: string;
  metrics: EvalMetric[];
}

export interface EvalSpec {
  spec_id: string;
  feature_name: string;
  evals: EvalAttempt[];
}

export interface EvalRun {
  run_id: string;
  timestamp: string;
  status: string;
  pass_rate: number;
  spec_count: number;
  specs: EvalSpec[];
}

/** A single swarm progress event emitted by the backend broker. */
export interface ProgressEvent {
  event: string;
  timestamp: number;
  feature?: string;
  agent?: string;
  messages?: number;
  layer?: number;
  total_layers?: number;
  features?: string[];
  spec_count?: number;
  status?: string;
  artifacts?: number;
  tests?: number;
  error?: string | null;
  reason?: string;
  [key: string]: unknown;
}

/** Payload from GET /api/graph/gaps — actionable gaps in the knowledge graph. */
export interface UnplannedRequirement {
  id: string;
  title: string | null;
  priority: string | null;
  source_file: string | null;
}

export interface SpecWithoutArtifacts {
  id: string;
  title: string | null;
  capability: string | null;
  requirement_ids: string[];
}

export interface SpecFailingEval {
  id: string;
  title: string | null;
  capability: string | null;
  requirement_ids: string[];
  metric_name: string;
  score: number;
  last_eval_at: string | null;
}

export interface StaleRequirement {
  id: string;
  spec_count: number;
  last_eval_at: string | null;
}

export interface GraphGaps {
  enabled_postgres: boolean;
  postgres_error?: string;
  stale_days: number;
  unplanned_requirements: UnplannedRequirement[];
  specs_without_artifacts: SpecWithoutArtifacts[];
  specs_failing_evals: SpecFailingEval[];
  stale_requirements: StaleRequirement[];
  totals: {
    requirements: number;
    specs: number;
  };
}

/** Postgres metrics store data shapes (returned by /api/metrics/*). */
export interface MetricsSummary {
  enabled: boolean;
  reason?: string;
  runs?: {
    total_runs?: number;
    success_runs?: number;
    partial_runs?: number;
    error_runs?: number;
    running_runs?: number;
    avg_pass_rate?: number | null;
    avg_duration_seconds?: number | null;
  };
  llm?: {
    total_calls?: number;
    input_tokens?: number;
    output_tokens?: number;
    cache_read_tokens?: number;
    total_cost_usd?: number;
    avg_latency_seconds?: number | null;
    rate_limited_count?: number;
    error_count?: number;
  };
  evals?: {
    total_evals?: number;
    avg_score?: number | null;
    passed?: number;
  };
  incidents?: {
    open_incidents?: number;
  };
  decomposition?: {
    total_sub_specs?: number;
    requirements_planned?: number;
    planner_fallbacks?: number;
  };
}

export interface MetricsRun {
  run_id: string;
  started_at: string;
  ended_at: string | null;
  status: string;
  spec_count: number;
  feature_count: number;
  pass_rate: number | null;
  duration_seconds: number | null;
  error: string | null;
}

export interface EvalTrendPoint {
  timestamp: string;
  metric_name: string;
  score: number;
  passed: boolean;
  attempt: number | null;
  run_id: string | null;
  spec_id: string | null;
  requirement_id: string | null;
  eval_type: string | null;
  reason: string | null;
}

export interface LlmUsageBucket {
  bucket: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  avg_latency_seconds: number | null;
}

export interface SwarmFeatureEvent {
  run_id: string | null;
  feature: string;
  event: string;
  status: string | null;
  artifact_count: number | null;
  test_count: number | null;
  handoff_count: number | null;
  layer: number | null;
  error: string | null;
  duration_seconds: number | null;
  agent_transitions?: number | null;
  unique_agents_visited?: number | null;
  planner_calls?: number | null;
  coder_calls?: number | null;
  reviewer_calls?: number | null;
  tester_calls?: number | null;
  tool_call_count?: number | null;
  tool_failure_count?: number | null;
  deep_agent_invocations?: number | null;
  worker_crash_count?: number | null;
  timestamp: string;
}

export interface CostPerRun {
  run_id: string;
  call_count: number;
  total_cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
}

export interface CostPerPhase {
  phase: string;
  call_count: number;
  total_cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  avg_latency_seconds: number | null;
}

export interface CostPerModel {
  model: string;
  calls: number;
  total_cost_usd: number;
  input_tokens: number;
  output_tokens: number;
}

export interface CostRollup {
  enabled: boolean;
  per_run: CostPerRun[];
  per_phase: CostPerPhase[];
  per_model: CostPerModel[];
}

export interface ThroughputDay {
  day: string;
  runs: number;
  success_runs: number;
  partial_runs: number;
  error_runs: number;
  avg_pass_rate: number | null;
  avg_duration_seconds: number | null;
}

export interface QualityMetric {
  metric_name: string;
  total: number;
  passed: number;
  pass_rate: number;
  avg_score: number;
  min_score: number;
  max_score: number;
}

export interface QualityResponse {
  enabled: boolean;
  per_metric: QualityMetric[];
  total_requirements: number;
  passed_requirements: number;
  first_attempt_pass_rate: number;
  mean_attempts_to_pass: number | null;
  max_attempt: number | null;
}

export interface IncidentRow {
  id: number;
  run_id: string | null;
  category: string;
  severity: string;
  message: string;
  stack: string | null;
  phase: string | null;
  feature: string | null;
  resolved: boolean;
  resolved_at: string | null;
  timestamp: string;
}

export interface AgentStatRow {
  run_id?: string;
  feature?: string;
  agent: string;
  activations: number;
  tool_calls: number;
  decisions: number;
  handoffs_in: number;
  handoffs_out: number;
  total_time_seconds: number | null;
}

export interface ToolCallBucket {
  bucket: string;
  calls: number;
  successes: number;
  failures: number;
  avg_latency_seconds: number | null;
}

export interface MemoryOperationRow {
  operation: string;
  count: number;
  avg_latency_seconds: number | null;
}

export interface MemoryActivity {
  enabled: boolean;
  per_operation: MemoryOperationRow[];
  per_type: Array<{ memory_type: string; operation: string; count: number }>;
  summary: {
    recall_hits?: number;
    recall_misses?: number;
    created?: number;
    boosts?: number;
    demotes?: number;
  };
}

export interface DecompositionRow {
  run_id: string | null;
  requirement_id: string | null;
  requirement_title: string | null;
  planned_sub_specs_count: number;
  fallback: boolean;
  empty_result: boolean;
  truncated: boolean;
  depends_on_declared: number;
  depends_on_resolved: number;
  depends_on_unresolved: number;
  timestamp: string;
}

export interface DecompositionResponse {
  enabled: boolean;
  summary: {
    requirements_planned?: number;
    total_sub_specs?: number;
    avg_sub_specs?: number | null;
    fallback_count?: number;
    empty_result_count?: number;
    truncated_count?: number;
    depends_on_declared?: number;
    depends_on_resolved?: number;
    depends_on_unresolved?: number;
  };
  rows: DecompositionRow[];
}

export interface ArtifactsResponse {
  enabled: boolean;
  summary: {
    files_written?: number;
    total_bytes?: number;
    test_files?: number;
    code_files?: number;
  };
  per_language: Array<{ language: string; files: number; total_bytes: number }>;
  per_feature: Array<{
    feature: string;
    files: number;
    test_files: number;
    total_bytes: number;
  }>;
}

export interface BackgroundLoopSample {
  active_task_count: number;
  pending_task_count: number;
  completed_task_count: number;
  loop_restarts: number;
  timestamp: string;
}

/** Per-run detail payload for the Run Detail popup (GET /api/metrics/runs/{run_id}). */
export interface RunDetailResponse {
  enabled: boolean;
  reason?: string;
  run_id?: string;
  run?: {
    run_id: string;
    started_at: string;
    ended_at: string | null;
    status: string;
    spec_count: number;
    feature_count: number;
    pass_rate: number | null;
    duration_seconds: number | null;
    error: string | null;
    metadata?: Record<string, unknown>;
  };
  llm?: {
    totals: {
      total_calls?: number;
      input_tokens?: number;
      output_tokens?: number;
      cache_read_tokens?: number;
      total_cost_usd?: number;
      avg_latency_seconds?: number | null;
      rate_limited_count?: number;
      error_count?: number;
    };
    per_phase: Array<{
      phase: string | null;
      calls: number;
      input_tokens: number;
      output_tokens: number;
      total_cost_usd: number;
      avg_latency_seconds: number | null;
    }>;
  };
  swarm_events?: SwarmFeatureEvent[];
  agent_stats?: AgentStatRow[];
  tool_calls?: ToolCallBucket[];
  incidents?: IncidentRow[];
  eval_metrics?: EvalTrendPoint[];
  artifacts?: {
    summary: {
      files_written?: number;
      total_bytes?: number;
      test_files?: number;
      code_files?: number;
    };
    per_language: Array<{ language: string; files: number; total_bytes: number }>;
  };
  decomposition?: DecompositionRow[];
  progress_log?: ProgressLogEntry[];
}

/** Single row from the progress_events table — raw audit log entry. */
export interface ProgressLogEntry {
  id: number;
  run_id: string | null;
  event: string;
  feature: string | null;
  agent: string | null;
  timestamp: string;
  payload: Record<string, unknown>;
}

/** File tree node returned by GET /api/runs/{run_id}/files. */
export interface RunFileNode {
  name: string;
  type: "file" | "dir";
  path: string;
  size?: number;
  children?: RunFileNode[];
  truncated?: boolean;
  error?: string;
}

export interface RunFilesResponse {
  run_id: string;
  root: string;
  tree: RunFileNode;
  file_count: number;
  total_bytes: number;
}

export interface RunFileContentResponse {
  run_id: string;
  path: string;
  size: number;
  content: string;
}

/** Model entry returned by POST /api/models/{anthropic,openai}. */
export interface ModelInfo {
  id: string;
  display_name: string;
  /** ISO 8601 timestamp (or null when the upstream doesn't provide one). */
  created_at: string | null;
}

/** Response from the model-list proxy endpoints. ``source="fallback"``
 * means the backend couldn't reach the upstream and served the curated
 * hardcoded list instead — the UI can show a "using defaults" indicator. */
export interface ModelListResponse {
  source: "live" | "fallback";
  models: ModelInfo[];
}

/** Response from POST /api/admin/clear-all. */
export interface AdminClearAllResponse {
  status: "completed" | "partial";
  cleared: {
    neo4j?: { nodes_deleted?: number; status?: string };
    qdrant?: {
      collections_cleared?: string[];
      status?: string;
      skipped?: Array<{ collection: string; error: string }>;
    };
    postgres?: {
      tables_truncated?: string[];
      status?: string;
      skipped?: Array<{ table: string; error: string }>;
    };
    prometheus?: {
      status?:
        | "completed"
        | "in_process_only"
        | "skipped"
        | "admin_api_disabled"
        | "delete_failed"
        | "unreachable";
      reason?: string;
      url?: string;
      series_deleted?: boolean;
      tombstones_cleaned?: boolean;
      http_status?: number;
      body?: string;
      hint?: string;
      error?: string;
      in_process?: {
        cleared_collectors?: number;
        reinitialised_collectors?: number;
        skipped_collectors?: number;
        error?: string;
      };
    };
    output_dir?: {
      files_deleted?: number;
      bytes_freed?: number;
      path?: string;
      status?: string;
    };
    progress_broker_history?: { cleared?: boolean };
  };
  errors: Record<string, string>;
}

/** Per-type memory statistics for the Memory metrics dashboard. */
export interface MemoryTypeStats {
  count: number;
  mean_relevance: number;
  median_relevance: number;
  min_relevance: number;
  max_relevance: number;
  histogram: number[];
}

export interface TopRecalledMemory {
  id: string;
  description: string;
  source_feature: string;
  relevance_score: number;
  times_recalled: number;
  times_applied: number;
  memory_type: string;
}

export interface RecallEffectiveness {
  window_days: number;
  boosted: number;
  demoted: number;
  decays: number;
  total_recalls: number;
  boost_rate: number;
}

export interface MemoryMetricsResponse {
  enabled: boolean;
  reason?: string;
  counts_by_type: Record<string, MemoryTypeStats>;
  top_recalled: TopRecalledMemory[];
  recall_effectiveness: RecallEffectiveness;
}

/** One episodic memory record. Mirrors the backend Episode pydantic
 * model with JSON fields already decoded into their nested shape. */
export interface EpisodeResponse {
  id: string;
  run_id: string;
  feature: string;
  outcome: "success" | "partial" | "failed" | string;
  summary: string;
  turns_used: number;
  duration_seconds: number;
  spec_ids: string[];
  agents_visited: string[];
  key_events: Array<{
    order: number;
    agent: string;
    event: string;
    description: string;
  }>;
  tool_calls_summary: Record<string, number>;
  final_eval_scores: Record<string, number>;
  started_at: string;
  ended_at: string;
}

export const api = {
  /** Gap finder: actionable insights about what's missing, failed,
   * or stale in the knowledge graph. */
  graphGaps: (staleDays = 7) =>
    get<GraphGaps>(`/api/graph/gaps?stale_days=${staleDays}`),

  history: (limit = 20) =>
    get<{ runs: Array<Record<string, unknown>> }>(`/api/history?limit=${limit}`),

  /** Episodes: per-feature autobiographical records written after
   * every swarm run. Used by the Run Detail popup's Episodes tab
   * to show what each feature's trajectory looked like. */
  runEpisodes: (runId: string, feature?: string) => {
    const qs = feature ? `?feature=${encodeURIComponent(feature)}` : "";
    return get<{ run_id: string; episodes: EpisodeResponse[] }>(
      `/api/metrics/episodes/${encodeURIComponent(runId)}${qs}`,
    );
  },

  /** Memory metrics: counts by type, relevance histogram, top-N
   * recalled memories, and recall effectiveness KPIs. Powers the
   * Memory section of the Metrics tab. */
  memoryMetrics: () => get<MemoryMetricsResponse>("/api/metrics/memory"),

  memorySearch: (keywords: string, type = "all") =>
    get<{ results: Array<Record<string, unknown>>; keywords: string }>(
      `/api/memory/search?keywords=${encodeURIComponent(keywords)}&type=${type}`
    ),

  memoryList: (type = "all", limit = 100) =>
    get<{ results: Array<Record<string, unknown>>; total: number; type: string }>(
      `/api/memory/list?type=${type}&limit=${limit}`
    ),

  /** List Claude models available to the configured API key. Pass an
   * optional apiKey to override the server's env var (the Settings
   * tab passes the user's typed-in key here). On network failure or
   * missing key, returns a curated fallback list with source="fallback". */
  listAnthropicModels: (apiKey?: string) =>
    post<ModelListResponse>(
      "/api/models/anthropic",
      apiKey ? { api_key: apiKey } : {},
    ),

  /** List chat-capable OpenAI models available to the configured API
   * key. Filtered server-side to exclude embeddings, whisper, tts,
   * dall-e, moderation, and legacy completion models so DeepEval only
   * ever sees models it can use as a judge. */
  listOpenaiModels: (apiKey?: string) =>
    post<ModelListResponse>(
      "/api/models/openai",
      apiKey ? { api_key: apiKey } : {},
    ),

  /** Nuclear wipe: clears Neo4j, Qdrant, Postgres metrics, Prometheus
   * TSDB + in-process collectors, and the output directory. Refuses
   * while a run is in progress. */
  adminClearAll: (
    options: {
      includeOutputDir?: boolean;
      includePrometheus?: boolean;
    } = {},
  ) => {
    const qs = new URLSearchParams({ confirm: "yes" });
    if (options.includeOutputDir === false) {
      qs.set("include_output_dir", "false");
    }
    if (options.includePrometheus === false) {
      qs.set("include_prometheus", "false");
    }
    return post<AdminClearAllResponse>(
      `/api/admin/clear-all?${qs.toString()}`,
    );
  },

  evalHistory: (specId: string, limit = 10) =>
    get<{ spec_id: string; history: Array<Record<string, unknown>> }>(
      `/api/eval/${encodeURIComponent(specId)}?limit=${limit}`
    ),

  evalRuns: (runLimit = 20, runId?: string) => {
    const qs = new URLSearchParams({ run_limit: String(runLimit) });
    if (runId) qs.set("run_id", runId);
    return get<{ runs: EvalRun[] }>(`/api/eval?${qs.toString()}`);
  },

  health: () =>
    get<Record<string, { ok: boolean; message: string }>>("/api/health"),

  getSettings: () => get<PipelineSettings>("/api/settings"),

  updateSettings: async (
    body: Partial<PipelineSettingsUpdate>,
  ): Promise<PipelineSettings> => {
    const res = await fetch(`${BASE}/api/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(await describeFailure("PATCH", "/api/settings", res));
    }
    return parseJsonOrThrow<PipelineSettings>("PATCH", "/api/settings", res);
  },

  watchStart: () => post<{ status: string; paths?: string[] }>("/api/watch/start"),
  watchStop: () => post<{ status: string }>("/api/watch/stop"),
  watchStatus: () =>
    get<{ running: boolean; paths?: string[]; last_event?: unknown }>("/api/watch/status"),

  /** Open an SSE stream of swarm progress events. Returns the EventSource
   * so the caller can close it. Events arrive as JSON dicts with at least
   * ``event`` and ``timestamp`` fields.
   *
   * ``onError`` receives ``readyState`` so the caller can distinguish
   * transient reconnects (CONNECTING) from fatal closes (CLOSED).
   * ``onOpen`` fires on initial connect AND every auto-reconnect. */
  streamAgentEvents: (
    onEvent: (event: ProgressEvent) => void,
    onError?: (err: Event, readyState: number) => void,
    onOpen?: () => void,
  ): EventSource => {
    const es = new EventSource(`${BASE}/api/agent/events`);
    es.onmessage = (e: MessageEvent) => {
      try {
        onEvent(JSON.parse(e.data) as ProgressEvent);
      } catch {
        /* ignore malformed */
      }
    };
    if (onError) {
      es.onerror = (err) => onError(err, es.readyState);
    }
    if (onOpen) {
      es.onopen = () => onOpen();
    }
    return es;
  },

  metricsSummary: () => get<MetricsSummary>("/api/metrics/summary"),

  metricsRuns: (limit = 20) =>
    get<{ enabled: boolean; runs: MetricsRun[] }>(`/api/metrics/runs?limit=${limit}`),

  metricsEvalTrend: (metricName?: string, limit = 200) => {
    const qs = new URLSearchParams();
    if (metricName) qs.set("metric_name", metricName);
    qs.set("limit", String(limit));
    return get<{ enabled: boolean; metric_name: string | null; events: EvalTrendPoint[] }>(
      `/api/metrics/eval_trend?${qs.toString()}`,
    );
  },

  metricsLlmUsage: (groupBy: "model" | "phase" | "client" = "model", limit = 50) =>
    get<{ enabled: boolean; group_by: string; buckets: LlmUsageBucket[] }>(
      `/api/metrics/llm_usage?group_by=${groupBy}&limit=${limit}`,
    ),

  metricsSwarmFeatures: (runId?: string, limit = 100) => {
    const qs = new URLSearchParams();
    if (runId) qs.set("run_id", runId);
    qs.set("limit", String(limit));
    return get<{ enabled: boolean; run_id: string | null; events: SwarmFeatureEvent[] }>(
      `/api/metrics/swarm_features?${qs.toString()}`,
    );
  },

  metricsCostRollup: (limit = 50) =>
    get<CostRollup>(`/api/metrics/cost_rollup?limit=${limit}`),

  metricsThroughput: (days = 30) =>
    get<{ enabled: boolean; days: ThroughputDay[] }>(
      `/api/metrics/throughput?days=${days}`,
    ),

  metricsQuality: () => get<QualityResponse>("/api/metrics/quality"),

  metricsIncidents: (
    opts: { category?: string; unresolvedOnly?: boolean; limit?: number } = {},
  ) => {
    const qs = new URLSearchParams();
    if (opts.category) qs.set("category", opts.category);
    if (opts.unresolvedOnly) qs.set("unresolved_only", "true");
    qs.set("limit", String(opts.limit ?? 50));
    return get<{ enabled: boolean; incidents: IncidentRow[] }>(
      `/api/metrics/incidents?${qs.toString()}`,
    );
  },

  metricsAgentStats: (runId?: string, limit = 100) => {
    const qs = new URLSearchParams();
    if (runId) qs.set("run_id", runId);
    qs.set("limit", String(limit));
    return get<{ enabled: boolean; agents: AgentStatRow[] }>(
      `/api/metrics/agent_stats?${qs.toString()}`,
    );
  },

  metricsToolCalls: (
    groupBy: "tool" | "agent" | "feature" = "tool",
    limit = 50,
  ) =>
    get<{ enabled: boolean; group_by: string; buckets: ToolCallBucket[] }>(
      `/api/metrics/tool_calls?group_by=${groupBy}&limit=${limit}`,
    ),

  metricsMemoryActivity: () => get<MemoryActivity>("/api/metrics/memory_activity"),

  metricsDecomposition: (limit = 100) =>
    get<DecompositionResponse>(`/api/metrics/decomposition?limit=${limit}`),

  metricsArtifacts: () => get<ArtifactsResponse>("/api/metrics/artifacts"),

  /** Per-run metrics detail for the Run Detail popup. */
  metricsRunDetail: (runId: string) =>
    get<RunDetailResponse>(`/api/metrics/runs/${encodeURIComponent(runId)}`),

  /** File tree under the run's output directory. */
  runFiles: (runId: string) =>
    get<RunFilesResponse>(`/api/runs/${encodeURIComponent(runId)}/files`),

  /** Text content for a single file inside a run's output directory. */
  runFileContent: (runId: string, path: string) =>
    get<RunFileContentResponse>(
      `/api/runs/${encodeURIComponent(runId)}/file?path=${encodeURIComponent(path)}`,
    ),

  metricsBackgroundLoop: (limit = 200) =>
    get<{ enabled: boolean; samples: BackgroundLoopSample[] }>(
      `/api/metrics/background_loop?limit=${limit}`,
    ),

  /** Kill-switch: ask the server to stop the in-flight pipeline run.
   *
   * Returns ``{cancelled: true}`` when a run was active and the signal
   * was installed, or ``{cancelled: false, reason: "no active run"}`` when
   * no run is in progress. Always 200 — callers don't need a try/catch
   * for the common "no active run" case. */
  cancelRun: () =>
    post<{ cancelled: boolean; reason?: string; already_pending?: boolean }>(
      "/api/agent/cancel",
    ),

  uploadFiles: async (
    files: File[],
  ): Promise<{ upload_id: string; path: string; files: string[] }> => {
    const formData = new FormData();
    for (const file of files) formData.append("files", file);
    const res = await fetch(`${BASE}/api/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw new Error(await describeFailure("POST", "/api/upload", res));
    }
    return parseJsonOrThrow<{ upload_id: string; path: string; files: string[] }>(
      "POST",
      "/api/upload",
      res,
    );
  },
};
