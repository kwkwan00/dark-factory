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
  e2e_timeout_seconds: number;
  e2e_browsers: string[];
  enable_episodic_memory: boolean;
  memory_dedup_threshold: number;
  llm_model: string;
  planning_model: string;
  eval_model: string;
  refinery_model: string;
  refinery_reasoning_effort: string;
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
  e2e_timeout_seconds?: number;
  e2e_browsers?: string[];
  enable_episodic_memory?: boolean;
  memory_dedup_threshold?: number;
  llm_model?: string;
  planning_model?: string;
  eval_model?: string;
  refinery_model?: string;
  refinery_reasoning_effort?: string;
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

/** Payload from GET /api/graph/gaps/{run_id} — run-scoped gap analysis. */
export interface SpecWithoutArtifacts {
  id: string;
  title: string | null;
  capability: string | null;
}

export interface SpecFailingEval {
  id: string;
  title: string | null;
  capability: string | null;
  eval_scores: Record<string, number>;
}

export interface UnimplementedRequirement {
  id: string;
  title: string | null;
  priority: string | null;
  reason: string;
}

export interface BrokenDependency {
  spec_id: string;
  spec_title: string | null;
  missing_dep_id: string;
}

export interface CapabilityIsland {
  capability: string;
  total_specs: number;
  disconnected_specs: string[];
}

export interface MissingEpisode {
  feature: string;
  spec_count: number;
}

export interface RunGaps {
  run_id: string;
  specs_without_artifacts: SpecWithoutArtifacts[];
  specs_failing_evals: SpecFailingEval[];
  unimplemented_requirements: UnimplementedRequirement[];
  broken_dependencies: BrokenDependency[];
  capability_islands: CapabilityIsland[];
  missing_episodes: MissingEpisode[];
  totals: {
    requirements: number;
    specs: number;
  };
}

// ── Requirements Refinery ─────────────────────────────────────────────────

export interface RequirementRelationship {
  target_id: string;
  type: string;
  rationale: string;
}

export interface SuggestedSpec {
  title: string;
  capability: string;
  description: string;
  acceptance_criteria: string[];
}

export interface RefinedRequirement {
  id: string;
  original_title: string;
  original_description: string;
  title: string;
  description: string;
  priority: string;
  tags: string[];
  relationships: RequirementRelationship[];
  suggested_specs: SuggestedSpec[];
  changes: string[];
  pass_context: string;
  // Phase 8/9 convergence metadata — optional; null on legacy payloads.
  convergence_status?: "converged" | "short_circuited" | "aborted" | null;
  /** 0..1 progress signal: 0 = no debate occurred, 1 = converged,
   *  anything between = debated but threshold-short. Lets operators
   *  gauge how much more detail a requirement still needs. */
  convergence_score?: number;
  unresolved_points?: string[];
  open_questions?: string[];
  explicit_tradeoffs?: string[];
  /** Per-debate summary surfaced under the Description disclosure. Set by
   *  the runner from ``final_trace``; legacy carry-forward paths leave
   *  this null so the disclosure renders only when there is something to
   *  show. The shape is intentionally flat (no typed contract import) so
   *  the wire format stays stable as the trace grows. */
  debate?: {
    rounds_executed?: number;
    convergence_status?: string | null;
    termination_reason?: string;
    escalation_level?: number;
    research_calls_used?: number;
    critiques_by_round?: Record<string, Array<{
      role: string;
      severity: string;
      dimension: string;
      finding: string;
      proposed_fix: string;
    }>>;
    rebuttals_by_round?: Record<string, {
      accepted_count: number;
      rejected_count: number;
      mode: string;
    }>;
    scores_by_round?: Record<string, {
      overall?: number | null;
      passed?: boolean | null;
      dimensions?: Record<string, number>;
    }>;
    /** Per-requirement Debate Episode — the operator-readable narrative
     *  of one debate, built deterministically from the trace. Acts both
     *  as documentation (rendered markdown) and as the structured record
     *  the Episodes tab groups by. */
    episode?: {
      requirement_id: string;
      refinery_run_id: string;
      title: string;
      outcome: string;
      rounds_executed: number;
      escalation_level: number;
      research_calls_used: number;
      summary: string;
      final_overall_score?: number | null;
      final_dimensions?: Record<string, number>;
      convergence_score?: number;
      key_events?: Array<{
        order: number;
        round_number: number;
        actor: string;
        kind: string;
        headline: string;
        detail?: string;
      }>;
      unresolved_points?: string[];
      open_questions?: string[];
      explicit_tradeoffs?: string[];
      participants?: string[];
      duration_seconds?: number;
    };
    episode_markdown?: string;
  } | null;
}

export interface SuggestedMemory {
  type: string;
  description: string;
  context: string;
  applicability: string;
  source_feature: string;
  rationale: string;
  /** V2 fields — optional for backward compat. ``kind`` is the canonical
   *  5-bucket label (decision/incident/pattern/constraint/conflict) the UI
   *  groups by. ``validation_status`` drives the auto-save default in the
   *  Apply split-action. Producers populate when available; legacy LLM-emitted
   *  memories carry only ``type`` and the UI infers a kind. */
  kind?: string | null;
  validation_status?: string | null;
  summary?: string;
  source_role?: string;
  source_requirement_id?: string | null;
}

export interface RefineryResponse {
  summary: string;
  pass_summaries: string[];
  refined_requirements: RefinedRequirement[];
  suggested_memories: SuggestedMemory[];
  new_relationships_count: number;
  requirements_modified_count: number;
  requirements_unchanged_count: number;
  source_run_id: string | null;
  methodology: string;
  evidence_summary: string;
  risk_areas: string[];
}

export interface RefineryHistoryItem {
  id: string;
  timestamp: string;
  source_run_id: string | null;
  source_mode: string;
  requirements_count: number;
  requirements_modified: number;
  requirements_unchanged: number;
  relationships_count: number;
  suggested_memories_count: number;
  duration_seconds: number;
}

/** A refinery run that didn't complete cleanly and is resumable
 *  via ``POST /api/refinery/resume/{refinery_run_id}``. */
export interface ResumableRefineryRun {
  refinery_run_id: string;
  source_mode: string;
  source_run_id: string | null;
  requirements_count: number;
  started_at: string | null;
  status: "in_progress" | "cancelled" | "failed" | string;
}

export interface MemoryDedupResult {
  is_duplicate: boolean;
  existing_id: string | null;
  existing_description: string | null;
  similarity: number | null;
}

/** Phase 5 goal-generation output — proposed additional requirements
 *  the operator can approve / dismiss individually. */
export interface ProposedRequirement {
  title: string;
  description: string;
  priority: string;
  tags: string[];
  rationale: string;
  related_to: string[];
  confidence: number;
}

export interface ProposedAdditions {
  summary: string;
  proposals: ProposedRequirement[];
}

export interface RefinerySSEEvent {
  phase:
    | "gathering"
    | "refining"
    | "reconciling"
    | "planning"
    | "proposed_additions"
    | "done"
    | "error";
  step?: string;
  turn?: number;
  max_turns?: number;
  message?: string;
  tools?: string[];
  text?: string;
  requirement_id?: string;
  data?: RefineryResponse | ProposedAdditions;
  result_id?: string;
  requirement_count?: number;
  duration_seconds?: number;
  proposals_count?: number;
  summary?: string;
  // Per-agent debate-graph event fields. Emitted by the LangGraph
  // debate subgraph (one event per node transition); the timeline UI
  // renders them as a live agent log.
  debate_event?:
    | "generator_started"
    | "draft_ready"
    | "critic_started"
    | "critic_ready"
    | "critic_placeholder"
    | "synthesize_started"
    | "synthesis_ready"
    | "score_started"
    | "score_ready"
    | "research_started"
    | "research_ready"
    | "escalation"
    | "reconcile_started"
    | "reconcile_ready"
    | "finalized";
  role?: string;
  round?: number;
  severity?: string;
  dimension?: string;
  overall?: number;
  passed?: boolean;
  insights?: number;
  source_tier_mix?: number[];
  escalation_level?: number;
  new_tier?: string;
  unresolved_count?: number;
  convergence_status?: string;
  rounds?: number;
  rebuttals_accepted?: number;
  reason?: string;
  // Single-shot LLM call events (refinery_llm_started / _ready / _failed).
  // Routed back into the refinery SSE stream by the shared LLM helper so
  // the in-flight panel activity is visible inline on RefineryTab as well
  // as on the Agent Log tab.
  event?: string;
  provider?: string;
  model?: string;
  reasoning_effort?: string;
  latency_ms?: number;
  tokens_in?: number;
  tokens_out?: number;
  error?: string;
}

// ── Run Diff ──────────────────────────────────────────────────────────────

export interface DiffFile {
  path: string;
  status: "added" | "removed" | "modified" | "unchanged" | "binary" | "error";
  diff?: string;
}

export interface RunDiffResponse {
  run_a: string;
  run_b: string;
  files: DiffFile[];
  stats: { added: number; removed: number; modified: number; unchanged: number };
}

// ── Run Compare ──────────────────────────────────────────────────────────

export interface RunCompareResponse {
  run_a: Record<string, unknown>;
  run_b: Record<string, unknown>;
}

// ── Traceability ─────────────────────────────────────────────────────────

export interface TraceabilityFile {
  path: string;
  preview_url: string;
  s3_url?: string;
}

export interface TraceabilitySpec {
  id: string;
  title: string | null;
  capability: string | null;
  files: TraceabilityFile[];
  test_files: TraceabilityFile[];
  eval_scores: Record<string, number>;
  all_passed: boolean | null;
}

export interface TraceabilityRow {
  requirement: { id: string; title: string | null; priority: string | null };
  specs: TraceabilitySpec[];
  overall_status: "pass" | "fail" | "no_specs" | "no_evals";
}

export interface TraceabilityResponse {
  run_id: string;
  rows: TraceabilityRow[];
}

// ── Graph Topology ───────────────────────────────────────────────────────

export interface TopologyNode {
  id: string;
  type: "requirement" | "spec";
  label: string;
  priority?: string;
  capability?: string;
  status?: string;
  file_count?: number;
  test_count?: number;
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  type: "IMPLEMENTS" | "DEPENDS_ON";
}

export interface GraphTopologyResponse {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
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

/** A single run in the history list from GET /api/history. */
export interface HistoryRun {
  id: string;
  status: string;
  pass_rate?: number;
  duration_seconds?: number;
  timestamp?: string;
  stage?: string;
  spec_count?: number;
  feature_count?: number;
  [key: string]: unknown;
}

export const api = {
  /** Run-scoped gap finder (used internally by refinery). */
  runGaps: (runId: string) =>
    get<RunGaps>(`/api/graph/gaps/${encodeURIComponent(runId)}`),

  /** Requirements Refinery — SSE stream.
   *
   * Three input modes (exactly one must be set):
   * - ``run_id``     — refine a historical run's evidence bundle
   * - ``input_path`` — refine from uploaded requirement documents
   * - ``direct``     — Phase 6 direct-input mode (one typed requirement)
   */
  streamRefinery: (
    body: {
      run_id?: string;
      input_path?: string;
      direct?: {
        title: string;
        description?: string;
        priority?: string;
        tags?: string[];
      };
    },
    signal?: AbortSignal,
  ) =>
    fetch(`/api/refinery`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),

  /** PATCH a single requirement in Neo4j.
   *
   * Phase 9 atomic write-back: include ``apply_memories`` in the body
   * to save selected suggested memories inside the same Postgres +
   * Neo4j transaction. Legacy callers that don't supply
   * ``apply_memories`` get today's behaviour unchanged.
   */
  patchRequirement: (reqId: string, body: Record<string, unknown>) =>
    fetch(`/api/graph/requirements/${encodeURIComponent(reqId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => {
      if (!r.ok) throw new Error(`Patch failed: ${r.status}`);
      return r.json();
    }),

  /** List historical refinery results. */
  refineryHistory: (limit = 20) =>
    get<{ results: RefineryHistoryItem[] }>(`/api/refinery/history?limit=${limit}`),

  /** List refinery runs that didn't complete cleanly and can be
   *  resumed via ``streamResumeRefinery``. */
  refineryResumable: (limit = 20) =>
    get<{ results: ResumableRefineryRun[]; message?: string }>(
      `/api/refinery/resumable?limit=${limit}`,
    ),

  /** Resume an interrupted refinery run. Same SSE shape as
   *  ``streamRefinery``; the orchestrator hydrates state from the
   *  resume registry and replays cached debate completions. */
  streamResumeRefinery: (refineryRunId: string, signal?: AbortSignal) =>
    fetch(`/api/refinery/resume/${encodeURIComponent(refineryRunId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
    }),

  /** Dismiss a resumable refinery run (tombstones status → cancelled).
   *  Removes it from the resumable list without dropping forensic
   *  telemetry. Use this for runs you no longer intend to resume.
   *  Throws on 404 (unknown id) or 409 (run is already completed). */
  dismissResumableRefinery: (refineryRunId: string) =>
    fetch(`/api/refinery/resumable/${encodeURIComponent(refineryRunId)}`, {
      method: "DELETE",
    }).then((r) => {
      if (!r.ok) throw new Error(`Dismiss failed: ${r.status}`);
      return r.json() as Promise<{ refinery_run_id: string; status: string }>;
    }),

  /** Load a saved refinery result. */
  loadRefineryResult: (resultId: string) =>
    get<RefineryResponse>(`/api/refinery/${encodeURIComponent(resultId)}`),

  /** Delete a saved refinery result. */
  deleteRefineryResult: (resultId: string) =>
    fetch(`/api/refinery/${encodeURIComponent(resultId)}`, { method: "DELETE" })
      .then((r) => { if (!r.ok) throw new Error(`Delete failed: ${r.status}`); return r.json(); }),

  /** Check if a memory is a semantic duplicate before saving.
   *
   * Phase 9 kind-scoped dedup: the optional ``kind`` field scopes the
   * check to the specified refinery memory kind. Legacy callers
   * (passing only ``type``) keep today's scoping.
   */
  checkMemoryDuplicate: (body: {
    type: string;
    description: string;
    context?: string;
    kind?: "decision" | "incident" | "pattern" | "constraint" | "conflict";
  }) => post<MemoryDedupResult>("/api/memory/check-duplicate", body),

  /** Export all requirements as JSON (for re-ingestion). */
  requirementsExportUrl: () => "/api/graph/requirements/export",

  /** Export refinery results as a ZIP (report + individual requirement files). */
  exportRefineryZip: (response: RefineryResponse) =>
    fetch("/api/refinery/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(response),
    }).then((r) => {
      if (!r.ok) throw new Error(`Export failed: ${r.status}`);
      return r.blob();
    }),

  history: (limit = 20) =>
    get<{ runs: HistoryRun[]; message?: string }>(`/api/history?limit=${limit}`),

  deleteRun: (runId: string) =>
    fetch(`/api/history/${encodeURIComponent(runId)}`, { method: "DELETE" })
      .then((r) => { if (!r.ok) throw new Error(`Delete failed: ${r.status}`); return r.json(); }),

  runDownloadUrl: (runId: string) =>
    `/api/runs/${encodeURIComponent(runId)}/download`,

  runDiff: (runA: string, runB: string) =>
    get<RunDiffResponse>(`/api/runs/diff?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`),

  runCompare: (runA: string, runB: string) =>
    get<RunCompareResponse>(`/api/runs/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`),

  traceability: (runId: string) =>
    get<TraceabilityResponse>(`/api/traceability/${encodeURIComponent(runId)}`),

  graphTopology: (runId: string) =>
    get<GraphTopologyResponse>(`/api/graph/topology/${encodeURIComponent(runId)}`),

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

  deleteMemory: (memoryId: string) =>
    fetch(`/api/memory/${encodeURIComponent(memoryId)}`, { method: "DELETE" })
      .then((r) => { if (!r.ok) throw new Error(`Delete failed: ${r.status}`); return r.json(); }),

  importMemories: (memories: Array<Record<string, unknown>>) =>
    post<{ imported: number; skipped: number; total: number }>(
      "/api/memory/import",
      { memories },
    ),

  /** Active-learning loop: tell the backend the operator decided
   * (accepted | dismissed | edited) on a suggested memory. Best-effort
   * — the response carries flags for which sinks succeeded but the UI
   * action proceeds regardless. */
  submitMemoryFeedback: (
    memoryId: string,
    body: {
      decision: "accepted" | "dismissed" | "edited";
      memory_kind: string;
      refinery_run_id?: string;
      source_role?: string;
      reason?: string;
    },
  ) =>
    post<{
      ok: boolean;
      telemetry_recorded?: boolean;
      relevance_adjusted?: boolean;
      conflict_emitted?: boolean;
      reason?: string;
    }>(`/api/refinery/memory/${encodeURIComponent(memoryId)}/feedback`, body),

  /** Active-learning loop: tell the backend which way the operator
   * decided on a refined requirement. Drives the Judge confidence-
   * calibration multiplier on subsequent runs. Best-effort — the
   * response carries a flag for whether telemetry landed but the UI
   * action proceeds regardless. */
  submitRequirementDecision: (
    refineryRunId: string,
    requirementId: string,
    body: {
      decision: "accepted" | "dismissed" | "edited";
      judge_overall_score?: number | null;
      convergence_status?: "converged" | "short_circuited" | "aborted" | null;
    },
  ) =>
    post<{ ok: boolean; telemetry_recorded?: boolean; reason?: string }>(
      `/api/refinery/${encodeURIComponent(refineryRunId)}/requirement/${encodeURIComponent(requirementId)}/decision`,
      body,
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
