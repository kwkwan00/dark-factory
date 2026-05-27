/**
 * Shared formatting helpers for agent log events.
 *
 * Used by both the live AgentLogsTab (SSE stream) and the historical
 * AgentLog pane inside RunDetailWindow (DB records).
 */

import type { ProgressEvent } from "../api/client";

// ── Badge definitions ──────────────────────────────────────────────────────

export const EVENT_BADGES: Record<string, { label: string; color: string }> = {
  log_connected: { label: "CONNECTED", color: "#8b949e" },
  layer_started: { label: "LAYER START", color: "#58a6ff" },
  layer_completed: { label: "LAYER DONE", color: "#58a6ff" },
  feature_started: { label: "FEATURE START", color: "#d29922" },
  feature_completed: { label: "FEATURE DONE", color: "#3fb950" },
  feature_skipped: { label: "SKIPPED", color: "#8b949e" },
  agent_active: { label: "AGENT", color: "#bc8cff" },
  agent_llm_start: { label: "LLM CALL", color: "#bc8cff" },
  agent_decision: { label: "DECISION", color: "#e3b341" },
  agent_handoff: { label: "HANDOFF", color: "#f0883e" },
  tool_call: { label: "TOOL CALL", color: "#39c5cf" },
  tool_result: { label: "TOOL RESULT", color: "#7ee2ec" },
  spec_gen_layer_started: { label: "SPEC LAYER", color: "#58a6ff" },
  spec_plan_started: { label: "SPEC PLAN", color: "#d29922" },
  spec_plan_completed: { label: "SPEC PLAN OK", color: "#3fb950" },
  spec_plan_failed: { label: "SPEC PLAN FAIL", color: "#f85149" },
  spec_plan_resolved: { label: "DEP RESOLVE", color: "#7ee2ec" },
  spec_gen_started: { label: "SPEC START", color: "#d29922" },
  spec_handoff: { label: "SPEC HANDOFF", color: "#bc8cff" },
  eval_rubric: { label: "EVAL RUBRIC", color: "#7ee2ec" },
  spec_gen_completed: { label: "SPEC DONE", color: "#3fb950" },
  spec_gen_failed: { label: "SPEC FAIL", color: "#f85149" },
  spec_gen_layer_completed: { label: "SPEC LAYER DONE", color: "#58a6ff" },
  reconciliation_started: { label: "RECON START", color: "#f0883e" },
  reconciliation_completed: { label: "RECON DONE", color: "#f0883e" },
  e2e_started: { label: "E2E START", color: "#d2a8ff" },
  e2e_completed: { label: "E2E DONE", color: "#d2a8ff" },
  e2e_validation_skipped: { label: "E2E SKIPPED", color: "#8b949e" },
  spec_gen_skipped: { label: "SPEC REUSED", color: "#7ee2ec" },
  ingest_completed: { label: "INGEST DONE", color: "#3fb950" },
  spec_reconciliation_completed: { label: "SPEC RECON DONE", color: "#3fb950" },
  graph_write_completed: { label: "GRAPH DONE", color: "#3fb950" },
  swarm_completed: { label: "SWARM DONE", color: "#3fb950" },
  pipeline_completed: { label: "PIPELINE DONE", color: "#3fb950" },
  deep_agent_turn: { label: "DEEP AGENT", color: "#bc8cff" },
  reflection_started: { label: "REFLECTING", color: "#e3b341" },
  reflection_completed: { label: "REFLECTED", color: "#e3b341" },
  pipeline_cancelled: { label: "CANCELLED", color: "#f85149" },
  // ── Refinery debate events (color-coded by agent role) ──
  refinery_phase:                { label: "REFINERY",     color: "#8b949e" },
  refinery_generator_started:    { label: "PRODUCT",      color: "#79c0ff" },
  refinery_draft_ready:          { label: "PRODUCT ✓",    color: "#79c0ff" },
  refinery_critic_started:       { label: "CRITIC",       color: "#f0b72f" },
  refinery_critic_ready:         { label: "CRITIC ✓",     color: "#f0b72f" },
  refinery_critic_placeholder:   { label: "CRITIC SKIP",  color: "#8b949e" },
  refinery_synthesize_started:   { label: "JUDGE SYNTH",  color: "#bc8cff" },
  refinery_synthesis_ready:      { label: "JUDGE SYNTH ✓", color: "#bc8cff" },
  refinery_score_started:        { label: "JUDGE SCORE",  color: "#bc8cff" },
  refinery_score_ready:          { label: "JUDGE SCORE ✓", color: "#bc8cff" },
  refinery_research_started:     { label: "RESEARCH",     color: "#39c5cf" },
  refinery_research_ready:       { label: "RESEARCH ✓",   color: "#39c5cf" },
  refinery_escalation:           { label: "ESCALATED",    color: "#f0883e" },
  refinery_reconcile_started:    { label: "RECONCILE",    color: "#d29922" },
  refinery_reconcile_ready:      { label: "RECONCILE ✓",  color: "#d29922" },
  refinery_finalized:            { label: "FINALIZED",    color: "#3fb950" },
  // LLM single-shot calls — badge colour is overridden by resolveBadge
  // via the ``agent`` field, so these defaults are only used when the
  // payload is missing an agent (shouldn't happen in practice).
  refinery_llm_started:          { label: "LLM CALL",     color: "#bc8cff" },
  refinery_llm_ready:            { label: "LLM CALL ✓",   color: "#bc8cff" },
  refinery_llm_failed:           { label: "LLM CALL ✗",   color: "#f85149" },
};

// ── Per-agent color palette ────────────────────────────────────────────────
//
// Every known agent role gets a stable badge label + color so the agent log
// identifies WHICH agent emitted each event (not just the generic event
// type). Events that carry an ``agent`` or ``role`` field in their payload
// route through ``resolveBadge`` below and pick up these styles at render
// time, falling back to EVENT_BADGES for events without agent identity.

export const AGENT_STYLE: Record<string, { label: string; color: string }> = {
  // Swarm agents (manufacture pipeline)
  planner:     { label: "PLANNER",    color: "#f0b72f" },
  coder:       { label: "CODER",      color: "#56d364" },
  reviewer:    { label: "REVIEWER",   color: "#79c0ff" },
  tester:      { label: "TESTER",     color: "#bc8cff" },
  // Refinery panel seats
  product:     { label: "PRODUCT",    color: "#79c0ff" },
  engineering: { label: "ENG",        color: "#56d364" },
  security:    { label: "SEC",        color: "#ff7b72" },
  operations:  { label: "OPS",        color: "#d29922" },
  cost:        { label: "COST",       color: "#e3b341" },
  judge:       { label: "JUDGE",      color: "#bc8cff" },
  research:    { label: "RESEARCH",   color: "#39c5cf" },
  rules:       { label: "RULES",      color: "#8b949e" },
  // Legacy refinery single-agent path
  "refinery-deep-agent":   { label: "REFINERY",  color: "#d29922" },
  "reconciliation-agent":  { label: "RECON",     color: "#f0883e" },
};

/**
 * Extract the agent identity from an event payload. The backend emits the
 * agent under different keys depending on the event family — ``role`` on
 * refinery debate events, ``agent`` on swarm events and deep-agent turns,
 * and ``to_agent`` on handoffs (handoff "active agent" = the destination).
 */
function eventAgentKey(ev: ProgressEvent): string | undefined {
  const asRecord = ev as Record<string, unknown>;
  const role = asRecord.role;
  if (typeof role === "string" && role) return role.toLowerCase();
  const agent = asRecord.agent;
  if (typeof agent === "string" && agent) return agent.toLowerCase();
  const toAgent = asRecord.to_agent;
  if (typeof toAgent === "string" && toAgent) return toAgent.toLowerCase();
  return undefined;
}

/**
 * Resolve the badge for an event. Prefers per-agent styling when the event
 * carries an identifiable agent; otherwise falls back to the static
 * EVENT_BADGES table.
 */
export function resolveBadge(ev: ProgressEvent): { label: string; color: string } {
  const agentKey = eventAgentKey(ev);
  if (agentKey && AGENT_STYLE[agentKey]) {
    const style = AGENT_STYLE[agentKey];
    const suffix = eventBadgeSuffix(ev.event);
    return { label: suffix ? `${style.label} ${suffix}` : style.label, color: style.color };
  }
  return EVENT_BADGES[ev.event] ?? { label: ev.event.toUpperCase(), color: "#8b949e" };
}

/** Small suffix (e.g. ``✓`` for *_ready, ``→`` for tool_call) that gets
 *  appended to agent-specific badges so "PRODUCT" vs "PRODUCT ✓" still
 *  visually distinguish start vs finish events. */
function eventBadgeSuffix(eventName: string): string {
  if (eventName.endsWith("_ready") || eventName.endsWith("_completed")) return "✓";
  if (eventName.endsWith("_failed")) return "✗";
  if (eventName.endsWith("_started")) return "…";
  if (eventName === "tool_call") return "→";
  if (eventName === "tool_result") return "←";
  if (eventName === "agent_handoff") return "↦";
  if (eventName === "agent_decision") return "·";
  if (eventName === "agent_llm_start") return "…";
  if (eventName === "refinery_escalation") return "⇡";
  if (eventName === "refinery_critic_placeholder") return "—";
  return "";
}

// ── String helpers ─────────────────────────────────────────────────────────

/** Truncate ``s`` to ``max`` chars and append an ellipsis when cut.
 * Returns empty string for undefined / null input. */
export function truncate(s: string | undefined | null, max: number): string {
  if (!s) return "";
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

// ── Time formatting ────────────────────────────────────────────────────────

/** Format a UNIX timestamp (seconds) into HH:MM:SS.mmm */
export function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return (
    d.toLocaleTimeString("en-US", { hour12: false }) +
    "." +
    String(d.getMilliseconds()).padStart(3, "0")
  );
}

/** Format an ISO-8601 / Postgres timestamp string into HH:MM:SS.mmm */
export function formatTimeFromISO(iso: string): string {
  const d = new Date(iso);
  return (
    d.toLocaleTimeString("en-US", { hour12: false }) +
    "." +
    String(d.getMilliseconds()).padStart(3, "0")
  );
}

// ── Event detail formatting ────────────────────────────────────────────────

export function formatEventDetails(ev: ProgressEvent): string {
  switch (ev.event) {
    case "log_connected":
      return "Connected to event stream";
    case "layer_started":
      return `Layer ${ev.layer}/${ev.total_layers} — ${(ev.features ?? []).join(", ") || "(no features)"}`;
    case "layer_completed":
      return `Layer ${ev.layer} complete`;
    case "feature_started":
      return `Feature "${ev.feature}" started (${ev.spec_count} spec${ev.spec_count === 1 ? "" : "s"})`;
    case "feature_completed": {
      const base = `Feature "${ev.feature}" ${ev.status} — ${ev.artifacts} artifact(s), ${ev.tests} test(s)`;
      return ev.error ? `${base} — ${ev.error}` : base;
    }
    case "feature_skipped":
      return `Feature "${ev.feature}" skipped — ${ev.reason}`;
    case "agent_active":
      return `"${ev.feature}" → ${ev.agent} active (handoff #${ev.messages ?? 0})`;
    case "agent_llm_start": {
      const e = ev as { feature?: string; agent?: string; model?: string };
      const model = e.model ? ` (${e.model})` : "";
      return `"${e.feature ?? "?"}" ${e.agent ?? "agent"} thinking${model}…`;
    }
    case "agent_decision": {
      const e = ev as { feature?: string; agent?: string; text?: string };
      const text = e.text ?? "";
      const snippet = text.length > 200 ? `${text.slice(0, 200)}…` : text;
      return `"${e.feature}" ${e.agent ?? "?"}: ${snippet}`;
    }
    case "agent_handoff": {
      const e = ev as { feature?: string; from_agent?: string; to_agent?: string };
      return `"${e.feature}" handoff: ${e.from_agent ?? "?"} → ${e.to_agent ?? "?"}`;
    }
    case "tool_call": {
      const e = ev as {
        feature?: string;
        agent?: string;
        tool?: string;
        args_preview?: string;
      };
      const args = e.args_preview ? ` ${e.args_preview}` : "";
      return `"${e.feature}" ${e.agent ?? "?"} → tool ${e.tool ?? "?"}${args}`;
    }
    case "tool_result": {
      const e = ev as {
        feature?: string;
        tool?: string;
        result_preview?: string;
      };
      const preview = e.result_preview ?? "";
      const snippet = preview.length > 80 ? `${preview.slice(0, 80)}…` : preview;
      return snippet
        ? `"${e.feature}" ${e.tool ?? "?"} ← ${snippet}`
        : `"${e.feature}" ${e.tool ?? "?"} done`;
    }
    case "spec_gen_layer_started": {
      const e = ev as {
        parallel?: number;
        planned_sub_specs?: number;
        decomposition_enabled?: boolean;
      };
      const parallel = e.parallel ?? 1;
      const total = ev.total ?? 0;
      const planned = e.planned_sub_specs ?? total;
      if (e.decomposition_enabled && planned !== total) {
        return `Generating ${planned} sub-spec(s) from ${total} requirement(s) with ${parallel} worker(s) in parallel`;
      }
      return `Generating ${total} spec(s) with ${parallel} worker(s) in parallel`;
    }
    case "spec_plan_started":
      return `Planning sub-specs for "${(ev as { requirement_title?: string }).requirement_title ?? "(unknown)"}"`;
    case "spec_plan_completed": {
      const e = ev as {
        requirement_title?: string;
        sub_spec_count?: number;
        titles?: string[];
      };
      const count = e.sub_spec_count ?? 0;
      const title = e.requirement_title ?? "(unknown)";
      const titles = e.titles ?? [];
      const preview = titles.slice(0, 5).join(", ");
      const more = titles.length > 5 ? "…" : "";
      return `Planned ${count} sub-spec(s) for "${title}"` +
        (preview ? `: ${preview}${more}` : "");
    }
    case "spec_plan_failed": {
      const e = ev as { requirement_title?: string; requirement_id?: string };
      const id = e.requirement_title ?? e.requirement_id ?? "(unknown)";
      return `Spec planning failed for "${id}" — ${ev.error ?? ""} (falling back to single spec)`;
    }
    case "spec_plan_resolved": {
      const e = ev as {
        requirement_id?: string;
        resolved?: number;
        unresolved?: number;
      };
      const resolved = e.resolved ?? 0;
      const unresolved = e.unresolved ?? 0;
      const tail = unresolved ? ` (${unresolved} unresolved)` : "";
      return `Resolved ${resolved} sub-spec dep(s) for "${e.requirement_id ?? "?"}"${tail}`;
    }
    case "spec_gen_started": {
      const idx = ((ev as { index?: number }).index ?? 0) + 1;
      const reqTitle = (ev as { requirement_title?: string }).requirement_title ?? "(unknown)";
      const subTitle = (ev as { sub_spec_title?: string | null }).sub_spec_title;
      const label = subTitle ? `${reqTitle} → ${subTitle}` : reqTitle;
      return `[${idx}/${ev.total}] Spec generating: "${label}"`;
    }
    case "eval_rubric": {
      const e = ev as {
        requirement_title?: string;
        attempt?: number;
        max_handoffs?: number;
        avg_score?: number;
        threshold?: number;
        metrics?: Array<{
          name: string;
          score: number;
          passed: boolean;
          reason?: string;
        }>;
      };
      const title = e.requirement_title ?? "(unknown)";
      const avg = (e.avg_score ?? 0).toFixed(2);
      const threshold = (e.threshold ?? 0).toFixed(2);
      const metrics = e.metrics ?? [];
      const lines = metrics.map((m) => {
        const marker = m.passed ? "✓" : "✕";
        const reason =
          m.reason && m.reason.length > 60
            ? `: ${m.reason.slice(0, 60)}…`
            : m.reason
            ? `: ${m.reason}`
            : "";
        return `${marker} ${m.name} ${m.score.toFixed(2)}${reason}`;
      });
      return `"${title}" rubric attempt ${e.attempt}/${e.max_handoffs} avg=${avg}/${threshold} | ${lines.join(" · ")}`;
    }
    case "spec_handoff": {
      const e = ev as {
        requirement_title?: string;
        attempt?: number;
        max_handoffs?: number;
        score?: number;
        threshold?: number;
        role?: string;
      };
      const title = e.requirement_title ?? "(unknown)";
      const score = (e.score ?? 0).toFixed(2);
      const threshold = (e.threshold ?? 0).toFixed(2);
      return `"${title}" handoff ${e.attempt}/${e.max_handoffs} (${e.role}) — score ${score} (threshold ${threshold})`;
    }
    case "spec_gen_completed": {
      const e = ev as {
        spec_title?: string;
        spec_id?: string;
        final_score?: number;
        attempts?: number;
      };
      const title = e.spec_title ?? "(unknown)";
      const sid = e.spec_id ?? "?";
      const score = (e.final_score ?? 0).toFixed(2);
      return `Spec done: "${title}" (${sid}) — score ${score} after ${e.attempts ?? 1} attempt(s)`;
    }
    case "spec_gen_failed": {
      const rid = (ev as { requirement_id?: string }).requirement_id ?? "(unknown)";
      return `Spec failed for "${rid}" — ${ev.error ?? ""}`;
    }
    case "spec_gen_layer_completed":
      return `Spec generation complete: ${ev.total} done${(ev as { failed?: number }).failed ? `, ${(ev as { failed?: number }).failed} failed` : ""}`;
    case "reconciliation_started": {
      const e = ev as { feature_count?: number; max_turns?: number; timeout_seconds?: number };
      return `Reconciliation starting — ${e.feature_count ?? 0} feature(s), max ${e.max_turns ?? "?"} turns, ${e.timeout_seconds ?? "?"}s timeout`;
    }
    case "reconciliation_completed": {
      const e = ev as {
        status?: string;
        summary?: string;
        duration_seconds?: number;
        report_path?: string;
      };
      const dur = e.duration_seconds != null ? ` in ${e.duration_seconds.toFixed(1)}s` : "";
      return `Reconciliation ${e.status ?? "done"}${dur}${e.summary ? ` — ${e.summary}` : ""}`;
    }
    case "e2e_started": {
      const e = ev as {
        feature_count?: number;
        browsers?: string[];
        max_turns?: number;
        timeout_seconds?: number;
      };
      const browsers = (e.browsers ?? []).join(", ") || "?";
      return `E2E validation starting — ${e.feature_count ?? 0} feature(s), browsers: ${browsers}, max ${e.max_turns ?? "?"} turns`;
    }
    case "e2e_completed": {
      const e = ev as {
        status?: string;
        summary?: string;
        duration_seconds?: number;
        tests_total?: number;
        tests_passed?: number;
        tests_failed?: number;
        browsers_run?: string[];
      };
      const dur = e.duration_seconds != null ? ` in ${e.duration_seconds.toFixed(1)}s` : "";
      const tests = e.tests_total != null ? ` — ${e.tests_passed ?? 0}/${e.tests_total} passed` : "";
      const failed = e.tests_failed ? `, ${e.tests_failed} failed` : "";
      const browsers = (e.browsers_run ?? []).length > 0 ? ` across ${(e.browsers_run ?? []).join(", ")}` : "";
      return `E2E validation ${e.status ?? "done"}${dur}${tests}${failed}${browsers}`;
    }
    case "deep_agent_turn": {
      const e = ev as {
        feature?: string;
        agent?: string;
        turn?: number;
        max_turns?: number;
        tools?: string[];
        text?: string;
        message?: string;
      };
      const turnLabel = e.turn != null ? `turn ${e.turn}/${e.max_turns ?? "?"}` : "";
      const tools = (e.tools ?? []).length > 0 ? ` → ${(e.tools ?? []).join(", ")}` : "";
      const body = e.text
        ? ` — ${e.text.length > 120 ? e.text.slice(0, 120) + "…" : e.text}`
        : e.message
          ? ` — ${e.message}`
          : "";
      // Identify the agent explicitly rather than the generic
      // "deep agent" — the badge already color-codes it, but surfacing
      // the name in text makes filtering + scanning easier.
      const who = e.agent ?? `${e.feature ?? "?"} deep agent`;
      return `${who} ${turnLabel}${tools}${body}`.trim();
    }
    case "reflection_started": {
      const e = ev as { layer?: number; attempt?: number; max_retries?: number };
      return `Reflecting on layer ${e.layer ?? "?"} failures (attempt ${e.attempt ?? 1}/${e.max_retries ?? 1})…`;
    }
    case "reflection_completed": {
      const e = ev as { layer?: number; diagnosis?: string; retryable?: string[]; terminal?: string[] };
      const retryable = (e.retryable ?? []).length;
      const terminal = (e.terminal ?? []).length;
      const action = retryable > 0 ? `retrying ${retryable} feature${retryable > 1 ? "s" : ""}` : "no retry";
      const diag = e.diagnosis ? ` — ${e.diagnosis.length > 200 ? e.diagnosis.slice(0, 200) + "…" : e.diagnosis}` : "";
      return `Layer ${e.layer ?? "?"}: ${action}${terminal > 0 ? `, ${terminal} terminal` : ""}${diag}`;
    }
    case "pipeline_cancelled": {
      const e = ev as { reason?: string; run_id?: string };
      return `Pipeline cancelled — ${e.reason ?? "user_requested"}${e.run_id ? ` (${e.run_id})` : ""}`;
    }
    // ── Refinery debate events ────────────────────────────────────
    case "refinery_phase": {
      const e = ev as { phase?: string; message?: string; step?: string; requirement_count?: number };
      if (e.step === "complete") return `Refinery gathered ${e.requirement_count ?? 0} requirement(s)`;
      if (e.step) return `Refinery gathering: ${e.step}`;
      return e.message ?? "Refinery phase";
    }
    case "refinery_generator_started": {
      const e = ev as { requirement_id?: string; title?: string; priority?: string };
      const title = e.title ? ` "${e.title}"` : "";
      const pri = e.priority ? ` [${e.priority}]` : "";
      return `Product drafting${title} (${e.requirement_id ?? "?"})${pri}`;
    }
    case "refinery_draft_ready": {
      const e = ev as {
        requirement_id?: string; title?: string; priority?: string;
        specs_count?: number; relationships_count?: number;
      };
      const title = e.title ? ` "${e.title}"` : "";
      return `Product draft ready${title} — ${e.specs_count ?? 0} spec(s), ${e.relationships_count ?? 0} relationship(s)${e.priority ? ` [${e.priority}]` : ""}`;
    }
    case "refinery_critic_started": {
      const e = ev as {
        role?: string; round?: number; requirement_id?: string;
        target_title?: string; target_iteration?: number;
      };
      const target = e.target_title ? ` "${e.target_title}"` : ` "${e.requirement_id ?? "?"}"`;
      const iter = e.target_iteration != null ? ` v${e.target_iteration}` : "";
      return `${e.role ?? "critic"} round ${e.round ?? "?"} reviewing${target}${iter}`;
    }
    case "refinery_critic_ready": {
      const e = ev as {
        role?: string; round?: number; severity?: string; dimension?: string;
        finding?: string; proposed_fix?: string; confidence?: number;
        cited_evidence_count?: number; is_placeholder?: boolean;
      };
      const sev = (e.severity ?? "?").toUpperCase();
      const head = `${e.role ?? "critic"} round ${e.round ?? "?"} — ${sev} / ${e.dimension ?? "?"}`;
      if (e.is_placeholder) return `${head} (placeholder — no substantive finding)`;
      const conf = e.confidence != null ? ` · conf ${e.confidence.toFixed(2)}` : "";
      const evid = e.cited_evidence_count ? ` · ${e.cited_evidence_count} cite(s)` : "";
      const finding = e.finding ? ` — “${truncate(e.finding, 180)}”` : "";
      const fix = e.proposed_fix ? ` · fix: ${truncate(e.proposed_fix, 140)}` : "";
      return `${head}${conf}${evid}${finding}${fix}`;
    }
    case "refinery_critic_placeholder": {
      const e = ev as {
        role?: string; round?: number; reason?: string;
        severity?: string; dimension?: string;
      };
      const dim = e.dimension ? ` / ${e.dimension}` : "";
      return `${e.role ?? "critic"} round ${e.round ?? "?"} skipped${dim} — ${e.reason ?? "unknown"}`;
    }
    case "refinery_synthesize_started": {
      const e = ev as {
        round?: number; requirement_id?: string; critique_count?: number;
        blocker_count?: number; warning_count?: number; critic_roles?: string[];
      };
      const breakdown =
        e.critique_count != null
          ? ` — ${e.critique_count} critique(s) (${e.blocker_count ?? 0} blocker, ${e.warning_count ?? 0} warning)`
          : "";
      const roles = e.critic_roles?.length ? ` from ${e.critic_roles.join(", ")}` : "";
      return `Judge synthesizing round ${e.round ?? "?"} on "${e.requirement_id ?? "?"}"${breakdown}${roles}`;
    }
    case "refinery_synthesis_ready": {
      const e = ev as {
        round?: number;
        counts?: { accepted?: number; rejected?: number; deferred?: number; partial?: number };
        entries_preview?: Array<{ role?: string; severity?: string; dimension?: string; action?: string; rationale?: string }>;
        tradeoffs_count?: number;
        tradeoffs_preview?: string[];
        unresolved_count?: number;
        open_questions_count?: number;
        convergence_status?: string;
      };
      const c = e.counts ?? {};
      const counts =
        `${c.accepted ?? 0} accept · ${c.rejected ?? 0} reject` +
        (c.deferred ? ` · ${c.deferred} defer` : "") +
        (c.partial ? ` · ${c.partial} partial` : "");
      const top = e.entries_preview?.[0];
      const sample = top
        ? ` — e.g. ${top.role ?? "?"}/${(top.severity ?? "?").toUpperCase()} ${top.action ?? "?"}: ${truncate(top.rationale ?? "", 140)}`
        : "";
      const trade = e.tradeoffs_count
        ? ` · ${e.tradeoffs_count} tradeoff(s)${e.tradeoffs_preview?.[0] ? `: ${truncate(e.tradeoffs_preview[0], 100)}` : ""}`
        : "";
      const status = e.convergence_status ? ` [${e.convergence_status}]` : "";
      const unresolved = e.unresolved_count ? ` · ${e.unresolved_count} unresolved` : "";
      return `Judge synthesis round ${e.round ?? "?"} done — ${counts}${trade}${unresolved}${sample}${status}`;
    }
    case "refinery_score_started": {
      const e = ev as { round?: number; requirement_id?: string; target_iteration?: number };
      const iter = e.target_iteration != null ? ` v${e.target_iteration}` : "";
      return `Judge scoring round ${e.round ?? "?"} on "${e.requirement_id ?? "?"}"${iter}`;
    }
    case "refinery_score_ready": {
      const e = ev as {
        round?: number; overall?: number; passed?: boolean;
        overall_threshold?: number;
        dimensions?: Record<string, number>;
        failing_dimensions?: string[];
        reasons?: Record<string, string>;
        disagreement_score?: number;
        missing_external_info?: boolean;
        rule_violations_count?: number;
        rule_warnings_count?: number;
        fallback_used?: boolean;
      };
      const verdict = e.passed ? "✓ passed" : "✗ failed";
      const thr = e.overall_threshold != null ? `/${e.overall_threshold.toFixed(2)}` : "";
      const dims = e.dimensions
        ? " · " + Object.entries(e.dimensions)
            .map(([k, v]) => `${k}=${(v as number).toFixed(2)}`)
            .join(" ")
        : "";
      const fail = e.failing_dimensions?.length ? ` · failing: ${e.failing_dimensions.join(", ")}` : "";
      const rules = (e.rule_violations_count || e.rule_warnings_count)
        ? ` · rules: ${e.rule_violations_count ?? 0} blocker / ${e.rule_warnings_count ?? 0} warn`
        : "";
      const disagree = e.disagreement_score && e.disagreement_score > 0
        ? ` · disagreement ${e.disagreement_score.toFixed(2)}`
        : "";
      const research = e.missing_external_info ? " · needs-research" : "";
      const fb = e.fallback_used ? " · fallback-judge" : "";
      // Surface the first failing dimension's reason for fast triage.
      let firstReason = "";
      if (!e.passed && e.failing_dimensions?.length && e.reasons) {
        const r = e.reasons[e.failing_dimensions[0]];
        if (r) firstReason = ` — ${e.failing_dimensions[0]}: ${truncate(r, 160)}`;
      }
      return `Judge score round ${e.round ?? "?"} — ${(e.overall ?? 0).toFixed(2)}${thr} ${verdict}${dims}${fail}${rules}${disagree}${research}${fb}${firstReason}`;
    }
    case "refinery_research_started": {
      const e = ev as { round?: number; requirement_id?: string };
      return `Research gathering external sources for round ${e.round ?? "?"} on "${e.requirement_id ?? "?"}"`;
    }
    case "refinery_research_ready": {
      const e = ev as { round?: number; insights?: number };
      return `Research round ${e.round ?? "?"} — ${e.insights ?? 0} validated insight(s)`;
    }
    case "refinery_escalation": {
      const e = ev as { new_tier?: string; escalation_level?: number };
      return `Escalated to ${e.new_tier ?? "strong"} tier (level ${e.escalation_level ?? "?"})`;
    }
    case "refinery_reconcile_started": {
      const e = ev as {
        round?: number; requirement_id?: string;
        rounds_executed?: number; total_blockers?: number;
      };
      const blockers = e.total_blockers ? ` — ${e.total_blockers} total blocker(s) across rounds` : "";
      return `Judge reconciling unresolved tensions (round ${e.round ?? "?"}) on "${e.requirement_id ?? "?"}"${blockers}`;
    }
    case "refinery_reconcile_ready": {
      const e = ev as {
        round?: number; unresolved_count?: number; unresolved_preview?: string[];
        open_questions_count?: number; open_questions_preview?: string[];
        tradeoffs_count?: number;
      };
      const tail = e.unresolved_preview?.[0] ? ` — first: ${truncate(e.unresolved_preview[0], 160)}` : "";
      const oq = e.open_questions_count ? ` · ${e.open_questions_count} open question(s)` : "";
      const tr = e.tradeoffs_count ? ` · ${e.tradeoffs_count} tradeoff(s)` : "";
      return `Reconciled round ${e.round ?? "?"} — ${e.unresolved_count ?? 0} unresolved${oq}${tr}${tail}`;
    }
    case "refinery_finalized": {
      const e = ev as { requirement_id?: string; convergence_status?: string; rounds?: number };
      return `Finalized "${e.requirement_id ?? "?"}": ${e.convergence_status ?? "?"} after ${e.rounds ?? 0} round(s)`;
    }
    case "refinery_llm_started": {
      const e = ev as {
        agent?: string; model?: string; provider?: string;
        reasoning_effort?: string;
      };
      const provider = e.provider ? ` via ${e.provider}` : "";
      const effort = e.reasoning_effort ? ` (${e.reasoning_effort})` : "";
      return `${e.agent ?? "?"} calling ${e.model ?? "?"}${provider}${effort}`;
    }
    case "refinery_llm_ready": {
      const e = ev as {
        agent?: string; model?: string; provider?: string;
        latency_ms?: number;
        tokens_in?: number; tokens_out?: number;
      };
      const latency = e.latency_ms != null ? `${(e.latency_ms / 1000).toFixed(1)}s` : "?";
      const tokens = `${e.tokens_in ?? 0} in / ${e.tokens_out ?? 0} out`;
      const provider = e.provider ? ` (${e.provider})` : "";
      return `${e.agent ?? "?"} ← ${e.model ?? "?"}${provider} ${latency}, ${tokens}`;
    }
    case "refinery_llm_failed": {
      const e = ev as {
        agent?: string; model?: string; provider?: string;
        latency_ms?: number; error?: string;
      };
      const latency = e.latency_ms != null ? ` after ${(e.latency_ms / 1000).toFixed(1)}s` : "";
      const provider = e.provider ? ` (${e.provider})` : "";
      return `${e.agent ?? "?"} LLM call failed${provider}${latency} — ${e.error ?? "unknown"}`;
    }
    default:
      return JSON.stringify(ev);
  }
}

// ── ProgressLogEntry → ProgressEvent adapter ───────────────────────────────

import type { ProgressLogEntry } from "../api/client";

/**
 * Flatten a DB-sourced ProgressLogEntry into a ProgressEvent shape so that
 * formatEventDetails / EVENT_BADGES work identically for historical logs.
 */
export function logEntryToProgressEvent(entry: ProgressLogEntry): ProgressEvent {
  // Parse ISO timestamp → unix seconds
  const ts = new Date(entry.timestamp).getTime() / 1000;
  return {
    event: entry.event,
    timestamp: ts,
    feature: entry.feature ?? undefined,
    agent: entry.agent ?? undefined,
    ...entry.payload,
  } as ProgressEvent;
}
