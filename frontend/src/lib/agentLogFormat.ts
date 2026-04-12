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
  pipeline_cancelled: { label: "CANCELLED", color: "#f85149" },
};

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
    case "pipeline_cancelled": {
      const e = ev as { reason?: string; run_id?: string };
      return `Pipeline cancelled — ${e.reason ?? "user_requested"}${e.run_id ? ` (${e.run_id})` : ""}`;
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
