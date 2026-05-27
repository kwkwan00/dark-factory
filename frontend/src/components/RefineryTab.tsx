import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AGENT_STYLE, truncate } from "../lib/agentLogFormat";
import { openAboutRequirementsRefinery } from "../lib/openAboutRequirementsRefinery";
import {
  api,
  type HistoryRun,
  type ProposedAdditions,
  type ProposedRequirement,
  type RefinedRequirement,
  type RefineryHistoryItem,
  type RefineryResponse,
  type RefinerySSEEvent,
  type ResumableRefineryRun,
  type SuggestedMemory,
} from "../api/client";

// ── Types ────────────────────────────────────────────────────────────────────

type Phase = "input" | "gathering" | "refining" | "reconciling" | "done" | "error";
type ResultTab = "requirements" | "insights" | "memories" | "episodes";
type ReqFilter = "all" | "modified" | "unchanged" | "unresolved";

// ── Helpers ──────────────────────────────────────────────────────────────────

function priorityBadge(priority: string): string {
  if (priority === "critical" || priority === "high") return "badge-error";
  if (priority === "medium") return "badge-warn";
  return "badge-info";
}

function severityColor(type: string): string {
  if (type === "conflicts_with") return "#f85149";
  if (type === "depends_on") return "#58a6ff";
  if (type === "extends") return "#3fb950";
  return "#8b949e";
}

// ── Memory kind classification ───────────────────────────────────────────────
//
// The wire format carries an optional ``kind`` field (decision / incident /
// pattern / constraint / conflict). Legacy LLM-emitted memories from the
// Product role only carry ``type`` (pattern | strategy) — those map to
// ``pattern`` so the UI can bucket every memory into one of five sub-tabs
// regardless of which producer wrote it.

type MemoryKind = "decision" | "incident" | "pattern" | "constraint" | "conflict";
const MEMORY_KINDS: readonly MemoryKind[] = [
  "decision", "incident", "pattern", "constraint", "conflict",
];

const MEMORY_KIND_COLOR: Record<MemoryKind, { bg: string; fg: string; border: string }> = {
  decision:   { bg: "#2d2144", fg: "#bc8cff", border: "#bc8cff40" },
  incident:   { bg: "#3a1f1f", fg: "#ff7b72", border: "#ff7b7240" },
  pattern:    { bg: "#1f2d44", fg: "#79c0ff", border: "#79c0ff40" },
  constraint: { bg: "#3a2a17", fg: "#f0b72f", border: "#f0b72f40" },
  conflict:   { bg: "#3a1f2a", fg: "#f778ba", border: "#f778ba40" },
};

const MEMORY_KIND_LABEL: Record<MemoryKind, string> = {
  decision: "Decision", incident: "Incident", pattern: "Pattern",
  constraint: "Constraint", conflict: "Conflict",
};

function memoryKindOf(mem: SuggestedMemory): MemoryKind {
  const k = (mem.kind || "").toLowerCase();
  if ((MEMORY_KINDS as readonly string[]).includes(k)) return k as MemoryKind;
  // Legacy fallback: type=strategy|pattern both map to pattern.
  return "pattern";
}

// ── Input Selection ──────────────────────────────────────────────────────────

// "direct" input — user types one requirement directly. The refinery
// wraps it in a minimal run_context with source_mode="direct" and runs
// the debate on exactly one item.
type InputMode = "run" | "upload" | "direct";

type StartRefineryBody = {
  run_id?: string;
  input_path?: string;
  direct?: {
    title: string;
    description?: string;
    priority?: string;
    tags?: string[];
  };
};

function InputPhase({
  onStart,
  onLoadResult,
  onResume,
  onDismissResumable,
  history,
  historyLoading,
  resumable,
  onDeleteHistory,
}: {
  onStart: (body: StartRefineryBody) => void;
  onLoadResult: (resultId: string) => void;
  onResume: (refineryRunId: string) => void;
  onDismissResumable: (refineryRunId: string) => void;
  history: RefineryHistoryItem[];
  historyLoading: boolean;
  resumable: ResumableRefineryRun[];
  onDeleteHistory: (resultId: string) => void;
}) {
  const [mode, setMode] = useState<InputMode>("run");

  return (
    <div>
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 8,
            marginBottom: 4,
          }}
        >
          <div className="card-title" style={{ margin: 0 }}>Requirements Refinery</div>
          <button
            className="btn btn-secondary"
            onClick={() => openAboutRequirementsRefinery()}
            title="Open the architectural whitepaper"
            style={{ fontSize: 12 }}
          >
            About
          </button>
        </div>
        <p style={{ color: "#8b949e", fontSize: 12, margin: "0 0 12px" }}>
          Transform requirements into a more detailed, structured, and actionable set.
        </p>
        <div
          role="tablist"
          aria-label="Input source"
          style={{ display: "flex", gap: 0, marginBottom: 16 }}
        >
          <button
            role="tab"
            aria-selected={mode === "run"}
            className={`tab-btn${mode === "run" ? " active" : ""}`}
            onClick={() => setMode("run")}
            style={{ borderRadius: "6px 0 0 6px", fontSize: 12, padding: "6px 16px" }}
          >
            From a Run
          </button>
          <button
            role="tab"
            aria-selected={mode === "upload"}
            className={`tab-btn${mode === "upload" ? " active" : ""}`}
            onClick={() => setMode("upload")}
            style={{ borderRadius: 0, fontSize: 12, padding: "6px 16px" }}
          >
            From Documents
          </button>
          <button
            role="tab"
            aria-selected={mode === "direct"}
            className={`tab-btn${mode === "direct" ? " active" : ""}`}
            onClick={() => setMode("direct")}
            style={{ borderRadius: "0 6px 6px 0", fontSize: 12, padding: "6px 16px" }}
          >
            From a Requirement
          </button>
        </div>
        {mode === "run" ? (
          <RunSelector onSelect={(runId) => onStart({ run_id: runId })} />
        ) : mode === "upload" ? (
          <DocumentUploader onUpload={(path) => onStart({ input_path: path })} />
        ) : (
          <DirectInput onSubmit={(direct) => onStart({ direct })} />
        )}
      </div>

      {/* Resumable runs — only render when there's something to resume */}
      {resumable.length > 0 && (
        <div className="card" style={{ marginTop: 12, borderColor: "#d29922" }}>
          <div className="card-title" style={{ fontSize: 13, color: "#d29922" }}>
            Resumable Runs <span style={{ fontWeight: 400, color: "#8b949e" }}>({resumable.length})</span>
          </div>
          <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 8px" }}>
            These runs didn't complete — resume to replay completed
            requirements from cache and re-run the rest.
          </p>
          <table className="table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Source</th>
                <th>Status</th>
                <th>Started</th>
                <th>Requirements</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {resumable.map((r) => (
                <tr key={r.refinery_run_id}>
                  <td><code style={{ fontSize: 10 }}>{r.refinery_run_id}</code></td>
                  <td style={{ color: "#8b949e" }}>{r.source_run_id || r.source_mode}</td>
                  <td>
                    <span
                      style={{
                        fontSize: 10,
                        padding: "1px 6px",
                        borderRadius: 3,
                        background: r.status === "in_progress" ? "#3a2a17" : "#3a1f1f",
                        color: r.status === "in_progress" ? "#f0b72f" : "#ff7b72",
                      }}
                    >
                      {r.status}
                    </span>
                  </td>
                  <td style={{ color: "#8b949e", fontSize: 10 }}>
                    {r.started_at ? new Date(r.started_at).toLocaleString() : "—"}
                  </td>
                  <td>{r.requirements_count}</td>
                  <td style={{ display: "flex", gap: 6 }}>
                    <button
                      className="btn"
                      style={{ fontSize: 10, padding: "2px 10px" }}
                      onClick={() => onResume(r.refinery_run_id)}
                    >
                      Resume
                    </button>
                    <button
                      className="btn"
                      style={{
                        fontSize: 10,
                        padding: "2px 10px",
                        background: "transparent",
                        color: "#8b949e",
                        borderColor: "#30363d",
                      }}
                      onClick={() => onDismissResumable(r.refinery_run_id)}
                      title="Hide this run from the resume list (keeps telemetry)"
                    >
                      Dismiss
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* History */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-title" style={{ fontSize: 13 }}>Recent Refinements</div>
        {historyLoading ? (
          <p style={{ color: "#8b949e", fontSize: 12 }}>Loading...</p>
        ) : history.length === 0 ? (
          <p style={{ color: "#8b949e", fontSize: 12 }}>No previous refinements.</p>
        ) : (
          <table className="table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Source</th>
                <th>Modified</th>
                <th>Relationships</th>
                <th>Duration</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.id}>
                  <td><code style={{ fontSize: 10 }}>{h.id}</code></td>
                  <td style={{ color: "#8b949e" }}>{h.source_run_id || h.source_mode}</td>
                  <td>{h.requirements_modified}/{h.requirements_count}</td>
                  <td>{h.relationships_count}</td>
                  <td style={{ color: "#8b949e" }}>{h.duration_seconds.toFixed(0)}s</td>
                  <td>
                    <div style={{ display: "flex", gap: 4 }}>
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: 10, padding: "2px 8px" }}
                        onClick={() => onLoadResult(h.id)}
                      >
                        View
                      </button>
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: 10, padding: "2px 8px", color: "#f85149" }}
                        onClick={() => onDeleteHistory(h.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function RunSelector({ onSelect }: { onSelect: (runId: string) => void }) {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    api.history(20).then((data) => {
      const r = (data as { runs: HistoryRun[] }).runs || [];
      setRuns(r);
      if (r.length > 0) setSelected(r[0].id);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <p style={{ color: "#8b949e", fontSize: 12 }}>Loading runs...</p>;
  if (runs.length === 0) return <p style={{ color: "#8b949e", fontSize: 12 }}>No historical runs found.</p>;

  return (
    <div>
      <table className="table" style={{ fontSize: 12, marginBottom: 12 }}>
        <thead>
          <tr><th></th><th>Run</th><th>Status</th><th>Pass Rate</th></tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr
              key={r.id}
              onClick={() => setSelected(r.id)}
              style={{ cursor: "pointer", background: selected === r.id ? "#161b22" : undefined }}
            >
              <td style={{ width: 24 }}>
                <input type="radio" checked={selected === r.id} onChange={() => setSelected(r.id)} />
              </td>
              <td><code>{r.id}</code></td>
              <td><span className={r.status === "success" ? "badge-success" : r.status === "error" ? "badge-error" : "badge-warn"}>{r.status}</span></td>
              <td>{r.pass_rate != null ? `${(r.pass_rate * 100).toFixed(0)}%` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <button className="btn" onClick={() => selected && onSelect(selected)} disabled={!selected}>
        Refine
      </button>
    </div>
  );
}

function DocumentUploader({ onUpload }: { onUpload: (path: string) => void }) {
  const [uploading, setUploading] = useState(false);
  const [files, setFiles] = useState<string[]>([]);
  const [path, setPath] = useState("");
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = async (fileList: FileList) => {
    setUploading(true);
    setError("");
    try {
      const resp = await api.uploadFiles(Array.from(fileList));
      setPath((resp as { path: string }).path);
      setFiles((resp as { files: string[] }).files || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div
        style={{
          border: "2px dashed #30363d", borderRadius: 8, padding: 32,
          textAlign: "center", color: "#8b949e", fontSize: 12,
          cursor: uploading ? "wait" : "pointer", marginBottom: 12,
          background: dragActive ? "#161b2280" : "transparent",
        }}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => { e.preventDefault(); setDragActive(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
        onClick={() => { if (!uploading) document.getElementById("refinery-file-input")?.click(); }}
      >
        {uploading ? "Uploading..." : "Drop requirement files here or click to browse"}
        <input
          id="refinery-file-input" type="file" multiple
          accept=".md,.txt,.json,.yaml,.yml,.docx,.xlsx,.pdf,.csv,.html"
          style={{ display: "none" }}
          onChange={(e) => e.target.files?.length && handleFiles(e.target.files)}
        />
      </div>
      {error && <p style={{ color: "#f85149", fontSize: 12 }}>{error}</p>}
      {files.length > 0 && (
        <div style={{ marginBottom: 12, fontSize: 11, color: "#8b949e" }}>
          {files.map((f) => <code key={f} style={{ display: "block", color: "#e6edf3" }}>{f}</code>)}
        </div>
      )}
      <button className="btn" onClick={() => path && onUpload(path)} disabled={!path || uploading}>
        Refine
      </button>
    </div>
  );
}

// DirectInput — user types one requirement directly.
function DirectInput({
  onSubmit,
}: {
  onSubmit: (direct: {
    title: string;
    description?: string;
    priority?: string;
    tags?: string[];
  }) => void;
}) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [priority, setPriority] = useState("medium");
  const [tagsText, setTagsText] = useState("");

  const trimmedTitle = title.trim();
  const disabled = trimmedTitle.length === 0;

  const handleSubmit = () => {
    if (disabled) return;
    const tags = tagsText
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    onSubmit({
      title: trimmedTitle,
      description: description.trim() || undefined,
      priority,
      tags: tags.length ? tags : undefined,
    });
  };

  return (
    <div>
      <div style={{ marginBottom: 10 }}>
        <label className="form-label" htmlFor="refinery-direct-title">
          Title <span style={{ color: "#f85149" }}>*</span>
        </label>
        <input
          id="refinery-direct-title"
          type="text"
          className="form-input"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Enforce 30-minute inactivity timeout on user sessions"
        />
      </div>
      <div style={{ marginBottom: 10 }}>
        <label className="form-label" htmlFor="refinery-direct-desc">Description</label>
        <textarea
          id="refinery-direct-desc"
          className="form-textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the requirement — constraints, edge cases, acceptance context..."
          rows={6}
        />
      </div>
      <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
        <div style={{ flex: 1 }}>
          <label className="form-label" htmlFor="refinery-direct-priority">Priority</label>
          <select
            id="refinery-direct-priority"
            className="form-select"
            value={priority}
            onChange={(e) => setPriority(e.target.value)}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <div style={{ flex: 2 }}>
          <label className="form-label" htmlFor="refinery-direct-tags">Tags (comma-separated)</label>
          <input
            id="refinery-direct-tags"
            type="text"
            className="form-input"
            value={tagsText}
            onChange={(e) => setTagsText(e.target.value)}
            placeholder="auth, security, performance"
          />
        </div>
      </div>
      <button
        className="btn"
        onClick={handleSubmit}
        disabled={disabled}
        style={{ opacity: disabled ? 0.5 : 1 }}
      >
        Refine
      </button>
      {disabled && (
        <p style={{ color: "#8b949e", fontSize: 11, margin: "8px 0 0" }}>
          Title is required to start the refinery.
        </p>
      )}
    </div>
  );
}

// ── Progress ─────────────────────────────────────────────────────────────────

// Steps drive themselves from the event stream so the server is free to
// add or rename gather steps without the UI silently dropping them. Each
// unique ``event.step`` becomes a row; the most recently seen step is
// live until a later step (or ``complete``) supersedes it.
function GatheringProgress({
  step, history,
}: {
  step: string;
  history: readonly string[];
}) {
  const seen = useMemo(() => {
    const out: string[] = [];
    const dedup = new Set<string>();
    for (const s of history) {
      if (s === "complete") continue;
      if (s && !dedup.has(s)) { dedup.add(s); out.push(s); }
    }
    return out;
  }, [history]);
  const isComplete = step === "complete";
  return (
    <div className="card" style={{ maxWidth: 480, margin: "40px auto", textAlign: "center" }}>
      <div className="card-title">Gathering data...</div>
      <div style={{ textAlign: "left", display: "inline-block" }}>
        {seen.length === 0 ? (
          <div style={{ fontSize: 12, color: "#8b949e" }}>
            \u2026 Waiting for first event
          </div>
        ) : (
          seen.map((s, i) => {
            const isLast = i === seen.length - 1;
            const done = isComplete || !isLast;
            return (
              <div
                key={s}
                style={{ fontSize: 12, color: done ? "#3fb950" : "#8b949e", marginBottom: 4 }}
              >
                {done ? "\u2713" : "\u2026"} {s}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ── Agent event timeline ─────────────────────────────────────────────────────
//
// Per-agent events streamed by the debate subgraph (generator → critic
// fan-out → synthesize → score → research/escalate → reconcile → finalize).
// Each row is one SSE event; the badge on the left identifies the agent.

type TimelineEvent = RefinerySSEEvent & { _ts: number };

const ROLE_COLOR: Record<string, { bg: string; fg: string }> = {
  product:     { bg: "#1f2d44", fg: "#79c0ff" },
  engineering: { bg: "#1b2e1e", fg: "#56d364" },
  security:    { bg: "#3a1f1f", fg: "#ff7b72" },
  operations:  { bg: "#3a2a17", fg: "#f0b72f" },
  cost:        { bg: "#3a351a", fg: "#e3b341" },
  judge:       { bg: "#2d2144", fg: "#bc8cff" },
  research:    { bg: "#13333a", fg: "#39c5cf" },
  rules:       { bg: "#22272e", fg: "#8b949e" },
  system:      { bg: "#22272e", fg: "#8b949e" },
};

function agentForEvent(ev: TimelineEvent): string {
  // refinery_llm_* events carry the actual role in ``role`` (set by
  // the helper from the agent kwarg), so prefer that.
  if (ev.role) return ev.role;
  const d = ev.debate_event;
  if (d) {
    if (d === "generator_started" || d === "draft_ready") return "product";
    if (d === "research_started" || d === "research_ready") return "research";
    if (
      d === "synthesize_started" || d === "synthesis_ready" ||
      d === "score_started" || d === "score_ready" ||
      d === "reconcile_started" || d === "reconcile_ready" ||
      d === "finalized"
    ) return "judge";
  }
  return "system";
}

function eventLabel(ev: TimelineEvent): string {
  // Single-shot LLM call events ride the same stream as the
  // node-lifecycle events. Payloads carry only agent/model/provider/
  // tokens — prompt and response content live in the forensic trace,
  // not the live stream.
  if (ev.event === "refinery_llm_started") {
    const provider = ev.provider ? ` via ${ev.provider}` : "";
    const effort = ev.reasoning_effort ? ` (${ev.reasoning_effort})` : "";
    return `→ ${ev.model || "?"}${provider}${effort}`;
  }
  if (ev.event === "refinery_llm_ready") {
    const latency = ev.latency_ms != null ? `${(ev.latency_ms / 1000).toFixed(1)}s` : "?";
    const tokens = `${ev.tokens_in ?? 0}/${ev.tokens_out ?? 0} tok`;
    return `← ${ev.model || "?"} ${latency}, ${tokens}`;
  }
  if (ev.event === "refinery_llm_failed") {
    const latency = ev.latency_ms != null ? ` after ${(ev.latency_ms / 1000).toFixed(1)}s` : "";
    return `LLM call failed${latency} — ${truncate(ev.error, 160)}`;
  }
  const d = ev.debate_event;
  if (!d) return ev.message || ev.step || "event";
  switch (d) {
    case "generator_started":  return "Drafting initial requirement";
    case "draft_ready":        return "Draft ready";
    case "critic_started":     return "Critiquing";
    case "critic_ready":       return `Critique: ${ev.severity || "?"} / ${ev.dimension || "?"}`;
    case "critic_placeholder": return `Skipped (${ev.reason || "not registered"})`;
    case "synthesize_started": return "Synthesizing rebuttal";
    case "synthesis_ready":    return `Rebuttal ready (${ev.rebuttals_accepted ?? 0} accepted)`;
    case "score_started":      return "Scoring draft";
    case "score_ready":        return `Score ${ev.overall?.toFixed(2) ?? "?"} ${ev.passed ? "✓ passed" : "✗ failed"}`;
    case "research_started":   return "Gathering external sources";
    case "research_ready":     return `${ev.insights ?? 0} validated insights`;
    case "escalation":         return `Escalated to ${ev.new_tier || "strong"} tier (level ${ev.escalation_level ?? "?"})`;
    case "reconcile_started":  return "Reconciling unresolved tensions";
    case "reconcile_ready":    return `Reconciled (${ev.unresolved_count ?? 0} unresolved)`;
    case "finalized":          return `Finalized: ${ev.convergence_status} after ${ev.rounds ?? 0} round(s)`;
  }
  return d;
}

// Heartbeat — render-only "last event Ns ago" indicator. The timer ticks
// every second; we don't need to be more precise than that. Without this,
// a stalled SSE looks identical to "thinking".
function useEventHeartbeat(lastTs: number | null): string | null {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (lastTs == null) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, [lastTs]);
  if (lastTs == null) return null;
  void tick; // referenced only to invalidate the closure on each tick
  const seconds = Math.floor((Date.now() - lastTs) / 1000);
  if (seconds < 2) return "live";
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.floor(seconds / 60)}m ago`;
}

function AgentEventTimeline({
  events, heading, fallback,
}: { events: TimelineEvent[]; heading: string; fallback: string }) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
    }
  }, [events.length]);

  const lastTs = events.length > 0 ? events[events.length - 1]._ts : null;
  const heartbeat = useEventHeartbeat(lastTs);
  const stalled = heartbeat != null && heartbeat !== "live"
    && !heartbeat.endsWith("s ago") ? true
    : heartbeat != null && /^\d+s ago$/.test(heartbeat)
      && parseInt(heartbeat, 10) >= 30;

  if (events.length === 0) {
    return (
      <div className="card" style={{ maxWidth: 560, margin: "40px auto", textAlign: "center" }}>
        <div className="card-title">{heading}</div>
        <div style={{ fontSize: 12, color: "#8b949e" }}>{fallback}</div>
      </div>
    );
  }

  return (
    <div className="card" style={{ maxWidth: 820, margin: "20px auto" }}>
      <div
        style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "baseline", marginBottom: 8,
        }}
      >
        <div className="card-title" style={{ margin: 0 }}>{heading}</div>
        {heartbeat && (
          <span
            style={{
              fontSize: 10,
              color: stalled ? "#d29922" : "#3fb950",
              fontWeight: 500,
            }}
            title="Time since last SSE event"
            aria-live="off"
          >
            {stalled ? "⚠ stalled" : "●"} last event {heartbeat}
          </span>
        )}
      </div>
      <div
        ref={scrollerRef}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-label={heading}
        style={{
          maxHeight: 520,
          overflowY: "auto",
          fontSize: 11,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          border: "1px solid #21262d",
          borderRadius: 4,
          padding: 0,
        }}
      >
        {events.map((ev, i) => {
          const agent = agentForEvent(ev);
          const color = ROLE_COLOR[agent] || ROLE_COLOR.system;
          const label = eventLabel(ev);
          const ts = new Date(ev._ts).toLocaleTimeString([], {
            hour: "2-digit", minute: "2-digit", second: "2-digit",
          });
          const ended = ev.debate_event?.endsWith("_ready")
            || ev.debate_event === "finalized"
            || ev.debate_event === "critic_placeholder"
            || ev.event === "refinery_llm_ready"
            || ev.event === "refinery_llm_failed";
          return (
            <div
              key={i}
              style={{
                display: "grid",
                gridTemplateColumns: "72px 100px 44px 1fr 140px",
                gap: 8, padding: "4px 10px",
                borderBottom: "1px solid #161b22",
                alignItems: "center",
                background: i % 2 === 0 ? "transparent" : "#0d1117",
              }}
            >
              <span style={{ color: "#484f58", fontSize: 10 }}>{ts}</span>
              <span
                style={{
                  background: color.bg, color: color.fg,
                  padding: "1px 6px", borderRadius: 3,
                  fontSize: 10, textTransform: "uppercase",
                  letterSpacing: 0.3, textAlign: "center",
                }}
                title={`agent: ${agent}`}
              >
                {agent}
              </span>
              <span style={{ color: "#8b949e", fontSize: 10, textAlign: "right" }}>
                {typeof ev.round === "number" ? `r${ev.round}` : ""}
              </span>
              <span style={{ color: ended ? "#e6edf3" : "#8b949e" }}>{label}</span>
              <span style={{ color: "#484f58", fontSize: 10, textAlign: "right" }}>
                {ev.requirement_id || ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Convergence score (0..1) ─────────────────────────────────────────────────

// Operators read this off the requirement card to decide how much more detail
// a requirement needs. 0 means no debate happened (carry-forward); 1 means the
// panel converged; values in between mean debated but the Judge's threshold
// was not cleared — closer to 1 = closer to convergence.

function convergenceColor(score: number): string {
  if (score <= 0)   return "#484f58";       // gray  — no debate
  if (score >= 0.8) return "#3fb950";       // green — converged
  if (score >= 0.6) return "#7ee2ec";       // cyan  — almost converged
  if (score >= 0.4) return "#d29922";       // amber — partial converge
  return "#f85149";                          // red   — low converge
}

function convergenceLabel(score: number): string {
  if (score <= 0)   return "No debate";
  if (score >= 1.0) return "Converged · 100%";
  if (score >= 0.8) return `Near converged · ${Math.round(score * 100)}%`;
  if (score >= 0.6) return `Almost converged · ${Math.round(score * 100)}%`;
  if (score >= 0.4) return `Partial · ${Math.round(score * 100)}%`;
  return `Low · ${Math.round(score * 100)}%`;
}

function ConvergenceBar({
  score, compact = false, title,
}: { score: number; compact?: boolean; title?: string }) {
  const safe = Math.max(0, Math.min(1, score));
  const pct = Math.round(safe * 100);
  const color = convergenceColor(safe);
  const tooltip = title ?? (
    `Convergence ${pct}% — 0% no debate, 100% converged, 80-99% near, 60-79% almost, 40-59% partial, <40% low.`
  );
  if (compact) {
    return (
      <span
        title={tooltip}
        aria-label={tooltip}
        style={{
          display: "inline-flex", alignItems: "center", gap: 4,
          fontSize: 9, color, fontWeight: 600, letterSpacing: 0.3,
        }}
      >
        <span
          aria-hidden
          style={{
            width: 28, height: 4, borderRadius: 2,
            background: "#21262d", overflow: "hidden",
          }}
        >
          <span style={{ display: "block", width: `${pct}%`, height: "100%", background: color }} />
        </span>
        {pct}%
      </span>
    );
  }
  return (
    <div title={tooltip} style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 200 }}>
      <span style={{ fontSize: 11, color: "#8b949e", fontWeight: 500 }}>Convergence</span>
      <span style={{
        flex: 1, height: 6, borderRadius: 3, background: "#21262d", overflow: "hidden",
      }}>
        <span style={{ display: "block", width: `${pct}%`, height: "100%", background: color }} />
      </span>
      <span style={{ fontSize: 11, color, fontWeight: 600, minWidth: 90, textAlign: "right" }}>
        {convergenceLabel(safe)}
      </span>
    </div>
  );
}

// ── Results: Requirement Sidebar Item ─────────────────────────────────────────

// A requirement is "unresolved" when the panel didn't converge, the generator
// crashed, or the converged draft still carries open questions the operator
// must answer. The list filter chip groups all three.
function isUnresolved(req: RefinedRequirement): boolean {
  return (
    req.convergence_status === "short_circuited" ||
    req.convergence_status === "aborted" ||
    (req.open_questions?.length || 0) > 0
  );
}

function ReqListItem({
  req, selected, applied, dismissed, onClick,
}: {
  req: RefinedRequirement; selected: boolean; applied: boolean; dismissed: boolean; onClick: () => void;
}) {
  const changed = req.changes.length > 0;
  const shortCircuited = req.convergence_status === "short_circuited";
  const aborted = req.convergence_status === "aborted";
  const hasOpenQs = (req.open_questions?.length || 0) > 0
    && !shortCircuited && !aborted;
  // Solid row tint for short_circuited / aborted so an operator scanning a
  // 20-row list spots them at a glance — the prior 9px text label was too
  // quiet against the 3px left border.
  const tint = shortCircuited
    ? "#f8514912" : aborted
    ? "#d2992212" : selected
    ? "#161b22" : "transparent";
  return (
    <div
      onClick={onClick}
      style={{
        padding: "8px 12px", cursor: "pointer", borderLeft: "3px solid",
        borderLeftColor: selected ? "#58a6ff" : shortCircuited ? "#f85149" : aborted ? "#d29922" : "transparent",
        background: dismissed ? "#0d1117" : tint,
        opacity: dismissed ? 0.4 : 1,
        borderBottom: "1px solid #21262d",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2, flexWrap: "wrap" }}>
        {/* Convergence chip first — load-bearing signal goes ahead of priority. */}
        {shortCircuited && (
          <span
            className="badge-error"
            style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.4 }}
            title="Panel did not converge within max_rounds"
            aria-label="Not converged — panel did not reach synthesis within the round budget"
          >
            ⚠ NOT CONVERGED
          </span>
        )}
        {aborted && (
          <span
            className="badge-warn"
            style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.4 }}
            title="Generator failed"
            aria-label="Aborted — the panel's generator crashed before producing a draft"
          >
            ⚠ ABORTED
          </span>
        )}
        {hasOpenQs && (
          <span
            className="badge-warn"
            style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.4 }}
            title="Converged but the panel left open questions"
            aria-label="Open questions — converged draft has unresolved questions"
          >
            ⚠ OPEN QUESTIONS
          </span>
        )}
        <span className={priorityBadge(req.priority)} style={{ fontSize: 10 }}>{req.priority}</span>
        {changed && !applied && !dismissed && !shortCircuited && !aborted && (
          <span style={{ fontSize: 9, color: "#58a6ff", fontWeight: 600 }}>MODIFIED</span>
        )}
        {applied && <span style={{ fontSize: 9, color: "#3fb950", fontWeight: 600 }}>APPLIED</span>}
        {dismissed && <span style={{ fontSize: 9, color: "#8b949e", fontWeight: 600 }}>DISMISSED</span>}
      </div>
      <div style={{ fontSize: 12, color: "#e6edf3", fontWeight: 500 }}>{req.title}</div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 2 }}>
        <code style={{ fontSize: 10, color: "#484f58" }}>{req.id}</code>
        <ConvergenceBar score={req.convergence_score ?? 0} compact />
      </div>
    </div>
  );
}

// ── Results: Requirement Detail Panel ─────────────────────────────────────────

function ReqDetailPanel({
  req, applied, dismissed, suggestedMemories, onApply, onDismiss, onEdit,
}: {
  req: RefinedRequirement; applied: boolean; dismissed: boolean;
  suggestedMemories: readonly SuggestedMemory[];
  onApply: (applyMemories: SuggestedMemory[]) => void;
  onDismiss: () => void; onEdit: () => void;
}) {
  const [showDiff, setShowDiff] = useState(false);
  const [showSpecs, setShowSpecs] = useState(false);
  const changed = req.changes.length > 0;
  const descChanged = req.original_description !== req.description;
  const shortCircuited = req.convergence_status === "short_circuited";
  const unresolvedPoints = req.unresolved_points || [];
  const openQuestions = req.open_questions || [];
  const explicitTradeoffs = req.explicit_tradeoffs || [];

  // Memories scoped to THIS requirement (set by backend producers via
  // ``source_requirement_id``). Auto-save default-on for ``validated`` /
  // ``produced``; default-off for ``unvalidated`` so the operator
  // explicitly opts in to suggestions the Judge didn't actually use.
  const memoriesForReq = useMemo(
    () => suggestedMemories.filter(
      (m) => (m.source_requirement_id || "") === req.id,
    ),
    [suggestedMemories, req.id],
  );
  const [memSelection, setMemSelection] = useState<Record<number, boolean>>({});
  // Reset selection whenever we switch requirements.
  useEffect(() => {
    const next: Record<number, boolean> = {};
    memoriesForReq.forEach((m, i) => {
      const status = (m.validation_status || "").toLowerCase();
      next[i] = status === "validated" || status === "produced";
    });
    setMemSelection(next);
  }, [req.id, memoriesForReq]);
  const checkedCount = useMemo(
    () => Object.values(memSelection).filter(Boolean).length,
    [memSelection],
  );

  const handleApply = () => {
    const checked = memoriesForReq.filter((_, i) => memSelection[i]);
    onApply(checked);
  };

  return (
    <div>
      {/* Short-circuit banner — impossible to miss. Renders before the
          title block so operators see it first. */}
      {shortCircuited && (
        <div
          className="card"
          style={{
            borderColor: "#f85149",
            background: "#f8514911",
            marginBottom: 12,
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 700, color: "#f85149", marginBottom: 4 }}>
            ⚠ PANEL DID NOT CONVERGE
          </div>
          <div style={{ fontSize: 11, color: "#e6edf3", lineHeight: 1.5 }}>
            The adversarial debate hit its max-rounds limit without reaching the
            score threshold. The draft below is the panel's best short-circuit
            consolidation — it documents what couldn't be resolved rather than
            forcing a synthesis that doesn't exist. Review the Unresolved
            section below before applying.
          </div>
        </div>
      )}

      {/* Unresolved / open-questions section — only shown when non-empty. */}
      {(unresolvedPoints.length > 0 || openQuestions.length > 0) && (
        <div
          className="card"
          style={{ borderColor: "#f8514940", marginBottom: 12 }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: "#f85149", marginBottom: 6 }}>
            Unresolved
          </div>
          {unresolvedPoints.length > 0 && (
            <>
              <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4, fontWeight: 500 }}>
                Points that couldn't be resolved
              </div>
              <ul style={{ fontSize: 12, color: "#e6edf3", margin: "0 0 10px", paddingLeft: 16 }}>
                {unresolvedPoints.map((p, i) => <li key={`u-${i}`}>{p}</li>)}
              </ul>
            </>
          )}
          {openQuestions.length > 0 && (
            <>
              <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4, fontWeight: 500 }}>
                Open questions
              </div>
              <ul style={{ fontSize: 12, color: "#e6edf3", margin: 0, paddingLeft: 16 }}>
                {openQuestions.map((q, i) => <li key={`q-${i}`}>{q}</li>)}
              </ul>
            </>
          )}
        </div>
      )}

      {/* Explicit tradeoffs surface prominently for converged debates
          where the Judge made a value-judgment between conflicting
          critiques. */}
      {explicitTradeoffs.length > 0 && (
        <div
          className="card"
          style={{ borderColor: "#d2992240", marginBottom: 12 }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: "#d29922", marginBottom: 6 }}>
            Explicit tradeoffs
          </div>
          <ul style={{ fontSize: 12, color: "#e6edf3", margin: 0, paddingLeft: 16 }}>
            {explicitTradeoffs.map((t, i) => <li key={`t-${i}`}>{t}</li>)}
          </ul>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, color: "#e6edf3" }}>
            {req.original_title !== req.title ? (
              <>{req.title} <span style={{ fontSize: 12, color: "#8b949e", fontWeight: 400 }}>(was: {req.original_title})</span></>
            ) : req.title}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
            <code style={{ fontSize: 11, color: "#8b949e" }}>{req.id}</code>
            <span className={priorityBadge(req.priority)}>{req.priority}</span>
            {req.tags.map((t) => <span key={t} style={{ fontSize: 10, padding: "1px 6px", background: "#21262d", borderRadius: 10, color: "#8b949e" }}>{t}</span>)}
          </div>
          <div style={{ marginTop: 8 }}>
            <ConvergenceBar score={req.convergence_score ?? 0} />
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
          {changed && !applied && !dismissed && (
            <button className="btn" style={{ fontSize: 11, padding: "4px 12px" }} onClick={handleApply}>
              Apply to Graph
              {checkedCount > 0 && (
                <span style={{ marginLeft: 4, opacity: 0.85 }}>+ {checkedCount} memor{checkedCount === 1 ? "y" : "ies"}</span>
              )}
            </button>
          )}
          {!applied && !dismissed && (
            <button className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 12px" }} onClick={onEdit}>Edit</button>
          )}
          {!dismissed && !applied && (
            <button className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 12px" }} onClick={onDismiss}>Dismiss</button>
          )}
          {applied && <span style={{ color: "#3fb950", fontSize: 12, fontWeight: 600, padding: "4px 0" }}>Applied</span>}
          {dismissed && <span style={{ color: "#8b949e", fontSize: 12, fontStyle: "italic", padding: "4px 0" }}>Dismissed</span>}
        </div>
      </div>

      {/* Auto-save memory checklist — surfaces just below the header so the
          operator reviews/unselects before clicking Apply. Only renders when
          the backend stamped a producer-emitted memory with this req's id;
          legacy LLM-emitted memories without source_requirement_id stay in
          the Memories tab and are saved per-card. */}
      {memoriesForReq.length > 0 && !applied && !dismissed && (
        <div className="card" style={{ borderColor: "#3fb95040", marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#3fb950", marginBottom: 6 }}>
            Memories to save with Apply ({checkedCount} of {memoriesForReq.length} selected)
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 8 }}>
            Memories the panel produced for this requirement. Validated and produced
            entries are checked by default; unvalidated entries default off.
          </div>
          {memoriesForReq.map((mem, i) => {
            const kind = memoryKindOf(mem);
            const palette = MEMORY_KIND_COLOR[kind];
            const status = (mem.validation_status || "unvalidated").toLowerCase();
            const id = `apply-mem-${req.id}-${i}`;
            return (
              <label
                key={i}
                htmlFor={id}
                style={{
                  display: "flex", gap: 8, alignItems: "flex-start",
                  padding: "4px 0", cursor: "pointer", fontSize: 12,
                }}
              >
                <input
                  id={id}
                  type="checkbox"
                  checked={!!memSelection[i]}
                  onChange={(e) => setMemSelection((s) => ({ ...s, [i]: e.target.checked }))}
                  style={{ marginTop: 3 }}
                />
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                    <span
                      style={{
                        background: palette.bg, color: palette.fg,
                        padding: "0 6px", borderRadius: 3,
                        fontSize: 9, fontWeight: 600, letterSpacing: 0.4,
                        textTransform: "uppercase",
                      }}
                    >
                      {MEMORY_KIND_LABEL[kind]}
                    </span>
                    {status !== "unvalidated" && (
                      <span
                        style={{
                          fontSize: 10,
                          color: status === "validated" ? "#3fb950" : "#79c0ff",
                          fontWeight: 600,
                        }}
                      >
                        {status}
                      </span>
                    )}
                  </div>
                  <div style={{ color: "#e6edf3", marginTop: 2 }}>
                    {mem.summary || mem.description}
                  </div>
                </div>
              </label>
            );
          })}
        </div>
      )}


      {req.changes.length > 0 && (
        <div className="card" style={{ borderColor: "#58a6ff40" }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#58a6ff", marginBottom: 6 }}>What changed</div>
          <ul style={{ fontSize: 12, color: "#8b949e", margin: 0, paddingLeft: 16 }}>
            {req.changes.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
          {req.pass_context && <div style={{ fontSize: 11, color: "#484f58", marginTop: 6, fontStyle: "italic" }}>{req.pass_context}</div>}
        </div>
      )}


      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3" }}>Description</div>
          {descChanged && (
            <button onClick={() => setShowDiff(!showDiff)} style={{ background: "none", border: "none", color: "#58a6ff", fontSize: 11, cursor: "pointer", padding: 0 }}>
              {showDiff ? "Hide original" : "Compare with original"}
            </button>
          )}
        </div>
        {showDiff && descChanged ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={{ background: "#1c1c1c", padding: 10, borderRadius: 6, borderLeft: "3px solid #f85149" }}>
              <div style={{ color: "#f85149", marginBottom: 6, fontWeight: 600, fontSize: 11 }}>Original</div>
              <div style={{ color: "#8b949e", fontSize: 12, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{req.original_description}</div>
            </div>
            <div style={{ background: "#1c1c1c", padding: 10, borderRadius: 6, borderLeft: "3px solid #3fb950" }}>
              <div style={{ color: "#3fb950", marginBottom: 6, fontWeight: 600, fontSize: 11 }}>Refined</div>
              <div style={{ color: "#e6edf3", fontSize: 12, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{req.description}</div>
            </div>
          </div>
        ) : (
          <div style={{ color: "#e6edf3", fontSize: 12, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{req.description}</div>
        )}
      </div>

      {/* Debate ledger — collapsed disclosure under Description so operators
          can drill into "why did Security and Cost disagree on round 2?"
          after the SSE timeline is gone. Renders only when the runner
          stamped a debate summary on this requirement. */}
      <DebateDisclosure debate={req.debate} />

      {req.relationships.length > 0 && (
        <div className="card">
          <div style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3", marginBottom: 8 }}>Relationships</div>
          <table className="table" style={{ fontSize: 12 }}>
            <thead><tr><th>Target</th><th>Type</th><th>Rationale</th></tr></thead>
            <tbody>
              {req.relationships.map((rel, i) => (
                <tr key={i}>
                  <td><code>{rel.target_id}</code></td>
                  <td><span style={{ color: severityColor(rel.type) }}>{rel.type}</span></td>
                  <td style={{ color: "#8b949e" }}>{rel.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}


      {req.suggested_specs.length > 0 && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3" }}>
              Suggested Specs ({req.suggested_specs.length})
            </div>
            <button onClick={() => setShowSpecs(!showSpecs)} style={{ background: "none", border: "none", color: "#58a6ff", fontSize: 11, cursor: "pointer", padding: 0 }}>
              {showSpecs ? "Collapse" : "Expand"}
            </button>
          </div>
          {showSpecs ? req.suggested_specs.map((s, i) => (
            <div key={i} style={{ background: "#0d1117", border: "1px solid #21262d", borderRadius: 6, padding: 10, marginBottom: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 12 }}>{s.title} <code style={{ color: "#8b949e", fontWeight: 400 }}>{s.capability}</code></div>
              <div style={{ color: "#8b949e", fontSize: 12, marginTop: 4 }}>{s.description}</div>
              {s.acceptance_criteria.length > 0 && (
                <ul style={{ margin: "6px 0 0", paddingLeft: 16, color: "#8b949e", fontSize: 11 }}>
                  {s.acceptance_criteria.map((c, j) => <li key={j}>{c}</li>)}
                </ul>
              )}
            </div>
          )) : (
            <div style={{ fontSize: 11, color: "#8b949e" }}>
              {req.suggested_specs.map((s) => s.title).join(" · ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Results: Debate Disclosure ───────────────────────────────────────────────
//
// Per-debate summary: rounds, critiques, rebuttals, scores. Renders as a
// collapsed disclosure under Description. The data comes from the runner's
// ``final_trace`` projected onto ``RefinedRequirement.debate``; the SSE
// timeline that played live is gone by the time the operator gets here.

function DebateDisclosure({
  debate,
}: {
  debate: RefinedRequirement["debate"];
}) {
  const [open, setOpen] = useState(false);
  if (!debate) return null;

  const rounds = debate.rounds_executed ?? 0;
  const escalation = debate.escalation_level ?? 0;
  const research = debate.research_calls_used ?? 0;
  const roundKeys = Object.keys(debate.scores_by_round || {})
    .map((k) => parseInt(k, 10))
    .filter((n) => !Number.isNaN(n))
    .sort((a, b) => a - b);

  return (
    <div className="card">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        style={{
          width: "100%",
          background: "none",
          border: "none",
          padding: 0,
          textAlign: "left",
          color: "#e6edf3",
          fontSize: 12,
          fontWeight: 600,
          cursor: "pointer",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>
          Debate ({rounds} round{rounds === 1 ? "" : "s"})
          {escalation > 0 && (
            <span style={{ marginLeft: 8, fontSize: 10, color: "#d29922", fontWeight: 500 }}>
              · escalated to strong tier (×{escalation})
            </span>
          )}
          {research > 0 && (
            <span style={{ marginLeft: 8, fontSize: 10, color: "#39c5cf", fontWeight: 500 }}>
              · {research} research call{research === 1 ? "" : "s"}
            </span>
          )}
        </span>
        <span style={{ color: "#58a6ff", fontSize: 11, fontWeight: 400 }}>
          {open ? "Collapse" : "Expand"}
        </span>
      </button>

      {open && (
        <div style={{ marginTop: 12 }}>
          {roundKeys.length === 0 ? (
            <div style={{ fontSize: 11, color: "#8b949e" }}>
              No round detail recorded.
            </div>
          ) : (
            roundKeys.map((rn) => (
              <DebateRound
                key={rn}
                roundNumber={rn}
                critiques={debate.critiques_by_round?.[String(rn)] || []}
                rebuttal={debate.rebuttals_by_round?.[String(rn)]}
                score={debate.scores_by_round?.[String(rn)]}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

function DebateRound({
  roundNumber, critiques, rebuttal, score,
}: {
  roundNumber: number;
  critiques: NonNullable<NonNullable<RefinedRequirement["debate"]>["critiques_by_round"]>[string];
  rebuttal: NonNullable<NonNullable<RefinedRequirement["debate"]>["rebuttals_by_round"]>[string] | undefined;
  score: NonNullable<NonNullable<RefinedRequirement["debate"]>["scores_by_round"]>[string] | undefined;
}) {
  const overall = score?.overall;
  const passed = score?.passed;

  return (
    <div
      style={{
        marginBottom: 10, padding: 8,
        background: "#0d1117",
        border: "1px solid #21262d",
        borderRadius: 4,
      }}
    >
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: "#8b949e", fontWeight: 600 }}>
          Round {roundNumber}
        </span>
        {overall != null && (
          <span style={{ fontSize: 11, color: passed ? "#3fb950" : "#d29922" }}>
            score {overall.toFixed(2)} {passed ? "✓" : "✗"}
          </span>
        )}
        {rebuttal && (
          <span style={{ fontSize: 11, color: "#8b949e" }}>
            rebuttal: {rebuttal.accepted_count} accepted, {rebuttal.rejected_count} rejected
          </span>
        )}
      </div>
      {critiques.length > 0 && (
        <div style={{ display: "grid", gap: 4 }}>
          {critiques.map((c, i) => {
            const role = c.role || "?";
            const palette = ROLE_COLOR[role] || ROLE_COLOR.system;
            const sevColor = c.severity === "blocker" ? "#f85149"
              : c.severity === "warning" ? "#d29922" : "#8b949e";
            return (
              <div key={i} style={{ display: "flex", gap: 8, fontSize: 11 }}>
                <span
                  style={{
                    background: palette.bg, color: palette.fg,
                    padding: "0 6px", borderRadius: 3,
                    fontSize: 9, fontWeight: 600, letterSpacing: 0.4,
                    textTransform: "uppercase", flexShrink: 0,
                    alignSelf: "flex-start",
                  }}
                >
                  {role}
                </span>
                <span style={{ color: sevColor, fontWeight: 600, flexShrink: 0 }}>
                  [{c.severity || "?"}]
                </span>
                <span style={{ color: "#8b949e", flexShrink: 0 }}>
                  {c.dimension || "?"}
                </span>
                <span style={{ color: "#e6edf3" }}>{c.finding}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Results: Inline Editor ───────────────────────────────────────────────────

function InlineEditor({
  req, onSave, onCancel,
}: {
  req: RefinedRequirement;
  onSave: (data: Record<string, unknown>) => void;
  onCancel: () => void;
}) {
  const [title, setTitle] = useState(req.title);
  const [description, setDescription] = useState(req.description);
  const [priority, setPriority] = useState(req.priority);
  const [tags, setTags] = useState(req.tags.join(", "));
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    const parsedTags = tags.split(",").map((t) => t.trim()).filter(Boolean);
    try {
      await api.patchRequirement(req.id, { title, description, priority, tags: parsedTags });
      onSave({ title, description, priority, tags: parsedTags });
    } catch { /* stay open on error */ } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <div style={{ fontSize: 14, fontWeight: 600, color: "#e6edf3", marginBottom: 12 }}>
        Editing: <code style={{ fontWeight: 400 }}>{req.id}</code>
      </div>
      <div style={{ display: "grid", gap: 12 }}>
        <label style={{ fontSize: 12, color: "#8b949e" }}>
          Title
          <input
            className="form-input"
            style={{ marginTop: 4 }}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>
        <label style={{ fontSize: 12, color: "#8b949e" }}>
          Description
          <textarea
            className="form-textarea"
            style={{ marginTop: 4 }}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={8}
          />
        </label>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <label style={{ fontSize: 12, color: "#8b949e" }}>
            Priority
            <select
              className="form-select"
              style={{ marginTop: 4 }}
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
            >
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
              <option value="critical">critical</option>
            </select>
          </label>
          <label style={{ fontSize: 12, color: "#8b949e" }}>
            Tags (comma-separated)
            <input
              className="form-input"
              style={{ marginTop: 4 }}
              value={tags}
              onChange={(e) => setTags(e.target.value)}
            />
          </label>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" onClick={handleSave} disabled={saving}>{saving ? "Saving..." : "Save"}</button>
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

// ── Results: Memory Card ─────────────────────────────────────────────────────

function _slugifyMemoryId(mem: SuggestedMemory): string {
  // Stable surrogate id for active-learning telemetry — the backend
  // doesn't yet hand out memory ids for unsaved suggestions, so we
  // hash the (type + description) into a short, URL-safe slug.
  const seed = `${mem.type}|${mem.description}`;
  let h = 5381;
  for (let i = 0; i < seed.length; i++) h = ((h << 5) + h + seed.charCodeAt(i)) & 0xffffffff;
  const slug = seed.slice(0, 24).replace(/[^A-Za-z0-9_-]/g, "_");
  return `${slug}-${(h >>> 0).toString(36)}`;
}

function _signalFeedback(
  mem: SuggestedMemory,
  decision: "accepted" | "dismissed" | "edited",
): void {
  // Best-effort active-learning signal — silent on failure so a
  // backend hiccup never blocks the operator's UI action.
  api
    .submitMemoryFeedback(_slugifyMemoryId(mem), {
      decision,
      memory_kind: mem.type,
      source_role: mem.source_feature,
    })
    .catch(() => undefined);
}

function _signalRequirementDecision(
  resultId: string | null,
  req: RefinedRequirement,
  decision: "accepted" | "dismissed" | "edited",
): void {
  // Best-effort calibration signal. No-op when the run has no
  // result_id (legacy view) so the operator action proceeds normally.
  if (!resultId) return;
  api
    .submitRequirementDecision(resultId, req.id, {
      decision,
      convergence_status: req.convergence_status ?? null,
    })
    .catch(() => undefined);
}

function MemoryCard({ mem, onSave, onDismiss }: { mem: SuggestedMemory; onSave: () => void; onDismiss: () => void }) {
  const [status, setStatus] = useState<"idle" | "checking" | "saved" | "duplicate">("idle");
  const [dupInfo, setDupInfo] = useState<{ description: string; similarity: number | null } | null>(null);

  const kind = memoryKindOf(mem);
  const palette = MEMORY_KIND_COLOR[kind];
  const headline = mem.summary || mem.description;

  const handleSave = async () => {
    setStatus("checking");
    try {
      const check = await api.checkMemoryDuplicate({
        type: mem.type,
        description: mem.description,
        context: mem.context || mem.applicability,
      });
      if (check.is_duplicate) {
        setDupInfo({
          description: check.existing_description || "",
          similarity: check.similarity,
        });
        setStatus("duplicate");
        return;
      }
      await api.importMemories([{ type: mem.type, description: mem.description, context: mem.context, applicability: mem.applicability, source_feature: mem.source_feature }]);
      setStatus("saved");
      _signalFeedback(mem, "accepted");
      onSave();
    } catch {
      setStatus("idle");
    }
  };

  const handleDismiss = () => {
    _signalFeedback(mem, "dismissed");
    onDismiss();
  };

  const borderColor = status === "saved" ? "#3fb95040"
    : status === "duplicate" ? "#d2992240"
    : palette.border;

  return (
    <div className="card" style={{ borderColor }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
            <span
              style={{
                background: palette.bg, color: palette.fg,
                padding: "1px 8px", borderRadius: 3,
                fontSize: 10, fontWeight: 600, letterSpacing: 0.4,
                textTransform: "uppercase",
              }}
              title={`Memory kind: ${MEMORY_KIND_LABEL[kind]}`}
            >
              {MEMORY_KIND_LABEL[kind]}
            </span>
            {mem.source_role && (
              <span style={{ fontSize: 10, color: "#8b949e" }}>
                from <strong>{mem.source_role}</strong>
              </span>
            )}
            {mem.validation_status && mem.validation_status !== "unvalidated" && (
              <span
                style={{
                  fontSize: 10,
                  color: mem.validation_status === "validated" ? "#3fb950" : "#79c0ff",
                  fontWeight: 600,
                }}
                title="Set by the Judge during synthesis"
              >
                {mem.validation_status}
              </span>
            )}
          </div>
          <div style={{ fontSize: 12, color: "#e6edf3" }}>{headline}</div>
          {mem.summary && mem.description !== mem.summary && (
            <div style={{ fontSize: 11, color: "#8b949e", marginTop: 4 }}>{mem.description}</div>
          )}
          {mem.rationale && <div style={{ fontSize: 11, color: "#8b949e", marginTop: 4 }}>Rationale: {mem.rationale}</div>}
          {mem.applicability && <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>Applicability: {mem.applicability}</div>}
          {status === "duplicate" && dupInfo && (
            <div style={{ marginTop: 8, padding: 8, background: "#0d1117", border: "1px solid #d2992240", borderRadius: 6, fontSize: 11 }}>
              <div style={{ color: "#d29922", fontWeight: 600, marginBottom: 4 }}>
                Similar memory already exists{dupInfo.similarity ? ` (${(dupInfo.similarity * 100).toFixed(0)}% match)` : ""}
              </div>
              <div style={{ color: "#8b949e" }}>{dupInfo.description}</div>
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
          <button
            className="btn btn-secondary"
            style={{ fontSize: 11, padding: "4px 10px" }}
            onClick={handleSave}
            disabled={status === "saved" || status === "duplicate" || status === "checking"}
          >
            {status === "checking" ? "Checking..." : status === "saved" ? "Saved" : status === "duplicate" ? "Already exists" : "Add to Memory Store"}
          </button>
          {status !== "saved" && (
            <button className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 10px" }} onClick={handleDismiss}>Dismiss</button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Results: Proposed Additions (Phase 5 goal generation) ──────────────────

function _confidenceColor(c: number): string {
  if (c >= 0.8) return "#3fb950";
  if (c >= 0.55) return "#d29922";
  return "#8b949e";
}

function ProposedAdditionsCard({
  proposed, dismissed, onDismiss,
}: {
  proposed: ProposedAdditions | null;
  dismissed: Set<number>;
  onDismiss: (idx: number) => void;
}) {
  const sorted = useMemo(() => {
    if (!proposed) return [];
    return proposed.proposals
      .map((p, i) => ({ proposal: p, idx: i }))
      .filter((x) => !dismissed.has(x.idx))
      .sort(
        (a, b) => b.proposal.confidence - a.proposal.confidence,
      );
  }, [proposed, dismissed]);

  if (!proposed || sorted.length === 0) return null;

  return (
    <div className="card" style={{ marginBottom: 16, borderColor: "#d29922" }}>
      <div className="card-title" style={{ fontSize: 13, color: "#d29922" }}>
        Suggested next requirements{" "}
        <span style={{ fontWeight: 400, color: "#8b949e" }}>
          ({sorted.length})
        </span>
      </div>
      {proposed.summary && (
        <p style={{ color: "#8b949e", fontSize: 12, margin: "0 0 8px" }}>
          {proposed.summary}
        </p>
      )}
      <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 12px" }}>
        The Planner reviewed the completed run and proposes these
        requirements as candidates for the next refinement. Each
        carries a rationale citing the existing items / risks that
        triggered it. Dismiss any you don't want; nothing runs
        without your explicit submission as a fresh run.
      </p>
      {sorted.map(({ proposal: p, idx }) => (
        <ProposedRequirementRow
          key={idx}
          proposal={p}
          onDismiss={() => onDismiss(idx)}
        />
      ))}
    </div>
  );
}

function ProposedRequirementRow({
  proposal: p, onDismiss,
}: { proposal: ProposedRequirement; onDismiss: () => void }) {
  return (
    <div
      style={{
        border: "1px solid #21262d",
        borderRadius: 6,
        padding: 12,
        marginBottom: 8,
        background: "#0d1117",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3", marginBottom: 4 }}>
            {p.title}
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 6 }}>
            <span className={priorityBadge(p.priority)} style={{ fontSize: 10 }}>
              {p.priority}
            </span>
            <span
              style={{
                fontSize: 10, padding: "1px 6px", borderRadius: 3,
                color: _confidenceColor(p.confidence),
                background: "#161b22",
              }}
              title="Planner self-rated confidence"
            >
              confidence {(p.confidence * 100).toFixed(0)}%
            </span>
            {p.tags.map((t, i) => (
              <span
                key={i}
                style={{
                  fontSize: 10, padding: "1px 6px", borderRadius: 3,
                  background: "#161b22", color: "#8b949e",
                }}
              >
                {t}
              </span>
            ))}
          </div>
          <p style={{ fontSize: 12, color: "#c9d1d9", margin: "0 0 6px" }}>
            {p.description}
          </p>
          {p.rationale && (
            <p style={{ fontSize: 11, color: "#8b949e", margin: 0, fontStyle: "italic" }}>
              <strong>Why:</strong> {p.rationale}
            </p>
          )}
          {p.related_to.length > 0 && (
            <p style={{ fontSize: 10, color: "#484f58", margin: "4px 0 0" }}>
              relates to:{" "}
              {p.related_to.map((rid, i) => (
                <code key={rid} style={{ fontSize: 10, marginRight: 4 }}>
                  {rid}{i < p.related_to.length - 1 ? "," : ""}
                </code>
              ))}
            </p>
          )}
        </div>
        <button
          onClick={onDismiss}
          className="btn btn-secondary"
          style={{ fontSize: 10, padding: "2px 8px", color: "#f85149" }}
          title="Dismiss this proposal"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

// ── Results: Episodes Tab ────────────────────────────────────────────────────
//
// One card per refined requirement that produced a Debate Episode. The
// episode is the operator-readable narrative of one panel debate —
// outcome, key turning points, unresolved tensions, dimension scores.
// Built deterministically server-side from the DebateTrace; we render
// the structured episode here (not the markdown blob) so the layout
// matches the rest of the Refinery UI palette.

const _EP_OUTCOME_STYLE: Record<string, { label: string; color: string; bg: string }> = {
  converged: { label: "✓ Converged", color: "#3fb950", bg: "#3fb95015" },
  short_circuited: { label: "⚠ Short-circuited", color: "#f0883e", bg: "#f0883e15" },
  aborted: { label: "✗ Aborted", color: "#f85149", bg: "#f8514915" },
};

function episodeActorColor(actor: string): string {
  // Reuse the single source of truth for panel-seat colors; fall back
  // for synthetic actors ("system") the agent log doesn't style.
  return AGENT_STYLE[actor]?.color ?? "#8b949e";
}

function EpisodeListSection({
  title, color, items,
}: { title: string; color: string; items: string[] | undefined }) {
  if (!items || items.length === 0) return null;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color, marginBottom: 4 }}>
        {title} ({items.length})
      </div>
      <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "#e6edf3", lineHeight: 1.5 }}>
        {items.map((s, i) => <li key={i}>{s}</li>)}
      </ul>
    </div>
  );
}

function EpisodesPanel({ refined }: { refined: RefinedRequirement[] }) {
  const withEpisodes = useMemo(
    () => refined.filter((r) => r.debate?.episode != null),
    [refined],
  );
  if (withEpisodes.length === 0) {
    return (
      <div style={{ color: "#8b949e", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
        No debate episodes recorded — this refinement may have used the
        carry-forward path or pre-dates episode tracking.
      </div>
    );
  }

  return (
    <div>
      <p style={{ color: "#8b949e", fontSize: 12, margin: "0 0 12px" }}>
        One episode per requirement debate — outcome, key turning points,
        and what remained unresolved. Episodes also write into Episode
        memory so future debates' recall can surface them.
      </p>
      {withEpisodes.map((r) => (
        <EpisodeCard key={r.id} refined={r} />
      ))}
    </div>
  );
}

function EpisodeCard({ refined }: { refined: RefinedRequirement }) {
  const [open, setOpen] = useState(false);
  const ep = refined.debate?.episode;
  if (!ep) return null;

  const outcomeStyle = _EP_OUTCOME_STYLE[ep.outcome] ?? {
    label: ep.outcome, color: "#8b949e", bg: "#21262d",
  };
  const dims = ep.final_dimensions ?? {};
  const score = ep.final_overall_score;

  return (
    <div
      className="card"
      style={{ marginBottom: 12, borderColor: `${outcomeStyle.color}40` }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 10 }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#e6edf3", marginBottom: 4 }}>
            {ep.title || refined.title}
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <code style={{ fontSize: 11, color: "#8b949e" }}>{ep.requirement_id}</code>
            <span
              style={{
                fontSize: 10, padding: "2px 8px", borderRadius: 10, fontWeight: 600,
                color: outcomeStyle.color, background: outcomeStyle.bg,
              }}
            >
              {outcomeStyle.label}
            </span>
            <span style={{ fontSize: 10, color: "#8b949e" }}>
              {ep.rounds_executed} round{ep.rounds_executed === 1 ? "" : "s"}
            </span>
            {ep.escalation_level > 0 && (
              <span style={{ fontSize: 10, color: "#d29922" }}>· escalated ×{ep.escalation_level}</span>
            )}
            {ep.research_calls_used > 0 && (
              <span style={{ fontSize: 10, color: "#39c5cf" }}>· {ep.research_calls_used} research call(s)</span>
            )}
            {typeof score === "number" && (
              <span style={{ fontSize: 10, color: "#8b949e" }}>· score {score.toFixed(2)}</span>
            )}
          </div>
          {typeof ep.convergence_score === "number" && (
            <div style={{ marginTop: 8 }}>
              <ConvergenceBar score={ep.convergence_score} />
            </div>
          )}
        </div>
        <button
          onClick={() => setOpen(!open)}
          style={{ background: "none", border: "none", color: "#58a6ff", fontSize: 11, cursor: "pointer", padding: 0, flexShrink: 0 }}
        >
          {open ? "Collapse" : "Expand"}
        </button>
      </div>

      <div style={{ fontSize: 12, color: "#e6edf3", lineHeight: 1.5, marginBottom: open ? 12 : 0 }}>
        {ep.summary}
      </div>

      {open && (
        <>
          {Object.keys(dims).length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e", marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 }}>
                Final dimension scores
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {Object.entries(dims).map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      fontSize: 11, padding: "3px 8px", borderRadius: 4,
                      background: "#161b22", border: "1px solid #21262d",
                      color: "#e6edf3",
                    }}
                  >
                    <span style={{ color: "#8b949e" }}>{k}:</span> {v.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>
          )}

          {ep.key_events && ep.key_events.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e", marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 }}>
                Key events
              </div>
              <div style={{ borderLeft: "2px solid #21262d", paddingLeft: 12 }}>
                {ep.key_events.map((kev) => {
                  const color = episodeActorColor(kev.actor);
                  return (
                    <div key={kev.order} style={{ marginBottom: 8, position: "relative" }}>
                      <div
                        style={{
                          position: "absolute", left: -17, top: 4,
                          width: 8, height: 8, borderRadius: 4,
                          background: color, border: "2px solid #0d1117",
                        }}
                      />
                      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 2, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 10, color: "#8b949e" }}>round {kev.round_number}</span>
                        <span
                          style={{
                            fontSize: 10, padding: "1px 6px", borderRadius: 3,
                            background: `${color}20`, color, fontWeight: 600,
                            textTransform: "uppercase",
                          }}
                        >
                          {kev.actor}
                        </span>
                        <span style={{ fontSize: 10, color: "#8b949e", fontStyle: "italic" }}>
                          {kev.kind}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: "#e6edf3", lineHeight: 1.4 }}>
                        {kev.headline}
                      </div>
                      {kev.detail && (
                        <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2, lineHeight: 1.4 }}>
                          {kev.detail}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <EpisodeListSection title="Unresolved points" color="#f85149" items={ep.unresolved_points} />
          <EpisodeListSection title="Open questions" color="#d29922" items={ep.open_questions} />
          <EpisodeListSection title="Explicit tradeoffs" color="#bc8cff" items={ep.explicit_tradeoffs} />

          {ep.participants && ep.participants.length > 0 && (
            <div style={{ fontSize: 10, color: "#8b949e", marginTop: 8 }}>
              Panel: {ep.participants.join(", ")}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Results: Insights Tab ────────────────────────────────────────────────────

function InsightsPanel({ response }: { response: RefineryResponse }) {
  return (
    <div>
      {response.summary && (
        <div className="card">
          <div className="card-title" style={{ fontSize: 13 }}>Summary</div>
          <div style={{ fontSize: 12, color: "#e6edf3", lineHeight: 1.6 }}>{response.summary}</div>
        </div>
      )}
      {response.evidence_summary && (
        <div className="card">
          <div className="card-title" style={{ fontSize: 13 }}>Evidence Summary</div>
          <div style={{ fontSize: 12, color: "#e6edf3", lineHeight: 1.6 }}>{response.evidence_summary}</div>
        </div>
      )}
      {response.methodology && (
        <div className="card">
          <div className="card-title" style={{ fontSize: 13 }}>Methodology</div>
          <div style={{ fontSize: 12, color: "#e6edf3", lineHeight: 1.6 }}>{response.methodology}</div>
        </div>
      )}
      {response.risk_areas.length > 0 && (
        <div className="card" style={{ borderColor: "#d2992240" }}>
          <div className="card-title" style={{ fontSize: 13, color: "#d29922" }}>
            Risk Areas ({response.risk_areas.length})
          </div>
          <ul style={{ fontSize: 12, color: "#e6edf3", margin: 0, paddingLeft: 16 }}>
            {response.risk_areas.map((r, i) => <li key={i} style={{ marginBottom: 4 }}>{r}</li>)}
          </ul>
        </div>
      )}
      {response.pass_summaries.length > 0 && (
        <div className="card">
          <div className="card-title" style={{ fontSize: 13 }}>Pass-by-Pass Analysis</div>
          <ol style={{ fontSize: 12, color: "#e6edf3", margin: 0, paddingLeft: 16 }}>
            {response.pass_summaries.map((s, i) => <li key={i} style={{ marginBottom: 6, lineHeight: 1.5 }}>{s}</li>)}
          </ol>
        </div>
      )}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

function autoSelectRequirement(reqs: RefinedRequirement[]): string | null {
  const firstMod = reqs.find((r) => r.changes.length > 0);
  if (firstMod) return firstMod.id;
  return reqs.length > 0 ? reqs[0].id : null;
}

export default function RefineryTab() {
  const [phase, setPhase] = useState<Phase>("input");
  const [gatherStep, setGatherStep] = useState("");
  const [gatherHistory, setGatherHistory] = useState<string[]>([]);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [response, setResponse] = useState<RefineryResponse | null>(null);
  const [proposedAdditions, setProposedAdditions] = useState<ProposedAdditions | null>(null);
  const [dismissedProposals, setDismissedProposals] = useState<Set<number>>(new Set());
  const [resultId, setResultId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  // History state (lifted from InputPhase to survive phase transitions)
  const [history, setHistory] = useState<RefineryHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [resumable, setResumable] = useState<ResumableRefineryRun[]>([]);

  const refreshResumable = useCallback(async () => {
    try {
      const data = await api.refineryResumable(20);
      setResumable(data.results || []);
    } catch {
      setResumable([]);
    }
  }, []);

  useEffect(() => {
    api.refineryHistory(20)
      .then((data) => setHistory(data.results || []))
      .catch(() => {})
      .finally(() => setHistoryLoading(false));
    void refreshResumable();
  }, [refreshResumable]);

  const handleDeleteHistory = useCallback(async (rid: string) => {
    if (!window.confirm(`Delete refinement ${rid}? This cannot be undone.`)) return;
    try {
      await api.deleteRefineryResult(rid);
      setHistory((h) => h.filter((r) => r.id !== rid));
    } catch { /* ignore */ }
  }, []);

  const handleDismissResumable = useCallback(async (rid: string) => {
    if (!window.confirm(`Dismiss resumable run ${rid}? Telemetry is preserved; the run just stops appearing here.`)) return;
    try {
      await api.dismissResumableRefinery(rid);
      setResumable((rs) => rs.filter((r) => r.refinery_run_id !== rid));
    } catch { /* ignore */ }
  }, []);

  // Results state
  const [resultTab, setResultTab] = useState<ResultTab>("requirements");
  const [selectedReqId, setSelectedReqId] = useState<string | null>(null);
  const [reqFilter, setReqFilter] = useState<ReqFilter>("all");
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [dismissedMems, setDismissedMems] = useState<Set<number>>(new Set());
  const [memoryKindTab, setMemoryKindTab] = useState<MemoryKind>("pattern");
  const [editing, setEditing] = useState(false);

  const consumeRefineryStream = useCallback(
    async (resp: Response, ctrl: AbortController) => {
      if (!resp.ok || !resp.body) {
        setPhase("error");
        setError(`Server error: ${resp.status}`);
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done || ctrl.signal.aborted) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: RefinerySSEEvent = JSON.parse(line.slice(6));
            if (event.phase !== "done") {
              const ts = Date.now();
              const tagged = { ...event, _ts: ts };
              setEvents((prev) =>
                prev.length >= 500
                  ? [...prev.slice(prev.length - 499), tagged]
                  : [...prev, tagged]
              );
            }
            if (event.phase === "gathering") {
              setPhase("gathering");
              const s = event.step || "";
              setGatherStep(s);
              if (s && s !== "complete") {
                setGatherHistory((h) => (h[h.length - 1] === s ? h : [...h, s]));
              }
            }
            else if (event.phase === "refining" || event.phase === "reconciling") {
              setPhase(event.phase);
            } else if (event.phase === "done" && event.data) {
              const refResp = event.data as RefineryResponse;
              setResponse(refResp);
              if (event.result_id) setResultId(event.result_id);
              setPhase("done");
              setSelectedReqId(autoSelectRequirement(refResp.refined_requirements));
              void refreshResumable();
            } else if (event.phase === "proposed_additions" && event.data) {
              setProposedAdditions(event.data as ProposedAdditions);
            } else if (event.phase === "planning") {
              // Planner is a Phase-5 progress event; no UI state change
              // beyond letting the timeline see it via setEvents above.
            } else if (event.phase === "error") {
              setError(event.message || "Unknown error");
              setPhase("error");
            }
          } catch { /* skip */ }
        }
      }
    },
    [refreshResumable],
  );

  const _resetForNewRun = useCallback(() => {
    setPhase("gathering");
    setGatherStep("");
    setGatherHistory([]);
    setEvents([]);
    setResponse(null);
    setProposedAdditions(null);
    setDismissedProposals(new Set());
    setError("");
    setResultTab("requirements");
    setSelectedReqId(null);
    setReqFilter("all");
    setApplied(new Set());
    setDismissed(new Set());
    setDismissedMems(new Set());
    setEditing(false);
  }, []);

  const startRefinery = useCallback(async (body: StartRefineryBody) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    _resetForNewRun();
    try {
      const resp = await api.streamRefinery(body, ctrl.signal);
      await consumeRefineryStream(resp, ctrl);
    } catch (e) {
      if (!ctrl.signal.aborted) {
        setPhase("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  }, [consumeRefineryStream, _resetForNewRun]);

  const resumeRefinery = useCallback(async (refineryRunId: string) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    _resetForNewRun();
    try {
      const resp = await api.streamResumeRefinery(refineryRunId, ctrl.signal);
      await consumeRefineryStream(resp, ctrl);
    } catch (e) {
      if (!ctrl.signal.aborted) {
        setPhase("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  }, [consumeRefineryStream, _resetForNewRun]);

  useEffect(() => () => { abortRef.current?.abort(); }, []);

  const loadResult = useCallback(async (id: string) => {
    setPhase("gathering");
    setGatherStep("loading saved result");
    try {
      const data = await api.loadRefineryResult(id);
      setResponse(data);
      setResultId(id);
      setPhase("done");
      setSelectedReqId(autoSelectRequirement(data.refined_requirements));
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const handleExport = async () => {
    if (!response) return;
    try {
      const blob = await api.exportRefineryZip(response);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `refinery-${response.source_run_id || "upload"}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { /* ignore */ }
  };

  // ── Done-phase derivations ───────────────────────────────────
  // Hoisted above the early returns so the hook count stays stable
  // across phase transitions (React's rules-of-hooks). When response
  // is null the early returns below short-circuit before these
  // values are read; the empty defaults keep the hook calls cheap.
  const allReqs = response?.refined_requirements ?? [];
  const filteredReqs = useMemo(
    () => allReqs.filter((r) => {
      if (reqFilter === "modified") return r.changes.length > 0;
      if (reqFilter === "unchanged") return r.changes.length === 0;
      if (reqFilter === "unresolved") return isUnresolved(r);
      return true;
    }),
    [allReqs, reqFilter],
  );
  const unresolvedCount = useMemo(
    () => allReqs.filter(isUnresolved).length,
    [allReqs],
  );
  const selectedReq = useMemo(
    () => allReqs.find((r) => r.id === selectedReqId) || null,
    [allReqs, selectedReqId],
  );
  const remainingCount = useMemo(
    () => allReqs.filter(
      (r) => r.changes.length > 0 && !applied.has(r.id) && !dismissed.has(r.id),
    ).length,
    [allReqs, applied, dismissed],
  );
  const visibleMems = useMemo(
    () => (response?.suggested_memories ?? []).filter(
      (_, i) => !dismissedMems.has(i),
    ),
    [response?.suggested_memories, dismissedMems],
  );
  const episodeCount = useMemo(
    () => (response?.refined_requirements ?? []).filter(
      (r) => r.debate?.episode != null,
    ).length,
    [response?.refined_requirements],
  );

  // Group visible memories by their kind so the Memories tab can render
  // five sub-tabs. Index in the original ``suggested_memories`` array is
  // preserved so dismissal still works against the source list.
  const memsByKind = useMemo(() => {
    const out: Record<MemoryKind, Array<{ mem: SuggestedMemory; idx: number }>> = {
      decision: [], incident: [], pattern: [], constraint: [], conflict: [],
    };
    (response?.suggested_memories ?? []).forEach((mem, idx) => {
      if (dismissedMems.has(idx)) return;
      out[memoryKindOf(mem)].push({ mem, idx });
    });
    return out;
  }, [response?.suggested_memories, dismissedMems]);
  const appliedCount = applied.size;
  const dismissedCount = dismissed.size;

  // ── Input / Progress / Error phases ──────────────────────────
  if (phase === "input") return (
    <InputPhase
      onStart={startRefinery}
      onLoadResult={loadResult}
      onResume={resumeRefinery}
      onDismissResumable={handleDismissResumable}
      history={history}
      historyLoading={historyLoading}
      resumable={resumable}
      onDeleteHistory={handleDeleteHistory}
    />
  );
  if (phase === "gathering") return <GatheringProgress step={gatherStep} history={gatherHistory} />;
  if (phase === "refining") return (
    <AgentEventTimeline
      events={events}
      heading="Refining requirements — live agent events"
      fallback="Waiting for the first agent event..."
    />
  );
  if (phase === "reconciling") return (
    <AgentEventTimeline
      events={events}
      heading="Reconciling requirements — live agent events"
      fallback="Checking for duplicates, coherence issues, and relationship gaps..."
    />
  );
  if (phase === "error") return (
    <div className="card" style={{ borderColor: "#da3633", maxWidth: 480, margin: "40px auto" }}>
      <div className="card-title" style={{ color: "#f85149" }}>Refinery error</div>
      <code style={{ fontSize: 12, wordBreak: "break-all" }}>{error}</code>
      <div style={{ marginTop: 12 }}>
        <button className="btn btn-secondary" onClick={() => setPhase("input")}>Back</button>
      </div>
    </div>
  );
  if (!response) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 120px)" }}>
      {/* Sticky header */}
      <div style={{ padding: "8px 0", borderBottom: "1px solid #21262d", flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#e6edf3" }}>Refinery Results</div>
            {resultId && <code style={{ fontSize: 10, color: "#484f58" }}>{resultId}</code>}
            <span style={{ fontSize: 11, color: "#8b949e" }}>
              {appliedCount > 0 && <span style={{ color: "#3fb950" }}>{appliedCount} applied</span>}
              {appliedCount > 0 && (dismissedCount > 0 || remainingCount > 0) && " · "}
              {dismissedCount > 0 && <span>{dismissedCount} dismissed</span>}
              {dismissedCount > 0 && remainingCount > 0 && " · "}
              {remainingCount > 0 && <span style={{ color: "#58a6ff" }}>{remainingCount} remaining</span>}
            </span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 10px" }} onClick={() => setPhase("input")}>New Refinement</button>
            <button className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 10px" }} onClick={handleExport}>Export ZIP</button>
            <button
              className="btn btn-secondary"
              style={{ fontSize: 11, padding: "4px 10px" }}
              onClick={() => openAboutRequirementsRefinery()}
              title="Open the architectural whitepaper"
            >
              About
            </button>
          </div>
        </div>
        {/* Sub-tabs */}
        <div
          role="tablist"
          aria-label="Refinery result section"
          style={{ display: "flex", gap: 0, marginTop: 8 }}
        >
          {(["requirements", "insights", "memories", "episodes"] as ResultTab[]).map((tab) => {
            return (
              <button
                key={tab}
                role="tab"
                aria-selected={resultTab === tab}
                className={`tab-btn${resultTab === tab ? " active" : ""}`}
                onClick={() => setResultTab(tab)}
                style={{ fontSize: 12, padding: "5px 14px", textTransform: "capitalize" }}
              >
                {tab}
                {tab === "memories" && visibleMems.length > 0 && (
                  <span style={{ marginLeft: 4, fontSize: 10, color: "#8b949e" }}>({visibleMems.length})</span>
                )}
                {tab === "episodes" && episodeCount > 0 && (
                  <span style={{ marginLeft: 4, fontSize: 10, color: "#8b949e" }}>({episodeCount})</span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        {resultTab === "requirements" && (
          <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", height: "100%" }}>
            {/* Left sidebar */}
            <div style={{ borderRight: "1px solid #21262d", overflowY: "auto" }}>
              {/* Filter buttons */}
              <div
                role="tablist"
                aria-label="Requirement filter"
                style={{ padding: "8px 12px", borderBottom: "1px solid #21262d", display: "flex", gap: 4, flexWrap: "wrap" }}
              >
                {(["all", "modified", "unchanged", "unresolved"] as ReqFilter[]).map((f) => {
                  const count = f === "unresolved" ? unresolvedCount : null;
                  const danger = f === "unresolved" && count && count > 0;
                  return (
                    <button
                      key={f}
                      role="tab"
                      aria-selected={reqFilter === f}
                      onClick={() => setReqFilter(f)}
                      style={{
                        background: reqFilter === f ? "#21262d" : "transparent",
                        border: "1px solid",
                        borderColor: reqFilter === f
                          ? (danger ? "#f8514960" : "#30363d")
                          : (danger ? "#f8514940" : "transparent"),
                        borderRadius: 4, padding: "2px 8px", fontSize: 10,
                        color: danger ? "#ff7b72" : "#8b949e",
                        cursor: "pointer", textTransform: "capitalize",
                        fontWeight: danger ? 600 : 400,
                      }}
                    >
                      {f}
                      {count != null && count > 0 && (
                        <span style={{ marginLeft: 4, fontSize: 9, opacity: 0.85 }}>
                          ({count})
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
              {/* Requirement list */}
              {filteredReqs.map((req) => (
                <ReqListItem
                  key={req.id}
                  req={req}
                  selected={selectedReqId === req.id}
                  applied={applied.has(req.id)}
                  dismissed={dismissed.has(req.id)}
                  onClick={() => { setSelectedReqId(req.id); setEditing(false); }}
                />
              ))}
              {filteredReqs.length === 0 && (
                <div style={{ padding: 16, color: "#8b949e", fontSize: 12, textAlign: "center" }}>
                  No requirements match filter.
                </div>
              )}
            </div>

            {/* Right detail panel */}
            <div style={{ overflowY: "auto", padding: 16 }}>
              {selectedReq ? (
                editing ? (
                  <InlineEditor
                    req={selectedReq}
                    onSave={() => {
                      setApplied((s) => new Set(s).add(selectedReq.id));
                      setEditing(false);
                      _signalRequirementDecision(resultId, selectedReq, "edited");
                    }}
                    onCancel={() => setEditing(false)}
                  />
                ) : (
                  <ReqDetailPanel
                    req={selectedReq}
                    applied={applied.has(selectedReq.id)}
                    dismissed={dismissed.has(selectedReq.id)}
                    suggestedMemories={response?.suggested_memories ?? []}
                    onApply={async (applyMemories) => {
                      try {
                        const body: Record<string, unknown> = {
                          title: selectedReq.title,
                          description: selectedReq.description,
                          priority: selectedReq.priority,
                          tags: selectedReq.tags,
                        };
                        if (applyMemories.length > 0) {
                          // Convert wire-format SuggestedMemory → backend
                          // apply_memories payload (kind required; default
                          // legacy ``pattern`` so legacy memories without an
                          // explicit kind still pass the schema regex).
                          body.apply_memories = applyMemories.map((m) => ({
                            kind: memoryKindOf(m),
                            summary: (m.summary || m.description || "").slice(0, 500),
                            body: m.description || "",
                            context: m.context || "",
                            source_requirement_id: m.source_requirement_id || selectedReq.id,
                            source_role: m.source_role || m.source_feature || "",
                            rationale: m.rationale || "",
                            provenance_refinery_run_id: resultId || "",
                          }));
                        }
                        await api.patchRequirement(selectedReq.id, body);
                        setApplied((s) => new Set(s).add(selectedReq.id));
                        _signalRequirementDecision(resultId, selectedReq, "accepted");
                      } catch { /* ignore */ }
                    }}
                    onDismiss={() => {
                      setDismissed((s) => new Set(s).add(selectedReq.id));
                      _signalRequirementDecision(resultId, selectedReq, "dismissed");
                    }}
                    onEdit={() => setEditing(true)}
                  />
                )
              ) : (
                <div style={{ color: "#8b949e", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
                  Select a requirement from the list to view details.
                </div>
              )}
            </div>
          </div>
        )}

        {resultTab === "insights" && (
          <div style={{ overflowY: "auto", padding: 16, height: "100%" }}>
            <ProposedAdditionsCard
              proposed={proposedAdditions}
              dismissed={dismissedProposals}
              onDismiss={(i) =>
                setDismissedProposals((s) => new Set(s).add(i))
              }
            />
            <InsightsPanel response={response} />
          </div>
        )}

        {resultTab === "memories" && (
          <div style={{ overflowY: "auto", padding: 16, height: "100%" }}>
            {visibleMems.length > 0 ? (
              <>
                <p style={{ color: "#8b949e", fontSize: 12, margin: "0 0 12px" }}>
                  Save these as institutional-memory entries — role-filtered retrieval will recall them
                  on future runs, scoped by kind.
                </p>
                {/* 5-kind sub-tab strip */}
                <div
                  role="tablist"
                  aria-label="Memory kind"
                  style={{
                    display: "flex", gap: 0, marginBottom: 12,
                    borderBottom: "1px solid #21262d",
                  }}
                >
                  {MEMORY_KINDS.map((kind) => {
                    const count = memsByKind[kind].length;
                    const active = memoryKindTab === kind;
                    const palette = MEMORY_KIND_COLOR[kind];
                    return (
                      <button
                        key={kind}
                        role="tab"
                        aria-selected={active}
                        onClick={() => setMemoryKindTab(kind)}
                        disabled={count === 0}
                        style={{
                          background: "transparent",
                          border: "none",
                          borderBottom: "2px solid",
                          borderBottomColor: active ? palette.fg : "transparent",
                          color: active ? palette.fg : count === 0 ? "#484f58" : "#8b949e",
                          cursor: count === 0 ? "not-allowed" : "pointer",
                          fontSize: 12,
                          padding: "6px 14px",
                          fontWeight: active ? 600 : 400,
                          transition: "all 0.12s",
                        }}
                      >
                        {MEMORY_KIND_LABEL[kind]}
                        {count > 0 && (
                          <span style={{ marginLeft: 4, fontSize: 10, opacity: 0.85 }}>
                            ({count})
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
                {/* Per-kind list */}
                {memsByKind[memoryKindTab].length > 0 ? (
                  memsByKind[memoryKindTab].map(({ mem, idx }) => (
                    <MemoryCard
                      key={idx}
                      mem={mem}
                      onSave={() => {}}
                      onDismiss={() => setDismissedMems((s) => new Set(s).add(idx))}
                    />
                  ))
                ) : (
                  <div style={{ color: "#8b949e", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
                    No {MEMORY_KIND_LABEL[memoryKindTab].toLowerCase()} memories from this refinement.
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: "#8b949e", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
                No suggested memories from this refinement.
              </div>
            )}
          </div>
        )}

        {resultTab === "episodes" && (
          <div style={{ overflowY: "auto", padding: 16, height: "100%" }}>
            <EpisodesPanel
              refined={response?.refined_requirements ?? []}
            />
          </div>
        )}
      </div>
    </div>
  );
}
