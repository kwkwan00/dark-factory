import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { useAgentMemoryList } from "../hooks/useDashboard";
import { MemoryActivitySection, MemoryGraphSection } from "./MemoryMetrics";
import RefreshIcon from "./RefreshIcon";

function MemoryMenu({ memoryId, onDelete }: { memoryId: string; onDelete: () => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        style={{
          background: "transparent",
          border: "1px solid #30363d",
          borderRadius: 4,
          color: "#8b949e",
          cursor: "pointer",
          padding: "2px 8px",
          fontSize: 13,
          lineHeight: 1,
        }}
        title="Actions"
      >
        ···
      </button>
      {open && (
        <div
          style={{
            position: "absolute",
            right: 0,
            top: "100%",
            marginTop: 4,
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: 6,
            boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
            zIndex: 100,
            minWidth: 140,
            padding: 4,
          }}
        >
          <button
            onClick={async (e) => {
              e.stopPropagation();
              setOpen(false);
              if (!confirm(`Delete memory ${memoryId}?`)) return;
              try {
                await api.deleteMemory(memoryId);
                onDelete();
              } catch (err) {
                console.error("Delete failed:", err);
              }
            }}
            style={{
              display: "block",
              width: "100%",
              background: "transparent",
              border: "none",
              borderRadius: 4,
              color: "#f85149",
              cursor: "pointer",
              padding: "6px 10px",
              fontSize: 12,
              textAlign: "left",
            }}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "#1c2128"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
          >
            Delete memory
          </button>
        </div>
      )}
    </div>
  );
}

const TYPE_ICON: Record<string, string> = {
  pattern: "🔵",
  mistake: "🔴",
  solution: "🟢",
  strategy: "🟡",
};

const TYPE_LABEL: Record<string, string> = {
  pattern: "Pattern",
  mistake: "Mistake",
  solution: "Solution",
  strategy: "Strategy",
};

const TYPES = ["all", "pattern", "mistake", "solution", "strategy"];
const LIMITS = [50, 100, 200, 500];
const SORT_OPTIONS = [
  { value: "relevance", label: "Relevance" },
  { value: "recalled", label: "Most recalled" },
  { value: "newest", label: "Newest" },
  { value: "oldest", label: "Oldest" },
] as const;
type SortKey = (typeof SORT_OPTIONS)[number]["value"];

interface ResolvedBy {
  id?: string;
  description?: string;
}

interface MemoryEntry {
  id?: string;
  type?: string;
  description?: string;
  context?: string;
  trigger_context?: string;
  applicability?: string;
  code_snippet?: string;
  agent?: string;
  source_feature?: string;
  source_spec_id?: string;
  run_id?: string;
  relevance_score?: number;
  times_applied?: number;
  times_seen?: number;
  times_recalled?: number;
  created_at?: string;
  updated_at?: string;
  last_recalled_at?: string;
  resolved_by?: ResolvedBy;
  [key: string]: unknown;
}

function getDetailLines(m: MemoryEntry): Array<[string, string]> {
  const lines: Array<[string, string]> = [];
  if (m.context) lines.push(["Context", String(m.context)]);
  if (m.trigger_context) lines.push(["Trigger", String(m.trigger_context)]);
  if (m.applicability) lines.push(["When to apply", String(m.applicability)]);
  if (m.code_snippet) lines.push(["Code", String(m.code_snippet)]);
  if (m.agent) lines.push(["Agent", String(m.agent)]);
  if (m.source_feature) lines.push(["Source feature", String(m.source_feature)]);
  if (m.source_spec_id) lines.push(["Source spec", String(m.source_spec_id)]);
  if (m.run_id) lines.push(["Run", String(m.run_id)]);
  if (m.times_applied != null) lines.push(["Times applied", String(m.times_applied)]);
  if (m.times_seen != null) lines.push(["Times seen", String(m.times_seen)]);
  if (m.times_recalled != null) lines.push(["Times recalled", String(m.times_recalled)]);
  if (m.created_at) lines.push(["Created", String(m.created_at)]);
  if (m.updated_at && m.updated_at !== m.created_at) lines.push(["Updated", String(m.updated_at)]);
  if (m.last_recalled_at) lines.push(["Last recalled", String(m.last_recalled_at)]);
  if (m.resolved_by?.description) {
    lines.push(["Resolved by", `${m.resolved_by.id ?? "?"}: ${m.resolved_by.description}`]);
  }
  return lines;
}

function sortEntries(entries: MemoryEntry[], sortKey: SortKey): MemoryEntry[] {
  const sorted = [...entries];
  switch (sortKey) {
    case "relevance":
      sorted.sort((a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0));
      break;
    case "recalled":
      sorted.sort((a, b) => (b.times_recalled ?? 0) - (a.times_recalled ?? 0));
      break;
    case "newest":
      sorted.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
      break;
    case "oldest":
      sorted.sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));
      break;
  }
  return sorted;
}

type View = "memory" | "metrics";

export default function AgentMemoryTab() {
  const [view, setView] = useState<View>("memory");
  const { state, type, setType, limit, setLimit, refresh } = useAgentMemoryList();
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("recalled");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [importing, setImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exportMemories = useCallback(() => {
    if (state.status !== "done") return;
    const blob = new Blob(
      [JSON.stringify({ memories: state.data.results }, null, 2)],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `agent-memory-export-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 200);
  }, [state]);

  const importMemories = useCallback(
    async (file: File) => {
      setImporting(true);
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        const memories = Array.isArray(data) ? data : data.memories;
        if (!Array.isArray(memories)) {
          alert("Invalid file: expected a JSON array or {memories: [...]}");
          return;
        }
        const result = await api.importMemories(memories);
        alert(`Imported ${result.imported} memories (${result.skipped} skipped)`);
        void refresh();
      } catch (err) {
        alert(`Import failed: ${err instanceof Error ? err.message : err}`);
      } finally {
        setImporting(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [refresh],
  );

  const toggleExpand = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Client-side text filter + sort on the loaded results
  const filtered = useMemo(() => {
    if (state.status !== "done") return [] as MemoryEntry[];
    const all = state.data.results as MemoryEntry[];
    const needle = filter.trim().toLowerCase();
    const matched = needle
      ? all.filter((m) => {
          const hay = [
            m.description ?? "",
            m.context ?? "",
            m.trigger_context ?? "",
            m.applicability ?? "",
            m.code_snippet ?? "",
            m.agent ?? "",
            m.source_feature ?? "",
            m.run_id ?? "",
            m.type ?? "",
          ]
            .join(" ")
            .toLowerCase();
          return hay.includes(needle);
        })
      : all;
    return sortEntries(matched, sortKey);
  }, [state, filter, sortKey]);

  // Type counts (for showing how many of each type exist)
  const typeCounts = useMemo(() => {
    if (state.status !== "done") return {} as Record<string, number>;
    const counts: Record<string, number> = {};
    for (const m of state.data.results as MemoryEntry[]) {
      const t = (m.type as string | undefined) ?? "?";
      counts[t] = (counts[t] ?? 0) + 1;
    }
    return counts;
  }, [state]);

  return (
    <div>
      {/* View toggle */}
      <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
        {([["memory", "Browse"], ["metrics", "Metrics"]] as const).map(
          ([key, label]) => (
            <button
              key={key}
              className={view === key ? "btn" : "btn btn-secondary"}
              onClick={() => setView(key)}
              style={{ fontSize: 12 }}
            >
              {label}
            </button>
          ),
        )}
      </div>

      {view === "metrics" ? (
        <div>
          <MemoryGraphSection />
          <MemoryActivitySection />
        </div>
      ) : (
      <>
      {/* Controls */}
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
            gap: 12,
          }}
        >
          <div className="card-title" style={{ margin: 0 }}>
            Agent Memory
            {state.status === "done" && (
              <span style={{ color: "#8b949e", fontWeight: 400, marginLeft: 8 }}>
                — {state.data.total} loaded
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button className="btn btn-secondary" onClick={exportMemories} disabled={state.status !== "done"} style={{ fontSize: 12 }}>
              Export
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={importing}
              style={{ fontSize: 12 }}
            >
              {importing ? "Importing…" : "Import"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void importMemories(file);
              }}
            />
            <button className="btn btn-secondary" onClick={() => void refresh()} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              <RefreshIcon /> Refresh
            </button>
          </div>
        </div>

        {/* Type tabs */}
        <div
          style={{
            display: "flex",
            gap: 4,
            marginBottom: 12,
            flexWrap: "wrap",
          }}
        >
          {TYPES.map((t) => {
            const isActive = type === t;
            const count = t === "all"
              ? Object.values(typeCounts).reduce((a, b) => a + b, 0)
              : typeCounts[t] ?? 0;
            return (
              <button
                key={t}
                onClick={() => setType(t)}
                className={isActive ? "btn" : "btn btn-secondary"}
                style={{ fontSize: 12, padding: "4px 12px" }}
              >
                {t === "all" ? "All" : TYPE_ICON[t]} {t === "all" ? "All" : TYPE_LABEL[t]}
                {count > 0 && (
                  <span style={{ marginLeft: 6, opacity: 0.7 }}>({count})</span>
                )}
              </button>
            );
          })}
        </div>

        {/* Filter + limit */}
        <div className="input-row" style={{ margin: 0 }}>
          <input
            className="input-text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter loaded memories (text, agent, feature, type)..."
          />
          <select
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "8px 12px",
              fontSize: 13,
            }}
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
          >
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                Sort: {o.label}
              </option>
            ))}
          </select>
          <select
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "8px 12px",
              fontSize: 13,
            }}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          >
            {LIMITS.map((l) => (
              <option key={l} value={l}>
                Show {l}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results */}
      {state.status === "loading" && (
        <div className="empty-state"><p>Loading procedural memory...</p></div>
      )}

      {state.status === "error" && (
        <div className="card" style={{ borderColor: "#da3633" }}>
          <code>{state.error}</code>
        </div>
      )}

      {state.status === "done" && (
        <div className="card">
          {filtered.length === 0 ? (
            <div className="empty-state">
              <p>
                {state.data.total === 0
                  ? "No memories yet — run the pipeline to start collecting patterns, mistakes, solutions, and strategies."
                  : `No memories match "${filter}"`}
              </p>
            </div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: 32 }}></th>
                  <th>Type</th>
                  <th>Description</th>
                  <th>Source</th>
                  <th style={{ width: 60 }}>Score</th>
                  <th style={{ width: 60 }}>Recalls</th>
                  <th style={{ width: 32 }}></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((m, i) => {
                  const mtype = (m.type as string | undefined) ?? "?";
                  const desc = (m.description as string | undefined) ?? "(no description)";
                  const score = (m.relevance_score as number | undefined) ?? 0;
                  const feature = (m.source_feature as string | undefined) ?? "";
                  const agent = (m.agent as string | undefined) ?? "";
                  const rowKey = (m.id as string | undefined) ?? `row-${i}`;
                  const isOpen = expanded.has(rowKey);
                  const detailLines = getDetailLines(m);
                  return (
                    <Fragment key={rowKey}>
                      <tr style={{ cursor: "pointer" }} onClick={() => toggleExpand(rowKey)}>
                        <td style={{ color: "#8b949e", textAlign: "center" }}>
                          {isOpen ? "▼" : "▶"}
                        </td>
                        <td>
                          {TYPE_ICON[mtype] ?? "❓"} {TYPE_LABEL[mtype] ?? mtype}
                        </td>
                        <td style={{ maxWidth: 500, wordBreak: "break-word" }}>
                          {desc.length > 120 ? desc.slice(0, 120) + "…" : desc}
                          {m.resolved_by?.description && (
                            <div style={{ fontSize: 11, color: "#3fb950", marginTop: 2 }}>
                              Fix: {m.resolved_by.description.length > 100
                                ? m.resolved_by.description.slice(0, 100) + "…"
                                : m.resolved_by.description}
                            </div>
                          )}
                        </td>
                        <td style={{ color: "#8b949e", fontSize: 12 }}>
                          {feature && <div>{feature}</div>}
                          {agent && <div>{agent}</div>}
                          {!feature && !agent && "—"}
                        </td>
                        <td>
                          <span
                            className={
                              score >= 0.7
                                ? "badge-success"
                                : score >= 0.4
                                ? "badge-warn"
                                : "badge-error"
                            }
                          >
                            {score.toFixed(2)}
                          </span>
                        </td>
                        <td style={{ color: "#8b949e", textAlign: "center", fontSize: 12 }}>
                          {(m.times_recalled as number | undefined) ?? 0}
                        </td>
                        <td>
                          {m.id && (
                            <MemoryMenu
                              memoryId={m.id as string}
                              onDelete={() => void refresh()}
                            />
                          )}
                        </td>
                      </tr>
                      {isOpen && (
                        <tr>
                          <td></td>
                          <td colSpan={6} style={{ background: "#0d1117" }}>
                            <div style={{ padding: 8, fontSize: 12 }}>
                              {desc.length > 120 && (
                                <div style={{ marginBottom: 8, lineHeight: 1.5, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                                  {desc}
                                </div>
                              )}
                              {detailLines.map(([k, v]) => (
                                <div key={k} style={{ marginBottom: 4 }}>
                                  <span style={{ color: "#8b949e", display: "inline-block", minWidth: 120 }}>
                                    {k}:
                                  </span>
                                  <span
                                    style={{
                                      color: "#c9d1d9",
                                      fontFamily:
                                        k === "Code" ? "SF Mono, Consolas, monospace" : "inherit",
                                      whiteSpace: "pre-wrap",
                                    }}
                                  >
                                    {v}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          )}
          {filter && state.data.total > 0 && (
            <div
              style={{
                marginTop: 8,
                color: "#8b949e",
                fontSize: 12,
                textAlign: "right",
              }}
            >
              {filtered.length} of {state.data.total} match
            </div>
          )}
        </div>
      )}

      </>
      )}
    </div>
  );
}
