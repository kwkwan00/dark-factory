import { Fragment, useMemo, useState } from "react";
import { useAgentMemoryList } from "../hooks/useDashboard";

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
  relevance_score?: number;
  times_applied?: number;
  times_seen?: number;
  created_at?: string;
  updated_at?: string;
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
  if (m.times_applied != null) lines.push(["Times applied", String(m.times_applied)]);
  if (m.times_seen != null) lines.push(["Times seen", String(m.times_seen)]);
  if (m.created_at) lines.push(["Created", String(m.created_at)]);
  return lines;
}

export default function AgentMemoryTab() {
  const { state, type, setType, limit, setLimit, refresh } = useAgentMemoryList();
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggleExpand = (i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  // Client-side text filter on the loaded results
  const filtered = useMemo(() => {
    if (state.status !== "done") return [] as MemoryEntry[];
    const all = state.data.results as MemoryEntry[];
    const needle = filter.trim().toLowerCase();
    if (!needle) return all;
    return all.filter((m) => {
      const hay = [
        m.description ?? "",
        m.context ?? "",
        m.trigger_context ?? "",
        m.applicability ?? "",
        m.code_snippet ?? "",
        m.agent ?? "",
        m.source_feature ?? "",
        m.type ?? "",
      ]
        .join(" ")
        .toLowerCase();
      return hay.includes(needle);
    });
  }, [state, filter]);

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
          <button className="btn btn-secondary" onClick={() => void refresh()}>
            Refresh
          </button>
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
                  <th style={{ width: 80 }}>Score</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((m, i) => {
                  const mtype = (m.type as string | undefined) ?? "?";
                  const desc = (m.description as string | undefined) ?? "(no description)";
                  const score = (m.relevance_score as number | undefined) ?? 0;
                  const feature = (m.source_feature as string | undefined) ?? "";
                  const agent = (m.agent as string | undefined) ?? "";
                  const isOpen = expanded.has(i);
                  const detailLines = getDetailLines(m);
                  // M18 fix: wrap the row + detail row in a keyed <Fragment>
                  // so React can track them across re-renders. Previously
                  // the outer wrapper was a keyless `<>` inside a `.map()`
                  // which triggers React's "unique key" warning and can
                  // cause the wrong detail row to expand after filtering.
                  const rowKey = (m.id as string | undefined) ?? `row-${i}`;
                  return (
                    <Fragment key={rowKey}>
                      <tr style={{ cursor: "pointer" }} onClick={() => toggleExpand(i)}>
                        <td style={{ color: "#8b949e", textAlign: "center" }}>
                          {isOpen ? "▼" : "▶"}
                        </td>
                        <td>
                          {TYPE_ICON[mtype] ?? "❓"} {TYPE_LABEL[mtype] ?? mtype}
                        </td>
                        <td style={{ maxWidth: 500, wordBreak: "break-word" }}>
                          {desc.length > 200 ? desc.slice(0, 200) + "…" : desc}
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
                      </tr>
                      {isOpen && detailLines.length > 0 && (
                        <tr>
                          <td></td>
                          <td colSpan={4} style={{ background: "#0d1117" }}>
                            <div style={{ padding: 8, fontSize: 12 }}>
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

    </div>
  );
}
