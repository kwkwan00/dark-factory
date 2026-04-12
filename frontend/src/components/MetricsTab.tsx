import {
  Component,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { LlmUsageBucket } from "../api/client";
import {
  useMetricsAgentStats,
  useMetricsArtifacts,
  useMetricsBackgroundLoop,
  useMetricsCostRollup,
  useMetricsDecomposition,
  useMetricsLlmUsage,
  useMetricsMemory,
  useMetricsMemoryActivity,
  useMetricsQuality,
  useMetricsSummary,
  useMetricsToolCalls,
} from "../hooks/useDashboard";

function fmtNumber(value: number | null | undefined, digits = 0): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

function fmtSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 1) return `${Math.round(value * 1000)}ms`;
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

function fmtTimestamp(ts: string | null): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function fmtCost(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 0.01) return `$${(value * 100).toFixed(3)}¢`;
  return `$${value.toFixed(2)}`;
}

function fmtBytes(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Per-section error boundary ─────────────────────────────────────────────
//
// The Metrics tab has ~15 independent sections, each backed by its own
// fetch hook. Without per-section isolation a single failing query (e.g.
// Postgres unreachable) crashes the entire tab under the root
// ErrorBoundary in App.tsx. SectionBoundary catches render-time errors
// in its child subtree and shows a compact inline error banner, leaving
// the other sections usable.

interface SectionBoundaryProps {
  name: string;
  children: ReactNode;
}

interface SectionBoundaryState {
  error: Error | null;
}

class SectionBoundary extends Component<SectionBoundaryProps, SectionBoundaryState> {
  state: SectionBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): SectionBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Log to the browser console for debugging; the user sees the
    // inline error banner.
    console.error(`MetricsTab section "${this.props.name}" crashed`, error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          className="card"
          style={{ borderColor: "#da363340" }}
          role="alert"
          aria-live="polite"
        >
          <div className="card-title" style={{ color: "#f85149" }}>
            {this.props.name} — error
          </div>
          <code style={{ fontSize: 12, color: "#8b949e" }}>
            {this.state.error.message}
          </code>
          <div style={{ marginTop: 8 }}>
            <button
              className="btn btn-secondary"
              onClick={() => this.setState({ error: null })}
              style={{ fontSize: 11, padding: "4px 10px" }}
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Summary cards (extended with cost + incidents + decomposition) ──────────

function SummaryCards() {
  const { state, refresh } = useMetricsSummary();

  if (state.status === "loading") {
    return (
      <div className="empty-state">
        <p>Loading metrics summary…</p>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <code>{state.error}</code>
      </div>
    );
  }

  if (state.status !== "done") return null;

  const summary = state.data;
  if (!summary.enabled) {
    return (
      <div className="card" style={{ borderColor: "#30363d" }}>
        <div className="card-title">Metrics store not enabled</div>
        <p style={{ color: "#8b949e", margin: 0, fontSize: 13 }}>
          Set <code>POSTGRES_ENABLED=true</code> and point{" "}
          <code>POSTGRES_URL</code> at a reachable Postgres instance, then
          restart the app.
        </p>
      </div>
    );
  }

  const runs = summary.runs ?? {};
  const llm = summary.llm ?? {};
  const evals = summary.evals ?? {};
  const incidents = summary.incidents ?? {};
  const decomp = summary.decomposition ?? {};

  return (
    <>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div className="card-title" style={{ margin: 0 }}>
          Metrics Summary
        </div>
        <button className="btn btn-secondary" onClick={() => void refresh()}>
          Refresh
        </button>
      </div>

      <div className="result-grid">
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(runs.total_runs)}</div>
          <div className="stat-label">Pipeline runs</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtPct(runs.avg_pass_rate)}</div>
          <div className="stat-label">Avg pass rate</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtSeconds(runs.avg_duration_seconds)}</div>
          <div className="stat-label">Avg run duration</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(llm.total_calls)}</div>
          <div className="stat-label">LLM calls</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {fmtNumber((llm.input_tokens ?? 0) + (llm.output_tokens ?? 0))}
          </div>
          <div className="stat-label">Tokens</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtCost(llm.total_cost_usd)}</div>
          <div className="stat-label">Total cost</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtSeconds(llm.avg_latency_seconds)}</div>
          <div className="stat-label">Avg LLM latency</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {evals.avg_score != null ? evals.avg_score.toFixed(2) : "—"}
          </div>
          <div className="stat-label">Avg eval score</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(incidents.open_incidents)}</div>
          <div className="stat-label">Open incidents</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(decomp.total_sub_specs)}</div>
          <div className="stat-label">Sub-specs planned</div>
        </div>
      </div>
    </>
  );
}

// ── Quality section (first-attempt pass rate + per-metric rollup) ───────────

function QualitySection() {
  const { state } = useMetricsQuality();
  if (state.status !== "done" || !state.data.enabled) return null;
  const d = state.data;

  return (
    <div className="card">
      <div className="card-title">Outcome quality</div>
      <div className="result-grid" style={{ marginBottom: 12 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtPct(d.first_attempt_pass_rate)}</div>
          <div className="stat-label">First-attempt pass rate</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {d.mean_attempts_to_pass != null
              ? d.mean_attempts_to_pass.toFixed(2)
              : "—"}
          </div>
          <div className="stat-label">Mean attempts to pass</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.total_requirements)}</div>
          <div className="stat-label">Requirements evaluated</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.passed_requirements)}</div>
          <div className="stat-label">Requirements passed</div>
        </div>
      </div>

      {d.per_metric.length === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>No per-metric data yet.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Total</th>
              <th>Pass rate</th>
              <th>Avg score</th>
              <th>Min</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody>
            {d.per_metric.map((m) => (
              <tr key={m.metric_name}>
                <td>
                  <strong>{m.metric_name}</strong>
                </td>
                <td>{fmtNumber(m.total)}</td>
                <td>{fmtPct(m.pass_rate)}</td>
                <td>{m.avg_score != null ? m.avg_score.toFixed(2) : "—"}</td>
                <td style={{ color: "#8b949e" }}>
                  {m.min_score != null ? m.min_score.toFixed(2) : "—"}
                </td>
                <td style={{ color: "#8b949e" }}>
                  {m.max_score != null ? m.max_score.toFixed(2) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Cost rollup section ─────────────────────────────────────────────────────

function CostRollupSection() {
  const { state } = useMetricsCostRollup();
  if (state.status !== "done" || !state.data.enabled) return null;
  const d = state.data;

  return (
    <div className="card">
      <div className="card-title">Cost rollup</div>
      {d.per_model.length === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>
          No LLM calls recorded yet. Cost is computed per model once calls arrive.
        </p>
      ) : (
        <>
          <div style={{ marginBottom: 16 }}>
            <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 4 }}>
              By model
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={d.per_model.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                <XAxis
                  dataKey="model"
                  stroke="#8b949e"
                  tick={{ fontSize: 10 }}
                  interval={0}
                  angle={-12}
                  height={40}
                />
                <YAxis
                  stroke="#8b949e"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => `$${Number(v).toFixed(2)}`}
                />
                <Tooltip
                  contentStyle={{
                    background: "#161b22",
                    border: "1px solid #30363d",
                    borderRadius: 6,
                    color: "#e6edf3",
                    fontSize: 12,
                  }}
                  formatter={(v) => [fmtCost(v as number), "Cost"]}
                />
                <Bar dataKey="total_cost_usd" fill="#1f6feb" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <table className="table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>Model</th>
                <th>Calls</th>
                <th>Input tokens</th>
                <th>Output tokens</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {d.per_model.map((m) => (
                <tr key={m.model}>
                  <td>
                    <code>{m.model}</code>
                  </td>
                  <td>{fmtNumber(m.calls)}</td>
                  <td>{fmtNumber(m.input_tokens)}</td>
                  <td>{fmtNumber(m.output_tokens)}</td>
                  <td>
                    <strong>{fmtCost(m.total_cost_usd)}</strong>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ── Agent stats section ─────────────────────────────────────────────────────

function AgentStatsSection() {
  const { state } = useMetricsAgentStats();
  if (state.status !== "done" || !state.data.enabled) return null;

  return (
    <div className="card">
      <div className="card-title">Agent stats (rollup)</div>
      {state.data.agents.length === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>No agent data yet.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Agent</th>
              <th>Activations</th>
              <th>Tool calls</th>
              <th>Decisions</th>
              <th>Handoffs in</th>
              <th>Handoffs out</th>
            </tr>
          </thead>
          <tbody>
            {state.data.agents.map((a, i) => (
              <tr key={`${a.agent}-${i}`}>
                <td>
                  <code>{a.agent}</code>
                </td>
                <td>{fmtNumber(a.activations)}</td>
                <td>{fmtNumber(a.tool_calls)}</td>
                <td>{fmtNumber(a.decisions)}</td>
                <td>{fmtNumber(a.handoffs_in)}</td>
                <td>{fmtNumber(a.handoffs_out)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Tool call stats section ─────────────────────────────────────────────────

function ToolCallsSection() {
  const [groupBy, setGroupBy] = useState<"tool" | "agent" | "feature">("tool");
  const { state, refresh } = useMetricsToolCalls(groupBy);
  if (state.status !== "done" || !state.data.enabled) return null;

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div className="card-title" style={{ margin: 0 }}>
          Tool calls by {groupBy}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <select
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value as typeof groupBy)}
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "6px 10px",
              fontSize: 13,
            }}
          >
            <option value="tool">By tool</option>
            <option value="agent">By agent</option>
            <option value="feature">By feature</option>
          </select>
          <button className="btn btn-secondary" onClick={() => void refresh()}>
            Refresh
          </button>
        </div>
      </div>

      {state.data.buckets.length === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>No tool call data yet.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>{groupBy}</th>
              <th>Calls</th>
              <th>Successes</th>
              <th>Failures</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {state.data.buckets.map((b, i) => (
              <tr key={`${b.bucket}-${i}`}>
                <td>
                  <code>{b.bucket ?? "(null)"}</code>
                </td>
                <td>{fmtNumber(b.calls)}</td>
                <td style={{ color: "#3fb950" }}>{fmtNumber(b.successes)}</td>
                <td style={{ color: "#f85149" }}>{fmtNumber(b.failures)}</td>
                <td>{fmtSeconds(b.avg_latency_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Memory activity section ─────────────────────────────────────────────────

function MemoryActivitySection() {
  const { state } = useMetricsMemoryActivity();
  if (state.status !== "done" || !state.data.enabled) return null;
  const d = state.data;

  return (
    <div className="card">
      <div className="card-title">Memory activity</div>
      <div className="result-grid" style={{ marginBottom: 12 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.recall_hits)}</div>
          <div className="stat-label">Recall hits</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.recall_misses)}</div>
          <div className="stat-label">Recall misses</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.created)}</div>
          <div className="stat-label">Memories created</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.boosts)}</div>
          <div className="stat-label">Boosts</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.demotes)}</div>
          <div className="stat-label">Demotes</div>
        </div>
      </div>
      {d.per_operation.length === 0 ? null : (
        <table className="table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Operation</th>
              <th>Count</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {d.per_operation.map((op) => (
              <tr key={op.operation}>
                <td>
                  <code>{op.operation}</code>
                </td>
                <td>{fmtNumber(op.count)}</td>
                <td>{fmtSeconds(op.avg_latency_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Memory graph (Tier A: dedup + relevance + recall observability) ────────

const _MEMORY_TYPE_COLORS: Record<string, string> = {
  Pattern: "#58a6ff",
  Mistake: "#f85149",
  Solution: "#3fb950",
  Strategy: "#d2a8ff",
  Episode: "#ffa657",
};

function MemoryGraphSection() {
  const { state } = useMetricsMemory();
  if (state.status === "loading") {
    return (
      <div className="card">
        <div className="card-title">Memory graph</div>
        <p style={{ color: "#8b949e", fontSize: 12 }}>Loading…</p>
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <div className="card-title" style={{ color: "#f85149" }}>
          Memory graph
        </div>
        <code style={{ fontSize: 11 }}>{state.error}</code>
      </div>
    );
  }
  if (state.status !== "done") return null;
  const d = state.data;
  if (!d.enabled) {
    return (
      <div className="card">
        <div className="card-title">Memory graph</div>
        <p style={{ color: "#8b949e", fontSize: 12 }}>
          {d.reason || "memory store disabled"}
        </p>
      </div>
    );
  }

  const types = Object.entries(d.counts_by_type);
  const totalNodes = types.reduce((acc, [, s]) => acc + s.count, 0);
  const eff = d.recall_effectiveness;

  return (
    <div className="card">
      <div className="card-title">Memory graph</div>
      <p
        style={{
          color: "#8b949e",
          fontSize: 11,
          margin: "0 0 12px",
        }}
      >
        Tier A observability: procedural memory node counts, relevance
        distribution, most-recalled workhorses, and the feedback loop's
        boost-rate over the last 7 days. Use these to tune{" "}
        <code>memory_dedup_threshold</code> and decide when the graph
        needs maintenance.
      </p>

      {/* Counts + summary KPIs */}
      <div className="result-grid" style={{ marginBottom: 16 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(totalNodes)}</div>
          <div className="stat-label">Total nodes</div>
        </div>
        {types.map(([label, stats]) => (
          <div key={label} className="stat-card">
            <div
              className="stat-value"
              style={{ color: _MEMORY_TYPE_COLORS[label] || "#58a6ff" }}
            >
              {fmtNumber(stats.count)}
            </div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Relevance histogram per non-Episode type */}
      <div
        style={{
          fontSize: 11,
          color: "#58a6ff",
          textTransform: "uppercase",
          letterSpacing: 0.5,
          marginBottom: 8,
        }}
      >
        Relevance distribution
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 12,
          marginBottom: 16,
        }}
      >
        {types
          .filter(([label, stats]) => label !== "Episode" && stats.count > 0)
          .map(([label, stats]) => {
            const data = stats.histogram.map((count, bucket) => ({
              bucket: `${(bucket / 10).toFixed(1)}-${((bucket + 1) / 10).toFixed(1)}`,
              count,
            }));
            return (
              <div
                key={label}
                style={{
                  background: "#0d1117",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  padding: 10,
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    color: _MEMORY_TYPE_COLORS[label] || "#c9d1d9",
                    marginBottom: 4,
                    fontWeight: 600,
                  }}
                >
                  {label} · mean {stats.mean_relevance.toFixed(2)}
                </div>
                <ResponsiveContainer width="100%" height={80}>
                  <BarChart
                    data={data}
                    margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
                  >
                    <XAxis
                      dataKey="bucket"
                      tick={{ fill: "#6e7681", fontSize: 9 }}
                      interval={1}
                    />
                    <YAxis
                      tick={{ fill: "#6e7681", fontSize: 9 }}
                      width={30}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#161b22",
                        border: "1px solid #30363d",
                        fontSize: 11,
                      }}
                    />
                    <Bar
                      dataKey="count"
                      fill={_MEMORY_TYPE_COLORS[label] || "#58a6ff"}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
          })}
      </div>

      {/* Recall effectiveness KPIs */}
      <div
        style={{
          fontSize: 11,
          color: "#58a6ff",
          textTransform: "uppercase",
          letterSpacing: 0.5,
          marginBottom: 8,
        }}
      >
        Recall effectiveness ({eff.window_days}d window)
      </div>
      <div className="result-grid" style={{ marginBottom: 16 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(eff.total_recalls)}</div>
          <div className="stat-label">Recalls</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: "#3fb950" }}>
            {fmtNumber(eff.boosted)}
          </div>
          <div className="stat-label">Boosted</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: "#f85149" }}>
            {fmtNumber(eff.demoted)}
          </div>
          <div className="stat-label">Demoted</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtPct(eff.boost_rate)}</div>
          <div className="stat-label">Boost rate</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(eff.decays)}</div>
          <div className="stat-label">Decays</div>
        </div>
      </div>

      {/* Top 10 most-recalled memories */}
      {d.top_recalled.length > 0 && (
        <>
          <div
            style={{
              fontSize: 11,
              color: "#58a6ff",
              textTransform: "uppercase",
              letterSpacing: 0.5,
              marginBottom: 8,
            }}
          >
            Top 10 most-recalled memories
          </div>
          <table className="table" style={{ fontSize: 11 }}>
            <thead>
              <tr>
                <th>Type</th>
                <th>Description</th>
                <th>Feature</th>
                <th style={{ textAlign: "right" }}>Recalls</th>
                <th style={{ textAlign: "right" }}>Relevance</th>
              </tr>
            </thead>
            <tbody>
              {d.top_recalled.map((m) => (
                <tr key={m.id}>
                  <td>
                    <code
                      style={{
                        color:
                          _MEMORY_TYPE_COLORS[
                            m.memory_type.charAt(0).toUpperCase() +
                              m.memory_type.slice(1)
                          ] || "#8b949e",
                      }}
                    >
                      {m.memory_type}
                    </code>
                  </td>
                  <td
                    style={{
                      maxWidth: 400,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {m.description}
                  </td>
                  <td>
                    <code>{m.source_feature || "—"}</code>
                  </td>
                  <td style={{ textAlign: "right" }}>
                    {fmtNumber(m.times_recalled)}
                  </td>
                  <td style={{ textAlign: "right" }}>
                    {m.relevance_score.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}


// ── Decomposition stats section ─────────────────────────────────────────────

function DecompositionSection() {
  const { state } = useMetricsDecomposition();
  if (state.status !== "done" || !state.data.enabled) return null;
  const d = state.data;

  return (
    <div className="card">
      <div className="card-title">Decomposition planner</div>
      <div className="result-grid" style={{ marginBottom: 12 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.requirements_planned)}</div>
          <div className="stat-label">Requirements planned</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.total_sub_specs)}</div>
          <div className="stat-label">Sub-specs produced</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {d.summary.avg_sub_specs != null
              ? d.summary.avg_sub_specs.toFixed(1)
              : "—"}
          </div>
          <div className="stat-label">Avg sub-specs/req</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.fallback_count)}</div>
          <div className="stat-label">Planner fallbacks</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.depends_on_resolved)}</div>
          <div className="stat-label">Deps resolved</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.depends_on_unresolved)}</div>
          <div className="stat-label">Deps unresolved</div>
        </div>
      </div>
    </div>
  );
}

// ── Artifact writes section ─────────────────────────────────────────────────

function ArtifactsSection() {
  const { state } = useMetricsArtifacts();
  if (state.status !== "done" || !state.data.enabled) return null;
  const d = state.data;

  return (
    <div className="card">
      <div className="card-title">Artifacts</div>
      <div className="result-grid" style={{ marginBottom: 12 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.files_written)}</div>
          <div className="stat-label">Files written</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.code_files)}</div>
          <div className="stat-label">Code files</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(d.summary.test_files)}</div>
          <div className="stat-label">Test files</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtBytes(d.summary.total_bytes)}</div>
          <div className="stat-label">Total bytes</div>
        </div>
      </div>
      {d.per_language.length > 0 && (
        <table className="table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Language</th>
              <th>Files</th>
              <th>Bytes</th>
            </tr>
          </thead>
          <tbody>
            {d.per_language.map((l) => (
              <tr key={l.language}>
                <td>
                  <code>{l.language}</code>
                </td>
                <td>{fmtNumber(l.files)}</td>
                <td>{fmtBytes(l.total_bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Background loop sample section ──────────────────────────────────────────

function BackgroundLoopSection() {
  const { state } = useMetricsBackgroundLoop(200);
  if (state.status !== "done" || !state.data.enabled) return null;

  // Chronological for the line chart
  const data = [...state.data.samples]
    .reverse()
    .map((s) => ({
      timestamp: s.timestamp,
      active: s.active_task_count,
      completed: s.completed_task_count,
    }));

  return (
    <div className="card">
      <div className="card-title">Background loop task samples</div>
      {data.length === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>
          No samples yet — the sampler runs every 10s.
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="timestamp"
              stroke="#8b949e"
              tick={{ fontSize: 10 }}
              tickFormatter={(t) => {
                try {
                  return new Date(t).toLocaleTimeString();
                } catch {
                  return t as string;
                }
              }}
            />
            <YAxis stroke="#8b949e" tick={{ fontSize: 10 }} width={28} />
            <Tooltip
              contentStyle={{
                background: "#161b22",
                border: "1px solid #30363d",
                borderRadius: 6,
                color: "#e6edf3",
                fontSize: 12,
              }}
              labelFormatter={(t) => fmtTimestamp(t as string)}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="active" stroke="#58a6ff" dot={false} />
            <Line type="monotone" dataKey="completed" stroke="#3fb950" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── LLM usage section ───────────────────────────────────────────────────────

function LlmUsageSection() {
  const [groupBy, setGroupBy] = useState<"model" | "phase" | "client">("model");
  const { state, refresh } = useMetricsLlmUsage(groupBy);

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div className="card-title" style={{ margin: 0 }}>
          LLM usage by {groupBy}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <select
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value as typeof groupBy)}
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "6px 10px",
              fontSize: 13,
            }}
          >
            <option value="model">By model</option>
            <option value="phase">By phase</option>
            <option value="client">By client</option>
          </select>
          <button className="btn btn-secondary" onClick={() => void refresh()}>
            Refresh
          </button>
        </div>
      </div>

      {state.status !== "done" || !state.data.enabled ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>No LLM usage data.</p>
      ) : state.data.buckets.length === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 13 }}>
          No LLM calls recorded yet.
        </p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>{groupBy}</th>
              <th>Calls</th>
              <th>Input tokens</th>
              <th>Output tokens</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {state.data.buckets.map((b: LlmUsageBucket) => (
              <tr key={b.bucket ?? "(null)"}>
                <td>
                  <code>{b.bucket ?? "(null)"}</code>
                </td>
                <td>{fmtNumber(b.calls)}</td>
                <td>{fmtNumber(b.input_tokens)}</td>
                <td>{fmtNumber(b.output_tokens)}</td>
                <td>{fmtSeconds(b.avg_latency_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Group header ────────────────────────────────────────────────────────────
//
// Small uppercase divider rendered between conceptual groups of
// sections in the tab root. Pure presentation — no data dependencies.

function GroupHeader({ label }: { label: string }) {
  return (
    <div
      style={{
        margin: "24px 0 8px",
        color: "#8b949e",
        fontSize: 11,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: 1.2,
        borderBottom: "1px solid #30363d",
        paddingBottom: 4,
      }}
    >
      {label}
    </div>
  );
}

// ── Tab root ────────────────────────────────────────────────────────────────
//
// Sections are arranged into five conceptual groups, scanning top-to-
// bottom from "what happened" down to "runtime internals":
//
//   1. Overview — big-picture KPIs + outcome pass rate
//   2. Cost & LLM efficiency — spend and per-model usage
//   3. Swarm behavior — what the agents + tools did
//   4. Pipeline output — planner decomposition and generated artifacts
//   5. System internals — memory ops and background-loop samples
//
// Each group is preceded by a GroupHeader divider.

export default function MetricsTab() {
  return (
    <div>
      {/* ── Overview ───────────────────────────────────────────────── */}
      <GroupHeader label="Overview" />
      <SectionBoundary name="Summary">
        <div className="card">
          <SummaryCards />
        </div>
      </SectionBoundary>
      <SectionBoundary name="Outcome quality">
        <QualitySection />
      </SectionBoundary>

      {/* ── Cost & LLM efficiency ─────────────────────────────────── */}
      <GroupHeader label="Cost & LLM efficiency" />
      <SectionBoundary name="Cost rollup">
        <CostRollupSection />
      </SectionBoundary>
      <SectionBoundary name="LLM usage">
        <LlmUsageSection />
      </SectionBoundary>

      {/* ── Swarm behavior ────────────────────────────────────────── */}
      <GroupHeader label="Swarm behavior" />
      <SectionBoundary name="Agent stats">
        <AgentStatsSection />
      </SectionBoundary>
      <SectionBoundary name="Tool calls">
        <ToolCallsSection />
      </SectionBoundary>

      {/* ── Pipeline output ───────────────────────────────────────── */}
      <GroupHeader label="Pipeline output" />
      <SectionBoundary name="Decomposition">
        <DecompositionSection />
      </SectionBoundary>
      <SectionBoundary name="Artifacts">
        <ArtifactsSection />
      </SectionBoundary>

      {/* ── System internals ──────────────────────────────────────── */}
      <GroupHeader label="System internals" />
      <SectionBoundary name="Memory graph">
        <MemoryGraphSection />
      </SectionBoundary>
      <SectionBoundary name="Memory activity">
        <MemoryActivitySection />
      </SectionBoundary>
      <SectionBoundary name="Background loop">
        <BackgroundLoopSection />
      </SectionBoundary>
    </div>
  );
}
