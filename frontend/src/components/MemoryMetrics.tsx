import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  useMetricsMemory,
  useMetricsMemoryActivity,
} from "../hooks/useDashboard";
import { fmtNumber, fmtPct, fmtSeconds } from "../lib/format";

// ── Constants ────────────────────────────────────────────────────────────────

const MEMORY_TYPE_COLORS: Record<string, string> = {
  Pattern: "#58a6ff",
  Mistake: "#f85149",
  Solution: "#3fb950",
  Strategy: "#d2a8ff",
  Episode: "#ffa657",
};

// ── Memory activity section ──────────────────────────────────────────────────

export function MemoryActivitySection() {
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

// ── Memory graph section ─────────────────────────────────────────────────────

export function MemoryGraphSection() {
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
        Procedural memory node counts, relevance distribution, most-recalled
        workhorses, and the feedback loop's boost-rate over the last 7 days.
        Use these to tune <code>memory_dedup_threshold</code> and decide when
        the graph needs maintenance.
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
              style={{ color: MEMORY_TYPE_COLORS[label] || "#58a6ff" }}
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
                    color: MEMORY_TYPE_COLORS[label] || "#c9d1d9",
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
                      fill={MEMORY_TYPE_COLORS[label] || "#58a6ff"}
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
                          MEMORY_TYPE_COLORS[
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
