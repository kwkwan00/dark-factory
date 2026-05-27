import { useEffect, useState } from "react";
import { api, type HistoryRun, type RunCompareResponse } from "../api/client";
import RunDiff from "./RunDiff";

function Stat({ label, a, b, format, better }: {
  label: string;
  a: unknown;
  b: unknown;
  format?: (v: unknown) => string;
  better?: "higher" | "lower";
}) {
  const fmt = format ?? String;
  const numA = typeof a === "number" ? a : null;
  const numB = typeof b === "number" ? b : null;
  let deltaColor = "#8b949e";
  if (better && numA != null && numB != null) {
    const improved = better === "higher" ? numB > numA : numB < numA;
    const regressed = better === "higher" ? numB < numA : numB > numA;
    deltaColor = improved ? "#3fb950" : regressed ? "#f85149" : "#8b949e";
  }
  return (
    <tr>
      <td style={{ color: "#8b949e", fontSize: 12, padding: "4px 12px 4px 0" }}>{label}</td>
      <td style={{ fontSize: 13, padding: "4px 12px" }}>{fmt(a)}</td>
      <td style={{ fontSize: 13, padding: "4px 12px", color: deltaColor }}>{fmt(b)}</td>
    </tr>
  );
}

export default function RunCompareTab({ runId }: { runId: string }) {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [compareWith, setCompareWith] = useState<string>("");
  const [data, setData] = useState<RunCompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showDiff, setShowDiff] = useState(false);

  // Load run history for the picker
  useEffect(() => {
    api.history(50).then((r) => {
      // Exclude the current run from the picker
      setRuns(r.runs.filter((h) => h.id !== runId));
    }).catch(() => {});
  }, [runId]);

  // Load comparison when a run is selected
  useEffect(() => {
    if (!compareWith) { setData(null); return; }
    setLoading(true);
    setError("");
    setShowDiff(false);
    api.runCompare(runId, compareWith)
      .then(setData)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [runId, compareWith]);

  if (showDiff && compareWith) {
    return <RunDiff runA={runId} runB={compareWith} onClose={() => setShowDiff(false)} />;
  }

  const pct = (v: unknown) => typeof v === "number" ? `${Math.round(v * 100)}%` : "—";
  const sec = (v: unknown) => typeof v === "number" ? `${(v as number).toFixed(1)}s` : "—";
  const num = (v: unknown) => v != null ? String(v) : "—";

  return (
    <div>
      {/* Run picker */}
      <div style={{ marginBottom: 16, display: "flex", gap: 12, alignItems: "center" }}>
        <label style={{ color: "#8b949e", fontSize: 12 }}>Compare with:</label>
        <select
          value={compareWith}
          onChange={(e) => setCompareWith(e.target.value)}
          style={{
            background: "#0d1117",
            border: "1px solid #30363d",
            borderRadius: 6,
            color: "#e6edf3",
            padding: "6px 12px",
            fontSize: 12,
            minWidth: 300,
          }}
        >
          <option value="">Select a run...</option>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} — {r.status ?? "?"} — {r.timestamp ? new Date(r.timestamp).toLocaleDateString() : ""}
            </option>
          ))}
        </select>
        {data && (
          <button className="btn btn-secondary" onClick={() => setShowDiff(true)} style={{ fontSize: 11 }}>
            View File Diffs
          </button>
        )}
      </div>

      {loading && <div className="empty-state"><p>Loading comparison...</p></div>}
      {error && <div className="card" style={{ borderColor: "#da3633" }}><code>{error}</code></div>}

      {data && (() => {
        const a = data.run_a as Record<string, unknown>;
        const b = data.run_b as Record<string, unknown>;

        const epsA = (a.episodes ?? []) as Array<Record<string, unknown>>;
        const epsB = (b.episodes ?? []) as Array<Record<string, unknown>>;
        const allFeatures = [...new Set([
          ...epsA.map((e) => String(e.feature)),
          ...epsB.map((e) => String(e.feature)),
        ])].sort();

        return (
          <>
            {/* Summary stats */}
            <div className="card">
              <div className="card-title">Summary</div>
              <table>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", fontSize: 11, color: "#8b949e" }}></th>
                    <th style={{ textAlign: "left", fontSize: 11, color: "#8b949e", padding: "4px 12px" }}>This run</th>
                    <th style={{ textAlign: "left", fontSize: 11, color: "#8b949e", padding: "4px 12px" }}>
                      {compareWith.length > 30 ? compareWith.slice(0, 30) + "…" : compareWith}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <Stat label="Status" a={a.status} b={b.status} />
                  <Stat label="Pass rate" a={a.pass_rate} b={b.pass_rate} format={pct} better="higher" />
                  <Stat label="Duration" a={a.duration_seconds} b={b.duration_seconds} format={sec} better="lower" />
                  <Stat label="Specs" a={a.spec_count} b={b.spec_count} format={num} />
                  <Stat label="Features" a={a.feature_count} b={b.feature_count} format={num} />
                </tbody>
              </table>
            </div>

            {/* Per-feature comparison */}
            {allFeatures.length > 0 && (
              <div className="card">
                <div className="card-title">Per-Feature Status</div>
                <table className="table" style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>This run</th>
                      <th>Compare</th>
                      <th>Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allFeatures.map((feat) => {
                      const epA = epsA.find((e) => e.feature === feat);
                      const epB = epsB.find((e) => e.feature === feat);
                      const statusA = String(epA?.outcome ?? "—");
                      const statusB = String(epB?.outcome ?? "—");
                      const changed = statusA !== statusB;
                      return (
                        <tr key={feat}>
                          <td><code>{feat}</code></td>
                          <td>
                            <span className={statusA === "success" ? "badge-success" : statusA === "partial" ? "badge-warn" : "badge-error"}>
                              {statusA}
                            </span>
                          </td>
                          <td>
                            <span className={statusB === "success" ? "badge-success" : statusB === "partial" ? "badge-warn" : "badge-error"}>
                              {statusB}
                            </span>
                          </td>
                          <td style={{ color: changed ? (statusB === "success" ? "#3fb950" : "#f85149") : "#8b949e" }}>
                            {changed ? `${statusA} → ${statusB}` : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {!allFeatures.length && !loading && (
              <p style={{ color: "#8b949e", fontSize: 12 }}>No per-feature episode data available for comparison.</p>
            )}
          </>
        );
      })()}

      {!compareWith && !loading && (
        <div className="empty-state">
          <p>Select a previous run from the dropdown above to compare against this run.</p>
        </div>
      )}
    </div>
  );
}
