import { useEffect, useState } from "react";
import { api, type RunCompareResponse } from "../api/client";
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

export default function RunCompare({
  runA,
  runB,
  onClose,
}: {
  runA: string;
  runB: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<RunCompareResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showDiff, setShowDiff] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.runCompare(runA, runB).then(setData).catch((e) => setError(String(e))).finally(() => setLoading(false));
  }, [runA, runB]);

  if (loading) return <div className="empty-state"><p>Loading comparison...</p></div>;
  if (error) return <div className="card" style={{ borderColor: "#da3633" }}><code>{error}</code></div>;
  if (!data) return null;

  if (showDiff) return <RunDiff runA={runA} runB={runB} onClose={() => setShowDiff(false)} />;

  const a = data.run_a as Record<string, unknown>;
  const b = data.run_b as Record<string, unknown>;

  const pct = (v: unknown) => typeof v === "number" ? `${Math.round(v * 100)}%` : "—";
  const sec = (v: unknown) => typeof v === "number" ? `${(v as number).toFixed(1)}s` : "—";
  const num = (v: unknown) => v != null ? String(v) : "—";

  const epsA = (a.episodes ?? []) as Array<Record<string, unknown>>;
  const epsB = (b.episodes ?? []) as Array<Record<string, unknown>>;
  const allFeatures = [...new Set([
    ...epsA.map((e) => String(e.feature)),
    ...epsB.map((e) => String(e.feature)),
  ])].sort();

  return (
    <div>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div className="card-title" style={{ margin: 0 }}>Run Comparison</div>
            <p style={{ color: "#8b949e", fontSize: 12, margin: "4px 0 0" }}>
              <code>{runA}</code> vs <code>{runB}</code>
            </p>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary" onClick={() => setShowDiff(true)}>View File Diffs</button>
            <button className="btn btn-secondary" onClick={onClose}>Close</button>
          </div>
        </div>
      </div>

      {/* Summary stats */}
      <div className="card">
        <div className="card-title">Summary</div>
        <table>
          <thead>
            <tr>
              <th style={{ textAlign: "left", fontSize: 11, color: "#8b949e" }}></th>
              <th style={{ textAlign: "left", fontSize: 11, color: "#8b949e", padding: "4px 12px" }}>Run A</th>
              <th style={{ textAlign: "left", fontSize: 11, color: "#8b949e", padding: "4px 12px" }}>Run B</th>
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
                <th>Run A</th>
                <th>Run B</th>
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
    </div>
  );
}
