import { useEffect, useState } from "react";
import { api, type TraceabilityResponse, type TraceabilityRow } from "../api/client";
import DependencyGraph from "./DependencyGraph";

const STATUS_BADGE: Record<string, string> = {
  pass: "badge-success",
  fail: "badge-error",
  no_specs: "badge-error",
  no_evals: "badge-warn",
};

type View = "table" | "graph";

export default function RunTraceability({ runId }: { runId: string }) {
  const [data, setData] = useState<TraceabilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [view, setView] = useState<View>("table");

  useEffect(() => {
    setLoading(true);
    api.traceability(runId).then(setData).catch((e) => setError(String(e))).finally(() => setLoading(false));
  }, [runId]);

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  if (loading) return <div className="empty-state"><p>Loading traceability matrix...</p></div>;
  if (error) return <div className="card" style={{ borderColor: "#da3633" }}><code>{error}</code></div>;
  if (!data || data.rows.length === 0) return <div className="empty-state"><p>No requirements found.</p></div>;

  const passCount = data.rows.filter((r) => r.overall_status === "pass").length;
  const failCount = data.rows.filter((r) => r.overall_status === "fail").length;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <p style={{ color: "#8b949e", fontSize: 12, margin: 0 }}>
          {data.rows.length} requirements — <span style={{ color: "#3fb950" }}>{passCount} passing</span>
          {failCount > 0 && <>, <span style={{ color: "#f85149" }}>{failCount} failing</span></>}
        </p>
        <div style={{ display: "flex", gap: 4 }}>
          <button
            className={view === "table" ? "btn" : "btn btn-secondary"}
            onClick={() => setView("table")}
            style={{ fontSize: 11, padding: "4px 10px" }}
          >
            Table
          </button>
          <button
            className={view === "graph" ? "btn" : "btn btn-secondary"}
            onClick={() => setView("graph")}
            style={{ fontSize: 11, padding: "4px 10px" }}
          >
            Graph
          </button>
        </div>
      </div>

      {view === "graph" && <DependencyGraph runId={runId} />}

      {view === "table" && (
      <table className="table" style={{ fontSize: 14 }}>
        <thead>
          <tr>
            <th style={{ width: 24 }}></th>
            <th>Requirement</th>
            <th>Priority</th>
            <th>Specs</th>
            <th>Files</th>
            <th>Tests</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row: TraceabilityRow) => {
            const reqId = row.requirement.id;
            const isOpen = expanded.has(reqId);
            const totalFiles = row.specs.reduce((s, sp) => s + sp.files.length, 0);
            const totalTests = row.specs.reduce((s, sp) => s + sp.test_files.length, 0);
            return (
              <tr key={reqId} style={{ cursor: "pointer" }} onClick={() => toggle(reqId)}>
                <td style={{ color: "#8b949e" }}>{isOpen ? "▼" : "▶"}</td>
                <td>
                  <code>{reqId}</code>
                  <div style={{ color: "#c9d1d9", fontSize: 14 }}>{row.requirement.title ?? ""}</div>
                  {isOpen && row.specs.map((sp) => (
                    <div key={sp.id} style={{ marginTop: 8, padding: 10, background: "#0d1117", borderRadius: 4, fontSize: 15 }}>
                      <div style={{ color: "#58a6ff" }}>
                        <code>{sp.id}</code> — {sp.title ?? ""}
                        {sp.capability && <span style={{ color: "#8b949e" }}> ({sp.capability})</span>}
                      </div>
                      {Object.keys(sp.eval_scores).length > 0 && (
                        <div style={{ marginTop: 6 }}>
                          <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 2 }}>Evaluations</div>
                          <ul style={{ margin: 0, paddingLeft: 16, listStyle: "disc" }}>
                            {Object.entries(sp.eval_scores).map(([metric, score]) => (
                              <li key={metric} style={{ fontSize: 13, color: "#c9d1d9", marginBottom: 2 }}>
                                <span style={{ color: "#8b949e" }}>{metric}:</span>{" "}
                                <span className={score >= 0.5 ? "badge-success" : "badge-error"}>
                                  {(score as number).toFixed(2)}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {sp.files.length > 0 && (
                        <div style={{ marginTop: 6 }}>
                          <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 2 }}>Files</div>
                          <ul style={{ margin: 0, paddingLeft: 16, listStyle: "disc" }}>
                            {sp.files.map((f) => {
                              const path = typeof f === "string" ? f : f.path;
                              const url = typeof f === "string" ? undefined : (f.s3_url || f.preview_url);
                              return (
                                <li key={path} style={{ fontSize: 13, color: "#c9d1d9", marginBottom: 2 }}>
                                  {url ? (
                                    <a
                                      href={url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      style={{ color: "#58a6ff" }}
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      <code>{path}</code>
                                    </a>
                                  ) : (
                                    <code>{path}</code>
                                  )}
                                </li>
                              );
                            })}
                          </ul>
                        </div>
                      )}
                      {sp.test_files.length > 0 && (
                        <div style={{ marginTop: 6 }}>
                          <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 2 }}>Tests</div>
                          <ul style={{ margin: 0, paddingLeft: 16, listStyle: "disc" }}>
                            {sp.test_files.map((f) => {
                              const path = typeof f === "string" ? f : f.path;
                              const url = typeof f === "string" ? undefined : (f.s3_url || f.preview_url);
                              return (
                                <li key={path} style={{ fontSize: 13, color: "#c9d1d9", marginBottom: 2 }}>
                                  {url ? (
                                    <a
                                      href={url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      style={{ color: "#58a6ff" }}
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      <code>{path}</code>
                                    </a>
                                  ) : (
                                    <code>{path}</code>
                                  )}
                                </li>
                              );
                            })}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </td>
                <td>
                  <span className={
                    row.requirement.priority === "high" || row.requirement.priority === "critical"
                      ? "badge-error"
                      : row.requirement.priority === "medium"
                      ? "badge-warn"
                      : "badge-info"
                  }>
                    {row.requirement.priority ?? "—"}
                  </span>
                </td>
                <td>{row.specs.length}</td>
                <td>{totalFiles}</td>
                <td>{totalTests}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <span className={STATUS_BADGE[row.overall_status] ?? "badge-warn"}>
                    {row.overall_status}
                  </span>
                  {row.overall_status !== "pass" && (() => {
                    const failingSpecs = row.specs.filter((s) => s.all_passed === false);
                    const noEvalSpecs = row.specs.filter((s) => s.all_passed === null);
                    const failingMetrics = failingSpecs.flatMap((s) =>
                      Object.entries(s.eval_scores)
                        .filter(([, score]) => (score as number) < 0.5)
                        .map(([metric, score]) => `${s.id}: ${metric} = ${(score as number).toFixed(2)}`)
                    );
                    const reasons: string[] = [];
                    if (row.overall_status === "no_specs") {
                      reasons.push("No specs implement this requirement");
                    } else if (row.overall_status === "no_evals") {
                      reasons.push("Specs exist but have no evaluation scores");
                    } else {
                      if (failingSpecs.length > 0) {
                        reasons.push(`${failingSpecs.length} spec(s) with failing evals:`);
                        reasons.push(...failingMetrics);
                      }
                      if (noEvalSpecs.length > 0) {
                        reasons.push(`${noEvalSpecs.length} spec(s) with no evals`);
                      }
                    }
                    return (
                      <span
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          marginLeft: 4,
                          position: "relative",
                          display: "inline-block",
                        }}
                      >
                        <span
                          style={{ cursor: "help", color: "#8b949e", fontSize: 12 }}
                          className="trace-info-trigger"
                        >
                          ⓘ
                        </span>
                        <span className="trace-info-popup">
                          {reasons.map((r, i) => (
                            <div key={i}>{r}</div>
                          ))}
                        </span>
                      </span>
                    );
                  })()}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      )}
    </div>
  );
}
