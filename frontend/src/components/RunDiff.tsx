import { useEffect, useState } from "react";
import { api, type DiffFile, type RunDiffResponse } from "../api/client";

const STATUS_COLORS: Record<string, string> = {
  added: "#3fb950",
  removed: "#f85149",
  modified: "#d29922",
  unchanged: "#8b949e",
  binary: "#8b949e",
  error: "#f85149",
};

export default function RunDiff({
  runA,
  runB,
  onClose,
}: {
  runA: string;
  runB: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<RunDiffResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<DiffFile | null>(null);

  useEffect(() => {
    setLoading(true);
    api.runDiff(runA, runB).then(setData).catch((e) => setError(String(e))).finally(() => setLoading(false));
  }, [runA, runB]);

  if (loading) return <div className="empty-state"><p>Computing diffs...</p></div>;
  if (error) return <div className="card" style={{ borderColor: "#da3633" }}><code>{error}</code></div>;
  if (!data) return null;

  const changed = data.files.filter((f) => f.status !== "unchanged");

  return (
    <div>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div className="card-title" style={{ margin: 0 }}>File Diff</div>
            <p style={{ color: "#8b949e", fontSize: 12, margin: "4px 0 0" }}>
              <code>{runA}</code> vs <code>{runB}</code>
              {" — "}
              <span style={{ color: "#3fb950" }}>+{data.stats.added}</span>
              {" "}
              <span style={{ color: "#f85149" }}>-{data.stats.removed}</span>
              {" "}
              <span style={{ color: "#d29922" }}>~{data.stats.modified}</span>
              {" "}
              <span style={{ color: "#8b949e" }}>{data.stats.unchanged} unchanged</span>
            </p>
          </div>
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12, minHeight: 400 }}>
        {/* File list */}
        <div className="card" style={{ width: 280, flexShrink: 0, overflow: "auto", maxHeight: 600 }}>
          {changed.length === 0 ? (
            <p style={{ color: "#8b949e", fontSize: 12 }}>No differences found.</p>
          ) : (
            changed.map((f) => (
              <div
                key={f.path}
                onClick={() => setSelected(f)}
                style={{
                  padding: "4px 8px",
                  cursor: "pointer",
                  borderRadius: 4,
                  fontSize: 12,
                  background: selected?.path === f.path ? "#1c2128" : "transparent",
                  color: STATUS_COLORS[f.status] ?? "#c9d1d9",
                  borderLeft: `3px solid ${STATUS_COLORS[f.status] ?? "#30363d"}`,
                  marginBottom: 2,
                }}
              >
                <code>{f.path}</code>
              </div>
            ))
          )}
        </div>

        {/* Diff content */}
        <div className="card" style={{ flex: 1, overflow: "auto", maxHeight: 600 }}>
          {!selected ? (
            <p style={{ color: "#8b949e", fontSize: 12 }}>Select a file to view its diff.</p>
          ) : selected.status === "added" ? (
            <p style={{ color: "#3fb950", fontSize: 12 }}>New file in {runB}</p>
          ) : selected.status === "removed" ? (
            <p style={{ color: "#f85149", fontSize: 12 }}>Removed in {runB}</p>
          ) : selected.diff ? (
            <pre style={{ fontSize: 11, lineHeight: 1.5, margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
              {selected.diff.split("\n").map((line, i) => (
                <div
                  key={i}
                  style={{
                    background: line.startsWith("+") && !line.startsWith("+++")
                      ? "#1a3a2a"
                      : line.startsWith("-") && !line.startsWith("---")
                      ? "#3a1a1a"
                      : line.startsWith("@@")
                      ? "#1a1a3a"
                      : "transparent",
                    color: line.startsWith("+") && !line.startsWith("+++")
                      ? "#3fb950"
                      : line.startsWith("-") && !line.startsWith("---")
                      ? "#f85149"
                      : line.startsWith("@@")
                      ? "#bc8cff"
                      : "#c9d1d9",
                    padding: "0 4px",
                  }}
                >
                  {line}
                </div>
              ))}
            </pre>
          ) : (
            <p style={{ color: "#8b949e", fontSize: 12 }}>{selected.status}</p>
          )}
        </div>
      </div>
    </div>
  );
}
