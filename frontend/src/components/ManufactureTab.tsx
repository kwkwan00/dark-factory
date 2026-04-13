import { useEffect, useRef, useState, type DragEvent, type ChangeEvent } from "react";
import { useManufacture } from "../contexts/ManufactureContext";
import { type Step } from "../hooks/useAgentRun";
import { useHistory } from "../hooks/useDashboard";
import { openRunDetail } from "../lib/openRunDetail";

const STATUS_ICON: Record<Step["status"], string> = {
  pending: "○",
  running: "●",
  done: "✓",
  error: "✕",
};

// Two tiers of accepted file types, matching the backend allowlist in
// ``routes_upload.py``. Native formats (.md .txt .json .yaml .yml) are
// parsed directly by the ingest stage. Rich formats (Office docs,
// PDFs, transcripts, HTML, XML, RTF, CSV, logs) are routed through a
// clean-context Claude Agent SDK invocation that extracts discrete
// requirements per document.
const ACCEPT_NATIVE = ".md,.txt,.json,.yaml,.yml";
const ACCEPT_RICH =
  ".docx,.xlsx,.pptx,.pdf,.rtf,.html,.htm,.xml,.csv,.vtt,.srt,.log";
const ACCEPT = `${ACCEPT_NATIVE},${ACCEPT_RICH}`;

export default function ManufactureTab() {
  // All persistent state lives in ManufactureContext (App-level provider) so it
  // survives tab unmounts. Only ephemeral local UI state (drag highlight)
  // stays here.
  const {
    state,
    startRun,
    cancelRun,
    cancelling,
    path,
    setPath,
    uploading,
    uploadError,
    uploadedFiles,
    handleFiles,
    clearUploads,
  } = useManufacture();
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Run History — auto-loads on mount and polls every 5s while any run is 'running'
  const { state: historyState, load: loadHistory } = useHistory(10);
  const pollRef = useRef<number | null>(null);
  // M17 fix: guard against setState after unmount. useHistory owns its own
  // mountedRef for the fetch path, but the polling interval here lives in
  // this component's lifecycle and can fire one last tick after unmount if
  // the component is torn down between a scheduled interval and its handler.
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  // Refresh history immediately when the active run finishes
  useEffect(() => {
    if (state.status === "done" || state.status === "error") {
      void loadHistory();
    }
  }, [state.status, loadHistory]);

  useEffect(() => {
    if (historyState.status !== "done") return;
    const hasRunning = historyState.data.runs.some(
      (r) => (r.status as string | undefined) === "running",
    );
    if (!hasRunning) {
      if (pollRef.current !== null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current === null) {
      pollRef.current = window.setInterval(() => {
        if (!mountedRef.current) return;
        void loadHistory();
      }, 5000);
    }
    return () => {
      if (pollRef.current !== null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [historyState, loadHistory]);

  const handleStart = () => {
    void startRun(path.trim() || "./openspec");
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      void handleFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      void handleFiles(e.target.files);
    }
  };

  const isRunning = state.status === "running";
  const result = state.result;

  // Derive the currently-running pipeline stage name from the step tree
  // so the run history table can show "running · Reconciliation" etc.
  const currentStageName = state.steps.findLast(
    (s) => s.status === "running",
  )?.name;

  return (
    <div>
      {/* Drop zone */}
      <div
        className={`dropzone${dragActive ? " dropzone-active" : ""}${uploading ? " dropzone-uploading" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ACCEPT}
          onChange={handleFileInputChange}
          style={{ display: "none" }}
          disabled={isRunning || uploading}
        />
        <div className="dropzone-inner">
          <div className="dropzone-icon">{uploading ? "⬆" : "📄"}</div>
          <div className="dropzone-text">
            {uploading
              ? "Uploading..."
              : dragActive
              ? "Drop files to upload"
              : "Drag & drop requirement files here, or click to browse"}
          </div>
          <div className="dropzone-hint">
            Native:{" "}
            <code style={{ color: "#79c0ff" }}>
              {ACCEPT_NATIVE.replaceAll(",", " ")}
            </code>
            <br />
            Rich (deep-agent extraction):{" "}
            <code style={{ color: "#d2a8ff" }}>
              {ACCEPT_RICH.replaceAll(",", " ")}
            </code>
            <br />
            Max 25 MB per file · 150 MB per upload
          </div>
        </div>
      </div>

      {uploadError && (
        <div className="card" style={{ borderColor: "#da3633", marginTop: -8 }}>
          <code style={{ color: "#f85149" }}>{uploadError}</code>
        </div>
      )}

      {uploadedFiles.length > 0 && (
        <div className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8,
            }}
          >
            <div className="card-title" style={{ margin: 0 }}>
              Uploaded files ({uploadedFiles.length})
            </div>
            <button
              className="btn btn-secondary"
              onClick={clearUploads}
              disabled={isRunning}
            >
              Clear
            </button>
          </div>
          <ul style={{ margin: 0, paddingLeft: 20, color: "#8b949e", fontSize: 13 }}>
            {uploadedFiles.map((f) => (
              <li key={f}>
                <code>{f}</code>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Path input + Run button */}
      <div className="card">
        <div className="card-title">Manufacture</div>
        <div className="input-row">
          <input
            className="input-text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="Requirements path, e.g. ./openspec"
            disabled={isRunning}
            onKeyDown={(e) => e.key === "Enter" && !isRunning && handleStart()}
          />
          {isRunning ? (
            <button
              className="btn btn-danger"
              onClick={() => void cancelRun()}
              disabled={cancelling}
              aria-label="Cancel the in-flight pipeline run"
              title="Stop the pipeline — server-side kill-switch, sub-second halt latency"
            >
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          ) : (
            <button className="btn" onClick={handleStart} disabled={uploading}>
              Run
            </button>
          )}
        </div>

      </div>

      {/* ── Run History ─────────────────────────────────────────────── */}
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
            Run History
            {historyState.status === "done" && (
              <span style={{ color: "#8b949e", fontWeight: 400, marginLeft: 6 }}>
                ({historyState.data.runs.length})
              </span>
            )}
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => void loadHistory()}
          >
            Refresh
          </button>
        </div>
        {historyState.status === "loading" && (
          <div style={{ color: "#8b949e", fontSize: 13 }}>Loading…</div>
        )}
        {historyState.status === "error" && (
          <div style={{ color: "#f85149", fontSize: 13 }}>
            Failed to load history.
          </div>
        )}
        {historyState.status === "done" &&
          historyState.data.runs.length === 0 && (
            <div style={{ color: "#8b949e", fontSize: 13 }}>
              No previous runs. Upload requirements and click Run to start.
            </div>
          )}
        {historyState.status === "done" &&
          historyState.data.runs.length > 0 && (
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Started</th>
                  <th>Status</th>
                  <th>Pass Rate</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {historyState.data.runs.map((r, i) => {
                  const status = r.status;
                  const passRate = r.pass_rate;
                  const duration = r.duration_seconds;
                  const id = r.id ?? `run-${i}`;
                  const timestamp = r.timestamp;
                  const startedLabel = timestamp
                    ? new Date(timestamp).toLocaleString()
                    : "—";
                  return (
                    <tr key={id}>
                      <td>
                        <button
                          onClick={() => openRunDetail(id)}
                          title="Open run detail in a new window"
                          style={{
                            background: "transparent",
                            border: "none",
                            color: "#58a6ff",
                            cursor: "pointer",
                            padding: 0,
                            fontFamily:
                              "SF Mono, Monaco, Consolas, monospace",
                            fontSize: "inherit",
                            textDecoration: "underline",
                          }}
                        >
                          {id}
                        </button>
                      </td>
                      <td style={{ color: "#8b949e", whiteSpace: "nowrap" }}>
                        {startedLabel}
                      </td>
                      <td>
                        <span
                          className={
                            status === "success"
                              ? "badge-success"
                              : status === "partial"
                              ? "badge-warn"
                              : status === "running"
                              ? "badge-info"
                              : "badge-error"
                          }
                        >
                          {status === "running" && (
                            <span
                              style={{
                                display: "inline-block",
                                width: 6,
                                height: 6,
                                background: "#58a6ff",
                                borderRadius: "50%",
                                marginRight: 4,
                                animation: "pulse 1s infinite",
                              }}
                            />
                          )}
                          {status}
                          {status === "running" && isRunning && currentStageName && (
                            <span style={{ color: "#8b949e", fontWeight: 400 }}>
                              {" "}· {currentStageName}
                            </span>
                          )}
                        </span>
                      </td>
                      <td>
                        {passRate != null
                          ? `${Math.round(passRate * 100)}%`
                          : "—"}
                      </td>
                      <td style={{ color: "#8b949e" }}>
                        {duration != null ? `${duration.toFixed(1)}s` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
      </div>

      {/* Steps */}
      {state.steps.length > 0 && (
        <div className="card">
          <div className="card-title">Pipeline Steps</div>
          <ul className="step-list">
            {state.steps.map((step) => (
              <li key={step.id} className="step-item">
                <div className={`step-icon ${step.status}`}>
                  {STATUS_ICON[step.status]}
                </div>
                <div className="step-body">
                  <div className="step-name">{step.name}</div>
                  <div className="step-messages">
                    {step.messages.map((msg, i) => (
                      <div key={i} className="step-message">
                        {msg}
                      </div>
                    ))}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Error */}
      {state.status === "error" && state.error && (
        <div className="card" style={{ borderColor: "#da3633" }}>
          <div className="card-title" style={{ color: "#f85149" }}>
            Pipeline Error
          </div>
          <code>{state.error}</code>
        </div>
      )}

      {/* Result summary */}
      {result && (
        <div className="card">
          <div className="card-title">Pipeline Complete</div>
          <div className="result-grid">
            <div className="stat-card">
              <div className="stat-value">
                {result.completed_features?.length ?? 0}
              </div>
              <div className="stat-label">Features</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {result.all_artifacts?.length ?? 0}
              </div>
              <div className="stat-label">Artifacts</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {result.all_tests?.length ?? 0}
              </div>
              <div className="stat-label">Tests</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {result.pass_rate != null
                  ? `${Math.round(result.pass_rate * 100)}%`
                  : "—"}
              </div>
              <div className="stat-label">Pass Rate</div>
            </div>
          </div>

          {result.completed_features && result.completed_features.length > 0 && (
            <table className="table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Status</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {result.completed_features.map((f, i) => (
                  <tr key={i}>
                    <td>
                      <code>{f.feature}</code>
                    </td>
                    <td>
                      <span
                        className={
                          f.status === "success"
                            ? "badge-success"
                            : f.status === "error"
                            ? "badge-error"
                            : "badge-warn"
                        }
                      >
                        {f.status}
                      </span>
                    </td>
                    <td style={{ color: "#8b949e", fontSize: "12px" }}>
                      {f.error ?? ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

    </div>
  );
}
