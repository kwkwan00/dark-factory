import { useEffect, useRef, useState, type DragEvent, type ChangeEvent } from "react";
import { api } from "../api/client";
import { useManufacture } from "../contexts/ManufactureContext";
import RefreshIcon from "./RefreshIcon";
import { type Step } from "../hooks/useAgentRun";
import { useHistory } from "../hooks/useDashboard";

function RunMenu({ runId, onDelete }: { runId: string; onDelete: () => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
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
              if (!confirm(`Delete run ${runId}?\n\nThis removes all data (episodes, evals, memories, files) for this run.`)) return;
              try {
                await api.deleteRun(runId);
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
            Delete run
          </button>
        </div>
      )}
    </div>
  );
}
import { openAboutDarkFactory } from "../lib/openAboutDarkFactory";
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

function StepItem({
  step,
  isFeature,
  featureName,
  keyMessages,
  hasMore,
  allMessages,
}: {
  step: Step;
  isFeature: boolean;
  featureName: string;
  keyMessages: string[];
  hasMore: boolean;
  allMessages: string[];
}) {
  const [expanded, setExpanded] = useState(false);
  const displayMessages = expanded ? allMessages : keyMessages;

  return (
    <li className="step-item">
      <div className={`step-icon ${step.status}`}>
        {STATUS_ICON[step.status]}
      </div>
      <div className="step-body">
        <div className="step-name" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {isFeature ? (
            <>
              <code style={{ color: "#d29922" }}>{featureName}</code>
              {step.status === "running" && (
                <span style={{ fontSize: 11, color: "#8b949e" }}>
                  {allMessages.filter((m) => m.includes("→ **")).length} handoff(s)
                </span>
              )}
            </>
          ) : (
            step.name
          )}
        </div>
        <div className="step-messages">
          {displayMessages.map((msg, i) => (
            <div key={i} className="step-message">
              {msg}
            </div>
          ))}
        </div>
        {hasMore && (
          <button
            onClick={() => setExpanded((v) => !v)}
            style={{
              background: "transparent",
              border: "none",
              color: "#58a6ff",
              cursor: "pointer",
              fontSize: 11,
              padding: "4px 0",
            }}
          >
            {expanded
              ? "Show less"
              : `Show all ${allMessages.length} messages`}
          </button>
        )}
      </div>
    </li>
  );
}

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

  // Refresh history when the active run changes status (started, done, error)
  useEffect(() => {
    void loadHistory();
  }, [state.status, loadHistory]);

  // Poll while any run is "running" or while the local pipeline is active.
  // 2s interval keeps the UI responsive without hammering the backend.
  const pipelineActive = state.status === "running";
  const hasRunningEntry = historyState.status === "done" &&
    historyState.data.runs.some((r) => (r.status as string | undefined) === "running");

  useEffect(() => {
    if (!pipelineActive && !hasRunningEntry) {
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
      }, 2000);
    }
    return () => {
      if (pollRef.current !== null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [pipelineActive, hasRunningEntry, loadHistory]);

  const handleStart = () => {
    void startRun(path.trim() || "./openspec");
    // Refresh history shortly after starting so the new entry appears
    setTimeout(() => void loadHistory(), 500);
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

  const [showNewRun, setShowNewRun] = useState(false);
  const isRunning = state.status === "running";
  const result = state.result;

  // Derive the currently-running pipeline stage name from the step tree
  // so the run history table can show "running · Reconciliation" etc.
  const currentStageName = state.steps.findLast(
    (s) => s.status === "running",
  )?.name;

  return (
    <div>
      {/* New Run Modal */}
      {showNewRun && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.6)",
            zIndex: 200,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          onClick={(e) => { if (e.target === e.currentTarget && !isRunning) setShowNewRun(false); }}
        >
          <div
            style={{
              background: "#161b22",
              border: "1px solid #30363d",
              borderRadius: 12,
              padding: 24,
              width: 600,
              maxHeight: "80vh",
              overflow: "auto",
              boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16, color: "#e6edf3" }}>New Run</h2>
              {!isRunning && (
                <button
                  onClick={() => setShowNewRun(false)}
                  style={{ background: "transparent", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18 }}
                >
                  ✕
                </button>
              )}
            </div>

            {/* Drop zone */}
            <div
              className={`dropzone${dragActive ? " dropzone-active" : ""}${uploading ? " dropzone-uploading" : ""}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragEnter={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              style={{ marginBottom: 12 }}
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
                    : "Drag & drop requirement files, or click to browse"}
                </div>
                <div className="dropzone-hint">
                  Native: <code style={{ color: "#79c0ff" }}>{ACCEPT_NATIVE.replaceAll(",", " ")}</code>
                  {" · "}
                  Rich: <code style={{ color: "#d2a8ff" }}>{ACCEPT_RICH.replaceAll(",", " ")}</code>
                </div>
              </div>
            </div>

            {uploadError && (
              <div style={{ color: "#f85149", fontSize: 12, marginBottom: 8 }}>
                <code>{uploadError}</code>
              </div>
            )}

            {uploadedFiles.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={{ color: "#8b949e", fontSize: 12 }}>
                    Uploaded ({uploadedFiles.length})
                  </span>
                  <button
                    className="btn btn-secondary"
                    onClick={clearUploads}
                    disabled={isRunning}
                    style={{ fontSize: 11, padding: "2px 8px" }}
                  >
                    Clear
                  </button>
                </div>
                <ul style={{ margin: 0, paddingLeft: 20, color: "#8b949e", fontSize: 12 }}>
                  {uploadedFiles.map((f) => (
                    <li key={f}><code>{f}</code></li>
                  ))}
                </ul>
              </div>
            )}

            {/* Path + Run */}
            <div className="input-row" style={{ margin: 0 }}>
              <input
                className="input-text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="Requirements path, e.g. ./openspec"
                disabled={isRunning}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isRunning) {
                    handleStart();
                    setShowNewRun(false);
                  }
                }}
              />
              {isRunning ? (
                <button
                  className="btn btn-danger"
                  onClick={() => void cancelRun()}
                  disabled={cancelling}
                >
                  {cancelling ? "Cancelling…" : "Cancel"}
                </button>
              ) : (
                <button
                  className="btn"
                  onClick={() => { handleStart(); setShowNewRun(false); }}
                  disabled={uploading}
                >
                  Run
                </button>
              )}
            </div>
          </div>
        </div>
      )}

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
          <div style={{ display: "flex", gap: 8 }}>
            {isRunning ? (
              <button
                className="btn btn-danger"
                onClick={() => void cancelRun()}
                disabled={cancelling}
                style={{ fontSize: 12 }}
              >
                {cancelling ? "Cancelling…" : "Cancel Run"}
              </button>
            ) : (
              <button className="btn" onClick={() => setShowNewRun(true)} style={{ fontSize: 12 }}>
                New Run
              </button>
            )}
            <button
              className="btn btn-secondary"
              onClick={() => void loadHistory()}
              style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12 }}
            >
              <RefreshIcon /> Refresh
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => openAboutDarkFactory()}
              title="Open the About Dark Factory window"
              style={{ fontSize: 12 }}
            >
              About
            </button>
          </div>
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
              {historyState.data.message
                ? `No runs available — ${historyState.data.message}.`
                : "No previous runs. Upload requirements and click Run to start."}
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
                  <th></th>
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
                              : status === "cancelled"
                              ? "badge-warn"
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
                      <td>
                        {status !== "running" && (
                          <RunMenu runId={id} onDelete={() => void loadHistory()} />
                        )}
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
            {state.steps.map((step) => {
              const isFeature = step.name.startsWith("Feature:");
              const featureName = isFeature ? step.name.replace("Feature: ", "") : "";
              // For feature steps, extract key messages (handoffs + final summary)
              const keyMessages = isFeature
                ? step.messages.filter((m) =>
                    m.includes("→ **") || // agent handoff
                    m.startsWith("✓") ||  // success
                    m.startsWith("✕") ||  // failure
                    m.startsWith("⏭") ||  // skipped
                    m.startsWith("Starting feature")
                  )
                : step.messages;
              const hasMore = isFeature && step.messages.length > keyMessages.length;
              return (
                <StepItem
                  key={step.id}
                  step={step}
                  isFeature={isFeature}
                  featureName={featureName}
                  keyMessages={keyMessages}
                  hasMore={hasMore}
                  allMessages={step.messages}
                />
              );
            })}
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
