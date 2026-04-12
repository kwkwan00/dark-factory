import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type EvalRun,
  type EvalSpec,
  type ProgressLogEntry,
  type RunDetailResponse,
  type RunFileContentResponse,
  type RunFileNode,
  type RunFilesResponse,
} from "../api/client";
import { useRunEvaluation } from "../hooks/useDashboard";

// ── Helpers ────────────────────────────────────────────────────────────────

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

function fmtBytes(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtCost(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 0.01) return `$${(value * 100).toFixed(3)}¢`;
  return `$${value.toFixed(2)}`;
}

function fmtTimestamp(ts: string | null | undefined): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function statusBadgeClass(status: string): string {
  if (status === "success") return "badge-success";
  if (status === "partial") return "badge-warn";
  if (status === "running") return "badge-info";
  return "badge-error";
}

function guessLanguageFromPath(path: string): string | undefined {
  const ext = path.split(".").pop()?.toLowerCase();
  if (!ext) return undefined;
  const map: Record<string, string> = {
    py: "python",
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    json: "json",
    md: "markdown",
    yaml: "yaml",
    yml: "yaml",
    toml: "toml",
    sh: "bash",
    html: "html",
    css: "css",
    sql: "sql",
  };
  return map[ext];
}

// ── File tree node (recursive) ─────────────────────────────────────────────

interface TreeNodeProps {
  node: RunFileNode;
  depth: number;
  selectedPath: string | null;
  expanded: Set<string>;
  onToggle: (path: string) => void;
  onSelect: (path: string) => void;
}

function TreeNode({
  node,
  depth,
  selectedPath,
  expanded,
  onToggle,
  onSelect,
}: TreeNodeProps) {
  const pad = depth * 12;
  if (node.type === "dir") {
    const isOpen = depth === 0 ? true : expanded.has(node.path);
    const label = depth === 0 ? "./" : node.name;
    return (
      <div>
        {depth > 0 && (
          <button
            onClick={() => onToggle(node.path)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              width: "100%",
              textAlign: "left",
              background: "transparent",
              border: "none",
              color: "#c9d1d9",
              padding: `2px 4px 2px ${pad}px`,
              cursor: "pointer",
              fontSize: 12,
              fontFamily: "SF Mono, Consolas, monospace",
            }}
          >
            <span style={{ color: "#8b949e", width: 10 }}>{isOpen ? "▼" : "▶"}</span>
            <span>📁 {label}</span>
            {node.truncated && (
              <span
                style={{ marginLeft: 6, color: "#d29922", fontSize: 10 }}
                title="Truncated: too many entries"
              >
                (truncated)
              </span>
            )}
          </button>
        )}
        {isOpen && node.children && (
          <div>
            {node.children.map((child) => (
              <TreeNode
                key={`${child.type}-${child.path || child.name}`}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                expanded={expanded}
                onToggle={onToggle}
                onSelect={onSelect}
              />
            ))}
          </div>
        )}
        {node.error && (
          <div
            style={{
              color: "#f85149",
              fontSize: 10,
              paddingLeft: pad + 16,
            }}
          >
            {node.error}
          </div>
        )}
      </div>
    );
  }

  // File
  const isSelected = selectedPath === node.path;
  return (
    <button
      onClick={() => onSelect(node.path)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        width: "100%",
        textAlign: "left",
        background: isSelected ? "#1f6feb33" : "transparent",
        border: "none",
        color: isSelected ? "#e6edf3" : "#c9d1d9",
        padding: `2px 4px 2px ${pad}px`,
        cursor: "pointer",
        fontSize: 12,
        fontFamily: "SF Mono, Consolas, monospace",
      }}
    >
      <span style={{ width: 10 }} />
      <span>📄 {node.name}</span>
      <span style={{ marginLeft: "auto", color: "#8b949e", fontSize: 10 }}>
        {fmtBytes(node.size)}
      </span>
    </button>
  );
}

// ── File explorer pane ──────────────────────────────────────────────────────

interface FileExplorerProps {
  runId: string;
}

function FileExplorer({ runId }: FileExplorerProps) {
  const [tree, setTree] = useState<RunFilesResponse | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [content, setContent] = useState<RunFileContentResponse | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setTreeError(null);
    api
      .runFiles(runId)
      .then((data) => {
        if (cancelled) return;
        setTree(data);
        // Auto-expand the first level so the user sees something immediately.
        const firstLevel = new Set<string>();
        for (const child of data.tree.children ?? []) {
          if (child.type === "dir") firstLevel.add(child.path);
        }
        setExpanded(firstLevel);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setTreeError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const toggleDir = useCallback((path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const selectFile = useCallback(
    (path: string) => {
      setSelectedPath(path);
      setContent(null);
      setContentError(null);
      setLoadingContent(true);
      api
        .runFileContent(runId, path)
        .then((data) => {
          setContent(data);
          setLoadingContent(false);
        })
        .catch((err: unknown) => {
          setContentError(err instanceof Error ? err.message : String(err));
          setLoadingContent(false);
        });
    },
    [runId],
  );

  const summaryLine = useMemo(() => {
    if (!tree) return null;
    return `${fmtNumber(tree.file_count)} files · ${fmtBytes(tree.total_bytes)}`;
  }, [tree]);

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
          Output files
          {summaryLine && (
            <span style={{ color: "#8b949e", fontWeight: 400, marginLeft: 8, fontSize: 12 }}>
              — {summaryLine}
            </span>
          )}
        </div>
      </div>

      {treeError && (
        <div className="card" style={{ borderColor: "#da3633" }}>
          <code>{treeError}</code>
        </div>
      )}

      {!tree && !treeError && (
        <div className="empty-state">
          <p>Loading file tree…</p>
        </div>
      )}

      {tree && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "300px 1fr",
            gap: 12,
            minHeight: 400,
          }}
        >
          {/* Tree pane */}
          <div
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              padding: 8,
              overflowY: "auto",
              maxHeight: 600,
            }}
          >
            <TreeNode
              node={tree.tree}
              depth={0}
              selectedPath={selectedPath}
              expanded={expanded}
              onToggle={toggleDir}
              onSelect={selectFile}
            />
          </div>

          {/* Content pane */}
          <div
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              padding: 12,
              overflow: "auto",
              maxHeight: 600,
            }}
          >
            {!selectedPath && (
              <div className="empty-state">
                <p>Select a file from the tree to preview its contents.</p>
              </div>
            )}
            {selectedPath && loadingContent && (
              <div style={{ color: "#8b949e", fontSize: 12 }}>Loading…</div>
            )}
            {selectedPath && contentError && (
              <div style={{ borderColor: "#da3633", color: "#f85149", fontSize: 12 }}>
                <code>{contentError}</code>
              </div>
            )}
            {selectedPath && content && (
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 8,
                    paddingBottom: 8,
                    borderBottom: "1px solid #30363d",
                  }}
                >
                  <code style={{ color: "#58a6ff", fontSize: 12 }}>
                    {content.path}
                  </code>
                  <span style={{ color: "#8b949e", fontSize: 11 }}>
                    {fmtBytes(content.size)}
                    {guessLanguageFromPath(content.path) && (
                      <span style={{ marginLeft: 8 }}>
                        · {guessLanguageFromPath(content.path)}
                      </span>
                    )}
                  </span>
                </div>
                <pre
                  style={{
                    margin: 0,
                    color: "#c9d1d9",
                    fontSize: 12,
                    fontFamily: "SF Mono, Consolas, monospace",
                    whiteSpace: "pre",
                    overflow: "auto",
                  }}
                >
                  {content.content}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Metrics summary pane ───────────────────────────────────────────────────

interface RunMetricsProps {
  runId: string;
}

function RunMetrics({ runId }: RunMetricsProps) {
  const [detail, setDetail] = useState<RunDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .metricsRunDetail(runId)
      .then((data) => {
        setDetail(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
  }, [runId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (loading) {
    return (
      <div className="card">
        <div className="empty-state">
          <p>Loading run detail…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <div className="card-title" style={{ color: "#f85149" }}>
          Run detail unavailable
        </div>
        <code style={{ fontSize: 12 }}>{error}</code>
        <div style={{ marginTop: 8 }}>
          <button className="btn btn-secondary" onClick={refresh}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!detail) return null;

  if (!detail.enabled) {
    return (
      <div className="card" style={{ borderColor: "#30363d" }}>
        <div className="card-title">Metrics store not enabled</div>
        <p style={{ color: "#8b949e", margin: 0, fontSize: 13 }}>
          Enable Postgres metrics to see per-run aggregates.
        </p>
      </div>
    );
  }

  const run = detail.run;
  const llmTotals = detail.llm?.totals ?? {};
  const llmPerPhase = detail.llm?.per_phase ?? [];
  const swarmEvents = detail.swarm_events ?? [];
  const agents = detail.agent_stats ?? [];
  const tools = detail.tool_calls ?? [];
  const incidents = detail.incidents ?? [];
  const evalMetrics = detail.eval_metrics ?? [];
  const artifactsSummary = detail.artifacts?.summary ?? {};
  const artifactsPerLang = detail.artifacts?.per_language ?? [];
  const decomposition = detail.decomposition ?? [];

  return (
    <div>
      {/* Header card: run status + basic stats */}
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
            Run {run && <code style={{ fontSize: 13 }}>{run.run_id}</code>}
          </div>
          <button className="btn btn-secondary" onClick={refresh}>
            Refresh
          </button>
        </div>

        {run && (
          <div className="result-grid">
            <div className="stat-card">
              <div className="stat-value">
                <span className={statusBadgeClass(run.status)}>{run.status}</span>
              </div>
              <div className="stat-label">Status</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{fmtPct(run.pass_rate)}</div>
              <div className="stat-label">Pass rate</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{fmtSeconds(run.duration_seconds)}</div>
              <div className="stat-label">Duration</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{fmtNumber(run.spec_count)}</div>
              <div className="stat-label">Specs</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{fmtNumber(run.feature_count)}</div>
              <div className="stat-label">Features</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ fontSize: 13 }}>
                {fmtTimestamp(run.started_at)}
              </div>
              <div className="stat-label">Started</div>
            </div>
          </div>
        )}

        {run?.error && (
          <div
            className="card"
            style={{ borderColor: "#da3633", marginTop: 12 }}
            role="alert"
          >
            <div className="card-title" style={{ color: "#f85149" }}>
              Error
            </div>
            <code style={{ fontSize: 12, whiteSpace: "pre-wrap" }}>{run.error}</code>
          </div>
        )}
      </div>

      {/* LLM usage */}
      <div className="card">
        <div className="card-title">LLM usage</div>
        <div className="result-grid">
          <div className="stat-card">
            <div className="stat-value">{fmtNumber(llmTotals.total_calls)}</div>
            <div className="stat-label">Calls</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {fmtNumber((llmTotals.input_tokens ?? 0) + (llmTotals.output_tokens ?? 0))}
            </div>
            <div className="stat-label">Tokens</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{fmtCost(llmTotals.total_cost_usd)}</div>
            <div className="stat-label">Cost</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {fmtSeconds(llmTotals.avg_latency_seconds)}
            </div>
            <div className="stat-label">Avg latency</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{fmtNumber(llmTotals.rate_limited_count)}</div>
            <div className="stat-label">Rate-limited</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{fmtNumber(llmTotals.error_count)}</div>
            <div className="stat-label">Errors</div>
          </div>
        </div>

        {llmPerPhase.length > 0 && (
          <table className="table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Phase</th>
                <th>Calls</th>
                <th>Input tokens</th>
                <th>Output tokens</th>
                <th>Cost</th>
                <th>Avg latency</th>
              </tr>
            </thead>
            <tbody>
              {llmPerPhase.map((p) => (
                <tr key={p.phase ?? "(unknown)"}>
                  <td>{p.phase ?? "—"}</td>
                  <td>{fmtNumber(p.calls)}</td>
                  <td>{fmtNumber(p.input_tokens)}</td>
                  <td>{fmtNumber(p.output_tokens)}</td>
                  <td>{fmtCost(p.total_cost_usd)}</td>
                  <td>{fmtSeconds(p.avg_latency_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Swarm feature events */}
      {swarmEvents.length > 0 && (
        <div className="card">
          <div className="card-title">Feature events ({swarmEvents.length})</div>
          <table className="table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Event</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Artifacts</th>
                <th>Tests</th>
                <th>Tool calls</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {swarmEvents.map((e, i) => (
                <tr key={`${e.feature}-${e.event}-${i}`}>
                  <td>{e.feature}</td>
                  <td>{e.event}</td>
                  <td>
                    {e.status && (
                      <span className={statusBadgeClass(e.status)}>{e.status}</span>
                    )}
                  </td>
                  <td>{fmtSeconds(e.duration_seconds)}</td>
                  <td>{fmtNumber(e.artifact_count)}</td>
                  <td>{fmtNumber(e.test_count)}</td>
                  <td>{fmtNumber(e.tool_call_count)}</td>
                  <td style={{ color: "#8b949e", fontSize: 11 }}>
                    {fmtTimestamp(e.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Agent stats */}
      {agents.length > 0 && (
        <div className="card">
          <div className="card-title">Agents ({agents.length})</div>
          <table className="table">
            <thead>
              <tr>
                <th>Agent</th>
                <th>Activations</th>
                <th>Tool calls</th>
                <th>Decisions</th>
                <th>Handoffs in</th>
                <th>Handoffs out</th>
                <th>Total time</th>
              </tr>
            </thead>
            <tbody>
              {agents.map((a) => (
                <tr key={a.agent}>
                  <td>{a.agent}</td>
                  <td>{fmtNumber(a.activations)}</td>
                  <td>{fmtNumber(a.tool_calls)}</td>
                  <td>{fmtNumber(a.decisions)}</td>
                  <td>{fmtNumber(a.handoffs_in)}</td>
                  <td>{fmtNumber(a.handoffs_out)}</td>
                  <td>{fmtSeconds(a.total_time_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Tool calls */}
      {tools.length > 0 && (
        <div className="card">
          <div className="card-title">Tool calls ({tools.length})</div>
          <table className="table">
            <thead>
              <tr>
                <th>Tool</th>
                <th>Calls</th>
                <th>Successes</th>
                <th>Failures</th>
                <th>Avg latency</th>
              </tr>
            </thead>
            <tbody>
              {tools.map((t) => (
                <tr key={t.bucket}>
                  <td>{t.bucket}</td>
                  <td>{fmtNumber(t.calls)}</td>
                  <td>{fmtNumber(t.successes)}</td>
                  <td>{fmtNumber(t.failures)}</td>
                  <td>{fmtSeconds(t.avg_latency_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Incidents */}
      {incidents.length > 0 && (
        <div className="card" style={{ borderColor: "#da363340" }}>
          <div className="card-title" style={{ color: "#f85149" }}>
            Incidents ({incidents.length})
          </div>
          <table className="table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Severity</th>
                <th>Phase</th>
                <th>Feature</th>
                <th>Message</th>
                <th>Resolved</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {incidents.map((inc) => (
                <tr key={inc.id}>
                  <td>{inc.category}</td>
                  <td>
                    <span
                      className={
                        inc.severity === "critical" || inc.severity === "error"
                          ? "badge-error"
                          : inc.severity === "warning"
                            ? "badge-warn"
                            : "badge-info"
                      }
                    >
                      {inc.severity}
                    </span>
                  </td>
                  <td>{inc.phase ?? "—"}</td>
                  <td>{inc.feature ?? "—"}</td>
                  <td style={{ maxWidth: 360, wordBreak: "break-word" }}>
                    {inc.message}
                  </td>
                  <td>{inc.resolved ? "yes" : "no"}</td>
                  <td style={{ color: "#8b949e", fontSize: 11 }}>
                    {fmtTimestamp(inc.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Eval metrics */}
      {evalMetrics.length > 0 && (
        <div className="card">
          <div className="card-title">Eval metrics ({evalMetrics.length})</div>
          <table className="table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Spec ID</th>
                <th>Requirement</th>
                <th>Type</th>
                <th>Score</th>
                <th>Passed</th>
                <th>Reason</th>
                <th>Attempt</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {evalMetrics.map((m, i) => (
                <tr key={`${m.metric_name}-${m.spec_id}-${i}`}>
                  <td>{m.metric_name}</td>
                  <td>
                    <code style={{ fontSize: 11 }}>{m.spec_id ?? "—"}</code>
                  </td>
                  <td>
                    <code style={{ fontSize: 11 }}>{m.requirement_id ?? "—"}</code>
                  </td>
                  <td style={{ color: "#8b949e", fontSize: 11 }}>
                    {m.eval_type ?? "—"}
                  </td>
                  <td>{fmtNumber(m.score, 2)}</td>
                  <td>
                    <span className={m.passed ? "badge-success" : "badge-error"}>
                      {m.passed ? "pass" : "fail"}
                    </span>
                  </td>
                  <td
                    style={{
                      color: "#8b949e",
                      fontSize: 11,
                      maxWidth: 220,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={m.reason ?? undefined}
                  >
                    {m.reason ?? "—"}
                  </td>
                  <td>{m.attempt ?? "—"}</td>
                  <td style={{ color: "#8b949e", fontSize: 11 }}>
                    {fmtTimestamp(m.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Artifacts */}
      {(artifactsSummary.files_written ?? 0) > 0 && (
        <div className="card">
          <div className="card-title">Artifacts</div>
          <div className="result-grid">
            <div className="stat-card">
              <div className="stat-value">
                {fmtNumber(artifactsSummary.files_written)}
              </div>
              <div className="stat-label">Files written</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{fmtBytes(artifactsSummary.total_bytes)}</div>
              <div className="stat-label">Total bytes</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {fmtNumber(artifactsSummary.code_files)}
              </div>
              <div className="stat-label">Code files</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {fmtNumber(artifactsSummary.test_files)}
              </div>
              <div className="stat-label">Test files</div>
            </div>
          </div>
          {artifactsPerLang.length > 0 && (
            <table className="table" style={{ marginTop: 12 }}>
              <thead>
                <tr>
                  <th>Language</th>
                  <th>Files</th>
                  <th>Total bytes</th>
                </tr>
              </thead>
              <tbody>
                {artifactsPerLang.map((l) => (
                  <tr key={l.language}>
                    <td>{l.language}</td>
                    <td>{fmtNumber(l.files)}</td>
                    <td>{fmtBytes(l.total_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Decomposition */}
      {decomposition.length > 0 && (
        <div className="card">
          <div className="card-title">Decomposition ({decomposition.length})</div>
          <table className="table">
            <thead>
              <tr>
                <th>Requirement</th>
                <th>Sub-specs</th>
                <th>Fallback</th>
                <th>Empty</th>
                <th>Deps declared</th>
                <th>Deps resolved</th>
              </tr>
            </thead>
            <tbody>
              {decomposition.map((d, i) => (
                <tr key={`${d.requirement_id}-${i}`}>
                  <td>
                    <code style={{ fontSize: 11 }}>{d.requirement_id ?? "—"}</code>
                    {d.requirement_title && (
                      <div style={{ color: "#8b949e", fontSize: 11 }}>
                        {d.requirement_title}
                      </div>
                    )}
                  </td>
                  <td>{fmtNumber(d.planned_sub_specs_count)}</td>
                  <td>{d.fallback ? "yes" : "no"}</td>
                  <td>{d.empty_result ? "yes" : "no"}</td>
                  <td>{fmtNumber(d.depends_on_declared)}</td>
                  <td>{fmtNumber(d.depends_on_resolved)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Evaluations pane ───────────────────────────────────────────────────────
//
// Per-run eval rendering — replaces the deleted top-level Evaluations
// tab. Hierarchy: spec → eval attempt → metrics breakdown. The same
// SpecAttempts component the old tab used, just inlined here so the
// popup is fully self-contained and we don't have to maintain an
// orphan top-level page just to host it.

function scoreBadgeClass(score: number): string {
  if (score >= 0.8) return "badge-success";
  if (score >= 0.5) return "badge-warn";
  return "badge-error";
}

function SpecAttempts({ spec }: { spec: EvalSpec }) {
  const [openAttempt, setOpenAttempt] = useState<string | null>(null);

  // M19 fix: use .at(-1) with an explicit fallback so an empty evals
  // array doesn't produce NaN / crash during render.
  const lastScore = spec.evals.at(-1)?.overall_score ?? 0;

  return (
    <div
      style={{
        marginTop: 8,
        marginLeft: 16,
        paddingLeft: 12,
        borderLeft: "2px solid #21262d",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <div style={{ fontSize: 13 }}>
          <code>{spec.spec_id}</code>
          {spec.feature_name && (
            <span style={{ color: "#8b949e", marginLeft: 8 }}>
              feature: <code>{spec.feature_name}</code>
            </span>
          )}
        </div>
        <span className={scoreBadgeClass(lastScore)}>
          {lastScore.toFixed(2)} ({spec.evals.length} attempt
          {spec.evals.length === 1 ? "" : "s"})
        </span>
      </div>

      <table className="table" style={{ fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ width: 24 }}></th>
            <th style={{ width: 60 }}>#</th>
            <th>Type</th>
            <th>Score</th>
            <th>Passed</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {spec.evals.map((attempt, i) => {
            const isOpen = openAttempt === attempt.id;
            return (
              <Fragment key={attempt.id}>
                <tr
                  style={{ cursor: "pointer" }}
                  onClick={() => setOpenAttempt(isOpen ? null : attempt.id)}
                >
                  <td style={{ color: "#8b949e", textAlign: "center" }}>
                    {isOpen ? "▼" : "▶"}
                  </td>
                  <td>{i + 1}</td>
                  <td>{attempt.eval_type}</td>
                  <td>
                    <span className={scoreBadgeClass(attempt.overall_score)}>
                      {attempt.overall_score.toFixed(2)}
                    </span>
                  </td>
                  <td>
                    {attempt.all_passed ? (
                      <span className="badge-success">✓</span>
                    ) : (
                      <span className="badge-error">✕</span>
                    )}
                  </td>
                  <td style={{ color: "#8b949e", fontSize: 11 }}>
                    {fmtTimestamp(attempt.timestamp)}
                  </td>
                </tr>
                {isOpen && (
                  <tr>
                    <td></td>
                    <td colSpan={5} style={{ background: "#0d1117" }}>
                      <div style={{ padding: 8 }}>
                        {attempt.metrics.length === 0 ? (
                          <div style={{ color: "#8b949e" }}>
                            No per-metric data available.
                          </div>
                        ) : (
                          attempt.metrics.map((m) => (
                            <div
                              key={m.name}
                              style={{ marginBottom: 6, fontSize: 12 }}
                            >
                              <span
                                className={
                                  m.passed ? "badge-success" : "badge-error"
                                }
                                style={{ marginRight: 8 }}
                              >
                                {m.passed ? "✓" : "✕"}
                              </span>
                              <strong>{m.name}</strong>{" "}
                              <span style={{ color: "#58a6ff" }}>
                                {m.score.toFixed(2)}
                              </span>
                              {m.reason && (
                                <div
                                  style={{
                                    marginLeft: 22,
                                    color: "#8b949e",
                                    fontSize: 11,
                                    whiteSpace: "pre-wrap",
                                    marginTop: 2,
                                  }}
                                >
                                  {m.reason}
                                </div>
                              )}
                            </div>
                          ))
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

interface RunEvaluationsProps {
  runId: string;
}

function RunEvaluations({ runId }: RunEvaluationsProps) {
  const { state, refresh } = useRunEvaluation(runId);

  if (state.status === "loading" || state.status === "idle") {
    return (
      <div className="card">
        <div className="empty-state">
          <p>Loading evaluations…</p>
        </div>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <div className="card-title" style={{ color: "#f85149" }}>
          Evaluations unavailable
        </div>
        <code style={{ fontSize: 12 }}>{state.error}</code>
        <div style={{ marginTop: 8 }}>
          <button className="btn btn-secondary" onClick={() => void refresh()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  const runs: EvalRun[] = state.data.runs ?? [];
  // The backend filter returns a 1-element array (or empty if no
  // evals exist for this run). Pull the single matching run out.
  const run = runs[0];

  if (!run) {
    return (
      <div className="card">
        <div className="card-title" style={{ margin: 0 }}>
          Evaluations
        </div>
        <p style={{ color: "#8b949e", fontSize: 13, margin: "8px 0 0" }}>
          No evaluations recorded for this run yet. Evals are created
          during the spec generation and codegen phases — they'll appear
          here once the run produces results.
        </p>
      </div>
    );
  }

  const totalEvals = run.specs.reduce((acc, s) => acc + s.evals.length, 0);
  const avgScore =
    totalEvals === 0
      ? 0
      : run.specs.reduce(
          (acc, s) =>
            acc + s.evals.reduce((a2, e) => a2 + e.overall_score, 0),
          0,
        ) / totalEvals;

  return (
    <div>
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div className="card-title" style={{ margin: 0 }}>
            Evaluations
            <span style={{ color: "#8b949e", fontWeight: 400, marginLeft: 8 }}>
              — {run.specs.length} spec{run.specs.length === 1 ? "" : "s"} ·{" "}
              {totalEvals} eval{totalEvals === 1 ? "" : "s"}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {totalEvals > 0 && (
              <span className={scoreBadgeClass(avgScore)}>
                avg {avgScore.toFixed(2)}
              </span>
            )}
            <button
              className="btn btn-secondary"
              onClick={() => void refresh()}
            >
              Refresh
            </button>
          </div>
        </div>

        {run.specs.length === 0 ? (
          <p style={{ color: "#8b949e", fontSize: 13, margin: "12px 0 0" }}>
            This run is recorded in memory but has no per-spec eval results.
          </p>
        ) : (
          <div style={{ marginTop: 12 }}>
            {run.specs.map((spec) => (
              <SpecAttempts key={spec.spec_id} spec={spec} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Top-level window ────────────────────────────────────────────────────────

// ── Episodes (episodic memory timeline) ────────────────────────────────────

interface RunEpisodesProps {
  runId: string;
}

function _outcomeBadgeClass(outcome: string): string {
  const o = (outcome || "").toLowerCase();
  if (o === "success" || o === "pass") return "badge-success";
  if (o === "failed" || o === "error" || o === "broken") return "badge-error";
  if (o === "partial" || o === "warning") return "badge-warn";
  return "badge-info";
}

function _formatDuration(seconds: number): string {
  if (!seconds || seconds < 0.1) return "<0.1s";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

function RunEpisodes({ runId }: RunEpisodesProps) {
  const [episodes, setEpisodes] = useState<
    import("../api/client").EpisodeResponse[] | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    import("../api/client")
      .then(({ api }) => api.runEpisodes(runId))
      .then((res) => {
        setEpisodes(res.episodes);
        setLoading(false);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
  }, [runId]);

  if (loading) {
    return (
      <div className="empty-state">
        <p>Loading episodes…</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <div className="card-title" style={{ color: "#f85149" }}>
          Failed to load episodes
        </div>
        <code>{error}</code>
      </div>
    );
  }
  if (!episodes || episodes.length === 0) {
    return (
      <div className="empty-state">
        <p>
          No episodes recorded for this run yet. Episodes are written at the
          end of each feature swarm — if the run is still in Phase 2/3 or
          episodic memory is disabled in Settings, this will be empty.
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          color: "#8b949e",
          fontSize: 12,
          marginBottom: 4,
        }}
      >
        {episodes.length} episode{episodes.length === 1 ? "" : "s"} recorded
        from this run — each one is queryable by future Planners via{" "}
        <code>recall_episodes</code>.
      </div>
      {episodes.map((ep) => (
        <div key={ep.id} className="card">
          <div
            className="card-title"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 8,
            }}
          >
            <span>{ep.feature}</span>
            <span className={`log-badge ${_outcomeBadgeClass(ep.outcome)}`}>
              {ep.outcome}
            </span>
            <span style={{ fontSize: 11, color: "#8b949e", fontWeight: 400 }}>
              {ep.turns_used} turns · {_formatDuration(ep.duration_seconds)}
            </span>
          </div>
          <p
            style={{
              fontSize: 12,
              color: "#c9d1d9",
              lineHeight: 1.55,
              margin: "0 0 12px",
            }}
          >
            {ep.summary}
          </p>

          {ep.key_events && ep.key_events.length > 0 && (
            <div style={{ marginBottom: 10 }}>
              <div
                style={{
                  fontSize: 10,
                  color: "#58a6ff",
                  textTransform: "uppercase",
                  letterSpacing: 0.5,
                  marginBottom: 4,
                }}
              >
                Key events
              </div>
              <ol
                style={{
                  margin: 0,
                  paddingLeft: 18,
                  fontSize: 11,
                  color: "#8b949e",
                  lineHeight: 1.6,
                }}
              >
                {ep.key_events.map((ke, i) => (
                  <li key={i}>
                    <strong style={{ color: "#d2a8ff" }}>{ke.agent}</strong>{" "}
                    · <code style={{ color: "#ffa657" }}>{ke.event}</code>{" "}
                    — {ke.description}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {ep.final_eval_scores && Object.keys(ep.final_eval_scores).length > 0 && (
            <div
              style={{
                display: "flex",
                gap: 10,
                flexWrap: "wrap",
                marginBottom: 10,
              }}
            >
              {Object.entries(ep.final_eval_scores).map(([metric, score]) => (
                <span
                  key={metric}
                  style={{
                    fontSize: 10,
                    color: score >= 0.5 ? "#3fb950" : "#f85149",
                    background: "#0d1117",
                    border: "1px solid #30363d",
                    padding: "2px 6px",
                    borderRadius: 4,
                  }}
                >
                  {metric}: {score.toFixed(2)}
                </span>
              ))}
            </div>
          )}

          <div
            style={{
              display: "flex",
              gap: 12,
              fontSize: 10,
              color: "#6e7681",
            }}
          >
            <span>agents: {(ep.agents_visited || []).join(" → ") || "—"}</span>
            {ep.spec_ids && ep.spec_ids.length > 0 && (
              <span>specs: {ep.spec_ids.length}</span>
            )}
            <span style={{ marginLeft: "auto" }}>{ep.id}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Agent Log pane ─────────────────────────────────────────────────────────

import {
  EVENT_BADGES,
  formatTimeFromISO,
  formatEventDetails,
  logEntryToProgressEvent,
} from "../lib/agentLogFormat";

interface AgentLogProps {
  runId: string;
}

function AgentLog({ runId }: AgentLogProps) {
  const [entries, setEntries] = useState<ProgressLogEntry[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const toggleExpand = useCallback((id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .metricsRunDetail(runId)
      .then((resp) => {
        if (cancelled) return;
        setEntries(resp.progress_log ?? []);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const filtered = useMemo(() => {
    if (!entries) return [];
    if (!filter.trim()) return entries;
    const needle = filter.toLowerCase();
    return entries.filter((e) => {
      const ev = logEntryToProgressEvent(e);
      const hay = `${e.event} ${e.feature ?? ""} ${e.agent ?? ""} ${formatEventDetails(ev)}`
        .toLowerCase();
      return hay.includes(needle);
    });
  }, [entries, filter]);

  if (loading) return <p style={{ padding: 12, color: "#8b949e" }}>Loading agent log…</p>;
  if (error)
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <code style={{ color: "#f85149" }}>{error}</code>
      </div>
    );
  if (!entries || entries.length === 0)
    return (
      <div className="card">
        <p style={{ color: "#8b949e" }}>
          No progress events recorded for this run.
        </p>
      </div>
    );

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
            Agent Log ({filtered.length}
            {filter.trim() ? ` / ${entries.length}` : ""} events)
          </div>
        </div>
        <div className="input-row" style={{ margin: 0 }}>
          <input
            className="input-text"
            placeholder="Filter events (event type, feature, agent, text)..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
      </div>

      {/* Log view — same layout as the live Agent Event Stream */}
      <div className="card" style={{ padding: 0 }}>
        <div className="log-view">
          {filtered.length === 0 ? (
            <div className="empty-state" style={{ padding: 24 }}>
              <p>No entries match &quot;{filter}&quot;</p>
            </div>
          ) : (
            filtered.map((entry) => {
              const ev = logEntryToProgressEvent(entry);
              const badge = EVENT_BADGES[entry.event] ?? {
                label: entry.event.toUpperCase(),
                color: "#8b949e",
              };
              const hasPayload =
                entry.payload && Object.keys(entry.payload).length > 0;
              const isExpanded = expandedIds.has(entry.id);
              return (
                <div key={entry.id}>
                  <div className="log-entry">
                    {hasPayload ? (
                      <button
                        onClick={() => toggleExpand(entry.id)}
                        title="Toggle payload detail"
                        style={{
                          background: "transparent",
                          border: "none",
                          color: "#58a6ff",
                          cursor: "pointer",
                          fontSize: 12,
                          padding: "0 4px 0 0",
                          flexShrink: 0,
                          width: 16,
                        }}
                      >
                        {isExpanded ? "▾" : "▸"}
                      </button>
                    ) : (
                      <span style={{ width: 16, flexShrink: 0 }} />
                    )}
                    <span className="log-time">
                      {formatTimeFromISO(entry.timestamp)}
                    </span>
                    <span
                      className="log-badge"
                      style={{ color: badge.color, borderColor: badge.color }}
                    >
                      {badge.label}
                    </span>
                    <span className="log-details">
                      {formatEventDetails(ev)}
                    </span>
                  </div>
                  {isExpanded && hasPayload && (
                    <pre
                      style={{
                        margin: 0,
                        padding: "8px 12px 8px 90px",
                        fontSize: 11,
                        color: "#c9d1d9",
                        background: "#161b22",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        maxHeight: 400,
                        overflow: "auto",
                        borderBottom: "1px solid #21262d",
                      }}
                    >
                      {JSON.stringify(entry.payload, null, 2)}
                    </pre>
                  )}
                </div>
              );
            })
          )}
        </div>
        <div className="log-footer">
          {filtered.length} of {entries.length} events
        </div>
      </div>
    </div>
  );
}

// ── RunDetailWindow main component ─────────────────────────────────────────

interface RunDetailWindowProps {
  runId: string;
}

type RunDetailView = "metrics" | "output" | "evaluations" | "episodes" | "agent-log";

export default function RunDetailWindow({ runId }: RunDetailWindowProps) {
  // Per-run detail is split into two separate screens so each one can
  // use the full scroll area — stacking both vertically (metrics above
  // the file explorer) pushed the tree out of view for any run with
  // more than a handful of metric cards.
  const [view, setView] = useState<RunDetailView>("metrics");

  // Set the window title so tab bars show the run id.
  useEffect(() => {
    if (runId) {
      document.title = `Run ${runId} · AI Dark Factory`;
    } else {
      document.title = "Run Detail · AI Dark Factory";
    }
  }, [runId]);

  if (!runId) {
    return (
      <div style={{ padding: 24 }}>
        <div className="card" style={{ borderColor: "#da3633" }}>
          <div className="card-title" style={{ color: "#f85149" }}>
            Missing run_id
          </div>
          <p style={{ color: "#8b949e", fontSize: 13 }}>
            This window must be opened with a <code>?run_id=…</code> query
            parameter. Example: <code>#/run-detail?run_id=run-20260410-1234</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <header className="app-header">
        <h1>
          Run Detail <code style={{ fontSize: 14, color: "#58a6ff" }}>{runId}</code>
        </h1>
        <span className="badge">popup</span>
      </header>

      {/* Screen switcher — mirrors the main App's tab-nav styling so the
          popup feels like part of the same UI. Three views: the metrics
          rollup (default), the per-spec evaluations tree (formerly the
          top-level Evaluations tab, now scoped to this run only), and
          the file-tree explorer for generated artifacts. */}
      <nav className="tab-nav">
        <button
          className={`tab-btn${view === "metrics" ? " active" : ""}`}
          onClick={() => setView("metrics")}
          aria-pressed={view === "metrics"}
        >
          Metrics
        </button>
        <button
          className={`tab-btn${view === "agent-log" ? " active" : ""}`}
          onClick={() => setView("agent-log")}
          aria-pressed={view === "agent-log"}
        >
          Agent Log
        </button>
        <button
          className={`tab-btn${view === "evaluations" ? " active" : ""}`}
          onClick={() => setView("evaluations")}
          aria-pressed={view === "evaluations"}
        >
          Evaluations
        </button>
        <button
          className={`tab-btn${view === "output" ? " active" : ""}`}
          onClick={() => setView("output")}
          aria-pressed={view === "output"}
        >
          Output
        </button>
        <button
          className={`tab-btn${view === "episodes" ? " active" : ""}`}
          onClick={() => setView("episodes")}
          aria-pressed={view === "episodes"}
        >
          Episodes
        </button>
      </nav>

      <main className="tab-content">
        {view === "metrics" && <RunMetrics runId={runId} />}
        {view === "agent-log" && <AgentLog runId={runId} />}
        {view === "evaluations" && <RunEvaluations runId={runId} />}
        {view === "output" && <FileExplorer runId={runId} />}
        {view === "episodes" && <RunEpisodes runId={runId} />}
      </main>
    </>
  );
}
