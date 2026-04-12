import { useEffect, useRef, useState } from "react";
import {
  api,
  type AdminClearAllResponse,
  type ModelInfo,
  type PipelineSettings,
  type PipelineSettingsUpdate,
} from "../api/client";
import { useManufacture } from "../contexts/ManufactureContext";
import { useHealth, usePipelineSettings, useWatcher } from "../hooks/useDashboard";

// L9 fix: proper interface + null guard instead of unchecked cast
interface WatchEvent {
  path: string;
  type: string;
  timestamp: number;
}

function isWatchEvent(v: unknown): v is WatchEvent {
  if (typeof v !== "object" || v == null) return false;
  const o = v as Record<string, unknown>;
  return typeof o.path === "string" && typeof o.type === "string" && typeof o.timestamp === "number";
}

function LastEvent({ event }: { event: unknown }) {
  if (!isWatchEvent(event)) return null;
  return (
    <p style={{ color: "#8b949e", margin: 0, fontSize: 12 }}>
      Last event: <code>{event.path}</code> ({event.type}){" "}
      {new Date(event.timestamp * 1000).toLocaleTimeString()}
    </p>
  );
}


// ── Danger Zone ─────────────────────────────────────────────────────────────

/** Literal phrase a user must type to arm the "Clear All Data" button.
 * Chosen to be long and annoying enough that muscle-memory copy/paste
 * is the only way to hit it twice by accident. */
const CLEAR_CONFIRM_PHRASE = "DELETE ALL DATA";

function fmtBytes(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// ── Clear-all confirmation modal ────────────────────────────────────────────
//
// Hoisted to module scope so the component identity is stable across
// DangerZone re-renders. Owns the ``confirmText`` local state so it resets
// to empty every time the modal is re-opened (unmount → remount).

interface ClearAllModalProps {
  open: boolean;
  clearing: boolean;
  includePrometheus: boolean;
  includeOutput: boolean;
  onTogglePrometheus: (v: boolean) => void;
  onToggleOutput: (v: boolean) => void;
  onCancel: () => void;
  onConfirm: () => void;
}

function ClearAllModal({
  open,
  clearing,
  includePrometheus,
  includeOutput,
  onTogglePrometheus,
  onToggleOutput,
  onCancel,
  onConfirm,
}: ClearAllModalProps) {
  const [confirmText, setConfirmText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const armed = confirmText.trim() === CLEAR_CONFIRM_PHRASE;

  // Reset input + focus it when the modal opens.
  useEffect(() => {
    if (!open) {
      setConfirmText("");
      return;
    }
    // Small timeout lets the browser finish mounting before focus
    // moves, otherwise Safari occasionally misses the call.
    const t = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(t);
  }, [open]);

  // Escape key closes the modal (unless we're mid-clear — can't cancel).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !clearing) {
        onCancel();
      }
      if (e.key === "Enter" && armed && !clearing) {
        e.preventDefault();
        onConfirm();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, clearing, armed, onCancel, onConfirm]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="clear-all-modal-title"
      aria-describedby="clear-all-modal-warning"
      onClick={(e) => {
        // Click on the backdrop (outside the dialog) → cancel.
        if (e.target === e.currentTarget && !clearing) {
          onCancel();
        }
      }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(1, 4, 9, 0.75)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: 24,
      }}
    >
      <div
        style={{
          background: "#161b22",
          border: "1px solid #da3633",
          borderRadius: 8,
          padding: 24,
          maxWidth: 560,
          width: "100%",
          maxHeight: "90vh",
          overflowY: "auto",
          boxShadow: "0 16px 64px rgba(0, 0, 0, 0.6)",
        }}
      >
        <div
          id="clear-all-modal-title"
          style={{
            color: "#f85149",
            fontSize: 16,
            fontWeight: 700,
            marginBottom: 12,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span>⚠ Delete all data?</span>
        </div>

        <p style={{ color: "#c9d1d9", margin: "0 0 8px", fontSize: 13 }}>
          This will permanently wipe <strong>every</strong> docker-compose
          managed data store:
        </p>
        <ul
          style={{
            color: "#8b949e",
            margin: "0 0 16px 20px",
            padding: 0,
            fontSize: 12,
            lineHeight: 1.6,
          }}
        >
          <li>
            <strong>Neo4j</strong> — all nodes and relationships (knowledge
            graph <em>and</em> procedural memory)
          </li>
          <li>
            <strong>Qdrant</strong> — every vector collection (specs, code,
            memories)
          </li>
          <li>
            <strong>Postgres</strong> — every metrics table (runs, eval rows,
            LLM calls, incidents, …)
          </li>
          {includePrometheus && (
            <li>
              <strong>Prometheus</strong> — every <code>dark_factory_*</code>{" "}
              time series from the TSDB + reset in-process counters
            </li>
          )}
          {includeOutput && (
            <li>
              <strong>Output directory</strong> — every file under the
              configured pipeline output path
            </li>
          )}
          <li>
            <strong>Progress broker history</strong> — the in-memory replay
            buffer
          </li>
        </ul>

        <div
          id="clear-all-modal-warning"
          role="alert"
          style={{
            background: "#0d1117",
            border: "1px solid #da3633",
            borderRadius: 6,
            padding: 12,
            marginBottom: 16,
            color: "#f85149",
            fontSize: 13,
            fontWeight: 600,
            textAlign: "center",
          }}
        >
          This cannot be undone. There is no backup. There is no trash.
        </div>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 12,
            color: "#8b949e",
            marginBottom: 6,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={includePrometheus}
            onChange={(e) => onTogglePrometheus(e.target.checked)}
            disabled={clearing}
          />
          Also wipe Prometheus time series + reset in-process collectors
        </label>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 12,
            color: "#8b949e",
            marginBottom: 16,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={includeOutput}
            onChange={(e) => onToggleOutput(e.target.checked)}
            disabled={clearing}
          />
          Also wipe the output directory
        </label>

        <div style={{ marginBottom: 16 }}>
          <label
            htmlFor="clear-all-modal-confirm"
            style={{
              display: "block",
              fontSize: 11,
              color: "#8b949e",
              marginBottom: 4,
            }}
          >
            Type{" "}
            <code style={{ color: "#f85149" }}>{CLEAR_CONFIRM_PHRASE}</code>{" "}
            to arm the confirm button:
          </label>
          <input
            id="clear-all-modal-confirm"
            ref={inputRef}
            type="text"
            className="input-text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            disabled={clearing}
            autoComplete="off"
            spellCheck={false}
            placeholder={CLEAR_CONFIRM_PHRASE}
            style={{ fontFamily: "SF Mono, Consolas, monospace" }}
          />
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: 8,
          }}
        >
          <button
            className="btn btn-secondary"
            onClick={onCancel}
            disabled={clearing}
          >
            Cancel
          </button>
          <button
            className="btn btn-danger"
            onClick={onConfirm}
            disabled={!armed || clearing}
            aria-label="Permanently delete all data"
          >
            {clearing ? "Clearing…" : "Clear ALL data"}
          </button>
        </div>
      </div>
    </div>
  );
}

function DangerZone() {
  const [includeOutput, setIncludeOutput] = useState(true);
  const [includePrometheus, setIncludePrometheus] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [result, setResult] = useState<AdminClearAllResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const clearingRef = useRef(false);

  const handleClear = async () => {
    // Synchronous ref guard so a double-click cannot fire two wipes.
    if (clearingRef.current) return;
    clearingRef.current = true;
    setClearing(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.adminClearAll({
        includeOutputDir: includeOutput,
        includePrometheus,
      });
      setResult(data);
      setModalOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // Keep the modal open on error so the user can retry without
      // re-typing the confirm phrase.
    } finally {
      clearingRef.current = false;
      setClearing(false);
    }
  };

  return (
    <div className="card" style={{ borderColor: "#da3633" }}>
      <div
        className="card-title"
        style={{ color: "#f85149", display: "flex", alignItems: "center", gap: 8 }}
      >
        <span>⚠ Danger Zone</span>
      </div>

      <p style={{ color: "#c9d1d9", margin: "0 0 12px", fontSize: 13 }}>
        Permanently wipe every docker-compose managed data store — Neo4j,
        Qdrant, Postgres metrics, Prometheus time series, and the pipeline
        output directory. You'll be asked to confirm in a separate dialog.
      </p>

      <button
        className="btn btn-danger"
        onClick={() => {
          setError(null);
          setModalOpen(true);
        }}
        disabled={clearing}
        aria-label="Open clear-all confirmation dialog"
      >
        Clear ALL data…
      </button>

      <ClearAllModal
        open={modalOpen}
        clearing={clearing}
        includePrometheus={includePrometheus}
        includeOutput={includeOutput}
        onTogglePrometheus={setIncludePrometheus}
        onToggleOutput={setIncludeOutput}
        onCancel={() => {
          if (clearing) return;
          setModalOpen(false);
        }}
        onConfirm={() => void handleClear()}
      />

      {error && (
        <div
          style={{
            marginTop: 12,
            padding: 8,
            borderRadius: 6,
            background: "#0d1117",
            border: "1px solid #da3633",
            color: "#f85149",
            fontSize: 12,
          }}
          role="alert"
        >
          {error}
        </div>
      )}

      {result && (
        <div
          style={{
            marginTop: 12,
            padding: 12,
            background: "#0d1117",
            borderRadius: 6,
            border: `1px solid ${result.status === "completed" ? "#3fb950" : "#d29922"}`,
            fontSize: 12,
          }}
          role="status"
          aria-live="polite"
        >
          <div
            style={{
              color: result.status === "completed" ? "#3fb950" : "#d29922",
              fontWeight: 600,
              marginBottom: 8,
            }}
          >
            {result.status === "completed"
              ? "✓ All stores cleared"
              : "⚠ Partial clear — some stores reported errors"}
          </div>
          <ul
            style={{
              color: "#8b949e",
              margin: 0,
              padding: "0 0 0 20px",
              lineHeight: 1.6,
            }}
          >
            {result.cleared.neo4j != null && (
              <li>
                Neo4j:{" "}
                {result.cleared.neo4j.status === "disabled"
                  ? "disabled"
                  : `${result.cleared.neo4j.nodes_deleted ?? 0} nodes deleted`}
              </li>
            )}
            {result.cleared.qdrant != null && (
              <li>
                Qdrant:{" "}
                {result.cleared.qdrant.status === "disabled"
                  ? "disabled"
                  : `${result.cleared.qdrant.collections_cleared?.length ?? 0} collections cleared`}
              </li>
            )}
            {result.cleared.postgres != null && (
              <li>
                Postgres:{" "}
                {result.cleared.postgres.status === "disabled"
                  ? "disabled"
                  : `${result.cleared.postgres.tables_truncated?.length ?? 0} tables truncated`}
              </li>
            )}
            {result.cleared.prometheus != null && (
              <li>
                Prometheus:{" "}
                {(() => {
                  const p = result.cleared.prometheus;
                  if (p.status === "skipped") return "skipped";
                  if (p.status === "completed")
                    return `TSDB series deleted, tombstones ${p.tombstones_cleaned ? "cleaned" : "skipped"}, ${p.in_process?.cleared_collectors ?? 0} collectors reset`;
                  if (p.status === "in_process_only")
                    return `in-process only (${p.reason ?? "remote disabled"}) — ${p.in_process?.cleared_collectors ?? 0} collectors reset`;
                  if (p.status === "admin_api_disabled")
                    return `${p.hint ?? "admin API disabled"} — in-process reset done (${p.in_process?.cleared_collectors ?? 0} collectors)`;
                  if (p.status === "unreachable")
                    return `remote unreachable (${p.error ?? ""}) — in-process reset done (${p.in_process?.cleared_collectors ?? 0} collectors)`;
                  if (p.status === "delete_failed")
                    return `delete failed (HTTP ${p.http_status ?? "?"}) — in-process reset done`;
                  return "unknown";
                })()}
              </li>
            )}
            {result.cleared.output_dir != null && (
              <li>
                Output dir:{" "}
                {result.cleared.output_dir.status === "skipped"
                  ? "skipped"
                  : `${result.cleared.output_dir.files_deleted ?? 0} files, ${fmtBytes(result.cleared.output_dir.bytes_freed)} freed`}
              </li>
            )}
            {result.cleared.progress_broker_history?.cleared && (
              <li>Progress broker history: cleared</li>
            )}
          </ul>
          {Object.keys(result.errors).length > 0 && (
            <div style={{ marginTop: 8, color: "#f85149" }}>
              <strong>Errors:</strong>
              <ul style={{ margin: "4px 0 0 20px", padding: 0 }}>
                {Object.entries(result.errors).map(([store, msg]) => (
                  <li key={store}>
                    <code>{store}</code>: {msg}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Settings form ────────────────────────────────────────────────────────────

interface SettingsFormProps {
  current: PipelineSettings;
  onSaved: (next: PipelineSettings) => void;
}

// M15 fix: SliderRow is hoisted out of SettingsForm so it isn't redefined
// on every render. The previous nested definition created a fresh function
// identity per render, which broke React's reconciliation on the <input>
// (losing focus mid-drag) and would defeat any future React.memo wrap.
interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  description: string;
  formatter?: (v: number) => string;
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
  description,
  formatter,
  disabled = false,
}: SliderRowProps & { disabled?: boolean }) {
  return (
    <div style={{ marginBottom: 16, opacity: disabled ? 0.5 : 1 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 4,
        }}
      >
        <label style={{ fontWeight: 600, fontSize: 13 }}>{label}</label>
        <span
          style={{
            fontFamily: "SF Mono, Consolas, monospace",
            color: "#58a6ff",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          {formatter ? formatter(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#58a6ff" }}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          color: "#6e7681",
          fontSize: 11,
          marginTop: 2,
        }}
      >
        <span>{formatter ? formatter(min) : min}</span>
        <span style={{ color: "#8b949e", fontSize: 11 }}>{description}</span>
        <span>{formatter ? formatter(max) : max}</span>
      </div>
    </div>
  );
}

// Compact integer input: label + number box on one line, small
// description underneath. Replaces the full-width slider for every
// setting where a discrete integer count is more natural than a drag
// (parallelism, handoff caps, sub-spec caps). Only ``spec_eval_threshold``
// kept its slider because dragging a 0.0–1.0 continuous value feels
// right there; everything else benefits from typing the exact number.
interface NumberRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  description: string;
  disabled?: boolean;
}

function NumberRow({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  description,
  disabled = false,
}: NumberRowProps) {
  return (
    <div style={{ marginBottom: 12, opacity: disabled ? 0.5 : 1 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <label style={{ fontWeight: 600, fontSize: 13, flex: 1 }}>
          {label}
        </label>
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => {
            const raw = Number(e.target.value);
            if (Number.isNaN(raw)) return;
            // Clamp to range on commit so the input can't send an
            // out-of-range value to the PATCH endpoint.
            onChange(Math.max(min, Math.min(max, raw)));
          }}
          style={{
            width: 72,
            background: "#0d1117",
            border: "1px solid #30363d",
            borderRadius: 6,
            color: "#e6edf3",
            padding: "4px 8px",
            fontSize: 13,
            fontFamily: "SF Mono, Consolas, monospace",
            textAlign: "right",
          }}
        />
      </div>
      <div
        style={{
          color: "#6e7681",
          fontSize: 11,
          marginTop: 2,
        }}
      >
        {description}{" "}
        <span style={{ color: "#484f58" }}>
          ({min}–{max})
        </span>
      </div>
    </div>
  );
}

// Small uppercase divider used to group related settings so the form
// scans top-to-bottom as parallelism → limits → behavior → quality.
function SectionLabel({ label }: { label: string }) {
  return (
    <div
      style={{
        margin: "16px 0 8px",
        color: "#8b949e",
        fontSize: 10,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: 1,
        borderBottom: "1px solid #21262d",
        paddingBottom: 4,
      }}
    >
      {label}
    </div>
  );
}

interface ToggleRowProps {
  label: string;
  value: boolean;
  description: string;
  onChange: (v: boolean) => void;
}

// Hoisted for the same reason as SliderRow — nested component definitions
// are recreated on every render and break input identity / memoisation.
function ToggleRow({ label, value, description, onChange }: ToggleRowProps) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 4,
        }}
      >
        <label style={{ fontWeight: 600, fontSize: 13, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => onChange(e.target.checked)}
            style={{ marginRight: 8, accentColor: "#58a6ff" }}
          />
          {label}
        </label>
        <span
          style={{
            fontFamily: "SF Mono, Consolas, monospace",
            color: value ? "#3fb950" : "#8b949e",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          {value ? "ENABLED" : "DISABLED"}
        </span>
      </div>
      <div style={{ color: "#8b949e", fontSize: 11, marginLeft: 24 }}>
        {description}
      </div>
    </div>
  );
}

// ── Model dropdowns ────────────────────────────────────────────────────────
//
// The dropdown options come from the backend — ``POST /api/models/anthropic``
// and ``POST /api/models/openai`` proxy the respective /v1/models endpoints
// using whatever API key is configured (server env var or per-session
// override from the API Keys section below). When no key is available or
// the upstream is unreachable, the backend falls back to a curated list
// so the dropdown is never empty.

const CUSTOM = "__custom__";

interface ModelSelectRowProps {
  label: string;
  description: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  /** "live" when fetched from the upstream /v1/models, "fallback" when
   * served from the hardcoded list, "loading" while the fetch is
   * pending. Controls the small indicator underneath the dropdown. */
  source?: "live" | "fallback" | "loading" | "error";
  /** Error message to surface next to the source indicator. */
  error?: string | null;
}

function ModelSelectRow({
  label,
  description,
  value,
  options,
  onChange,
  source,
  error,
}: ModelSelectRowProps) {
  // If the current value isn't in the curated list, treat it as
  // "custom" on load so the text input is shown pre-populated.
  const isKnown = options.includes(value);
  const [custom, setCustom] = useState(!isKnown);

  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 4,
        }}
      >
        <label style={{ fontWeight: 600, fontSize: 13, flex: 1 }}>
          {label}
        </label>
        {custom ? (
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="model-name"
            autoComplete="off"
            spellCheck={false}
            style={{
              width: 220,
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "4px 8px",
              fontSize: 12,
              fontFamily: "SF Mono, Consolas, monospace",
            }}
          />
        ) : (
          <select
            value={options.includes(value) ? value : options[0]}
            onChange={(e) => {
              if (e.target.value === CUSTOM) {
                setCustom(true);
                return;
              }
              onChange(e.target.value);
            }}
            style={{
              width: 220,
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "4px 8px",
              fontSize: 12,
              fontFamily: "SF Mono, Consolas, monospace",
            }}
          >
            {options.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
            <option value={CUSTOM}>Custom…</option>
          </select>
        )}
      </div>
      <div
        style={{
          color: "#6e7681",
          fontSize: 11,
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <span>
          {description}
          {source && (
            <span
              style={{
                marginLeft: 8,
                color:
                  source === "live"
                    ? "#3fb950"
                    : source === "loading"
                      ? "#58a6ff"
                      : source === "error"
                        ? "#f85149"
                        : "#d29922",
              }}
            >
              •{" "}
              {source === "live"
                ? "live list"
                : source === "loading"
                  ? "loading…"
                  : source === "error"
                    ? `error: ${error ?? "unknown"}`
                    : "using defaults"}
            </span>
          )}
        </span>
        {custom && (
          <button
            type="button"
            onClick={() => setCustom(false)}
            style={{
              background: "transparent",
              border: "none",
              color: "#58a6ff",
              cursor: "pointer",
              fontSize: 11,
              padding: 0,
            }}
          >
            ← back to list
          </button>
        )}
      </div>
    </div>
  );
}

// ── API key rows ───────────────────────────────────────────────────────────
//
// Thin wrapper around <input type="password"> with a label + help text.
// Values are stored in the ManufactureContext (in-memory only, never
// persisted) and sent per-run in the POST body to /api/agent/run.

interface KeyInputRowProps {
  label: string;
  help: string;
  value: string;
  placeholder: string;
  onChange: (v: string) => void;
}

function KeyInputRow({
  label,
  help,
  value,
  placeholder,
  onChange,
}: KeyInputRowProps) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 4,
        }}
      >
        <label style={{ fontWeight: 600, fontSize: 13, flex: 1 }}>
          {label}
        </label>
        <input
          type="password"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck={false}
          style={{
            width: 320,
            background: "#0d1117",
            border: "1px solid #30363d",
            borderRadius: 6,
            color: "#e6edf3",
            padding: "4px 8px",
            fontSize: 12,
            fontFamily: "SF Mono, Consolas, monospace",
          }}
        />
      </div>
      <div style={{ color: "#6e7681", fontSize: 11 }}>{help}</div>
    </div>
  );
}

// Debounce delay before re-fetching a model list after the user edits
// the API key input. Keeps us from hammering the upstream for every
// keystroke while still feeling responsive.
const MODEL_FETCH_DEBOUNCE_MS = 500;

type ModelListState =
  | { status: "loading"; models: string[] }
  | { status: "done"; source: "live" | "fallback"; models: string[] }
  | { status: "error"; error: string; models: string[] };

/** Fetch the model list for a given provider, re-running whenever the
 * API key changes (debounced). The returned ``models`` list is always
 * populated — during loading or error we keep showing the previous
 * results so the dropdown doesn't blank out. */
function useModelList(
  fetcher: (apiKey?: string) => Promise<{
    source: "live" | "fallback";
    models: ModelInfo[];
  }>,
  apiKey: string,
): ModelListState {
  const [state, setState] = useState<ModelListState>({
    status: "loading",
    models: [],
  });
  // Keep the latest successful models so transient errors don't clear
  // the dropdown options.
  const lastModelsRef = useRef<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    setState((prev) => ({ status: "loading", models: prev.models }));

    const timer = window.setTimeout(() => {
      fetcher(apiKey.trim() || undefined)
        .then((data) => {
          if (cancelled) return;
          const ids = data.models.map((m) => m.id);
          lastModelsRef.current = ids;
          setState({ status: "done", source: data.source, models: ids });
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          const message = err instanceof Error ? err.message : String(err);
          setState({
            status: "error",
            error: message,
            models: lastModelsRef.current,
          });
        });
    }, MODEL_FETCH_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [apiKey, fetcher]);

  return state;
}

function SettingsForm({ current, onSaved }: SettingsFormProps) {
  const [draft, setDraft] = useState<PipelineSettings>(current);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedFlash, setSavedFlash] = useState(false);

  // API keys live in the top-level Manufacture context (in-memory React
  // state only, never persisted) so switching tabs doesn't lose them and
  // the Manufacture tab's run button picks up the same values.
  const {
    anthropicApiKey,
    setAnthropicApiKey,
    openaiApiKey,
    setOpenaiApiKey,
    clearApiKeys,
  } = useManufacture();

  // Live model lists from the backend. Re-fetches whenever the
  // corresponding API key input changes (debounced). On error we keep
  // the last successful list so the dropdown never goes empty.
  const anthropicModels = useModelList(api.listAnthropicModels, anthropicApiKey);
  const openaiModels = useModelList(api.listOpenaiModels, openaiApiKey);
  // M16 fix: track the flash timeout so we can clear it on unmount, and so
  // repeated saves don't stack overlapping timers.
  const flashTimerRef = useRef<number | null>(null);

  // Re-sync draft when the parent re-fetches
  useEffect(() => {
    setDraft(current);
  }, [current]);

  // Clear any pending flash timer on unmount.
  useEffect(() => {
    return () => {
      if (flashTimerRef.current !== null) {
        window.clearTimeout(flashTimerRef.current);
        flashTimerRef.current = null;
      }
    };
  }, []);

  const dirty =
    draft.max_parallel_features !== current.max_parallel_features ||
    draft.max_parallel_specs !== current.max_parallel_specs ||
    draft.max_spec_handoffs !== current.max_spec_handoffs ||
    draft.max_codegen_handoffs !== current.max_codegen_handoffs ||
    draft.spec_eval_threshold !== current.spec_eval_threshold ||
    draft.enable_spec_decomposition !== current.enable_spec_decomposition ||
    draft.reuse_existing_specs !== current.reuse_existing_specs ||
    draft.max_specs_per_requirement !== current.max_specs_per_requirement ||
    draft.max_reconciliation_turns !== current.max_reconciliation_turns ||
    draft.reconciliation_timeout_seconds !==
      current.reconciliation_timeout_seconds ||
    draft.requirement_dedup_threshold !==
      current.requirement_dedup_threshold ||
    draft.enable_e2e_validation !== current.enable_e2e_validation ||
    draft.max_e2e_turns !== current.max_e2e_turns ||
    draft.e2e_timeout_seconds !== current.e2e_timeout_seconds ||
    draft.e2e_browsers.join(",") !== current.e2e_browsers.join(",") ||
    draft.enable_episodic_memory !== current.enable_episodic_memory ||
    draft.memory_dedup_threshold !== current.memory_dedup_threshold ||
    draft.llm_model !== current.llm_model ||
    draft.eval_model !== current.eval_model;

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSavedFlash(false);
    try {
      const body: PipelineSettingsUpdate = {
        max_parallel_features: draft.max_parallel_features,
        max_parallel_specs: draft.max_parallel_specs,
        max_spec_handoffs: draft.max_spec_handoffs,
        max_codegen_handoffs: draft.max_codegen_handoffs,
        spec_eval_threshold: draft.spec_eval_threshold,
        enable_spec_decomposition: draft.enable_spec_decomposition,
        reuse_existing_specs: draft.reuse_existing_specs,
        max_specs_per_requirement: draft.max_specs_per_requirement,
        max_reconciliation_turns: draft.max_reconciliation_turns,
        reconciliation_timeout_seconds: draft.reconciliation_timeout_seconds,
        requirement_dedup_threshold: draft.requirement_dedup_threshold,
        enable_e2e_validation: draft.enable_e2e_validation,
        max_e2e_turns: draft.max_e2e_turns,
        e2e_timeout_seconds: draft.e2e_timeout_seconds,
        e2e_browsers: draft.e2e_browsers,
        enable_episodic_memory: draft.enable_episodic_memory,
        memory_dedup_threshold: draft.memory_dedup_threshold,
        llm_model: draft.llm_model,
        eval_model: draft.eval_model,
      };
      const next = await api.updateSettings(body);
      onSaved(next);
      setSavedFlash(true);
      if (flashTimerRef.current !== null) {
        window.clearTimeout(flashTimerRef.current);
      }
      flashTimerRef.current = window.setTimeout(() => {
        setSavedFlash(false);
        flashTimerRef.current = null;
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setDraft(current);
    setError(null);
  };

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <div className="card-title" style={{ margin: 0 }}>
          Pipeline Tuning
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {savedFlash && (
            <span style={{ color: "#3fb950", fontSize: 12 }}>✓ Saved</span>
          )}
          {dirty && (
            <button className="btn btn-secondary" onClick={handleReset} disabled={saving}>
              Reset
            </button>
          )}
          <button
            className="btn"
            onClick={() => void handleSave()}
            disabled={!dirty || saving}
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      <p style={{ color: "#8b949e", fontSize: 12, margin: "0 0 16px" }}>
        Changes apply to the <strong>next</strong> pipeline run. Currently
        running pipelines are not affected.
      </p>

      <SectionLabel label="Models" />
      <ModelSelectRow
        label="Main LLM (spec gen + codegen swarm)"
        description="Anthropic model used by the architect/critic/swarm agents"
        value={draft.llm_model}
        options={anthropicModels.models}
        onChange={(v) => setDraft({ ...draft, llm_model: v })}
        source={
          anthropicModels.status === "loading"
            ? "loading"
            : anthropicModels.status === "error"
              ? "error"
              : anthropicModels.source
        }
        error={
          anthropicModels.status === "error" ? anthropicModels.error : null
        }
      />
      <ModelSelectRow
        label="Eval model (DeepEval judge)"
        description="OpenAI model used to score specs, tests, and generated code"
        value={draft.eval_model}
        options={openaiModels.models}
        onChange={(v) => setDraft({ ...draft, eval_model: v })}
        source={
          openaiModels.status === "loading"
            ? "loading"
            : openaiModels.status === "error"
              ? "error"
              : openaiModels.source
        }
        error={openaiModels.status === "error" ? openaiModels.error : null}
      />

      <SectionLabel label="API keys (optional)" />
      <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 10px" }}>
        When set, these override the server's{" "}
        <code>ANTHROPIC_API_KEY</code> / <code>OPENAI_API_KEY</code> env
        vars for the NEXT pipeline run. Values are kept in browser memory
        only — never persisted anywhere, never sent to any endpoint other
        than <code>/api/agent/run</code>. Leave blank to use the server's
        configured keys.
      </p>
      <KeyInputRow
        label="Anthropic API key"
        help="powers the main LLM (spec gen + codegen swarm)"
        value={anthropicApiKey}
        placeholder="sk-ant-… (leave empty to use server default)"
        onChange={setAnthropicApiKey}
      />
      <KeyInputRow
        label="OpenAI API key"
        help="powers the DeepEval judge (evaluation model)"
        value={openaiApiKey}
        placeholder="sk-… (leave empty to use server default)"
        onChange={setOpenaiApiKey}
      />
      {(anthropicApiKey || openaiApiKey) && (
        <div style={{ marginBottom: 12 }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={clearApiKeys}
            style={{ fontSize: 11, padding: "4px 10px" }}
          >
            Clear keys
          </button>
          <span
            style={{
              color: "#3fb950",
              marginLeft: 8,
              fontSize: 11,
              fontWeight: 600,
            }}
          >
            • custom keys set
          </span>
        </div>
      )}

      <SectionLabel label="Behavior" />
      <ToggleRow
        label="Spec decomposition (planner phase)"
        value={draft.enable_spec_decomposition}
        onChange={(v) => setDraft({ ...draft, enable_spec_decomposition: v })}
        description="Decompose each requirement into multiple granular sub-specs before refinement. Each sub-spec runs the full refinement loop independently."
      />
      <ToggleRow
        label="Reuse existing specs (preflight skip)"
        value={draft.reuse_existing_specs}
        onChange={(v) => setDraft({ ...draft, reuse_existing_specs: v })}
        description="Before running the refinement swarm, query Neo4j for specs whose target id already exists and pass them through unchanged. Makes re-runs on unchanged requirements effectively free. Turn OFF to force full regeneration of every spec."
      />

      <SectionLabel label="Parallelism" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          columnGap: 24,
        }}
      >
        <NumberRow
          label="Spec gen workers"
          value={draft.max_parallel_specs}
          min={1}
          max={8}
          onChange={(v) => setDraft({ ...draft, max_parallel_specs: v })}
          description="requirements processed concurrently"
        />
        <NumberRow
          label="Codegen swarm workers"
          value={draft.max_parallel_features}
          min={1}
          max={8}
          onChange={(v) => setDraft({ ...draft, max_parallel_features: v })}
          description="features executed concurrently per layer"
        />
      </div>

      <SectionLabel label="Refinement limits" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          columnGap: 24,
        }}
      >
        <NumberRow
          label="Max spec handoffs"
          value={draft.max_spec_handoffs}
          min={1}
          max={10}
          onChange={(v) => setDraft({ ...draft, max_spec_handoffs: v })}
          description="generate→evaluate→refine iterations"
        />
        <NumberRow
          label="Max sub-specs / requirement"
          value={draft.max_specs_per_requirement}
          min={1}
          max={32}
          onChange={(v) =>
            setDraft({ ...draft, max_specs_per_requirement: v })
          }
          description="hard cap on planner output"
          disabled={!draft.enable_spec_decomposition}
        />
        <NumberRow
          label="Max codegen swarm handoffs"
          value={draft.max_codegen_handoffs}
          min={5}
          max={100}
          step={5}
          onChange={(v) => setDraft({ ...draft, max_codegen_handoffs: v })}
          description="planner↔coder↔reviewer↔tester transitions"
        />
      </div>

      <SectionLabel label="Reconciliation" />
      <p
        style={{
          color: "#8b949e",
          fontSize: 11,
          margin: "0 0 10px",
        }}
      >
        After every feature swarm completes, a single Claude Agent SDK
        pass on the full run output reviews the code, fixes
        cross-feature issues, runs validation (build + tests), and
        writes a <code>RECONCILIATION_REPORT.md</code>. Always runs on
        non-empty output. Best-effort — failures do NOT fail the run.
      </p>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          columnGap: 24,
        }}
      >
        <NumberRow
          label="Max reconciliation turns"
          value={draft.max_reconciliation_turns}
          min={1}
          max={500}
          onChange={(v) =>
            setDraft({ ...draft, max_reconciliation_turns: v })
          }
          description="SDK max_turns for the deep-agent pass"
        />
        <NumberRow
          label="Reconciliation timeout (s)"
          value={draft.reconciliation_timeout_seconds}
          min={60}
          max={7200}
          step={60}
          onChange={(v) =>
            setDraft({ ...draft, reconciliation_timeout_seconds: v })
          }
          description="hard cap on the reconciliation phase"
        />
      </div>

      <SectionLabel label="Quality threshold" />
      <SliderRow
        label="Spec eval threshold"
        value={draft.spec_eval_threshold}
        min={0}
        max={1}
        step={0.05}
        onChange={(v) =>
          setDraft({ ...draft, spec_eval_threshold: Math.round(v * 100) / 100 })
        }
        description="early-exit avg score that stops the refinement loop"
        formatter={(v) => v.toFixed(2)}
      />

      <SectionLabel label="Requirement dedup" />
      <p
        style={{
          color: "#8b949e",
          fontSize: 11,
          margin: "0 0 10px",
        }}
      >
        Semantic deduplication of requirements runs automatically at
        the end of the Ingest phase, before Spec generation. A mix of
        uploaded documents (meeting notes + Word brief + spreadsheet)
        routinely describes the same requirement multiple ways; this
        threshold is the cosine similarity
        (<code>text-embedding-3-large</code>) above which two
        requirements are collapsed into a single canonical entry.
        Paraphrases typically land at 0.92–0.97, distinct requirements
        well below 0.85. Raise to tighten clusters; lower to catch
        more paraphrases at the risk of false-positive merges.
      </p>
      <SliderRow
        label="Dedup similarity threshold"
        value={draft.requirement_dedup_threshold}
        min={0.5}
        max={1}
        step={0.01}
        onChange={(v) =>
          setDraft({
            ...draft,
            requirement_dedup_threshold: Math.round(v * 100) / 100,
          })
        }
        description="cosine similarity ≥ threshold → merge into one canonical requirement"
        formatter={(v) => v.toFixed(2)}
      />

      <SectionLabel label="Episodic memory" />
      <p
        style={{
          color: "#8b949e",
          fontSize: 11,
          margin: "0 0 10px",
        }}
      >
        After every feature swarm, a clean-context LLM call
        synthesises the trajectory as an <em>Episode</em> — a
        narrative record of what strategy was picked, what went
        wrong, what fixed it, and whether the outcome was success /
        partial / failed. Episodes are embedded with{" "}
        <code>text-embedding-3-large</code> and stored in Neo4j +
        Qdrant so the Planner can call <code>recall_episodes</code>{" "}
        at the start of each feature and bias strategy selection by
        what actually worked last time. Costs ~1k LLM tokens per
        feature; unique knob in the L4 story — disable if you're not
        running repeat features.
      </p>
      <ToggleRow
        label="Enable episodic memory"
        value={draft.enable_episodic_memory}
        onChange={(v) => setDraft({ ...draft, enable_episodic_memory: v })}
        description="Synthesise + recall per-feature trajectories across runs"
      />
      <p
        style={{
          color: "#8b949e",
          fontSize: 11,
          margin: "10px 0",
        }}
      >
        <strong style={{ color: "#c9d1d9" }}>Write-time dedup:</strong>{" "}
        before creating a new Pattern / Mistake / Solution / Strategy,
        the repository embeds the candidate and cosine-matches against
        existing same-type same-feature memories. Matches above this
        threshold get boosted instead of duplicated — keeps the memory
        graph clean and prevents the recall list from filling up with
        paraphrases of the same idea. Higher than the requirement
        dedup threshold (0.90) because memory false-positives are
        harder to untangle than requirement false-positives. Set to{" "}
        <code>0.00</code> to disable dedup entirely.
      </p>
      <SliderRow
        label="Memory dedup similarity threshold"
        value={draft.memory_dedup_threshold}
        min={0}
        max={1}
        step={0.01}
        onChange={(v) =>
          setDraft({
            ...draft,
            memory_dedup_threshold: Math.round(v * 100) / 100,
          })
        }
        description="cosine similarity ≥ threshold → boost existing memory instead of creating a duplicate"
        formatter={(v) => v.toFixed(2)}
      />

      <SectionLabel label="E2E validation (Phase 6)" />
      <p
        style={{
          color: "#8b949e",
          fontSize: 11,
          margin: "0 0 10px",
        }}
      >
        After reconciliation completes, a clean-context Claude Agent
        SDK pass runs Playwright smoke tests across every selected
        browser. The agent detects whether the run produced a web
        app, installs Playwright if needed, writes smoke tests
        derived from the specs' acceptance criteria, starts the
        server, runs the tests, and writes{" "}
        <code>E2E_REPORT.md</code> plus a Playwright HTML report.
        Best-effort — E2E failures never fail the run. Browser
        binaries for chromium, firefox, and webkit are bundled in
        the Docker image.
      </p>
      <ToggleRow
        label="Enable E2E validation"
        value={draft.enable_e2e_validation}
        onChange={(v) => setDraft({ ...draft, enable_e2e_validation: v })}
        description="Run Playwright smoke tests after reconciliation"
      />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          columnGap: 24,
        }}
      >
        <NumberRow
          label="Max E2E turns"
          value={draft.max_e2e_turns}
          min={1}
          max={500}
          onChange={(v) => setDraft({ ...draft, max_e2e_turns: v })}
          description="SDK max_turns for the E2E deep-agent pass"
        />
        <NumberRow
          label="E2E timeout (s)"
          value={draft.e2e_timeout_seconds}
          min={60}
          max={7200}
          step={60}
          onChange={(v) => setDraft({ ...draft, e2e_timeout_seconds: v })}
          description="hard cap on the E2E validation phase"
        />
      </div>
      <div style={{ marginTop: 12 }}>
        <div
          style={{
            fontSize: 11,
            color: "#c9d1d9",
            marginBottom: 6,
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          Browser matrix
        </div>
        <div
          style={{ display: "flex", gap: 16, flexWrap: "wrap" }}
        >
          {(["chromium", "firefox", "webkit"] as const).map((b) => {
            const enabled = draft.e2e_browsers.includes(b);
            return (
              <label
                key={b}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 12,
                  color: enabled ? "#c9d1d9" : "#6e7681",
                  cursor: "pointer",
                  userSelect: "none",
                }}
              >
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => {
                    const next = e.target.checked
                      ? Array.from(new Set([...draft.e2e_browsers, b]))
                      : draft.e2e_browsers.filter((x) => x !== b);
                    // Force at least one browser selected so the
                    // backend validator won't reject the save with
                    // a 400. The pattern mirrors the multi-select
                    // control we use elsewhere.
                    if (next.length === 0) return;
                    // Preserve the canonical order so saved lists
                    // don't drift by click order.
                    const canonical = ["chromium", "firefox", "webkit"].filter(
                      (x) => next.includes(x),
                    );
                    setDraft({ ...draft, e2e_browsers: canonical });
                  }}
                />
                {b}
              </label>
            );
          })}
        </div>
        <div
          style={{ fontSize: 10, color: "#8b949e", marginTop: 4 }}
        >
          At least one browser must remain selected.
        </div>
      </div>

      {error && (
        <div
          style={{
            marginTop: 12,
            padding: 8,
            borderRadius: 6,
            background: "#0d1117",
            border: "1px solid #da3633",
            color: "#f85149",
            fontSize: 12,
          }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          marginTop: 16,
          padding: 12,
          background: "#0d1117",
          borderRadius: 6,
          fontSize: 11,
          color: "#8b949e",
        }}
      >
        <div>
          <strong style={{ color: "#c9d1d9" }}>Output dir:</strong>{" "}
          <code>{current.output_dir}</code>
        </div>
      </div>
    </div>
  );
}


export default function SettingsTab() {
  const { state: healthState, refresh: refreshHealth } = useHealth();
  const { statusState, start, stop, refresh: refreshStatus } = useWatcher();
  const { state: settingsState, refresh: refreshSettings } = usePipelineSettings();

  const watching =
    statusState.status === "done" && statusState.data.running;

  return (
    <div>
      {/* Pipeline Tuning — runtime-mutable settings */}
      {settingsState.status === "loading" && (
        <div className="empty-state"><p>Loading settings…</p></div>
      )}
      {settingsState.status === "error" && (
        <div className="card" style={{ borderColor: "#da3633" }}>
          <code>{settingsState.error}</code>
        </div>
      )}
      {settingsState.status === "done" && (
        <SettingsForm
          current={settingsState.data}
          onSaved={() => void refreshSettings()}
        />
      )}

      {/* Service Health */}
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
            Service Health
          </div>
          <button className="btn btn-secondary" onClick={() => void refreshHealth()}>
            Refresh
          </button>
        </div>

        {healthState.status === "loading" && (
          <p style={{ color: "#8b949e" }}>Checking services...</p>
        )}

        {healthState.status === "error" && (
          <code style={{ color: "#f85149" }}>{healthState.error}</code>
        )}

        {healthState.status === "done" && (
          <table className="table">
            <thead>
              <tr>
                <th>Service</th>
                <th>Status</th>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(healthState.data).map(([name, info]) => (
                <tr key={name}>
                  <td style={{ textTransform: "capitalize", fontWeight: 600 }}>
                    {name}
                  </td>
                  <td>
                    <span className={`health-dot ${info.ok ? "ok" : "err"}`} />
                    <span className={info.ok ? "badge-success" : "badge-error"}>
                      {info.ok ? "OK" : "Error"}
                    </span>
                  </td>
                  <td style={{ color: "#8b949e", fontSize: 12 }}>{info.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* File Watcher */}
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
            File Watcher
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary" onClick={() => void refreshStatus()}>
              Refresh
            </button>
            {watching ? (
              <button className="btn btn-danger" onClick={() => void stop()}>
                Stop Watcher
              </button>
            ) : (
              <button className="btn" onClick={() => void start()}>
                Start Watcher
              </button>
            )}
          </div>
        </div>

        {statusState.status === "done" && (
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
              }}
            >
              <span className={`health-dot ${watching ? "ok" : "err"}`} />
              <span style={{ fontWeight: 600 }}>
                {watching ? "Running" : "Stopped"}
              </span>
            </div>

            {watching && statusState.data.paths && (
              <p style={{ color: "#8b949e", margin: "0 0 8px", fontSize: 12 }}>
                Watching: {statusState.data.paths.join(", ")}
              </p>
            )}

            {watching && statusState.data.last_event != null ? (
              <LastEvent event={statusState.data.last_event} />
            ) : null}
          </div>
        )}
      </div>

      {/* Settings Summary */}
      <div className="card">
        <div className="card-title">Quick Tips</div>
        <ul style={{ color: "#8b949e", fontSize: 13, margin: 0, paddingLeft: 20 }}>
          <li>Start the file watcher to auto-detect spec changes</li>
          <li>Green services indicate successful connectivity to Neo4j and Qdrant</li>
          <li>
            Pipeline runs output to <code>output/&lt;run-id&gt;/</code> on the server
          </li>
        </ul>
      </div>

      {/* Danger Zone — last on the page so it can't be hit by accident */}
      <DangerZone />
    </div>
  );
}
