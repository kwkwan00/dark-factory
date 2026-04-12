import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type ProgressEvent } from "../api/client";
import { EVENT_BADGES, formatTime, formatEventDetails } from "../lib/agentLogFormat";

// Ring buffer size — keep the most recent N events in memory
const MAX_EVENTS = 2000;

interface LogEntry extends ProgressEvent {
  _id: string;
}


export default function AgentLogsTab() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [paused, setPaused] = useState(false);
  const [filter, setFilter] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [connected, setConnected] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Refs for things the SSE handler needs to read without re-subscribing
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // M5 fix: batched append via rAF — avoids setState-per-event churn
  const pendingRef = useRef<LogEntry[]>([]);
  const rafRef = useRef<number | null>(null);

  const flushPending = useCallback(() => {
    rafRef.current = null;
    const pending = pendingRef.current;
    if (pending.length === 0) return;
    pendingRef.current = [];
    setEntries((prev) => {
      const combined = prev.length + pending.length > MAX_EVENTS
        ? [...prev.slice(pending.length + prev.length - MAX_EVENTS), ...pending]
        : [...prev, ...pending];
      return combined;
    });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(flushPending);
  }, [flushPending]);

  // Open the SSE stream on mount
  useEffect(() => {
    const es = api.streamAgentEvents(
      (ev) => {
        if (ev.event === "log_connected") {
          setConnected(true);
        }
        if (pausedRef.current) return;
        // M4 fix: use crypto.randomUUID() for keys (unique across mounts)
        const entry: LogEntry = { ...ev, _id: crypto.randomUUID() };
        pendingRef.current.push(entry);
        scheduleFlush();
      },
      (_err, readyState) => {
        // M7: distinguish transient (CONNECTING) vs fatal (CLOSED)
        if (readyState === EventSource.CONNECTING) {
          setConnected(false);
        } else if (readyState === EventSource.CLOSED) {
          setConnected(false);
        }
      },
      // M3: onOpen handler — flip back to Live after an auto-reconnect
      () => setConnected(true),
    );
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      es.close();
    };
  }, [scheduleFlush]);

  const filtered = useMemo(() => {
    if (!filter.trim()) return entries;
    const needle = filter.toLowerCase();
    return entries.filter((e) => {
      const hay = `${e.event} ${e.feature ?? ""} ${e.agent ?? ""} ${formatEventDetails(e)}`
        .toLowerCase();
      return hay.includes(needle);
    });
  }, [entries, filter]);

  // M6 fix: auto-scroll effect depends on filtered (what's actually rendered)
  useEffect(() => {
    if (!autoScroll || !containerRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [filtered, autoScroll]);

  const jumpToLatest = () => {
    setAutoScroll(true);
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  };

  const handleClear = () => {
    // L3 fix: clear resets filter too for a clean slate
    setEntries([]);
    pendingRef.current = [];
    setFilter("");
  };

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
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div className="card-title" style={{ margin: 0 }}>
              Agent Event Stream
            </div>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                color: connected ? "#3fb950" : "#8b949e",
              }}
            >
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: connected ? "#3fb950" : "#8b949e",
                }}
              />
              {connected ? "Live" : "Disconnected"}
            </span>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {!autoScroll && filtered.length > 0 && (
              <button
                className="btn btn-secondary"
                onClick={jumpToLatest}
                aria-label="Scroll to latest event"
              >
                Jump to latest
              </button>
            )}
            <button
              className="btn btn-secondary"
              onClick={() => setPaused((p) => !p)}
              aria-label={paused ? "Resume event stream" : "Pause event stream"}
              aria-pressed={paused}
            >
              {paused ? "Resume" : "Pause"}
            </button>
            <button
              className="btn btn-secondary"
              onClick={handleClear}
              aria-label="Clear all events"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="input-row" style={{ margin: 0 }}>
          <input
            className="input-text"
            placeholder="Filter events (event type, feature, agent, text)..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              color: "#8b949e",
              fontSize: 13,
              whiteSpace: "nowrap",
              padding: "0 12px",
            }}
          >
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>
        </div>
      </div>

      {/* Log view */}
      <div className="card" style={{ padding: 0 }}>
        <div
          ref={containerRef}
          className="log-view"
          onScroll={(e) => {
            const el = e.currentTarget;
            const atBottom =
              el.scrollHeight - el.scrollTop - el.clientHeight < 40;
            if (!atBottom && autoScroll) setAutoScroll(false);
          }}
        >
          {filtered.length === 0 ? (
            <div className="empty-state" style={{ padding: 24 }}>
              <p>
                {entries.length === 0
                  ? "Waiting for agent events. Run a pipeline from the Manufacture tab."
                  : `No entries match "${filter}"`}
              </p>
            </div>
          ) : (
            filtered.map((entry) => {
              const badge = EVENT_BADGES[entry.event] ?? {
                label: entry.event.toUpperCase(),
                color: "#8b949e",
              };
              return (
                <div key={entry._id} className="log-entry">
                  <span className="log-time">{formatTime(entry.timestamp)}</span>
                  <span
                    className="log-badge"
                    style={{ color: badge.color, borderColor: badge.color }}
                  >
                    {badge.label}
                  </span>
                  <span className="log-details">
                    {formatEventDetails(entry)}
                  </span>
                </div>
              );
            })
          )}
        </div>
        <div className="log-footer">
          {filtered.length} of {entries.length} events
          {paused && " · PAUSED"}
        </div>
      </div>
    </div>
  );
}
