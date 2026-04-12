import { useState } from "react";
import type {
  SpecFailingEval,
  SpecWithoutArtifacts,
  StaleRequirement,
  UnplannedRequirement,
} from "../api/client";
import { useGraphGaps } from "../hooks/useDashboard";

// ── Helpers ────────────────────────────────────────────────────────────────

function fmtTimestamp(ts: string | null | undefined): string {
  if (!ts) return "never";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function priorityBadge(priority: string | null): string {
  if (priority === "high") return "badge-error";
  if (priority === "medium") return "badge-warn";
  return "badge-info";
}

// ── Shared section shell ───────────────────────────────────────────────────

interface GapCardProps {
  title: string;
  count: number;
  emptyHint: string;
  children: React.ReactNode;
}

function GapCard({ title, count, emptyHint, children }: GapCardProps) {
  const borderColor = count > 0 ? "#da363340" : "#30363d";
  return (
    <div className="card" style={{ borderColor }}>
      <div
        className="card-title"
        style={{ display: "flex", alignItems: "center", gap: 8 }}
      >
        <span>{title}</span>
        <span
          style={{
            color: count > 0 ? "#f85149" : "#3fb950",
            fontWeight: 400,
            fontSize: 12,
          }}
        >
          ({count})
        </span>
      </div>
      {count === 0 ? (
        <p style={{ color: "#8b949e", fontSize: 12, margin: 0 }}>{emptyHint}</p>
      ) : (
        children
      )}
    </div>
  );
}

// ── Individual gap tables ──────────────────────────────────────────────────

function UnplannedTable({ rows }: { rows: UnplannedRequirement[] }) {
  return (
    <table className="table" style={{ fontSize: 12 }}>
      <thead>
        <tr>
          <th>ID</th>
          <th>Priority</th>
          <th>Title</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}>
            <td>
              <code>{r.id}</code>
            </td>
            <td>
              <span className={priorityBadge(r.priority)}>
                {r.priority ?? "—"}
              </span>
            </td>
            <td>{r.title ?? "—"}</td>
            <td style={{ color: "#8b949e" }}>
              <code>{r.source_file ?? "—"}</code>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function NoArtifactsTable({ rows }: { rows: SpecWithoutArtifacts[] }) {
  return (
    <table className="table" style={{ fontSize: 12 }}>
      <thead>
        <tr>
          <th>Spec ID</th>
          <th>Title</th>
          <th>Capability</th>
          <th>Requirements</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((s) => (
          <tr key={s.id}>
            <td>
              <code>{s.id}</code>
            </td>
            <td>{s.title ?? "—"}</td>
            <td style={{ color: "#8b949e" }}>{s.capability ?? "—"}</td>
            <td style={{ color: "#8b949e", fontSize: 11 }}>
              {s.requirement_ids.length > 0
                ? s.requirement_ids.map((id) => <code key={id}>{id} </code>)
                : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FailingEvalsTable({ rows }: { rows: SpecFailingEval[] }) {
  return (
    <table className="table" style={{ fontSize: 12 }}>
      <thead>
        <tr>
          <th>Spec ID</th>
          <th>Title</th>
          <th>Metric</th>
          <th>Score</th>
          <th>Last eval</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((s) => (
          <tr key={s.id}>
            <td>
              <code>{s.id}</code>
            </td>
            <td>{s.title ?? "—"}</td>
            <td style={{ color: "#8b949e" }}>
              <code>{s.metric_name}</code>
            </td>
            <td>
              <span className="badge-error">{s.score.toFixed(2)}</span>
            </td>
            <td style={{ color: "#8b949e", fontSize: 11 }}>
              {fmtTimestamp(s.last_eval_at)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StaleTable({ rows }: { rows: StaleRequirement[] }) {
  return (
    <table className="table" style={{ fontSize: 12 }}>
      <thead>
        <tr>
          <th>Requirement ID</th>
          <th>Specs</th>
          <th>Most recent eval</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}>
            <td>
              <code>{r.id}</code>
            </td>
            <td>{r.spec_count}</td>
            <td style={{ color: "#8b949e", fontSize: 11 }}>
              {fmtTimestamp(r.last_eval_at)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Tab root ──────────────────────────────────────────────────────────────

export default function GapFinderTab() {
  const [staleDays, setStaleDays] = useState(7);
  const { state, refresh } = useGraphGaps(staleDays);

  if (state.status === "loading" || state.status === "idle") {
    return (
      <div className="empty-state">
        <p>Analyzing the knowledge graph…</p>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="card" style={{ borderColor: "#da3633" }}>
        <div className="card-title" style={{ color: "#f85149" }}>
          Gap finder unavailable
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

  const gaps = state.data;
  const totalGaps =
    gaps.unplanned_requirements.length +
    gaps.specs_without_artifacts.length +
    gaps.specs_failing_evals.length +
    gaps.stale_requirements.length;

  return (
    <div>
      {/* Header + controls */}
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div className="card-title" style={{ margin: 0 }}>
              Gap Finder
            </div>
            <p
              style={{
                color: "#8b949e",
                fontSize: 12,
                margin: "4px 0 0",
              }}
            >
              {gaps.totals.requirements} requirements · {gaps.totals.specs}{" "}
              specs ·{" "}
              <span
                style={{ color: totalGaps > 0 ? "#f85149" : "#3fb950" }}
              >
                {totalGaps} open gap{totalGaps === 1 ? "" : "s"}
              </span>
            </p>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label
              style={{
                color: "#8b949e",
                fontSize: 11,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              Stale after
              <input
                type="number"
                min={1}
                max={365}
                value={staleDays}
                onChange={(e) =>
                  setStaleDays(Math.max(1, Number(e.target.value) || 7))
                }
                style={{
                  background: "#0d1117",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  color: "#e6edf3",
                  padding: "4px 8px",
                  fontSize: 12,
                  width: 56,
                }}
              />
              days
            </label>
            <button
              className="btn btn-secondary"
              onClick={() => void refresh()}
            >
              Refresh
            </button>
          </div>
        </div>

        {!gaps.enabled_postgres && (
          <div
            style={{
              marginTop: 12,
              padding: 8,
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#d29922",
              fontSize: 12,
            }}
          >
            ⚠ Postgres metrics store is {gaps.postgres_error ? "unreachable" : "disabled"} —
            only the <em>unplanned requirements</em> category can be computed.
            Enable Postgres to see artifact, eval, and staleness gaps.
            {gaps.postgres_error && (
              <div style={{ marginTop: 4, color: "#8b949e", fontSize: 11 }}>
                <code>{gaps.postgres_error}</code>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Happy path: no gaps */}
      {totalGaps === 0 && (
        <div
          className="card"
          style={{ borderColor: "#3fb95040", textAlign: "center" }}
        >
          <div
            style={{
              fontSize: 36,
              marginBottom: 8,
              color: "#3fb950",
            }}
          >
            ✓
          </div>
          <div style={{ color: "#3fb950", fontWeight: 600, fontSize: 14 }}>
            No gaps found
          </div>
          <p style={{ color: "#8b949e", fontSize: 12, margin: "4px 0 0" }}>
            Every requirement is planned, every spec has artifacts, every eval
            is passing, and nothing is stale.
          </p>
        </div>
      )}

      {/* Gap cards */}
      <GapCard
        title="Unplanned requirements"
        count={gaps.unplanned_requirements.length}
        emptyHint="Every requirement has at least one implementing spec."
      >
        <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 8px" }}>
          Requirements without an IMPLEMENTS edge. The planner either missed
          these or they were added after the last run.
        </p>
        <UnplannedTable rows={gaps.unplanned_requirements} />
      </GapCard>

      <GapCard
        title="Specs without artifacts"
        count={gaps.specs_without_artifacts.length}
        emptyHint="Every spec has at least one generated file."
      >
        <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 8px" }}>
          Specs with no rows in the ``artifact_writes`` table — codegen either
          failed or was skipped for these.
        </p>
        <NoArtifactsTable rows={gaps.specs_without_artifacts} />
      </GapCard>

      <GapCard
        title="Specs failing evals"
        count={gaps.specs_failing_evals.length}
        emptyHint="Every spec's most recent eval passed."
      >
        <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 8px" }}>
          Specs whose most recent eval metric row has <code>passed=false</code>.
          Quality gap — worth re-running the swarm on these.
        </p>
        <FailingEvalsTable rows={gaps.specs_failing_evals} />
      </GapCard>

      <GapCard
        title={`Stale requirements (> ${staleDays}d)`}
        count={gaps.stale_requirements.length}
        emptyHint={`Every requirement has been re-evaluated in the last ${staleDays} days.`}
      >
        <p style={{ color: "#8b949e", fontSize: 11, margin: "0 0 8px" }}>
          Requirements whose implementing specs haven't been evaluated in the
          last {staleDays} days. May indicate drift or abandoned work.
        </p>
        <StaleTable rows={gaps.stale_requirements} />
      </GapCard>
    </div>
  );
}
