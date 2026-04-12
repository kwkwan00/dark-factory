import type { ReactNode } from "react";
import {
  Background,
  BackgroundVariant,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// ── Primitives ───────────────────────────────────────────────────────────────

interface SectionProps {
  id?: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
}

function Section({ id, title, subtitle, children }: SectionProps) {
  return (
    <div className="card" id={id} style={{ marginBottom: 16 }}>
      <div className="card-title">{title}</div>
      {subtitle && (
        <p
          style={{
            color: "#8b949e",
            fontSize: 12,
            margin: "0 0 12px",
            fontStyle: "italic",
          }}
        >
          {subtitle}
        </p>
      )}
      {children}
    </div>
  );
}

interface CalloutProps {
  tone?: "info" | "warn" | "success";
  title?: string;
  children: ReactNode;
}

function Callout({ tone = "info", title, children }: CalloutProps) {
  const border =
    tone === "warn" ? "#d29922" : tone === "success" ? "#3fb950" : "#58a6ff";
  return (
    <div
      style={{
        margin: "12px 0",
        padding: 12,
        borderLeft: `3px solid ${border}`,
        background: "#0d1117",
        borderRadius: 4,
        fontSize: 12,
        color: "#c9d1d9",
      }}
    >
      {title && (
        <div
          style={{
            fontWeight: 600,
            color: border,
            marginBottom: 4,
            fontSize: 11,
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function Para({ children }: { children: ReactNode }) {
  return (
    <p
      style={{
        margin: "0 0 10px",
        fontSize: 13,
        lineHeight: 1.6,
        color: "#c9d1d9",
      }}
    >
      {children}
    </p>
  );
}

function SubHeading({ children }: { children: ReactNode }) {
  return (
    <h3
      style={{
        fontSize: 13,
        color: "#58a6ff",
        margin: "16px 0 6px",
        textTransform: "uppercase",
        letterSpacing: 0.6,
        fontWeight: 600,
      }}
    >
      {children}
    </h3>
  );
}

function Code({ children }: { children: ReactNode }) {
  return (
    <code
      style={{
        background: "#161b22",
        border: "1px solid #30363d",
        borderRadius: 3,
        padding: "1px 5px",
        fontSize: 11,
        color: "#d2a8ff",
      }}
    >
      {children}
    </code>
  );
}

interface FactTableProps {
  rows: Array<[string, string]>;
}

function FactTable({ rows }: FactTableProps) {
  return (
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        fontSize: 12,
        margin: "8px 0 12px",
      }}
    >
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} style={{ borderBottom: "1px solid #21262d" }}>
            <td
              style={{
                padding: "6px 12px 6px 0",
                color: "#8b949e",
                width: "35%",
                verticalAlign: "top",
              }}
            >
              {k}
            </td>
            <td style={{ padding: "6px 0", color: "#c9d1d9" }}>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── React Flow helpers ───────────────────────────────────────────────────────

type Category =
  | "ui"
  | "api"
  | "service"
  | "phase"
  | "memory"
  | "agent"
  | "terminal";

const CATEGORY_STYLES: Record<
  Category,
  { bg: string; border: string; color: string }
> = {
  ui: { bg: "#0b1e35", border: "#58a6ff", color: "#79c0ff" },
  api: { bg: "#0a1f15", border: "#3fb950", color: "#7ee787" },
  service: { bg: "#2b1f05", border: "#d29922", color: "#e3b341" },
  phase: { bg: "#1e1530", border: "#bc8cff", color: "#d2a8ff" },
  memory: { bg: "#2a0f1d", border: "#f778ba", color: "#ff9bc6" },
  agent: { bg: "#2a1a08", border: "#ffa657", color: "#ffa657" },
  terminal: { bg: "#161b22", border: "#30363d", color: "#8b949e" },
};

interface NodeOpts {
  width?: number;
  height?: number;
  horizontal?: boolean;
}

function makeNode(
  id: string,
  label: ReactNode,
  x: number,
  y: number,
  cat: Category,
  opts: NodeOpts = {},
): Node {
  const { width = 160, height, horizontal = false } = opts;
  const s = CATEGORY_STYLES[cat];
  return {
    id,
    position: { x, y },
    data: { label },
    sourcePosition: horizontal ? Position.Right : Position.Bottom,
    targetPosition: horizontal ? Position.Left : Position.Top,
    style: {
      background: s.bg,
      border: `1.5px solid ${s.border}`,
      color: s.color,
      borderRadius: 6,
      fontSize: 11,
      fontFamily:
        "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
      padding: "8px 10px",
      width,
      ...(height ? { height } : {}),
      textAlign: "center" as const,
      lineHeight: 1.35,
    },
    draggable: false,
    selectable: false,
    connectable: false,
  };
}

interface EdgeOpts {
  label?: string;
  dashed?: boolean;
  color?: string;
  sourceHandle?: string;
  targetHandle?: string;
}

function makeEdge(
  id: string,
  source: string,
  target: string,
  opts: EdgeOpts = {},
): Edge {
  const stroke = opts.color ?? "#6e7681";
  return {
    id,
    source,
    target,
    type: "smoothstep",
    animated: false,
    label: opts.label,
    labelStyle: {
      fill: "#c9d1d9",
      fontSize: 10,
      fontFamily:
        "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    },
    labelBgStyle: { fill: "#0d1117" },
    labelBgPadding: [4, 2],
    labelBgBorderRadius: 2,
    style: {
      stroke,
      strokeWidth: 1.5,
      ...(opts.dashed ? { strokeDasharray: "5 4" } : {}),
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
  };
}

interface FlowFigureProps {
  nodes: Node[];
  edges: Edge[];
  caption?: string;
  height?: number;
}

function FlowFigure({
  nodes,
  edges,
  caption,
  height = 340,
}: FlowFigureProps) {
  return (
    <figure
      style={{
        margin: "16px 0",
        padding: 0,
        background: "#0d1117",
        border: "1px solid #30363d",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      <div style={{ height, width: "100%" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
          panOnScroll={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
          minZoom={0.1}
          maxZoom={2}
        >
          <Background
            variant={BackgroundVariant.Dots}
            color="#21262d"
            gap={18}
            size={1}
          />
        </ReactFlow>
      </div>
      {caption && (
        <figcaption
          style={{
            padding: "10px 16px 12px",
            color: "#6e7681",
            fontSize: 11,
            fontStyle: "italic",
            textAlign: "center",
            borderTop: "1px solid #21262d",
            background: "#0a0d12",
          }}
        >
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

// ── Multi-line label helper ──────────────────────────────────────────────────

function Label({ lines }: { lines: string[] }) {
  return (
    <div>
      {lines.map((l, i) => (
        <div
          key={i}
          style={{
            fontWeight: i === 0 ? 600 : 400,
            opacity: i === 0 ? 1 : 0.75,
            fontSize: i === 0 ? 11 : 10,
          }}
        >
          {l}
        </div>
      ))}
    </div>
  );
}

// ── Figure 1: System topology ────────────────────────────────────────────────

const SYSTEM_NODES: Node[] = [
  makeNode(
    "spa",
    <Label lines={["React SPA (Vite)", "7 tabs"]} />,
    260,
    0,
    "ui",
    { width: 180 },
  ),
  makeNode(
    "api",
    <Label lines={["FastAPI", "dark_factory.api.app"]} />,
    260,
    100,
    "api",
    { width: 180 },
  ),
  makeNode("neo4j", <Label lines={["Neo4j", "graph + memory"]} />, 0, 210, "service"),
  makeNode(
    "qdrant",
    <Label lines={["Qdrant", "vectors"]} />,
    180,
    210,
    "service",
  ),
  makeNode(
    "postgres",
    <Label lines={["Postgres", "forensic"]} />,
    360,
    210,
    "service",
  ),
  makeNode(
    "prom",
    <Label lines={["Prometheus", "+ Grafana"]} />,
    540,
    210,
    "service",
  ),
  makeNode(
    "sdk",
    <Label lines={["Claude Agent SDK", "Read/Write/Edit/Bash"]} />,
    260,
    320,
    "agent",
    { width: 200 },
  ),
];

const SYSTEM_EDGES: Edge[] = [
  makeEdge("e-spa-api", "spa", "api", { label: "AG-UI SSE + REST" }),
  makeEdge("e-api-neo4j", "api", "neo4j"),
  makeEdge("e-api-qdrant", "api", "qdrant"),
  makeEdge("e-api-postgres", "api", "postgres"),
  makeEdge("e-api-prom", "api", "prom"),
  makeEdge("e-api-sdk", "api", "sdk"),
];

// ── Figure 2: Pipeline ───────────────────────────────────────────────────────

const PIPELINE_NODES: Node[] = [
  makeNode("req", "requirements/", 200, 0, "terminal", { width: 180 }),
  makeNode(
    "p1",
    <Label lines={["Phase 1", "Ingest"]} />,
    200,
    90,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p2",
    <Label lines={["Phase 2", "Spec Generation"]} />,
    200,
    200,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p3",
    <Label lines={["Phase 3", "Knowledge Graph"]} />,
    200,
    310,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p4",
    <Label lines={["Phase 4", "Swarm Orchestration"]} />,
    200,
    420,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p5",
    <Label lines={["Phase 5", "Reconciliation"]} />,
    200,
    530,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p6",
    <Label lines={["Phase 6", "E2E Validation"]} />,
    200,
    640,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "playwright",
    <Label lines={["Playwright", "chromium · firefox · webkit"]} />,
    500,
    640,
    "agent",
    { width: 220 },
  ),
  makeNode("out", "production code", 200, 750, "terminal", { width: 180 }),
  makeNode(
    "mem",
    <Label lines={["Memory", "Neo4j + Qdrant"]} />,
    500,
    310,
    "memory",
    { width: 180 },
  ),
  makeNode(
    "eval",
    <Label lines={["DeepEval", "GPT judge"]} />,
    -120,
    200,
    "api",
    { width: 150 },
  ),
  // Phase 1 sub-steps: rich-document extraction (per-file clean-context
  // deep agent) + semantic requirement dedup (cosine similarity).
  makeNode(
    "extract",
    <Label lines={["Doc extractor", "per-file deep agent"]} />,
    -120,
    60,
    "agent",
    { width: 180 },
  ),
  makeNode(
    "dedup",
    <Label lines={["Semantic dedup", "text-embedding-3-large"]} />,
    500,
    90,
    "api",
    { width: 200 },
  ),
];

const PIPELINE_EDGES: Edge[] = [
  makeEdge("e-req-p1", "req", "p1"),
  makeEdge("e-p1-p2", "p1", "p2"),
  makeEdge("e-p2-p3", "p2", "p3"),
  makeEdge("e-p3-p4", "p3", "p4"),
  makeEdge("e-p4-p5", "p4", "p5"),
  makeEdge("e-p5-p6", "p5", "p6"),
  makeEdge("e-p6-out", "p6", "out"),
  makeEdge("e-p6-playwright", "p6", "playwright", {
    label: "smoke tests",
    color: "#ffa657",
  }),
  makeEdge("e-playwright-p6", "playwright", "p6", {
    label: "E2E_REPORT.md",
    dashed: true,
    color: "#ffa657",
  }),
  // Phase 1 sub-step branches — rich-doc extraction happens per
  // uploaded file; dedup runs once on the merged list before the
  // handoff to Phase 2.
  makeEdge("e-p1-extract", "p1", "extract", {
    label: ".docx/.xlsx/.pdf/...",
    color: "#ffa657",
  }),
  makeEdge("e-extract-p1", "extract", "p1", {
    label: "Requirement[]",
    dashed: true,
    color: "#ffa657",
  }),
  makeEdge("e-p1-dedup", "p1", "dedup", {
    label: "cluster + collapse",
    dashed: true,
    color: "#58a6ff",
  }),
  makeEdge("e-p2-eval", "p2", "eval", {
    label: "score",
    dashed: true,
    color: "#3fb950",
  }),
  makeEdge("e-p4-eval", "p4", "eval", { dashed: true, color: "#3fb950" }),
  makeEdge("e-p2-mem", "p2", "mem", { dashed: true, color: "#f778ba" }),
  makeEdge("e-p4-mem", "p4", "mem", {
    label: "recall / upsert",
    dashed: true,
    color: "#f778ba",
  }),
  makeEdge("e-p5-mem", "p5", "mem", { dashed: true, color: "#f778ba" }),
];

// ── Figure 3: Swarm ──────────────────────────────────────────────────────────

const SWARM_NODES: Node[] = [
  makeNode(
    "planner",
    <Label lines={["Planner", "eval + strategy"]} />,
    200,
    0,
    "agent",
  ),
  makeNode(
    "coder",
    <Label lines={["Coder", "RAG + SDK"]} />,
    20,
    130,
    "agent",
  ),
  makeNode(
    "tester",
    <Label lines={["Tester", "tests + eval"]} />,
    380,
    130,
    "agent",
  ),
  makeNode(
    "reviewer",
    <Label lines={["Reviewer", "DeepEval code"]} />,
    20,
    260,
    "agent",
  ),
  makeNode(
    "memory",
    <Label lines={["MemoryRepository", "pattern · mistake · solution · strategy"]} />,
    160,
    400,
    "memory",
    { width: 240 },
  ),
];

const SWARM_EDGES: Edge[] = [
  makeEdge("e-pl-co", "planner", "coder", {
    label: "transfer_to_coder",
    color: "#ffa657",
  }),
  makeEdge("e-pl-te", "planner", "tester", {
    label: "transfer_to_tester",
    color: "#ffa657",
  }),
  makeEdge("e-co-rv", "coder", "reviewer", {
    label: "transfer_to_reviewer",
    color: "#ffa657",
  }),
  makeEdge("e-rv-pl", "reviewer", "planner", {
    label: "transfer_to_planner",
    color: "#ffa657",
  }),
  makeEdge("e-te-pl", "tester", "planner", {
    label: "transfer_to_planner",
    color: "#ffa657",
  }),
  makeEdge("e-co-mem", "coder", "memory", { dashed: true, color: "#f778ba" }),
  makeEdge("e-rv-mem", "reviewer", "memory", {
    dashed: true,
    color: "#f778ba",
  }),
  makeEdge("e-te-mem", "tester", "memory", { dashed: true, color: "#f778ba" }),
];

// ── Figure 4: Memory ─────────────────────────────────────────────────────────

const MEMORY_NODES: Node[] = [
  makeNode(
    "repo",
    <Label lines={["MemoryRepository", "recall / upsert API"]} />,
    240,
    0,
    "memory",
    { width: 200 },
  ),
  makeNode(
    "neo4j",
    <Label lines={["Neo4j", "graph"]} />,
    40,
    130,
    "service",
  ),
  makeNode(
    "qdrant",
    <Label lines={["Qdrant", "vectors"]} />,
    260,
    130,
    "service",
  ),
  makeNode(
    "postgres",
    <Label lines={["Postgres", "forensic"]} />,
    480,
    130,
    "service",
  ),
  makeNode(
    "types",
    <Label lines={["Pattern · Mistake", "Solution · Strategy"]} />,
    40,
    250,
    "terminal",
    { width: 180 },
  ),
  makeNode(
    "episodes",
    <Label lines={["Episode", "narrative trajectories"]} />,
    40,
    340,
    "memory",
    { width: 180 },
  ),
  makeNode(
    "vecs",
    <Label lines={["text-embedding-3-large", "3072 dim"]} />,
    240,
    250,
    "terminal",
    { width: 200 },
  ),
  makeNode(
    "forensic",
    <Label lines={["llm_calls · eval_results", "incidents · tool_calls"]} />,
    480,
    250,
    "terminal",
    { width: 200 },
  ),
  makeNode(
    "rrf",
    <Label lines={["Reciprocal Rank Fusion", "hybrid merge on every recall"]} />,
    190,
    380,
    "api",
    { width: 260 },
  ),
];

const MEMORY_EDGES: Edge[] = [
  makeEdge("e-repo-neo4j", "repo", "neo4j"),
  makeEdge("e-repo-qdrant", "repo", "qdrant"),
  makeEdge("e-repo-postgres", "repo", "postgres"),
  makeEdge("e-neo4j-types", "neo4j", "types"),
  makeEdge("e-neo4j-episodes", "neo4j", "episodes", {
    label: "Episode nodes",
    color: "#f778ba",
  }),
  makeEdge("e-qdrant-vecs", "qdrant", "vecs"),
  makeEdge("e-postgres-forensic", "postgres", "forensic"),
  makeEdge("e-types-rrf", "types", "rrf", { dashed: true, color: "#58a6ff" }),
  makeEdge("e-episodes-rrf", "episodes", "rrf", {
    dashed: true,
    color: "#f778ba",
  }),
  makeEdge("e-vecs-rrf", "vecs", "rrf", { dashed: true, color: "#58a6ff" }),
];

// ── Figure 5: Observability ──────────────────────────────────────────────────

const OBS_NODES: Node[] = [
  makeNode(
    "instr",
    <Label lines={["phase · tool · llm · eval", "instrumentation points"]} />,
    240,
    0,
    "terminal",
    { width: 220 },
  ),
  makeNode(
    "helpers",
    <Label lines={["metrics/helpers.py", "observe_* · record_incident"]} />,
    240,
    110,
    "api",
    { width: 220 },
  ),
  makeNode("prom", "Prometheus", 20, 230, "service", { width: 160 }),
  makeNode("pg", "Postgres", 230, 230, "service", { width: 160 }),
  makeNode("broker", "ProgressBroker", 440, 230, "service", { width: 180 }),
  makeNode("grafana", "Grafana dashboards", 20, 350, "ui", { width: 160 }),
  makeNode("apiep", "/api/metrics/*", 230, 350, "ui", { width: 160 }),
  makeNode("logs", "Agent Logs tab", 440, 350, "ui", { width: 180 }),
];

const OBS_EDGES: Edge[] = [
  makeEdge("e-instr-helpers", "instr", "helpers"),
  makeEdge("e-helpers-prom", "helpers", "prom"),
  makeEdge("e-helpers-pg", "helpers", "pg"),
  makeEdge("e-helpers-broker", "helpers", "broker"),
  makeEdge("e-prom-grafana", "prom", "grafana"),
  makeEdge("e-pg-apiep", "pg", "apiep"),
  makeEdge("e-broker-logs", "broker", "logs"),
];

// ── Figure 6: Cancellation ───────────────────────────────────────────────────

const CANCEL_NODES: Node[] = [
  makeNode("btn", "UI: Cancel button", 20, 0, "ui", { width: 180 }),
  makeNode(
    "post",
    "POST /api/agent/cancel",
    20,
    100,
    "api",
    { width: 180 },
  ),
  makeNode(
    "flag",
    <Label lines={["set_cancel_flag()", "threading.Event.set()"]} />,
    20,
    200,
    "service",
    { width: 200 },
  ),
  makeNode(
    "worker",
    <Label lines={["Agent worker thread", "running a phase"]} />,
    380,
    0,
    "agent",
    { width: 200 },
  ),
  makeNode(
    "check",
    <Label lines={["is_cancelled()", "polled at every checkpoint"]} />,
    380,
    130,
    "api",
    { width: 220 },
  ),
  makeNode(
    "raise",
    "raise PipelineCancelled",
    380,
    250,
    "service",
    { width: 220 },
  ),
  makeNode(
    "cleanup",
    <Label lines={["finally: cleanup", "status = \"cancelled\""]} />,
    380,
    360,
    "terminal",
    { width: 220 },
  ),
];

const CANCEL_EDGES: Edge[] = [
  makeEdge("e-btn-post", "btn", "post"),
  makeEdge("e-post-flag", "post", "flag"),
  makeEdge("e-flag-check", "flag", "check", {
    label: "flag",
    dashed: true,
    color: "#d29922",
  }),
  makeEdge("e-worker-check", "worker", "check"),
  makeEdge("e-check-raise", "check", "raise"),
  makeEdge("e-raise-cleanup", "raise", "cleanup"),
];

// ── Main component ───────────────────────────────────────────────────────────

export default function AboutTab() {
  return (
    <div>
      {/* Hero */}
      <div
        className="card"
        style={{
          marginBottom: 16,
          background:
            "linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%)",
          borderColor: "#30363d",
        }}
      >
        <div
          style={{
            fontSize: 20,
            fontWeight: 700,
            color: "#e6edf3",
            marginBottom: 6,
          }}
        >
          AI Dark Factory — Architecture Whitepaper
        </div>
        <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 12 }}>
          An autonomous L4 code generation platform. Requirements enter,
          production-quality code comes out. This document explains how.
        </div>
        <FactTable
          rows={[
            ["Author", "Kevin Quon — linkedin.com/in/kwkwan00"],
            ["Backend", "FastAPI + Uvicorn (Python 3.12+)"],
            ["Frontend", "React 18 + Vite + TypeScript"],
            [
              "Agent runtime",
              "LangGraph swarm (create_swarm) + Claude Agent SDK",
            ],
            ["Knowledge store", "Neo4j (graph + procedural memory)"],
            [
              "Vector store",
              "Qdrant (text-embedding-3-large, 3072 dim)",
            ],
            [
              "Metrics",
              "Prometheus + Grafana + optional Postgres forensic store",
            ],
            ["Protocol", "AG-UI SSE events for real-time UI streaming"],
            ["Agentic level", "L4 — Fully Autonomous / Explorer"],
          ]}
        />
      </div>

      {/* Business Value */}
      <Section
        title="Business Value"
        subtitle="What this platform delivers — in plain terms"
        id="business-value"
      >
        <Para>
          AI Dark Factory takes the gap between{" "}
          <em>&ldquo;here is what we want&rdquo;</em> and{" "}
          <em>&ldquo;here is running, tested code&rdquo;</em> and closes
          it autonomously. A human writes requirements; the system
          produces production-quality code, tests, evaluation reports,
          and a reconciled integration — without human intervention
          between phases. Every run teaches the platform something that
          makes the next run faster and better.
        </Para>
        <SubHeading>Outcomes</SubHeading>
        <FactTable
          rows={[
            [
              "Faster time-to-code",
              "Requirements become working, tested code in a single invocation. Engineers supervise outcomes instead of typing them.",
            ],
            [
              "Continuous improvement",
              "Procedural memory persists across runs. The system remembers what worked, what broke, and how it was fixed — and applies those lessons automatically next time.",
            ],
            [
              "Predictable quality",
              "Every spec and every artifact is evaluated by an independent LLM judge against an explicit rubric. No artifact ships without a score.",
            ],
            [
              "Full cost visibility",
              "Every LLM call, token count, tool invocation, and eval result is recorded. Operators see exactly what a run cost, where the time went, and which features drove the spend.",
            ],
            [
              "Graceful degradation",
              "A single failing feature does not take down a run. Reconciliation failures do not block delivery. Cancellation is clean and leaves no orphaned state.",
            ],
            [
              "Low operational overhead",
              "Runtime settings are live-mutable from the UI — parallelism, handoff limits, model selection, eval thresholds — all without redeploying.",
            ],
            [
              "Auditable decisions",
              "Every agent action, handoff, tool call, and evaluation is logged, timestamped, and replayable in the Run Detail popup. Every run leaves a complete forensic trail.",
            ],
            [
              "Vendor flexibility",
              "Model selection is per-run and per-agent. Anthropic for codegen, OpenAI for evaluation — or swap either independently. API keys are injected per-request, not baked in.",
            ],
          ]}
        />
        <SubHeading>Core features for business</SubHeading>
        <FactTable
          rows={[
            [
              "Requirements ingestion",
              "Drop any mix of Markdown, JSON, YAML, OpenSpec, Word, Excel, PDF, HTML, XML, RTF, CSV, or transcript files. Native formats parse directly; rich business documents are routed through a clean-context Claude Agent SDK invocation that extracts discrete requirements per document.",
            ],
            [
              "Automatic specification",
              "Requirements become formal specs with acceptance criteria, dependencies, and capability tags — scored for correctness, coherence, instruction-following, and safety.",
            ],
            [
              "Per-feature codegen swarms",
              "Four specialised agents (Planner, Coder, Reviewer, Tester) collaborate per feature. Features run in parallel within dependency layers.",
            ],
            [
              "Cross-feature reconciliation",
              "After every run, a final polishing pass reviews the entire output, fixes broken integrations, runs the test suite, and produces a reconciliation report.",
            ],
            [
              "Cross-browser E2E validation",
              "After reconciliation, a Playwright smoke-test pass exercises the generated application in chromium, firefox, and webkit. Test results, failure screenshots, and a browsable HTML report are surfaced in the Run Detail popup so the operator can verify the app actually runs in a real browser — not just that it compiled.",
            ],
            [
              "Run history + replay",
              "Every run is persisted with spec counts, pass rate, mean eval scores, worst features, and full event timeline. The Manufacture tab shows the 10 most recent runs with datetime, status, pass rate, and duration. Click any run ID to open a detail popup with five tabs: Metrics, Agent Log, Evaluations, Output, and Episodes.",
            ],
            [
              "Gap analysis",
              "A dedicated dashboard surfaces unplanned requirements, stale specs, unimplemented features, and failing evaluations — so nothing silently falls through the cracks.",
            ],
            [
              "Live observability",
              "Real-time agent logs with color-coded event badges, comprehensive metrics dashboards, Prometheus + Grafana integration, per-run cost rollups, and per-run agent log replay with expandable JSON payloads. Know what the system is doing and what it has done.",
            ],
            [
              "File-watcher automation",
              "Optional: watch a requirements directory and auto-run the pipeline on change. Enables continuous delivery from spec edits to code.",
            ],
            [
              "Safe cancellation",
              "A single Cancel button cleanly stops any running pipeline at the next checkpoint, with no orphaned state or half-written records.",
            ],
            [
              "Deep agent resilience",
              "Claude Agent SDK subprocess crashes are caught and converted to soft errors via _safe_tool_deep_agent. Stderr is buffered for diagnostics, and --debug-to-stderr can be enabled via DEEP_AGENT_DEBUG_STDERR for investigating silent exits. A regression test verifies all tool-decorated deep agents use the safe wrapper.",
            ],
            [
              "Danger-zone reset",
              "One click (with confirmation) wipes all state and starts fresh — for environment hand-offs, demo resets, or recovery scenarios.",
            ],
          ]}
        />
        <Callout tone="success" title="Who this is for">
          Engineering teams that want to let a coordinated agent swarm
          handle routine feature delivery while humans focus on
          architectural direction, requirements quality, and review of
          the final output. Not a replacement for engineers — a force
          multiplier for them.
        </Callout>
      </Section>

      {/* 1. Philosophy */}
      <Section title="1. Design Philosophy" id="philosophy">
        <Para>
          AI Dark Factory is built on a single premise:{" "}
          <strong>
            a coordinated swarm of specialised agents with persistent
            memory produces higher-quality code than a single monolithic
            agent with a bigger context window.
          </strong>{" "}
          Every architectural choice in this system flows from that
          premise.
        </Para>
        <Para>
          The pipeline is decomposed into six well-defined phases. Each
          phase has a narrow contract, its own evaluation criteria, and a
          clean failure boundary. Agents operate in tight feedback loops
          with external judges (DeepEval + GPT), write what they learn
          into a shared memory graph, and benefit from every prior run —
          even partial ones.
        </Para>
        <SubHeading>Core tenets</SubHeading>
        <FactTable
          rows={[
            [
              "Specialisation over scale",
              "Four small agents with distinct roles beat one big agent told to do everything.",
            ],
            [
              "Memory is first-class",
              "Procedural memory is a graph, not a cache. Every agent reads it before acting and writes lessons after.",
            ],
            [
              "Evaluation is a dependency",
              "LLM judges run on every spec and every artifact. Scores feed back into memory relevance and future threshold tuning.",
            ],
            [
              "Best-effort polishing",
              "Phase 5 reconciliation never fails the pipeline. The worst case is that feature output ships as-is.",
            ],
            [
              "Observability by construction",
              "Every tool call, phase, handoff, and incident is counted, timed, and streamed to the UI in real time.",
            ],
            [
              "Cooperative cancellation",
              "A single threading.Event is polled at every checkpoint so the user can stop a run cleanly at any moment.",
            ],
          ]}
        />
      </Section>

      {/* 2. System overview */}
      <Section
        title="2. System Overview"
        subtitle="How the pieces fit together"
        id="system"
      >
        <Para>
          The runtime is a single-process FastAPI application that hosts
          both the REST API and the React SPA (served from{" "}
          <Code>src/dark_factory/api/static/</Code>). A background daemon
          asyncio loop (the <Code>BackgroundLoop</Code> singleton) runs
          every Claude Agent SDK invocation on a dedicated thread so that
          subprocess cleanup callbacks always have a valid loop to land
          on — this eliminates a whole class of &ldquo;Event loop is
          closed&rdquo; errors that plagued earlier iterations.
        </Para>
        <FlowFigure
          nodes={SYSTEM_NODES}
          edges={SYSTEM_EDGES}
          caption="Figure 1 — High-level system topology"
          height={420}
        />
        <Para>
          Agents communicate with the outside world through a small set
          of services that are each replaceable and individually
          health-checked:
        </Para>
        <FactTable
          rows={[
            [
              "Neo4j",
              "Primary store for specs, requirements, dependencies, runs, and procedural memory nodes.",
            ],
            [
              "Qdrant",
              "Hybrid RAG: similar-spec / similar-code / memory semantic search via text-embedding-3-large.",
            ],
            [
              "Postgres",
              "Optional forensic row store for LLM calls, tool calls, eval results, incidents, progress events.",
            ],
            [
              "Prometheus",
              "Always-on in-process counters and histograms — zero external dependency.",
            ],
            [
              "Grafana",
              "Pre-provisioned dashboards over the Prometheus scrape target.",
            ],
            [
              "Claude Agent SDK",
              "File I/O + Bash tools for code generation and Phase 5 reconciliation.",
            ],
          ]}
        />
      </Section>

      {/* 3. Pipeline */}
      <Section
        title="3. The Six-Phase Pipeline"
        subtitle="From requirements to production-quality code"
        id="pipeline"
      >
        <Para>
          A single <Code>POST /api/agent/run</Code> call drives the
          entire pipeline. The request returns immediately; subsequent
          events stream to the browser over{" "}
          <Code>GET /api/agent/events</Code> using the AG-UI SSE format.
        </Para>
        <FlowFigure
          nodes={PIPELINE_NODES}
          edges={PIPELINE_EDGES}
          caption="Figure 2 — Pipeline data flow across all six phases"
          height={680}
        />

        <SubHeading>Phase 1 — Ingest</SubHeading>
        <Para>
          <Code>IngestStage</Code> parses a requirements directory into{" "}
          <Code>Requirement</Code> domain models. Two tiers of input
          are supported. <strong>Native formats</strong> (
          <Code>.md</Code>, <Code>.txt</Code>, <Code>.json</Code>,{" "}
          <Code>.yaml</Code>, OpenSpec <Code>specs/</Code> trees) are
          parsed directly; OpenSpec inputs extract WHEN/THEN scenarios
          into structured acceptance criteria and large text documents
          go through an LLM splitter that breaks each discrete testable
          requirement into its own entry.
        </Para>
        <Para>
          <strong>Rich business documents</strong> (Word, Excel,
          PowerPoint, PDF, HTML, XML, RTF, CSV, transcripts) are routed
          through a <strong>clean-context Claude Agent SDK invocation</strong>{" "}
          per file. Each invocation runs in a fresh subprocess with its
          cwd set to the upload directory and has access to{" "}
          <Code>Read / Write / Edit / Glob / Grep / Bash</Code>. The
          agent reads the document with the appropriate Python library
          (<Code>python-docx</Code>, <Code>openpyxl</Code>,{" "}
          <Code>pypdf</Code>, <Code>beautifulsoup4</Code>,{" "}
          <Code>striprtf</Code>, <Code>lxml</Code>), extracts discrete
          testable requirements, and writes a staging JSON file that
          the ingest stage loads back into <Code>Requirement</Code>{" "}
          models. Because each document gets its own isolated agent
          context, raw meeting-transcript noise never pollutes the
          main pipeline's LLM context.
        </Para>
        <Para>
          After all files are parsed, the stage runs a{" "}
          <strong>semantic deduplication pass</strong>. Every
          requirement is embedded with{" "}
          <Code>text-embedding-3-large</Code>; near-duplicates (cosine
          similarity ≥ <Code>requirement_dedup_threshold</Code>,
          default 0.90) are clustered and collapsed into a single
          canonical entry — preferring the highest-priority + most
          detailed member, breaking ties by original position so
          re-runs are deterministic. Tags from every merged requirement
          are unioned onto the canonical so source-document attribution
          is never lost. This is a <em>correctness guarantee</em>: a
          corpus assembled from a meeting transcript, a Word brief,
          and a spreadsheet routinely describes the same requirement
          multiple ways, and without dedup the Spec stage would burn
          LLM budget generating duplicate specs for each. Embedding
          outages fall back to the un-deduped list rather than
          blocking the pipeline.
        </Para>
        <Callout tone="info" title="Observability">
          When dedup merges anything, the Agent Logs tab shows a
          <Code>requirements_deduped</Code> event with the input/
          output counts and a preview of each merge cluster (
          <em>&ldquo;kept X, merged Y, Z&rdquo;</em>). The first 5
          clusters are listed inline; larger dedups link to the
          structured log.
        </Callout>

        <SubHeading>Phase 2 — Spec Generation</SubHeading>
        <Para>
          Each requirement optionally runs through a{" "}
          <strong>planner decomposition</strong> step that splits it into
          multiple granular sub-specs. Every sub-spec then enters a
          dedicated <strong>architect → critic → refine</strong> loop
          that iterates until either the DeepEval score crosses{" "}
          <Code>spec_eval_threshold</Code> or the handoff budget is
          exhausted. A preflight skip checks Neo4j for specs with
          already-existing target ids and passes them through unchanged
          — this is what makes re-running the pipeline on an unchanged
          requirements set effectively free.
        </Para>
        <Callout tone="info" title="DeepEval rubric">
          Every spec is scored on four GEval metrics: Correctness,
          Coherence, Instruction Following (does it match the OpenSpec
          shape?), and Safety &amp; Ethics. GPT acts as judge. Scores
          feed both the early-exit threshold and the adaptive memory
          decay in Phase 4.
        </Callout>

        <SubHeading>Phase 3 — Knowledge Graph</SubHeading>
        <Para>
          Specs and requirements are persisted to Neo4j with{" "}
          <Code>IMPLEMENTS</Code> and <Code>DEPENDS_ON</Code>{" "}
          relationships. Specs are simultaneously auto-indexed into the
          Qdrant <Code>dark_factory_specs</Code> collection so that
          downstream Coder agents can pull in semantically-similar work
          from other features as RAG context.
        </Para>

        <SubHeading>Phase 4 — Swarm Orchestration</SubHeading>
        <Para>
          The orchestrator groups specs by <Code>capability</Code> and
          uses Tarjan&apos;s SCC algorithm to compute a{" "}
          <strong>cycle-tolerant topological order</strong>. Features
          within the same dependency layer run in parallel, bounded by{" "}
          <Code>max_parallel_features</Code>. Each feature spawns an
          isolated LangGraph swarm with four specialised agents (see
          Section 4).
        </Para>
        <Para>
          After each layer, the orchestrator reviews the layer&apos;s
          pass rate. If it drops below threshold, the Coder is forced
          onto the SDK path for all remaining features and the handoff
          budget is tightened. When performance recovers on a later
          layer, the overrides relax. This is the{" "}
          <strong>adaptive strategy override</strong> — a cheap but
          powerful lever that prevents a degrading run from eating its
          whole budget.
        </Para>

        <SubHeading>Phase 5 — Reconciliation</SubHeading>
        <Para>
          The youngest and arguably most important phase. A{" "}
          <strong>single extended Claude Agent SDK invocation</strong>{" "}
          runs over the full run output directory with the complete{" "}
          <Code>Read / Write / Edit / Glob / Grep / Bash</Code> tool
          set. It follows a rigid six-step checklist: inventory →
          review → fix → validate → iterate → report. Because it can
          see every feature at once, it catches the class of bugs that
          per-feature swarms structurally cannot — broken cross-feature
          imports, inconsistent API shapes between frontend and backend,
          missing <Code>main.py</Code> / <Code>package.json</Code> /{" "}
          <Code>requirements.txt</Code> glue, and runtime-only failures
          that manifest when the pieces are assembled.
        </Para>
        <Callout tone="warn" title="Best-effort contract">
          Phase 5 is intentionally wrapped in a broad try/except.
          Timeouts, crashes, and SDK failures are logged, recorded as
          incidents, surfaced in the Run Detail popup — and then{" "}
          <em>swallowed</em>. The pipeline always delivers feature
          output, even if reconciliation couldn&apos;t run. Polishing,
          not gatekeeping.
        </Callout>

        <SubHeading>Phase 6 — End-to-End Validation</SubHeading>
        <Para>
          The final phase. After reconciliation completes with a{" "}
          <Code>clean</Code> status (not just &ldquo;not error&rdquo; —{" "}
          <Code>partial</Code>, <Code>skipped</Code>, and{" "}
          <Code>error</Code> all gate this phase off), a second{" "}
          <strong>clean-context Claude Agent SDK invocation</strong>{" "}
          runs a <strong>Playwright cross-browser smoke test suite</strong>{" "}
          against the generated application. The agent follows a rigid
          six-step checklist — detect → install → plan → write → run
          → report — and has access to{" "}
          <Code>Read / Write / Edit / Glob / Grep / Bash</Code>. Browser
          binaries for <Code>chromium</Code>, <Code>firefox</Code>, and{" "}
          <Code>webkit</Code> are pre-installed in the Docker image at{" "}
          <Code>/ms-playwright</Code>, so the agent does not have to
          pay the download cost on every run.
        </Para>
        <Para>
          In Step 1 the agent checks whether the output is actually a
          web application (package manifests, <Code>Dockerfile</Code>{" "}
          ports, HTML entry points, framework markers); if not, it
          writes a <Code>skipped</Code> report and stops. Step 3 picks
          3–8 user-facing acceptance criteria from the specs and
          generates a minimal <Code>e2e/smoke.spec.ts</Code> plus a{" "}
          <Code>playwright.config.ts</Code> that enables every browser
          in the matrix. Step 5 starts the server in the background
          with a shell trap so it is always killed on exit, polls the
          health endpoint for readiness, and runs the suite with the{" "}
          <Code>line</Code> + <Code>html</Code> reporters. Step 6
          writes <Code>E2E_REPORT.md</Code> at the run output root
          with a per-test / per-browser table, failure reasons, and
          an <Code>Overall status</Code> of <Code>pass</Code> /{" "}
          <Code>partial</Code> / <Code>broken</Code>.
        </Para>
        <Para>
          The Playwright HTML report is captured to{" "}
          <Code>e2e_artifacts/html-report/</Code> and any failure
          screenshots end up under <Code>e2e_artifacts/</Code> — both
          are surfaced inline in the Run Detail popup so the operator
          can click through a failing test without leaving the UI.
          Per-browser test counts are fanned out to the{" "}
          <Code>dark_factory_e2e_tests_total{"{"}browser, status{"}"}</Code>{" "}
          Prometheus counter for dashboards that need to answer
          &ldquo;is webkit the flaky one?&rdquo;.
        </Para>
        <Callout tone="warn" title="Best-effort contract">
          Phase 6 inherits the same policy as Phase 5: agent crashes,
          timeouts, server startup failures, and flaky tests are all
          logged, recorded as incidents, and then <em>swallowed</em>.
          A broken E2E pass never fails the run. The pipeline always
          delivers the reconciled feature output, and the operator
          reads the E2E report to decide whether the delivery is
          ship-ready. When E2E is skipped due to reconciliation
          status, a skip-reason text event is emitted to the Agent
          Log so operators see exactly why.
        </Callout>
      </Section>

      {/* 3.5 Deep Agent Resilience */}
      <Section
        title="3.5 Deep Agent Resilience"
        subtitle="How the system handles Claude Agent SDK subprocess crashes"
        id="deep-agent-resilience"
      >
        <Para>
          The Claude Agent SDK runs as a Node.js subprocess that can
          crash silently — OOM, segfault, network timeout — with no
          useful stderr output. The SDK&apos;s default error message is
          a placeholder (&ldquo;Check stderr output for details&rdquo;)
          that provides zero diagnostics. Three layers of defense keep
          these crashes from cascading into feature failures:
        </Para>
        <FactTable
          rows={[
            [
              "_safe_tool_deep_agent",
              "A wrapper used by all 9 @tool-decorated deep-agent functions (codegen, dependency analysis, risk/security/performance/compliance review, unit/integration/edge-case test gen). Catches Exception and returns a structured error string instead of re-raising, so the LangGraph swarm treats it as a soft tool failure and continues. Non-tool callers (reconciliation, doc extraction, E2E) still see the exception for their own error handling. Only catches Exception, not BaseException — KeyboardInterrupt and SystemExit propagate normally.",
            ],
            [
              "Stderr capture",
              "Every SDK invocation buffers up to 200 lines / 16 KiB of subprocess stderr via a callback. When a crash occurs, the tail is logged alongside the incident for diagnostics. A truncation flag fires when the buffer caps are reached so the operator knows output was clipped.",
            ],
            [
              "DEEP_AGENT_DEBUG_STDERR",
              "When set to 1/true, the underlying Node CLI is spawned with --debug-to-stderr so it emits startup, transport, and protocol-state lines. Operators flip this on while investigating silent exits to see what the subprocess was doing before it died. Off by default to keep logs clean.",
            ],
          ]}
        />
        <Callout tone="info" title="Regression guard">
          A source-level test (<Code>test_all_tool_decorated_deep_agents_use_safe_wrapper</Code>)
          parses <Code>tools.py</Code> and verifies that every{" "}
          <Code>@tool</Code> function calling <Code>_run_deep_agent</Code>{" "}
          actually calls <Code>_safe_tool_deep_agent</Code>. If a
          developer adds a new deep-agent tool and forgets the safe
          wrapper, the test fails.
        </Callout>
      </Section>

      {/* 4. Swarm */}
      <Section
        title="4. Swarm Mechanics"
        subtitle="Four agents, one feature, bounded handoffs"
        id="swarm"
      >
        <Para>
          Each feature in Phase 4 runs its own isolated swarm built on{" "}
          <Code>langgraph-swarm.create_swarm</Code>. The agents rotate
          via named handoff tools — there is no central dispatcher,
          each agent decides who should act next based on its own
          system prompt and the current state.
        </Para>
        <FlowFigure
          nodes={SWARM_NODES}
          edges={SWARM_EDGES}
          caption="Figure 3 — Per-feature swarm handoff topology. Solid orange edges are LangGraph transfer tools; dashed pink edges are memory reads/writes."
          height={480}
        />
        <FactTable
          rows={[
            [
              "Planner",
              "Reads the spec, queries eval history, recalls strategies from memory, and decides whether to code, test, or call it done.",
            ],
            [
              "Coder",
              "Searches similar specs and similar code via Qdrant RAG, recalls patterns + mistakes, then either writes code directly or delegates to claude_agent_codegen (a Claude Agent SDK deep-agent).",
            ],
            [
              "Reviewer",
              "Scores generated code on DeepEval correctness / coherence / security / style. Records mistakes and solutions that future runs will recall.",
            ],
            [
              "Tester",
              "Writes tests against acceptance criteria. Evaluates test correctness, coherence, and completeness. Records failures as mistakes so that future Testers don't repeat them.",
            ],
          ]}
        />
        <Para>
          A swarm is bounded by a handoff budget (
          <Code>max_codegen_handoffs</Code>, default 50). If the budget
          runs out without a completed feature, the orchestrator records
          a failure, surfaces an incident, and continues to the next
          layer. The run still proceeds — a single failing feature does
          not take down the pipeline.
        </Para>
      </Section>

      {/* 5. Memory */}
      <Section
        title="5. Procedural Memory"
        subtitle="How the system learns across runs"
        id="memory"
      >
        <Para>
          Memory is the mechanism by which the system improves over
          time. It is split into two tiers — <strong>semantic</strong>{" "}
          memory (generalised lessons — what you should do) and{" "}
          <strong>episodic</strong> memory (specific past trajectories
          — what actually happened last time). Both tiers are stored
          in a dedicated Neo4j database with embeddings mirrored into
          Qdrant for hybrid semantic + keyword recall. Every agent
          reads memory before acting and writes memory after.
        </Para>
        <FlowFigure
          nodes={MEMORY_NODES}
          edges={MEMORY_EDGES}
          caption="Figure 4 — Memory topology. Solid edges persist to a store; dashed edges feed the hybrid RRF merge used by every recall_memories() and recall_episodes() call. Pink nodes are the semantic tier + new episodic tier; blue is the hybrid merge."
          height={600}
        />
        <SubHeading>Semantic tier</SubHeading>
        <Para>
          Four node types that encode generalised lessons — strip the
          temporal context and keep the teachable nugget.
        </Para>
        <FactTable
          rows={[
            [
              "Pattern",
              "Reusable code structures the Coder recognises as good. Written by Coder, read by Coder.",
            ],
            [
              "Mistake",
              "A concrete failure mode with a root cause. Written by Reviewer + Tester, read by all agents.",
            ],
            [
              "Solution",
              "A fix that resolved a mistake. Paired with the Mistake node it fixes. Written by Reviewer + Tester, read by all agents.",
            ],
            [
              "Strategy",
              "High-level approach decisions. Written by Planner, read by Planner in future runs.",
            ],
          ]}
        />
        <SubHeading>Episodic tier</SubHeading>
        <Para>
          After every feature swarm completes, the orchestrator
          spawns a small clean-context LLM call that synthesises a{" "}
          <strong>200-word narrative summary</strong> of the
          trajectory plus <strong>3–8 key turning-point events</strong>{" "}
          (strategy picks, rejections, pivots, test passes). The
          result is stored as an <Code>Episode</Code> node in Neo4j
          linked to its <Code>Run</Code> via <Code>PRODUCED_IN</Code>,
          and embedded into the <Code>dark_factory_episodes</Code>{" "}
          Qdrant collection via <Code>text-embedding-3-large</Code>.
          The writer is{" "}
          <strong>best-effort</strong>: a Neo4j outage, Qdrant hiccup,
          or embedding failure logs a warning but never propagates
          out — losing an episode is acceptable, breaking the run
          because we couldn&apos;t log one is not.
        </Para>
        <Para>
          At the start of every feature, the Planner calls{" "}
          <Code>recall_episodes(feature_name=&lt;current&gt;)</Code>{" "}
          which runs the same hybrid RRF merge used by{" "}
          <Code>recall_memories</Code> — Neo4j keyword match on the
          summary text + Qdrant vector match on the embedding —
          returning the top-ranked past trajectories for the same
          feature. If a previous run succeeded with a specific
          approach, the Planner biases toward it. If a previous run
          failed a particular way, the Planner steers clear of the
          same mode. This gives the Dark Factory a{" "}
          <strong>temporal reasoning layer</strong> the semantic
          tier structurally cannot provide.
        </Para>
        <Callout tone="info" title="Why episodic when you have semantic?">
          Semantic memory answers <em>&ldquo;what should I do?&rdquo;</em>{" "}
          with lessons stripped of context; episodic memory answers{" "}
          <em>&ldquo;what actually happened last time I was in this
          exact situation?&rdquo;</em>. For a Planner picking between
          five possible strategies, the episodic record of which one
          worked on the last three runs of the same feature is often
          more actionable than a generalised Pattern hit. The two
          tiers complement each other — semantic for transfer
          learning across features, episodic for continuity within a
          feature.
        </Callout>
        <SubHeading>Feedback loop</SubHeading>
        <Para>
          When an agent recalls a memory and the resulting evaluation
          passes, the memory&apos;s relevance score is{" "}
          <strong>boosted</strong>. When the evaluation fails, the
          memory is <strong>demoted</strong>. Every run decays all
          memory relevance by 5%. This produces a natural forgetting
          curve — stale lessons fade, proven lessons strengthen, and
          contradictory lessons compete.
        </Para>
        <SubHeading>Hybrid retrieval</SubHeading>
        <Para>
          The <Code>recall_memories</Code> tool uses Reciprocal Rank
          Fusion to merge Neo4j keyword matches with Qdrant vector
          matches. This gets you exact-string matches (like a specific
          error code) and semantic matches (like &ldquo;similar to this
          situation&rdquo;) in one call, weighted together.
        </Para>
        <SubHeading>Cross-feature briefing</SubHeading>
        <Para>
          Memory crosses feature boundaries <em>within the same run</em>
          — when feature B starts, the patterns and mistakes from
          feature A are already indexed and retrievable. Agents don&apos;t
          have to wait for the next run to benefit from what the current
          run just learned.
        </Para>

        <SubHeading>Memory hygiene (Tier A)</SubHeading>
        <Para>
          Three improvements keep the graph clean and the recall
          path sharp. <strong>Write-time dedup</strong>: before
          creating a new Pattern / Mistake / Solution / Strategy,
          the repository embeds the candidate and cosine-matches
          against existing same-type same-feature memories above{" "}
          <Code>memory_dedup_threshold</Code> (default 0.92). Matches
          get their <Code>relevance_score</Code> boosted and{" "}
          <Code>times_applied</Code> bumped instead of being
          duplicated — a Coder that learns the same lesson across
          five features ends up with one high-relevance Pattern
          instead of five near-identical ones.{" "}
          <strong>Relevance-weighted RRF</strong>: the hybrid recall
          merge now multiplies each rank contribution by the
          memory&apos;s relevance score, so the boost/demote
          feedback loop actually influences retrieval ordering.
          Demoted memories stay visible (floored at 0.1 weight) so
          the operator can see stale hits for cleanup.{" "}
          <strong>Memory graph dashboard</strong>: the Metrics tab
          has a Memory section with per-type node counts, relevance
          histograms, top-10 recalled memories, and a 7-day
          boost/demote effectiveness KPI row. Can&apos;t improve
          what you can&apos;t measure.
        </Para>
        <Callout tone="success" title="Why dedup matters for recall quality">
          Without write-time dedup, a Coder recording &ldquo;use
          parameterised queries&rdquo; across five features creates
          five near-identical Pattern nodes that all rank high for
          SQL-related queries. The agent&apos;s recall list fills
          with paraphrases of the same idea and burns context. With
          dedup, that single idea accumulates boost signal every
          time it&apos;s rediscovered, reaching high relevance
          faster and getting recalled first in every future run
          that needs it.
        </Callout>
      </Section>

      {/* 6. Observability */}
      <Section
        title="6. Observability"
        subtitle="Three independent telemetry pipelines"
        id="observability"
      >
        <Para>
          Every meaningful event in the system is fanned out to three
          independent sinks. No single telemetry pipeline&apos;s failure
          blocks any other — the Prometheus counters fire even if
          Postgres is down; the ProgressBroker emits events to the UI
          even if Prometheus is disabled.
        </Para>
        <FlowFigure
          nodes={OBS_NODES}
          edges={OBS_EDGES}
          caption="Figure 5 — Observability fan-out from instrumentation points through three independent sinks"
          height={500}
        />
        <FactTable
          rows={[
            [
              "Prometheus",
              "Always-on in-process counters + histograms for runs, phases, tool calls, LLM invocations, reconciliation outcomes, deep-agent timeouts, and BackgroundLoop sampler ticks.",
            ],
            [
              "Postgres",
              "Optional forensic store. Writes high-cardinality rows: every LLM call with token counts and cost, every eval result with score breakdowns, every tool call, every incident with a stack trace.",
            ],
            [
              "ProgressBroker",
              "In-process pub/sub. Subscribers receive AG-UI events in real time — used by both the /api/agent/events SSE stream and the server-side metric recorders.",
            ],
            [
              "Grafana",
              "Pre-provisioned dashboards over Prometheus for pipeline throughput, cost rollups, quality trends, and incident budgets.",
            ],
          ]}
        />
        <Para>
          The Metrics tab in the UI surfaces multiple dashboards backed by{" "}
          <Code>/api/metrics/*</Code> endpoints. Those endpoints read
          from Postgres when available and fall back to Prometheus
          scrapes otherwise.
        </Para>
        <SubHeading>Run Detail popup</SubHeading>
        <Para>
          Clicking any run ID in the Manufacture tab&apos;s history
          opens a dedicated popup window with five tabs:
        </Para>
        <FactTable
          rows={[
            [
              "Metrics",
              "Status, pass rate, duration, spec/feature counts, LLM cost, incidents, eval metrics (with spec ID, requirement, type, reason), tool calls, artifacts, decomposition stats.",
            ],
            [
              "Agent Log",
              "Historical progress events for the run — same color-coded badge layout as the main Agent Logs tab, with text filter and expandable JSON payload detail per event.",
            ],
            [
              "Evaluations",
              "Per-spec evaluation tree with requirements, metric scores, attempt history.",
            ],
            [
              "Output",
              "File explorer for generated code/artifacts with syntax highlighting.",
            ],
            [
              "Episodes",
              "Episodic memory timeline — feature narratives, outcomes, key turning-point events, eval scores.",
            ],
          ]}
        />
        <Para>
          The Agent Log tab in Run Detail shares the same formatting
          code (<Code>lib/agentLogFormat.ts</Code>) as the main Agent
          Logs tab — event badges, color coding, and human-readable
          event descriptions are identical. The only difference is
          data source: the main tab reads from a live SSE stream; the
          Run Detail tab loads historical records from the{" "}
          <Code>progress_events</Code> Postgres table via{" "}
          <Code>GET /api/metrics/runs/{"{run_id}"}</Code>.
        </Para>
      </Section>

      {/* 7. Cancellation */}
      <Section
        title="7. Cooperative Cancellation"
        subtitle="Why the Cancel button actually works"
        id="cancellation"
      >
        <Para>
          Stopping a distributed multi-agent pipeline cleanly is
          deceptively hard. The naive approach (<Code>thread.kill</Code>,
          subprocess termination) leaves half-written Neo4j nodes,
          orphaned Qdrant embeddings, and never-flushed metric rows. AI
          Dark Factory uses <strong>cooperative cancellation</strong>: a
          single module-level <Code>threading.Event</Code> is polled at
          every hot-path checkpoint across all six phases.
        </Para>
        <FlowFigure
          nodes={CANCEL_NODES}
          edges={CANCEL_EDGES}
          caption="Figure 6 — Cancel signal propagation. The UI path (left) sets a flag; the worker path (right) polls it at every checkpoint."
          height={520}
        />
        <Para>
          Every pipeline phase, every tool call, every loop body calls{" "}
          <Code>raise_if_cancelled()</Code> at its start. When the flag
          is set, the call raises <Code>PipelineCancelled</Code>, which
          propagates up through each phase&apos;s finally blocks, gets
          caught by the AG-UI bridge&apos;s top-level handler, and is
          translated into a clean{" "}
          <Code>status=&quot;cancelled&quot;</Code> run record — not a
          generic error. The flag is auto-reset at the start of every
          run so a cancel signal cannot bleed into the next invocation.
        </Para>
      </Section>

      {/* 8. L4 agentic behavior */}
      <Section
        title="8. L4 Agentic Classification"
        subtitle="Where this sits on the Vellum scale"
        id="l4"
      >
        <Para>
          This system operates at{" "}
          <strong>L4 — Fully Autonomous / Explorer</strong> on the{" "}
          <a
            href="https://www.vellum.ai/blog/levels-of-agentic-behavior"
            target="_blank"
            rel="noreferrer"
            style={{ color: "#58a6ff" }}
          >
            Vellum agentic behavior scale
          </a>
          . The four L4 traits are all implemented:
        </Para>
        <FactTable
          rows={[
            [
              "Persistent state across sessions",
              "Neo4j procedural memory + Qdrant embeddings + eval history + run history all survive restarts.",
            ],
            [
              "Refines execution from feedback",
              "DeepEval scores drive memory boosts/demotes, adaptive thresholds, and mid-run strategy overrides.",
            ],
            [
              "Parallel execution",
              "Concurrent feature swarms within each dependency layer, bounded by max_parallel_features.",
            ],
            [
              "Real-time adaptation",
              "Layer-level strategy switches + cross-feature briefing within the same run.",
            ],
            [
              "Cross-feature reconciliation",
              "Phase 5 extended Claude Agent SDK pass that sees the full output and polishes the integration.",
            ],
          ]}
        />
      </Section>

      {/* 9. Data model */}
      <Section
        title="9. Data Model Reference"
        subtitle="Key Neo4j node labels and relationships"
        id="data-model"
      >
        <Para>
          The Neo4j graph is the single source of truth for pipeline
          state. The primary labels and their roles:
        </Para>
        <FactTable
          rows={[
            [
              "Requirement",
              "Parsed from input files. One per meaningful requirement. Carries id, description, source path.",
            ],
            [
              "Spec",
              "Generated in Phase 2. Carries capability, scenarios, acceptance criteria, eval scores. Connected to its Requirement via IMPLEMENTS.",
            ],
            [
              "EvalResult",
              "One per spec evaluation or artifact evaluation. Stores per-metric scores and the judge's reasoning.",
            ],
            [
              "Run",
              "One per pipeline invocation. Stores spec_count, feature_count, pass_rate, mean_eval_scores, worst_features, duration, status.",
            ],
            [
              "Memory",
              "Pattern / Mistake / Solution / Strategy nodes. Connected via RELATED_TO edges.",
            ],
          ]}
        />
        <Para>
          Key relationships:{" "}
          <Code>Spec-[:IMPLEMENTS]-&gt;Requirement</Code>,{" "}
          <Code>Spec-[:DEPENDS_ON]-&gt;Spec</Code>,{" "}
          <Code>EvalResult-[:EVALUATES]-&gt;Spec</Code>,{" "}
          <Code>Run-[:PRODUCED]-&gt;Spec</Code>,{" "}
          <Code>Mistake-[:FIXED_BY]-&gt;Solution</Code>.
        </Para>
      </Section>

      {/* 10. Extensibility */}
      <Section
        title="10. Extensibility"
        subtitle="Where to cut in"
        id="extensibility"
      >
        <SubHeading>Adding a new pipeline phase</SubHeading>
        <Para>
          Create a new module in <Code>src/dark_factory/stages/</Code>{" "}
          with a class exposing a <Code>run()</Code> method. Wire it
          into <Code>src/dark_factory/api/ag_ui_bridge.py</Code>{" "}
          between the existing phases, surrounding it with{" "}
          <Code>StepStartedEvent</Code> /{" "}
          <Code>StepFinishedEvent</Code> emissions and a{" "}
          <Code>raise_if_cancelled()</Code> checkpoint. Add a test file
          in <Code>tests/</Code>.
        </Para>
        <SubHeading>Adding a new agent to the swarm</SubHeading>
        <Para>
          Define a new agent in <Code>src/dark_factory/agents/swarm.py</Code>{" "}
          using <Code>create_agent</Code>, add handoff tools that name
          it, wire the new agent into the <Code>create_swarm</Code>{" "}
          list, and extend every other agent&apos;s system prompt to
          know when to transfer to it.
        </Para>
        <SubHeading>Adding a new tool</SubHeading>
        <Para>
          Tools live in <Code>src/dark_factory/agents/tools.py</Code>{" "}
          and use the <Code>@tool</Code> decorator from LangChain. Add
          a new decorated function, then add its name to the{" "}
          <Code>allowed_tools</Code> list of the agent(s) that should
          be able to call it.
        </Para>
        <SubHeading>Adding a new metric</SubHeading>
        <Para>
          Declare the Counter / Histogram in{" "}
          <Code>src/dark_factory/metrics/prometheus.py</Code> and add
          an <Code>observe_*</Code> helper. Call the helper from the
          relevant instrumentation point. If you also want a Postgres
          row, add a recorder method to{" "}
          <Code>metrics/recorder.py</Code>.
        </Para>
      </Section>

      {/* Footer */}
      <div
        style={{
          marginTop: 24,
          padding: 16,
          borderTop: "1px solid #21262d",
          color: "#6e7681",
          fontSize: 11,
          textAlign: "center",
          lineHeight: 1.7,
        }}
      >
        <div>
          AI Dark Factory — Architecture Whitepaper · Diagrams rendered with{" "}
          <a
            href="https://reactflow.dev"
            target="_blank"
            rel="noreferrer"
            style={{ color: "#58a6ff" }}
          >
            React Flow
          </a>{" "}
          · See <Code>README.md</Code> for quick start and operational docs.
        </div>
        <div style={{ marginTop: 6 }}>
          Author:{" "}
          <a
            href="https://www.linkedin.com/in/kwkwan00/"
            target="_blank"
            rel="noreferrer"
            style={{ color: "#58a6ff" }}
          >
            Kevin Quon
          </a>
        </div>
      </div>
    </div>
  );
}
