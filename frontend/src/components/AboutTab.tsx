import { useEffect, useRef, useState } from "react";
import { type Edge, type Node } from "@xyflow/react";
import { exportWhitepaperPdf } from "./whitepaper/exportPdf";
import {
  Callout,
  Code,
  FactTable,
  FlowFigure,
  Label,
  Para,
  Section,
  SubHeading,
  makeEdge,
  makeNode,
} from "./whitepaper/primitives";

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
    "p2b",
    <Label lines={["Phase 2b", "Spec Reconciliation"]} />,
    200,
    310,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p3",
    <Label lines={["Phase 3", "Knowledge Graph"]} />,
    200,
    420,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p4",
    <Label lines={["Phase 4", "Swarm Orchestration"]} />,
    200,
    530,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p5",
    <Label lines={["Phase 5", "Reconciliation"]} />,
    200,
    640,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "p6",
    <Label lines={["Phase 6", "E2E Validation"]} />,
    200,
    750,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "playwright",
    <Label lines={["Playwright", "chromium · firefox · webkit"]} />,
    500,
    750,
    "agent",
    { width: 220 },
  ),
  makeNode("out", "production code", 200, 870, "terminal", { width: 180 }),
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
  makeEdge("e-p2-p2b", "p2", "p2b"),
  makeEdge("e-p2b-p3", "p2b", "p3"),
  makeEdge("e-p3-p4", "p3", "p4"),
  makeEdge("e-p4-p5", "p4", "p5"),
  makeEdge("e-p5-p6", "p5", "p6"),
  makeEdge("e-p6-out", "p6", "out"),
  makeEdge("e-p6-playwright", "p6", "playwright", {
    label: "smoke tests",
    color: "#ea580c",
  }),
  makeEdge("e-playwright-p6", "playwright", "p6", {
    label: "E2E_REPORT.md",
    dashed: true,
    color: "#ea580c",
  }),
  // Phase 1 sub-step branches — rich-doc extraction happens per
  // uploaded file; dedup runs once on the merged list before the
  // handoff to Phase 2.
  makeEdge("e-p1-extract", "p1", "extract", {
    label: ".docx/.xlsx/.pdf/...",
    color: "#ea580c",
  }),
  makeEdge("e-extract-p1", "extract", "p1", {
    label: "Requirement[]",
    dashed: true,
    color: "#ea580c",
  }),
  makeEdge("e-p1-dedup", "p1", "dedup", {
    label: "cluster + collapse",
    dashed: true,
    color: "#2563eb",
  }),
  makeEdge("e-p2-eval", "p2", "eval", {
    label: "score",
    dashed: true,
    color: "#16a34a",
  }),
  makeEdge("e-p4-eval", "p4", "eval", { dashed: true, color: "#16a34a" }),
  makeEdge("e-p2-mem", "p2", "mem", { dashed: true, color: "#db2777" }),
  makeEdge("e-p4-mem", "p4", "mem", {
    label: "recall / upsert",
    dashed: true,
    color: "#db2777",
  }),
  makeEdge("e-p5-mem", "p5", "mem", { dashed: true, color: "#db2777" }),
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
    color: "#ea580c",
  }),
  makeEdge("e-pl-te", "planner", "tester", {
    label: "transfer_to_tester",
    color: "#ea580c",
  }),
  makeEdge("e-co-rv", "coder", "reviewer", {
    label: "transfer_to_reviewer",
    color: "#ea580c",
  }),
  makeEdge("e-rv-pl", "reviewer", "planner", {
    label: "transfer_to_planner",
    color: "#ea580c",
  }),
  makeEdge("e-te-pl", "tester", "planner", {
    label: "transfer_to_planner",
    color: "#ea580c",
  }),
  makeEdge("e-co-mem", "coder", "memory", { dashed: true, color: "#db2777" }),
  makeEdge("e-rv-mem", "reviewer", "memory", {
    dashed: true,
    color: "#db2777",
  }),
  makeEdge("e-te-mem", "tester", "memory", { dashed: true, color: "#db2777" }),
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
    -100,
    330,
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
    230,
    460,
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
    color: "#db2777",
  }),
  makeEdge("e-qdrant-vecs", "qdrant", "vecs"),
  makeEdge("e-postgres-forensic", "postgres", "forensic"),
  makeEdge("e-types-rrf", "types", "rrf", { dashed: true, color: "#2563eb" }),
  makeEdge("e-episodes-rrf", "episodes", "rrf", {
    dashed: true,
    color: "#db2777",
  }),
  makeEdge("e-vecs-rrf", "vecs", "rrf", { dashed: true, color: "#2563eb" }),
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

// ── Storage Architecture Diagram ───────────────────────────────────────────

const STORAGE_NODES: Node[] = [
  makeNode("app", "Pipeline write", 0, 0, "phase", { width: 150 }),
  makeNode("rs", <Label lines={["RunStorage", "write_output() / sync_output_from_local()"]} />, 0, 100, "service", { width: 300 }),
  makeNode("md5", <Label lines={["MD5 manifest", ".md5-manifest.json"]} />, 350, 100, "memory", { width: 180 }),
  makeNode("repl", <Label lines={["ReplicatedStorage", "write to both"]} />, 0, 220, "api", { width: 300 }),
  makeNode("local", <Label lines={["LocalStorage", "fast local disk"]} />, 0, 340, "service", { width: 180 }),
  makeNode("s3", <Label lines={["S3Storage", "durable off-host"]} />, 220, 340, "memory", { width: 180 }),
  makeNode("read", <Label lines={["Read fallback", "local → S3 on miss"]} />, 440, 340, "api", { width: 170 }),
  makeNode("finally", <Label lines={["finally: sync", "catch-all on exit"]} />, 0, 450, "terminal", { width: 180 }),
];
const STORAGE_EDGES: Edge[] = [
  makeEdge("e-app-rs", "app", "rs"),
  makeEdge("e-rs-md5", "rs", "md5", { label: "hash", dashed: true }),
  makeEdge("e-rs-repl", "rs", "repl"),
  makeEdge("e-repl-local", "repl", "local"),
  makeEdge("e-repl-s3", "repl", "s3", { label: "best-effort", dashed: true }),
  makeEdge("e-s3-read", "s3", "read", { label: "fallback", dashed: true, color: "#d97706" }),
  makeEdge("e-local-read", "local", "read"),
  makeEdge("e-local-finally", "local", "finally", { dashed: true }),
];

// ── Spec Reconciliation Flow Diagram ──────────────────────────────────────

const RECON_FLOW_NODES: Node[] = [
  makeNode("input", "Specs from Phase 2", 0, 0, "phase", { width: 180 }),
  makeNode("phantom", "Strip phantom refs", 0, 90, "service", { width: 180 }),
  makeNode("cycles", "Detect + break cycles", 0, 170, "service", { width: 180 }),
  makeNode("islands", "Detect cap islands", 0, 250, "service", { width: 180 }),
  makeNode("coverage", "Flag uncovered reqs", 0, 330, "service", { width: 180 }),
  makeNode("llm", <Label lines={["LLM pass", "implicit deps + capability fix"]} />, 240, 130, "agent", { width: 200 }),
  makeNode("recheck", <Label lines={["Safety re-check", "cycles + phantoms"]} />, 240, 250, "api", { width: 200 }),
  makeNode("output", "Reconciled specs → Phase 3", 120, 420, "phase", { width: 220 }),
];
const RECON_FLOW_EDGES: Edge[] = [
  makeEdge("e-in-ph", "input", "phantom"),
  makeEdge("e-ph-cy", "phantom", "cycles"),
  makeEdge("e-cy-is", "cycles", "islands"),
  makeEdge("e-is-co", "islands", "coverage"),
  makeEdge("e-co-llm", "coverage", "llm", { label: "best-effort" }),
  makeEdge("e-llm-re", "llm", "recheck"),
  makeEdge("e-re-out", "recheck", "output"),
  makeEdge("e-co-out", "coverage", "output", { dashed: true, label: "if LLM fails" }),
];

// ── Traceability Data Flow Diagram ────────────────────────────────────────

const TRACE_NODES: Node[] = [
  makeNode("req", "Requirement", 0, 0, "ui", { width: 140 }),
  makeNode("spec", "Spec", 190, 0, "phase", { width: 140 }),
  makeNode("code", "Code artifact", 390, 0, "agent", { width: 140 }),
  makeNode("test", "Test file", 590, 0, "api", { width: 140 }),
  makeNode("eval", "Eval score", 370, 170, "service", { width: 140 }),
  makeNode("episode", "Episode", 560, 170, "memory", { width: 140 }),
  makeNode("neo4j", "Neo4j", 60, 340, "terminal", { width: 120 }),
  makeNode("postgres", "Postgres", 310, 340, "terminal", { width: 120 }),
  makeNode("qdrant", "Qdrant", 520, 340, "terminal", { width: 120 }),
  makeNode("s3t", "S3 / Local", 710, 340, "terminal", { width: 120 }),
];
const TRACE_EDGES: Edge[] = [
  makeEdge("e-req-spec", "req", "spec", { label: "IMPLEMENTS" }),
  makeEdge("e-spec-code", "spec", "code", { label: "generates" }),
  makeEdge("e-code-test", "code", "test", { label: "tested by" }),
  makeEdge("e-code-eval", "code", "eval"),
  makeEdge("e-test-eval", "test", "eval"),
  makeEdge("e-eval-ep", "eval", "episode", { label: "synthesized" }),
  makeEdge("e-req-neo", "req", "neo4j"),
  makeEdge("e-spec-neo", "spec", "neo4j"),
  makeEdge("e-eval-pg", "eval", "postgres"),
  makeEdge("e-code-s3", "code", "s3t"),
  makeEdge("e-ep-qdrant", "episode", "qdrant"),
  makeEdge("e-ep-neo", "episode", "neo4j", { dashed: true }),
];

// ── Memory Lifecycle Diagram ──────────────────────────────────────────────

const MEM_LIFE_NODES: Node[] = [
  makeNode("recall", <Label lines={["Agent recalls", "recall_memories()"]} />, 0, 0, "agent", { width: 180 }),
  makeNode("use", <Label lines={["Agent uses memory", "in code/review/test"]} />, 220, 0, "phase", { width: 180 }),
  makeNode("evalm", <Label lines={["Eval runs", "evaluate_spec / evaluate_tests"]} />, 440, 0, "api", { width: 200 }),
  makeNode("boost", <Label lines={["Boost relevance", "+0.1 score, +1 times_applied"]} />, 560, 130, "service", { width: 220 }),
  makeNode("demote", <Label lines={["Demote relevance", "−0.05 score"]} />, 440, 280, "memory", { width: 220 }),
  makeNode("decay", <Label lines={["Decay all ×0.95", "every pipeline start"]} />, 260, 420, "terminal", { width: 200 }),
  makeNode("prune", <Label lines={["Prune < 0.05", "garbage collection"]} />, -120, 420, "terminal", { width: 180 }),
  makeNode("write", <Label lines={["Agent writes memory", "record_pattern / record_mistake"]} />, 0, 130, "agent", { width: 200 }),
  makeNode("dedup", <Label lines={["Dedup check", "cosine ≥ 0.92 → boost existing"]} />, 0, 280, "service", { width: 220 }),
];
const MEM_LIFE_EDGES: Edge[] = [
  makeEdge("e-rec-use", "recall", "use"),
  makeEdge("e-use-eval", "use", "evalm"),
  makeEdge("e-eval-boost", "evalm", "boost", { label: "pass", color: "#16a34a" }),
  makeEdge("e-eval-demote", "evalm", "demote", { label: "fail", color: "#dc2626" }),
  makeEdge("e-boost-decay", "boost", "decay", { dashed: true }),
  makeEdge("e-demote-decay", "demote", "decay", { dashed: true }),
  makeEdge("e-decay-prune", "decay", "prune"),
  makeEdge("e-prune-rec", "prune", "recall", { dashed: true, label: "next run" }),
  makeEdge("e-rec-write", "recall", "write", { dashed: true }),
  makeEdge("e-write-dedup", "write", "dedup"),
];

// ── Cancellation Diagram ──────────────────────────────────────────────────

const CANCEL_NODES: Node[] = [
  makeNode("btn", "UI: Cancel button", 0, 0, "ui", { width: 160 }),
  makeNode("post", "POST /api/agent/cancel", 0, 80, "api", { width: 180 }),
  makeNode("flag", <Label lines={["threading.Event", ".set()"]} />, 0, 170, "service", { width: 160 }),
  makeNode("bridge", <Label lines={["ag_ui_bridge", "raise_if_cancelled()"]} />, 220, 0, "phase", { width: 180 }),
  makeNode("orch", <Label lines={["Orchestrator", "ThreadPoolExecutor"]} />, 220, 90, "agent", { width: 180 }),
  makeNode("spec", <Label lines={["SpecStage", "ThreadPoolExecutor"]} />, 220, 180, "agent", { width: 180 }),
  makeNode("deep", <Label lines={["Deep Agent SDK", "subprocess"]} />, 220, 270, "agent", { width: 180 }),
  makeNode("check", <Label lines={["is_cancelled()", "atomic Event.is_set()"]} />, 450, 90, "api", { width: 190 }),
  makeNode("raise", "raise PipelineCancelled", 450, 200, "service", { width: 190 }),
  makeNode("cleanup", <Label lines={["finally: cleanup", "status = 'cancelled'"]} />, 450, 310, "terminal", { width: 190 }),
];

const CANCEL_EDGES: Edge[] = [
  makeEdge("e-btn-post", "btn", "post"),
  makeEdge("e-post-flag", "post", "flag"),
  makeEdge("e-flag-check", "flag", "check", { label: "flag", dashed: true, color: "#d97706" }),
  makeEdge("e-bridge-check", "bridge", "check"),
  makeEdge("e-orch-check", "orch", "check"),
  makeEdge("e-spec-check", "spec", "check"),
  makeEdge("e-deep-check", "deep", "check", { dashed: true, label: "B2 fix" }),
  makeEdge("e-check-raise", "check", "raise"),
  makeEdge("e-raise-cleanup", "raise", "cleanup"),
];

// ── PDF export constants ─────────────────────────────────────────────────────

const PDF_EXPORT_BTN_ID = "pdf-export-btn";
const TOC_NAV_ID = "dark-factory-whitepaper-toc";

// Section title + anchor list for the sticky TOC. Order mirrors the
// rendered sections below; ids match each Section's ``id`` prop.
const TOC_SECTIONS: ReadonlyArray<{ id: string; label: string }> = [
  { id: "business-value", label: "Business Value" },
  { id: "philosophy", label: "1. Design Philosophy" },
  { id: "system", label: "2. System Overview" },
  { id: "pipeline", label: "3. The Seven-Phase Pipeline" },
  { id: "deep-agents", label: "3.5 Deep Agents" },
  { id: "swarm", label: "4. Swarm Mechanics" },
  { id: "memory", label: "5. Procedural Memory" },
  { id: "observability", label: "6. Observability" },
  { id: "traceability-arch", label: "6.5 Traceability & Storage" },
  { id: "cancellation", label: "7. Cooperative Cancellation" },
  { id: "l4", label: "8. L4 Agentic Classification" },
  { id: "data-model", label: "9. Data Model Reference" },
  { id: "extensibility", label: "10. Extensibility" },
];

// ── Table of Contents ────────────────────────────────────────────────────────
//
// Sticky left rail. Excluded from PDF export by id (the exportPdf helper
// only captures ``containerRef``'s subtree, and the TOC is a sibling).

function TableOfContents() {
  const [activeId, setActiveId] = useState<string>(TOC_SECTIONS[0]?.id || "");

  useEffect(() => {
    const targets = TOC_SECTIONS
      .map((s) => document.getElementById(s.id))
      .filter((el): el is HTMLElement => el !== null);
    if (targets.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0 && visible[0].target.id) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: "-20% 0px -70% 0px", threshold: [0, 1] },
    );
    targets.forEach((t) => observer.observe(t));
    return () => observer.disconnect();
  }, []);

  return (
    <nav
      id={TOC_NAV_ID}
      aria-label="Whitepaper sections"
      style={{
        position: "sticky",
        top: 16,
        maxHeight: "calc(100vh - 32px)",
        overflowY: "auto",
        padding: "12px 0",
        fontSize: 12,
        borderRight: "1px solid #d0d7de",
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: 0.6,
          color: "#57606a",
          padding: "0 12px 6px",
        }}
      >
        Contents
      </div>
      <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
        {TOC_SECTIONS.map((s) => {
          const active = activeId === s.id;
          return (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                aria-current={active ? "true" : undefined}
                style={{
                  display: "block",
                  padding: "5px 12px",
                  borderLeft: "2px solid",
                  borderLeftColor: active ? "#0550ae" : "transparent",
                  color: active ? "#0550ae" : "#24292f",
                  background: active ? "#f0f4f8" : "transparent",
                  fontWeight: active ? 600 : 400,
                  textDecoration: "none",
                  lineHeight: 1.35,
                  transition: "all 0.12s",
                }}
              >
                {s.label}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export default function AboutTab() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [exporting, setExporting] = useState(false);

  const exportPDF = async () => {
    if (!containerRef.current) return;
    setExporting(true);
    try {
      await exportWhitepaperPdf({
        container: containerRef.current,
        filename: "ai-dark-factory-whitepaper.pdf",
        title: "AI Dark Factory — Architecture Whitepaper",
        hideElementId: PDF_EXPORT_BTN_ID,
      });
    } finally {
      setExporting(false);
    }
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(180px, 220px) minmax(0, 1fr)",
        gap: 24,
        alignItems: "start",
      }}
    >
      <TableOfContents />
      <div className="about-whitepaper" ref={containerRef}>
      {/* Hero */}
      <div
        className="card"
        style={{
          marginBottom: 16,
          background: "#f0f0f0",
          borderColor: "#d0d0d0",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            marginBottom: 6,
          }}
        >
        <div
          style={{
            fontSize: 20,
            fontWeight: 700,
          }}
        >
          AI Dark Factory — Architecture Whitepaper
        </div>
        <button
          id={PDF_EXPORT_BTN_ID}
          onClick={exportPDF}
          disabled={exporting}
          style={{
            flexShrink: 0,
            padding: "6px 14px",
            fontSize: 12,
            fontWeight: 600,
            background: exporting ? "#e2e8f0" : "#0550ae",
            color: exporting ? "#64748b" : "#ffffff",
            border: "none",
            borderRadius: 4,
            cursor: exporting ? "not-allowed" : "pointer",
            letterSpacing: 0.3,
          }}
        >
          {exporting ? "Exporting…" : "Export PDF"}
        </button>
        </div>
        <div style={{ fontSize: 13, marginBottom: 12 }}>
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
            ["Protocol", "AG-UI Server-Sent Events for real-time UI streaming"],
            [
              "Storage",
              "Pluggable local, S3, or replicated (local + S3) storage — run-ID-scoped layout with input/, requirements/, specs/, output/ subfolders. Auto-replicates to S3 when S3_BUCKET is configured alongside local storage.",
            ],
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
              "Every agent action, handoff, tool call, and evaluation is logged, timestamped, and accessible in the Run Detail popup. Every run leaves a complete forensic trail.",
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
              "Run history",
              "Every run is persisted with spec counts, pass rate, mean eval scores, worst features, and full event timeline. The Manufacture tab shows recent runs with live polling, status badges (success/partial/cancelled/error), and an actions menu for deletion. Click any run to open a detail popup with tabs for: Agent Log, Metrics, Evaluations, Episodes, Output (with Download ZIP), Compare (side-by-side against another run with file diffs), and Traceability (requirements → specs → files → tests → evals matrix with interactive dependency graph).",
            ],
            [
              "Requirements Refinery",
              "A dedicated Refine tab lets operators transform raw requirements into a more detailed, structured, and actionable set. Powered by concurrent deep agents (OpenAI Responses API), it analyzes each requirement for clarity, completeness, and testability, discovers inter-requirement relationships, suggests spec decompositions, and proposes reusable strategy/pattern memories. A reconciliation phase detects duplicates, coherence issues, and priority inversions across the full set. Results are persisted to S3/local storage with full history and export to ZIP.",
            ],
            [
              "Live observability",
              "Real-time agent logs with color-coded event badges, comprehensive metrics dashboards, Prometheus + Grafana integration, per-run cost rollups, and per-run agent logs with expandable JSON payloads. Know what the system is doing and what it has done.",
            ],
            [
              "Safe cancellation",
              "A single Cancel button cleanly stops any running pipeline at the next checkpoint, with no orphaned state or half-written records.",
            ],
            [
              "Deep agent resilience",
              "Claude Agent SDK subprocess crashes are caught and converted to soft errors via _safe_tool_deep_agent. Stderr is buffered for diagnostics, and --debug-to-stderr can be enabled via DEEP_AGENT_DEBUG_STDERR for investigating silent exits. A regression test verifies all tool-decorated deep agents use the safe wrapper.",
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
          The pipeline is decomposed into seven well-defined phases. Each
          phase has a narrow contract, its own evaluation criteria, and a
          clean failure boundary. Agents operate in tight feedback loops
          with external judges (DeepEval + GPT), write what they learn
          into a shared memory graph, and benefit from every prior run —
          even partial ones.
        </Para>

        <SubHeading>Why specialisation beats scale</SubHeading>
        <Para>
          The natural instinct when a model underperforms is to reach for
          a larger model with a bigger context window. Dark Factory takes
          the opposite bet: keep each agent's context narrow and its
          responsibility singular, then coordinate. A Planner that only
          ever reads specs and eval history stays focused. A Coder that
          only writes code — informed by RAG-retrieved patterns and
          explicit mistakes from the Reviewer — accumulates a tighter
          internal model of what good code looks like for this codebase.
          A Reviewer that only evaluates and records catches problems the
          Coder cannot self-diagnose. No single agent needs to juggle the
          full problem space; the swarm does.
        </Para>
        <Para>
          This mirrors how high-performing engineering teams actually
          work. A senior engineer does not also run quality assurance, write the design
          doc, and manage the deployment — those roles exist because
          division of labour produces better outcomes than a single
          generalist doing everything. The swarm encodes that intuition
          in software.
        </Para>

        <SubHeading>Memory as infrastructure, not an afterthought</SubHeading>
        <Para>
          Most agent systems treat memory as a convenience feature bolted
          on after the core loop is working. Dark Factory treats it as
          load-bearing infrastructure. Every agent reads from the shared
          Neo4j + Qdrant memory graph before its first tool call, and
          every agent writes back after its last. The memory graph is not
          a log — it is a living, scored, decaying knowledge base. Lessons
          that prove useful accumulate relevance and surface first.
          Lessons that fail in practice decay and eventually prune
          themselves. The system does not just remember; it forgets
          intelligently.
        </Para>
        <Para>
          The episodic tier adds temporal reasoning on top of semantic
          generalisation. Semantic memory answers "what should I do?"
          Episodic memory answers "what actually happened the last three
          times I was in this exact situation?" For a Planner choosing
          between strategies, the episodic record is often more actionable
          than a generalised pattern hit — it carries the full trajectory,
          including the wrong turns and the recovery.
        </Para>

        <SubHeading>Evaluation as a first-class dependency</SubHeading>
        <Para>
          Evaluation is not a reporting step that runs after the pipeline
          finishes. It is wired into the pipeline as a dependency that
          gates phase transitions, drives memory feedback, and triggers
          adaptive strategy switches mid-run. A spec that scores below
          threshold does not advance to the graph write. A feature swarm
          whose pass rate drops below the layer threshold causes the
          orchestrator to tighten handoff budgets and force the SDK path
          for remaining features. Evaluation results do not sit in a
          dashboard waiting to be read — they change what happens next.
        </Para>
        <Callout tone="info" title="External judging">
          Evaluation uses DeepEval with GPT as the judge, intentionally
          separate from the Anthropic models used for codegen. A model
          cannot reliably judge its own output. Using a different model
          family as judge introduces the adversarial distance needed for
          scores to be meaningful.
        </Callout>

        <SubHeading>Failure isolation and graceful degradation</SubHeading>
        <Para>
          Every failure boundary in the system is explicit and narrow.
          A single feature swarm failing does not abort the run — the
          orchestrator records the failure, emits an incident, and
          continues to the next layer. Phase 5 reconciliation failing
          does not block delivery — the pipeline ships the unpolished
          feature output and logs the reconciliation error as an incident.
          Phase 6 E2E validation failing does not remove the output from
          the operator's hands — they see the E2E report and decide
          whether the delivery is ship-ready themselves.
        </Para>
        <Para>
          This is a deliberate design choice, not a compromise. Automated
          systems that hard-fail on any error are fragile in production.
          A system that delivers partial results with full transparency
          about what failed — and why — is more useful than one that
          delivers nothing and requires a human to restart from scratch.
        </Para>

        <SubHeading>Observability by construction</SubHeading>
        <Para>
          Observability is not instrumented as a separate concern — it is
          woven into the execution path. Every phase, tool call, LLM
          invocation, handoff, eval result, and incident flows through
          the same instrumentation layer that emits to Prometheus, writes
          to Postgres, and publishes to the ProgressBroker simultaneously.
          The three sinks are independent: a Postgres outage does not
          silence the Prometheus counters; a ProgressBroker subscriber
          crash does not interrupt the forensic row store. Operators can
          watch a run live in the Agent Logs tab, query it historically
          in Run Detail, and alert on it via Grafana — all from the same
          underlying event stream.
        </Para>

        <SubHeading>Core tenets</SubHeading>
        <FactTable
          rows={[
            [
              "Specialisation over scale",
              "Four small agents with distinct roles beat one big agent told to do everything. Narrow context produces tighter reasoning.",
            ],
            [
              "Memory is first-class",
              "Procedural memory is a scored, decaying graph — not a cache. Every agent reads before acting, writes after. Lessons that work accumulate; lessons that fail decay.",
            ],
            [
              "Evaluation is a dependency",
              "LLM judges (DeepEval + GPT) run on every spec and artifact. Scores gate phase transitions, drive memory feedback, and trigger mid-run strategy switches.",
            ],
            [
              "Failure isolation",
              "Every failure boundary is explicit and narrow. A failed feature, a crashed reconciliation, or a broken E2E pass never aborts the run — partial delivery with full transparency beats no delivery.",
            ],
            [
              "Best-effort polishing",
              "Phase 5 reconciliation and Phase 6 E2E are intentionally wrapped in broad try/except. They polish and validate; they never gatekeep.",
            ],
            [
              "Observability by construction",
              "Every tool call, phase, handoff, and incident flows through three independent sinks (Prometheus, Postgres, ProgressBroker) simultaneously. No single telemetry failure silences the others.",
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
          both the REST API and the React single-page application (served from{" "}
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
              "File I/O + Bash tools for Phase 5 reconciliation and Phase 6 E2E validation. Deep-agent tools (codegen, analysis, test gen) use direct Anthropic API calls.",
            ],
            [
              "Storage backend",
              "Pluggable local filesystem, S3, or replicated (local + S3) storage. When STORAGE_BACKEND=local and S3_BUCKET is set, a ReplicatedStorage backend auto-writes to both local disk (fast pipeline I/O) and S3 (durable off-host persistence). S3 failures are best-effort and never block the pipeline. A catch-all sync in the pipeline's finally block ensures the full run directory reaches S3 even on mid-pipeline crashes.",
            ],
          ]}
        />
      </Section>

      {/* 3. Pipeline */}
      <Section
        title="3. The Seven-Phase Pipeline"
        subtitle="From requirements to production-quality code"
        id="pipeline"
      >
        <Para>
          A single <Code>POST /api/agent/run</Code> call drives the
          entire pipeline. The request returns immediately; subsequent
          events stream to the browser over{" "}
          <Code>GET /api/agent/events</Code> using the AG-UI Server-Sent Events format.
        </Para>
        <FlowFigure
          nodes={PIPELINE_NODES}
          edges={PIPELINE_EDGES}
          caption="Figure 2 — Pipeline data flow across all seven phases"
          height={780}
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
          exhausted.
        </Para>
        <Para>
          An <strong>early-exit preflight</strong> queries Neo4j via{" "}
          <Code>IMPLEMENTS</Code> edges before the planner even runs.
          If every requirement already has specs, the entire decomposition +
          refinement loop is skipped — re-runs on unchanged inputs
          complete in a single Neo4j query with zero LLM spend. Partial
          coverage falls through to the normal path, reusing existing
          specs and only generating missing ones.
        </Para>
        <Callout tone="info" title="DeepEval rubric">
          Every spec is scored on four GEval metrics: Correctness,
          Coherence, Instruction Following (does it match the OpenSpec
          shape?), and Safety &amp; Ethics. GPT acts as judge. Scores
          feed both the early-exit threshold and the adaptive memory
          decay in Phase 4.
        </Callout>

        <SubHeading>Phase 2b — Spec Reconciliation</SubHeading>
        <Para>
          After spec generation and before the graph write, a
          reconciliation stage validates and repairs the spec dependency
          graph. A <strong>deterministic pass</strong> strips phantom
          references, detects and breaks circular dependencies, flags
          uncovered requirements, and identifies disconnected capability
          islands. An <strong>LLM-assisted pass</strong> then reviews
          spec descriptions, acceptance criteria, and WHEN/THEN scenarios
          to surface implicit dependencies the planner missed, fix
          requirement coverage gaps, and correct capability groupings.
          The LLM pass is sandboxed — cycles and phantoms are re-checked
          after patches are applied so the LLM cannot corrupt the graph.
        </Para>
        <Callout tone="info" title="Capability islands">
          Specs that share a <Code>capability</Code> but have no dependency
          path between them are flagged as &ldquo;islands.&rdquo; This
          usually means a missing dependency edge (the data model spec
          should be a dependency of the API handler spec) or a wrong
          capability assignment. The reconciliation surfaces these for
          both the LLM and the operator.
        </Callout>
        <FlowFigure
          nodes={RECON_FLOW_NODES}
          edges={RECON_FLOW_EDGES}
          caption="Figure — Spec reconciliation flow. Deterministic checks (left) always run; LLM pass (right) is best-effort with safety re-check."
          height={500}
        />

        <SubHeading>Phase 3 — Knowledge Graph</SubHeading>
        <Para>
          Specs and requirements are persisted to Neo4j with{" "}
          <Code>IMPLEMENTS</Code> and <Code>DEPENDS_ON</Code>{" "}
          relationships. Specs are simultaneously auto-indexed into the
          Qdrant <Code>dark_factory_specs</Code> collection with enriched
          metadata (eval scores, attempts, scenarios, dependencies) so
          that downstream Coder agents can pull in semantically-similar
          work from other features as RAG context.
        </Para>

        <SubHeading>Phase 4 — Swarm Orchestration</SubHeading>
        <Para>
          The orchestrator groups specs by <Code>capability</Code> and
          uses Tarjan&apos;s Strongly Connected Components algorithm to compute a{" "}
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

      {/* 3.5 Deep Agents */}
      <Section
        title="3.5 Deep Agents"
        subtitle="How Claude Agent SDK subprocess agents are used throughout the pipeline"
        id="deep-agents"
      >
        <Para>
          Dark Factory makes extensive use of the{" "}
          <strong>Claude Agent SDK</strong> to spawn isolated, clean-context
          subprocess agents at several points in the pipeline. Unlike the
          LangGraph swarm agents — which share a persistent graph state
          across handoffs — deep agents run in a completely fresh context,
          with their own working directory and a full file-system tool set:{" "}
          <Code>Read / Write / Edit / Glob / Grep / Bash</Code>. Each
          invocation starts with zero memory of previous runs and exits
          cleanly when its task is complete. This isolation is a feature,
          not a limitation: it means a deep agent's context is never
          polluted by earlier pipeline state, and it can operate directly
          on the filesystem without coordination overhead.
        </Para>

        <SubHeading>Where deep agents are used</SubHeading>
        <FactTable
          rows={[
            [
              "Phase 1 — Document extraction",
              "Rich business documents (Word, Excel, PDF, HTML, XML, RTF, CSV, transcripts) are routed to a per-file deep agent. Each invocation runs in a fresh subprocess with its cwd set to the upload directory, reads the document with the appropriate Python library, and writes a staging JSON file of discrete requirements. Because each document gets its own isolated context, raw meeting-transcript noise never pollutes the main pipeline's LLM context.",
            ],
            [
              "Phase 4 — Code generation",
              "The Coder agent in the LangGraph swarm can delegate to a claude_agent_codegen deep agent for complex implementation tasks. The deep agent has full Read / Write / Edit / Glob / Grep / Bash access to the run output directory, so it can create files, run the linter, check imports, and iterate — all within a single SDK invocation. The swarm Coder receives the result as a structured tool response and continues.",
            ],
            [
              "Phase 4 — Code review and test generation",
              "Specialised deep agents handle dependency analysis, risk review, security review, performance review, compliance review, and unit / integration / edge-case test generation. Each is a separate @tool-decorated function backed by its own SDK invocation, scoped to the files it needs to read and the output location it writes to.",
            ],
            [
              "Phase 5 — Reconciliation",
              "A single extended deep agent runs over the full run output directory after all feature swarms complete. It follows a rigid six-step checklist — inventory → review → fix → validate → iterate → report — and is the only agent in the pipeline that can see every feature's output simultaneously. This gives it the authority to fix cross-feature import breakage, inconsistent API shapes, and missing glue files that per-feature swarms structurally cannot detect.",
            ],
            [
              "Phase 6 — E2E validation",
              "A second extended deep agent runs after reconciliation to execute a Playwright cross-browser smoke test suite. It detects whether the output is a web application, writes a smoke.spec.ts against the acceptance criteria, starts the server, and runs the test suite across chromium, firefox, and webkit — all within a single subprocess invocation.",
            ],
          ]}
        />

        <SubHeading>Why subprocess isolation matters</SubHeading>
        <Para>
          Each deep agent runs as a Node.js subprocess managed by the{" "}
          <Code>BackgroundLoop</Code> singleton — a daemon asyncio event
          loop that ensures subprocess cleanup callbacks always have a
          valid loop to land on. Isolation delivers three concrete
          benefits: the agent's file-system tools operate on real disk
          paths with no translation layer; its LLM context contains
          exactly what its task needs and nothing else; and its exit is
          clean — the subprocess terminates, releasing all resources,
          before the calling swarm agent processes the result.
        </Para>
        <Callout tone="info" title="Clean-context advantage">
          The reconciliation and E2E agents deliberately start with no
          knowledge of how individual features were built. They evaluate
          the output as a whole — the same way an engineer picking up an
          unfamiliar codebase would. This cold-start perspective is what
          makes them effective at catching integration problems that the
          feature authors are too close to see.
        </Callout>
      </Section>

      {/* 4. Swarm */}
      <Section
        title="4. Swarm Mechanics"
        subtitle="Four agents, one feature, bounded handoffs"
        id="swarm"
      >
        <SubHeading>Why a swarm?</SubHeading>
        <Para>
          The canonical alternative to a swarm is a single large-context
          agent given the full spec and told to produce code, tests, and
          a review in one pass. This approach has a well-documented
          failure mode: as the context fills — with the spec, the code
          draft, the review notes, the revised draft, the test suite —
          the model's attention dilutes. Early instructions get
          overweighted relative to late corrections. The agent loses the
          thread of its own critique. Quality degrades as a function of
          context length, not task complexity.
        </Para>
        <Para>
          The swarm sidesteps this entirely. Each agent receives only the
          context relevant to its role. The Coder's context is the spec,
          the RAG results, and the recalled patterns — not the Reviewer's
          notes from three rounds ago. The Reviewer's context is the
          generated code and the evaluation rubric — not the Coder's
          internal reasoning about why it made a particular choice. Each
          handoff resets the active context to exactly what the next
          agent needs. This means a 50-handoff swarm maintains sharp
          attention at every step, rather than a single agent running out
          of effective context after the first revision cycle.
        </Para>
        <Para>
          There is a second benefit that compounds over time:{" "}
          <strong>specialisation enables targeted memory.</strong>{" "}
          Because the Reviewer always writes Mistake nodes and the Coder
          always reads Pattern nodes, the memory graph accumulates
          role-specific signal. A Reviewer mistake from run 3 surfaces
          in the Coder's recall on run 7 — not as noise in a general
          conversation history, but as a structured, scored, typed memory
          node that the retrieval system can rank against the current
          feature's capability tag. A monolithic agent has no clean
          boundary between "what I generated" and "what I evaluated" —
          the swarm makes that boundary explicit, and the memory system
          exploits it.
        </Para>
        <Callout tone="success" title="The compounding argument">
          A single-agent system plateaus at whatever quality level its
          context window and base model allow. A swarm with persistent
          memory does not plateau — each run deposits lessons that make
          the next run start from a higher baseline. The quality ceiling
          of the system rises with usage.
        </Callout>

        <Para>
          Each feature in Phase 4 runs its own isolated LangGraph swarm
          built on <Code>langgraph-swarm.create_swarm</Code>. Features
          within the same dependency layer run concurrently — each in its
          own swarm instance, on its own thread, with its own memory
          context. There is no shared mutable state between concurrent
          swarms. The agents within a single swarm rotate via named
          handoff tools — there is no central dispatcher. Each agent
          reads the current graph state, decides what needs to happen
          next, and either acts or transfers control to the agent best
          positioned to continue.
        </Para>
        <FlowFigure
          nodes={SWARM_NODES}
          edges={SWARM_EDGES}
          caption="Figure 3 — Per-feature swarm handoff topology. Solid orange edges are LangGraph transfer tools; dashed pink edges are memory reads/writes."
          height={480}
        />

        <SubHeading>The four agents</SubHeading>
        <Para>
          Every swarm contains exactly the same four agents. Their roles
          are narrow by design — each agent has a focused system prompt,
          a restricted tool set, and a clear definition of done.
        </Para>
        <FactTable
          rows={[
            [
              "Planner",
              "The entry point for every swarm invocation. Reads the spec in full, queries eval history for prior attempts on this feature, and recalls strategies from memory. Decides the opening move: delegate to the Coder if implementation is needed, transfer to the Tester if code exists but tests are missing, or mark the feature complete if both pass threshold. After each Reviewer or Tester round-trip, the Planner re-evaluates and either signs off or requests another pass. It is the only agent that can terminate the swarm successfully.",
            ],
            [
              "Coder",
              "Responsible for producing the implementation. Before writing a single line, the Coder runs two Qdrant RAG queries — one for similar specs (to understand the expected interface) and one for similar code artifacts (to reuse proven patterns). It then recalls patterns and mistakes from memory. For straightforward tasks it writes code directly using its tool set; for complex implementations it delegates to a claude_agent_codegen deep agent that has full filesystem access and can iterate — run the linter, fix import errors, restructure files — before returning a result. The Coder then transfers to the Reviewer.",
            ],
            [
              "Reviewer",
              "Evaluates the generated code against a DeepEval rubric covering correctness, coherence, security, and style. The Reviewer does not generate code — its role is adversarial evaluation. When it finds problems, it records them as Mistake nodes in memory (with root cause and file context) and records the fix as a paired Solution node. It then transfers back to the Planner with a structured verdict. Future Coder agents on similar features will recall these mistakes before writing their first line.",
            ],
            [
              "Tester",
              "Writes a test suite against the spec's acceptance criteria — unit tests, integration tests, and edge cases. Evaluates its own output on correctness, coherence, and completeness using DeepEval. Records test failures and coverage gaps as Mistake nodes so future Testers on similar features start with that knowledge. Transfers back to the Planner with pass/fail status and a structured test report.",
            ],
          ]}
        />

        <SubHeading>Handoff topology and flow control</SubHeading>
        <Para>
          The handoff graph is intentionally not a strict pipeline.
          The Planner can send control to the Coder or the Tester
          independently — if tests already exist from a prior attempt,
          the Planner skips straight to the Tester rather than
          regenerating code. The Reviewer always returns to the Planner
          (never directly to the Coder) so that the Planner can weigh
          the review verdict against the remaining handoff budget before
          deciding whether another coding pass is warranted. This
          hub-and-spoke pattern around the Planner prevents runaway
          Coder-Reviewer cycles from consuming the entire budget on a
          single feature.
        </Para>
        <Callout tone="info" title="Handoff budget">
          Every swarm is bounded by <Code>max_codegen_handoffs</Code>{" "}
          (default 50). Each agent-to-agent transfer consumes one unit.
          If the budget is exhausted before the Planner signs off, the
          orchestrator records the feature as failed, surfaces an
          incident, and moves to the next layer. A single failing feature
          never stalls the pipeline.
        </Callout>

        <SubHeading>Adaptive strategy overrides</SubHeading>
        <Para>
          After each dependency layer completes, the orchestrator
          reviews the layer's aggregate pass rate. If it drops below
          the configured threshold, two overrides activate for all
          remaining features: the Coder is forced onto the deep-agent
          SDK path (disabling the direct-write shortcut), and the
          handoff budget is tightened. The premise is that when a layer
          underperforms, lightweight codegen is not keeping up with the
          complexity of the remaining specs — the SDK path's ability to
          iterate on disk and run the linter between attempts produces
          higher-quality output at the cost of more wall-clock time.
          When a subsequent layer's pass rate recovers, the overrides
          relax automatically.
        </Para>

        <SubHeading>Memory integration within the swarm</SubHeading>
        <Para>
          Memory is not a post-processing step — it is woven into the
          hot path of every agent. Before the Coder writes its first
          line, it has already queried the shared memory graph for
          patterns relevant to this feature's capability tag and for
          mistakes recorded by prior Reviewers on similar code. Before
          the Planner picks a strategy, it has already recalled what
          worked on the last three runs of this feature. This means the
          swarm improves within a single run — a mistake recorded by
          the Reviewer on feature A is immediately available to the
          Coder on feature B when B's swarm starts, even before the
          run finishes.
        </Para>
        <Callout tone="success" title="Cross-feature learning within a run">
          Because memory is written to Neo4j and Qdrant after each
          agent action — not batched at run end — features that start
          later in the dependency order benefit from lessons learned by
          features that ran earlier in the same run. The system gets
          smarter as a run progresses, not just across runs.
        </Callout>
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

        <SubHeading>How procedural memory is created</SubHeading>
        <Para>
          Memory nodes are not manually authored — they are a byproduct
          of the swarm doing its job. Every time the Coder produces an
          implementation it considers reusable, it calls{" "}
          <Code>record_pattern</Code> to write a Pattern node describing
          the structure, the language idiom, and the capability context.
          Every time the Reviewer finds a problem — a broken import, an
          unsafe query, an inconsistent return type — it calls{" "}
          <Code>record_mistake</Code> with a root cause description and
          a reference to the offending file. If the fix is known, it
          calls <Code>record_solution</Code> to pair a Solution node with
          the Mistake. Every time the Planner makes a strategy decision
          — whether to use the SDK path, how to structure the output
          directory, which dependency to implement first — it calls{" "}
          <Code>record_strategy</Code>. These writes happen during the
          live swarm, not in a post-processing step, which means memory
          accumulates continuously across the run.
        </Para>
        <Para>
          Episodic memories are created differently. After a feature
          swarm terminates — whether by success or budget exhaustion —
          the orchestrator synthesises the full event trajectory into a
          narrative summary and stores it as an Episode node. This
          synthesis uses the swarm's event log, the eval scores, and
          the IDs of every semantic memory recalled during the run.
          The result is a structured record of what happened, what
          worked, and which prior memories influenced the outcome.
        </Para>
        <Callout tone="info" title="Memory is earned, not configured">
          There are no manually defined rules, templates, or knowledge
          bases to maintain. The memory graph is entirely self-generated
          from agent observations during real runs. The first run on a
          new codebase starts with no memory; subsequent runs start
          progressively more informed. The system improves by doing,
          not by being told.
        </Callout>

        <SubHeading>How procedural memory is leveraged</SubHeading>
        <Para>
          Before any agent takes its first action in a swarm, it issues
          a <Code>recall_memories</Code> call scoped to the current
          feature's capability tag. The call runs a hybrid Reciprocal
          Rank Fusion query — merging Neo4j keyword matches with Qdrant
          vector matches — and returns the top-ranked memories weighted
          by relevance score. The Coder receives Pattern nodes (reusable
          structures) and Mistake + Solution pairs (known failure modes
          with their fixes) before writing a single line of code. The
          Planner receives Strategy nodes (proven approach decisions)
          and Episode summaries (what happened on the last few runs of
          this exact feature) before choosing its opening move. The
          Reviewer and Tester receive Mistake nodes (patterns to watch
          for) before evaluating.
        </Para>
        <Para>
          This recall step is not advisory — it directly shapes agent
          behaviour. A Coder that recalls "parameterised queries prevent
          injection in this codebase" will use parameterised queries
          without the Reviewer having to catch the violation. A Planner
          that recalls "the last two runs of the auth feature failed
          when using the direct-write path; the third succeeded with the
          SDK path" will pick the SDK path immediately. A Tester that
          recalls "edge-case tests for this API must include null
          payload variants" will include them without being told. Memory
          turns prior failure into current constraint, and prior success
          into current prior.
        </Para>
        <FlowFigure
          nodes={MEMORY_NODES}
          edges={MEMORY_EDGES}
          caption="Figure 4 — Memory topology. Solid edges persist to a store; dashed edges feed the hybrid Reciprocal Rank Fusion merge used by every recall_memories() and recall_episodes() call. Pink nodes are the semantic tier + new episodic tier; blue is the hybrid merge."
          height={660}
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
          synthesises a{" "}
          <strong>narrative summary</strong> (up to 300 words) of the
          trajectory plus <strong>key turning-point events</strong> (up to 15){" "}
          (strategy picks, rejections, pivots, test passes). The
          result is stored as an <Code>Episode</Code> node in Neo4j
          linked to its <Code>Run</Code> via <Code>PRODUCED_IN</Code>,
          and embedded into the <Code>dark_factory_episodes</Code>{" "}
          Qdrant collection with enriched payloads (spec IDs, eval
          scores, recalled memory IDs).
        </Para>
        <Para>
          Episodes now capture <strong>which semantic memories
          influenced the outcome</strong> — the synthesis prompt
          includes the IDs of every pattern, strategy, and prior
          episode recalled during the feature&apos;s lifecycle.
          This lets future Planners see not just &ldquo;the last
          run succeeded&rdquo; but &ldquo;the last run succeeded{" "}
          <em>because it applied pattern-abc</em>.&rdquo;{" "}
          <Code>(Episode)-[:APPLIED]-&gt;(Pattern/Strategy/Episode)</Code>{" "}
          graph edges enable traversals like &ldquo;which episodes
          used this pattern?&rdquo; and &ldquo;which patterns come
          from successful runs?&rdquo;
        </Para>
        <Para>
          At the start of every feature, the Planner calls{" "}
          <Code>recall_episodes(feature_name=&lt;current&gt;)</Code>{" "}
          which runs the same hybrid Reciprocal Rank Fusion merge used by{" "}
          <Code>recall_memories</Code> — Neo4j keyword match on the
          summary text + Qdrant vector match on the embedding —
          returning the top-ranked past trajectories for the same
          feature. This gives the Dark Factory a{" "}
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
          <strong>boosted</strong> (and <Code>times_applied</Code>{" "}
          incremented). When the evaluation fails, the memory is{" "}
          <strong>demoted</strong> (relevance decreases but{" "}
          <Code>times_applied</Code> is NOT incremented — a demote
          means the memory was recalled but didn&apos;t help). Every
          run decays all memory relevance by 5%. Relevance scores
          are synced to Qdrant after every boost, demote, and decay
          so vector search ranking stays consistent with Neo4j. This
          produces a natural forgetting curve — stale lessons fade,
          proven lessons strengthen, and contradictory lessons compete.
        </Para>
        <Para>
          Recalled memory IDs are tracked in two sets:{" "}
          <strong>per-eval</strong> (cleared after each evaluation,
          used for boost/demote feedback) and{" "}
          <strong>per-feature</strong> (accumulated across the entire
          swarm lifecycle, used for episode synthesis). This ensures
          strategic memories recalled by the Planner get credit even
          when the evaluation happens several handoffs later in the
          Reviewer.
        </Para>
        <SubHeading>Hybrid retrieval + cross-feature recall</SubHeading>
        <Para>
          The <Code>recall_memories</Code> tool uses Reciprocal Rank
          Fusion to merge Neo4j keyword matches with Qdrant vector
          matches — exact-string matches (like a specific error code)
          and semantic matches (like &ldquo;similar to this
          situation&rdquo;) in one call, weighted together.
        </Para>
        <Para>
          Recall runs <strong>two Qdrant passes</strong>: one
          feature-scoped (high precision) and one unscoped
          (cross-feature). A pattern about parameterised SQL recorded
          by feature &ldquo;auth&rdquo; now surfaces when working on
          feature &ldquo;user-profile.&rdquo; When a Mistake is
          recalled, its associated Solution (via{" "}
          <Code>RESOLVED_BY</Code> edge) is returned inline so the
          agent sees the problem AND the fix in a single hit.
        </Para>
        <SubHeading>Cross-feature briefing</SubHeading>
        <Para>
          Memory crosses feature boundaries <em>within the same run</em>
          — when feature B starts, the patterns and mistakes from
          feature A are already indexed and retrievable. The briefing
          excludes the current feature&apos;s own memories so retried
          features don&apos;t see redundant data. Agents don&apos;t
          have to wait for the next run to benefit from what the current
          run just learned.
        </Para>

        <SubHeading>Memory lifecycle</SubHeading>
        <Para>
          The feedback loop drives memory evolution across runs:
          agents recall → use → eval → boost or demote → decay →
          prune → next run recalls with updated rankings.
        </Para>
        <FlowFigure
          nodes={MEM_LIFE_NODES}
          edges={MEM_LIFE_EDGES}
          caption="Figure — Memory lifecycle. Green edge = eval passed (boost). Red edge = eval failed (demote). Dashed = cross-run boundary."
          height={540}
        />

        <SubHeading>Memory hygiene</SubHeading>
        <Para>
          Five mechanisms keep the graph clean.{" "}
          <strong>Write-time dedup</strong>: before creating a new
          memory, the repository embeds the candidate and cosine-matches
          against existing memories above{" "}
          <Code>memory_dedup_threshold</Code> (default 0.92). For
          Patterns and Strategies, dedup is{" "}
          <strong>cross-feature</strong> — the same pattern discovered
          by different features consolidates into one high-relevance
          node. Mistakes and Solutions stay feature-scoped to avoid
          conflating distinct failure modes.{" "}
          <strong>Relevance-weighted Reciprocal Rank Fusion</strong>: the hybrid recall
          merge multiplies each rank contribution by the memory&apos;s
          relevance score, so the feedback loop influences retrieval
          ordering.{" "}
          <strong>Garbage collection</strong>:{" "}
          <Code>prune_low_relevance(threshold=0.05)</Code> runs at
          every pipeline start, deleting memories that have decayed
          below usefulness from both Neo4j and Qdrant.{" "}
          <strong>Qdrant payload sync</strong>: relevance scores,
          run IDs, timestamps, and recall counts are kept in sync
          between Neo4j and Qdrant payloads.{" "}
          <strong>Memory graph dashboard</strong>: the Metrics tab
          has a Memory section with per-type node counts, relevance
          histograms, top-10 recalled memories, and a 7-day
          boost/demote effectiveness KPI row.
        </Para>
        <Callout tone="success" title="Why cross-feature dedup matters">
          Without cross-feature dedup, a Coder recording &ldquo;use
          parameterised queries&rdquo; across five features creates
          five near-identical Pattern nodes that all rank high for
          SQL-related queries. The agent&apos;s recall list fills
          with paraphrases of the same idea and burns context. With
          cross-feature dedup, that single idea accumulates boost
          signal every time it&apos;s rediscovered, reaching high
          relevance faster and getting recalled first in every future
          run that needs it.
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
              "In-process pub/sub. Subscribers receive AG-UI events in real time — used by both the /api/agent/events Server-Sent Events stream and the server-side metric recorders.",
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
          opens a dedicated popup window with these tabs:
        </Para>
        <FactTable
          rows={[
            [
              "Agent Log",
              "Historical progress events for the run — same color-coded badge layout as the main Agent Logs tab, with text filter and expandable JSON payload detail per event. Default tab on open.",
            ],
            [
              "Metrics",
              "Status, pass rate, duration, spec/feature counts, LLM cost (broken down by pipeline phase), incidents, eval metrics, tool calls, artifacts, decomposition stats.",
            ],
            [
              "Evaluations",
              "Per-spec evaluation tree with requirements, metric scores, attempt history.",
            ],
            [
              "Episodes",
              "Episodic memory timeline — feature narratives, outcomes, key turning-point events, eval scores, recalled memory IDs.",
            ],
            [
              "Output",
              "File explorer for generated code/artifacts with syntax highlighting. Download ZIP button streams the full output as a zip archive.",
            ],
            [
              "Compare",
              "Side-by-side comparison against another run selected from a dropdown. Shows pass rate, duration, and feature-level status deltas with green/red coloring. 'View File Diffs' opens a unified diff viewer with line-level syntax coloring, powered by pre-computed MD5 manifests for efficient comparison.",
            ],
            [
              "Traceability",
              "Requirements → specs → files → tests → eval scores matrix. Table/Graph toggle: table view shows expandable rows with bulleted evaluations, files (linked to S3 presigned URLs), and tests; graph view renders an interactive React Flow dependency graph with status-colored nodes (green=pass, red=fail, yellow=no evals). Info tooltip on non-pass statuses explains which specs/metrics failed.",
            ],
          ]}
        />
        <Para>
          The Agent Log tab in Run Detail shares the same formatting
          code (<Code>lib/agentLogFormat.ts</Code>) as the main Agent
          Logs tab — event badges, color coding, and human-readable
          event descriptions are identical. The only difference is
          data source: the main tab reads from a live Server-Sent Events stream; the
          Run Detail tab loads historical records from the{" "}
          <Code>progress_events</Code> Postgres table via{" "}
          <Code>GET /api/metrics/runs/{"{run_id}"}</Code>.
        </Para>
      </Section>

      {/* 6.5 Traceability */}
      <Section
        title="6.5 Traceability & Storage"
        subtitle="How data flows from requirements to S3"
        id="traceability-arch"
      >
        <SubHeading>End-to-end data flow</SubHeading>
        <Para>
          A single requirement traces through the entire system:
          Requirement → Spec (via IMPLEMENTS) → Code artifact (via swarm
          coder) → Test file (via swarm tester) → Eval score (via DeepEval
          judge) → Episode (via post-mortem synthesis). Each step persists
          to a different store.
        </Para>
        <FlowFigure
          nodes={TRACE_NODES}
          edges={TRACE_EDGES}
          caption="Figure — Traceability data flow. Top row: pipeline artifacts. Bottom row: backing stores. Each artifact persists to the store below it."
          height={460}
        />

        <SubHeading>Storage architecture</SubHeading>
        <Para>
          All pipeline output flows through{" "}
          <Code>RunStorage</Code> which delegates to the configured
          backend. When <Code>STORAGE_BACKEND=local</Code> and{" "}
          <Code>S3_BUCKET</Code> is set, a{" "}
          <Code>ReplicatedStorage</Code> writes to both local disk
          (fast) and S3 (durable). Reads fall back to S3 when local
          is empty (handles container restarts). MD5 hashes are
          computed at write time and stored in{" "}
          <Code>.md5-manifest.json</Code> for efficient cross-run
          diffing.
        </Para>
        <FlowFigure
          nodes={STORAGE_NODES}
          edges={STORAGE_EDGES}
          caption="Figure — Storage architecture. Writes go to both backends; reads fall back to S3 on local miss. MD5 manifest enables O(1) diff comparisons."
          height={550}
        />
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
            style={{ color: "#2563eb" }}
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
              "Episode",
              "Autobiographical record of a feature swarm's execution. Linked to Run via PRODUCED_IN, and to recalled memories via APPLIED edges.",
            ],
            [
              "Memory",
              "Pattern / Mistake / Solution / Strategy nodes. Mistake-[:RESOLVED_BY]->Solution pairs the problem with its fix.",
            ],
          ]}
        />
        <Para>
          Key relationships:{" "}
          <Code>Spec-[:IMPLEMENTS]-&gt;Requirement</Code>,{" "}
          <Code>Spec-[:DEPENDS_ON]-&gt;Spec</Code>,{" "}
          <Code>EvalResult-[:EVALUATED_IN]-&gt;Run</Code>,{" "}
          <Code>Episode-[:PRODUCED_IN]-&gt;Run</Code>,{" "}
          <Code>Mistake-[:RESOLVED_BY]-&gt;Solution</Code>.
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
            style={{ color: "#2563eb" }}
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
            style={{ color: "#2563eb" }}
          >
            Kevin Quon
          </a>
        </div>
      </div>
      </div>
    </div>
  );
}
