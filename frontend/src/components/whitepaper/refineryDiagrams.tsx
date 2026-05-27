import { type Edge, type Node } from "@xyflow/react";
import { FlowFigure, Label, makeEdge, makeNode } from "./primitives";

// ─────────────────────────────────────────────────────────────────────────────
// Refinery whitepaper diagrams.
//
// Four ReactFlow figures matching the Dark Factory whitepaper's visual
// language (same shape vocabulary via the shared category palette in
// ``primitives.tsx``):
//
//   1. Debate graph topology  — generator → critic fan-out → synthesize →
//                               score → router → finalize / reconcile /
//                               research / escalate.
//   2. Combined evaluation pipeline — Phase A rules → Phase B LLM
//                               (rule-aware) → Phase C penalty clamp →
//                               unified EvaluationScore.
//   3. Layered research pipeline — six tier providers feeding the
//                               Librarian → Analyst → Editor → Historian
//                               sequence and writing back as
//                               SuggestedMemoryV2.
//   4. Memory atomic write-back — kind-scoped dedup → one Neo4j tx →
//                               Qdrant upsert → compensating delete on
//                               failure.
//
// Every node + edge is laid out by hand. ReactFlow's ``fitView`` rescales
// on render so the diagrams adapt to the popup width.
// ─────────────────────────────────────────────────────────────────────────────

// ── 1. Debate graph topology ────────────────────────────────────────────────

const DEBATE_NODES: Node[] = [
  makeNode("start", "START", 280, 0, "terminal", { width: 110 }),
  makeNode(
    "gen",
    <Label lines={["generator", "ProductRole.propose"]} />,
    260,
    80,
    "agent",
    { width: 160 },
  ),
  makeNode(
    "eng",
    <Label lines={["engineering", "feasibility"]} />,
    20,
    180,
    "agent",
    { width: 130 },
  ),
  makeNode(
    "sec",
    <Label lines={["security", "risk_coverage"]} />,
    180,
    180,
    "agent",
    { width: 130 },
  ),
  makeNode(
    "ops",
    <Label lines={["operations", "completeness"]} />,
    340,
    180,
    "agent",
    { width: 130 },
  ),
  makeNode(
    "cost",
    <Label lines={["cost", "feasibility"]} />,
    500,
    180,
    "agent",
    { width: 130 },
  ),
  makeNode(
    "synth",
    <Label lines={["synthesize", "JudgeRole.defend"]} />,
    260,
    280,
    "agent",
    { width: 160 },
  ),
  makeNode(
    "score",
    <Label lines={["score", "Combined pipeline"]} />,
    260,
    370,
    "agent",
    { width: 160 },
  ),
  makeNode(
    "router",
    <Label lines={["_route_after_score", "passed? max_rounds? research? escalate?"]} />,
    220,
    470,
    "phase",
    { width: 240 },
  ),
  makeNode(
    "research",
    <Label lines={["research", "ResearchAgent.run"]} />,
    560,
    470,
    "agent",
    { width: 160 },
  ),
  makeNode(
    "escalate",
    <Label lines={["escalate_model", "strong-tier next round"]} />,
    -40,
    470,
    "phase",
    { width: 180 },
  ),
  makeNode(
    "reconcile",
    <Label lines={["reconcile_node", "documents unresolved"]} />,
    -40,
    580,
    "agent",
    { width: 180 },
  ),
  makeNode(
    "finalize",
    <Label lines={["finalize", "convergence_status"]} />,
    260,
    580,
    "agent",
    { width: 180 },
  ),
  makeNode("end", "END", 318, 680, "terminal", { width: 110 }),
];

const DEBATE_EDGES: Edge[] = [
  makeEdge("start-gen", "start", "gen"),
  makeEdge("gen-eng", "gen", "eng", { label: "Send" }),
  makeEdge("gen-sec", "gen", "sec", { label: "Send" }),
  makeEdge("gen-ops", "gen", "ops", { label: "Send" }),
  makeEdge("gen-cost", "gen", "cost", { label: "Send" }),
  makeEdge("eng-synth", "eng", "synth"),
  makeEdge("sec-synth", "sec", "synth"),
  makeEdge("ops-synth", "ops", "synth"),
  makeEdge("cost-synth", "cost", "synth"),
  makeEdge("synth-score", "synth", "score"),
  makeEdge("score-router", "score", "router"),
  makeEdge("router-finalize", "router", "finalize", { label: "passed" }),
  makeEdge("router-reconcile", "router", "reconcile", {
    label: "max_rounds",
    color: "#dc2626",
  }),
  makeEdge("router-research", "router", "research", {
    label: "missing_external_info",
    dashed: true,
  }),
  makeEdge("router-escalate", "router", "escalate", {
    label: "disagree ≥ 0.7",
    dashed: true,
  }),
  makeEdge("research-synth", "research", "synth", { dashed: true }),
  makeEdge("escalate-eng", "escalate", "eng", {
    label: "fan-out",
    dashed: true,
  }),
  makeEdge("reconcile-finalize", "reconcile", "finalize"),
  makeEdge("finalize-end", "finalize", "end"),
];

export function DebateGraphFigure() {
  return (
    <FlowFigure
      nodes={DEBATE_NODES}
      edges={DEBATE_EDGES}
      height={620}
      caption={
        "Per-requirement debate graph. The Send fan-out runs the four critics " +
        "in parallel; the router branches to research / escalate (loops back to " +
        "synthesize / critics) or to finalize / reconcile (terminal)."
      }
    />
  );
}

// ── 2. Combined evaluation pipeline ─────────────────────────────────────────

const EVAL_NODES: Node[] = [
  makeNode(
    "draft",
    <Label lines={["Draft", "from synthesize"]} />,
    0,
    100,
    "memory",
    { width: 150, horizontal: true },
  ),
  makeNode(
    "rules",
    <Label lines={["Phase A — Rules", "RulesJudge.validate", "deterministic ~ms"]} />,
    200,
    20,
    "phase",
    { width: 180, horizontal: true },
  ),
  makeNode(
    "llm",
    <Label lines={["Phase B — LLM judge", "DeepEval GEval", "(rule findings in prompt)"]} />,
    200,
    180,
    "phase",
    { width: 200, horizontal: true },
  ),
  makeNode(
    "shortcircuit",
    <Label lines={["short-circuit?", "blockers ≥ N → skip B"]} />,
    200,
    310,
    "terminal",
    { width: 180, horizontal: true },
  ),
  makeNode(
    "fallback",
    <Label lines={["FallbackJudge", "heuristic"]} />,
    430,
    280,
    "agent",
    { width: 140, horizontal: true },
  ),
  makeNode(
    "penalty",
    <Label lines={["Phase C — penalty clamp", "blocker → cap 0.5", "warning → −0.1"]} />,
    470,
    100,
    "phase",
    { width: 200, horizontal: true },
  ),
  makeNode(
    "score",
    <Label lines={["EvaluationScore", "(post-penalty + raw_llm)"]} />,
    760,
    100,
    "memory",
    { width: 180, horizontal: true },
  ),
];

const EVAL_EDGES: Edge[] = [
  makeEdge("draft-rules", "draft", "rules"),
  makeEdge("draft-llm", "draft", "llm", { dashed: true }),
  makeEdge("rules-llm", "rules", "llm", {
    label: "rule findings",
    color: "#0550ae",
  }),
  makeEdge("rules-sc", "rules", "shortcircuit", { dashed: true }),
  makeEdge("sc-llm", "shortcircuit", "llm", { label: "no" }),
  makeEdge("llm-fallback", "llm", "fallback", {
    label: "exception",
    color: "#dc2626",
    dashed: true,
  }),
  makeEdge("rules-penalty", "rules", "penalty", {
    label: "violations",
    color: "#0550ae",
  }),
  makeEdge("llm-penalty", "llm", "penalty", { label: "raw dims" }),
  makeEdge("fallback-penalty", "fallback", "penalty", { dashed: true }),
  makeEdge("penalty-score", "penalty", "score"),
];

export function EvalPipelineFigure() {
  return (
    <FlowFigure
      nodes={EVAL_NODES}
      edges={EVAL_EDGES}
      height={420}
      caption={
        "Combined evaluation pipeline. Phase A's rule findings ride into Phase " +
        "B's prompt so the LLM scores in line with deterministic findings; Phase " +
        "C clamps as a final backstop. ``passed`` requires all three to agree."
      }
    />
  );
}

// ── 3. Layered research pipeline ────────────────────────────────────────────

const RESEARCH_NODES: Node[] = [
  // Tier providers (left column)
  makeNode("t0", <Label lines={["T0 Structured", "Neo4j + Qdrant"]} />, 0, 0, "memory"),
  makeNode("t1", <Label lines={["T1 Internal", "PRDs / ADRs"]} />, 0, 70, "memory"),
  makeNode("t2", <Label lines={["T2 Observability", "metrics"]} />, 0, 140, "service"),
  makeNode("t3", <Label lines={["T3 Official", "vendor docs"]} />, 0, 210, "service"),
  makeNode("t4", <Label lines={["T4 Academic", "arXiv"]} />, 0, 280, "service"),
  makeNode("t5", <Label lines={["T5 Web", "search"]} />, 0, 350, "service"),

  // Pipeline (centre column)
  makeNode(
    "lib",
    <Label lines={["Librarian", "internal-first triage"]} />,
    240,
    60,
    "agent",
    { width: 180 },
  ),
  makeNode(
    "ana",
    <Label lines={["Analyst", "claim extraction"]} />,
    240,
    180,
    "agent",
    { width: 180 },
  ),
  makeNode(
    "edt",
    <Label lines={["Editor", "cross-tier weighting"]} />,
    240,
    280,
    "agent",
    { width: 180 },
  ),
  makeNode(
    "his",
    <Label lines={["Historian", "→ SuggestedMemoryV2"]} />,
    240,
    380,
    "agent",
    { width: 180 },
  ),

  // Outputs (right column)
  makeNode(
    "insights",
    <Label lines={["ValidatedInsight", "confidence + tier_mix"]} />,
    480,
    240,
    "memory",
    { width: 180 },
  ),
  makeNode(
    "memory",
    <Label lines={["SuggestedMemoryV2", "with provenance"]} />,
    480,
    380,
    "memory",
    { width: 180 },
  ),
];

const RESEARCH_EDGES: Edge[] = [
  // Tiers feed the librarian
  makeEdge("t0-lib", "t0", "lib", { color: "#16a34a" }),
  makeEdge("t1-lib", "t1", "lib", { color: "#16a34a" }),
  makeEdge("t2-lib", "t2", "lib", { dashed: true }),
  makeEdge("t3-lib", "t3", "lib", { dashed: true }),
  makeEdge("t4-lib", "t4", "lib", { dashed: true }),
  makeEdge("t5-lib", "t5", "lib", { dashed: true }),
  // Pipeline
  makeEdge("lib-ana", "lib", "ana"),
  makeEdge("ana-edt", "ana", "edt"),
  makeEdge("edt-his", "edt", "his"),
  // Outputs
  makeEdge("edt-insights", "edt", "insights", { label: "validated" }),
  makeEdge("his-mem", "his", "memory", { label: "write-back" }),
];

export function ResearchPipelineFigure() {
  return (
    <FlowFigure
      nodes={RESEARCH_NODES}
      edges={RESEARCH_EDGES}
      height={460}
      caption={
        "Layered research pipeline. Solid edges from T0 / T1 are the " +
        "internal-first short-circuit path; dashed external tiers fire only " +
        "when the internal signal is below threshold and the per-tier budget " +
        "has slots remaining."
      }
    />
  );
}

// ── 4. Memory atomic write-back ─────────────────────────────────────────────

const MEMORY_NODES: Node[] = [
  makeNode(
    "user",
    <Label lines={["User Apply", "PATCH /api/graph/requirements/{id}"]} />,
    0,
    100,
    "ui",
    { width: 220, horizontal: true },
  ),
  makeNode(
    "dedup",
    <Label lines={["Dedup (kind-scoped)", "Qdrant similarity ≥ 0.92"]} />,
    280,
    100,
    "phase",
    { width: 200, horizontal: true },
  ),
  makeNode(
    "tx",
    <Label lines={["Neo4j transaction", "CREATE all-or-nothing"]} />,
    540,
    20,
    "service",
    { width: 200, horizontal: true },
  ),
  makeNode(
    "qdrant",
    <Label lines={["Qdrant upsert", "post-commit"]} />,
    540,
    180,
    "service",
    { width: 200, horizontal: true },
  ),
  makeNode(
    "compensate",
    <Label lines={["Compensating", "DETACH DELETE"]} />,
    810,
    180,
    "phase",
    { width: 180, horizontal: true },
  ),
  makeNode(
    "ok",
    <Label lines={["committed", "(both stores in sync)"]} />,
    810,
    20,
    "agent",
    { width: 180, horizontal: true },
  ),
];

const MEMORY_EDGES: Edge[] = [
  makeEdge("user-dedup", "user", "dedup"),
  makeEdge("dedup-tx", "dedup", "tx", { label: "new" }),
  makeEdge("dedup-ok", "dedup", "ok", { label: "boost only", dashed: true }),
  makeEdge("tx-qdrant", "tx", "qdrant", { label: "Neo4j commit OK" }),
  makeEdge("qdrant-ok", "qdrant", "ok", { label: "OK" }),
  makeEdge("qdrant-compensate", "qdrant", "compensate", {
    label: "fail",
    color: "#dc2626",
    dashed: true,
  }),
];

export function MemoryWriteBackFigure() {
  return (
    <FlowFigure
      nodes={MEMORY_NODES}
      edges={MEMORY_EDGES}
      height={320}
      caption={
        "Atomic memory write-back on Apply. Dedup runs read-only outside the " +
        "transaction. New nodes commit in one Neo4j tx; Qdrant upserts happen " +
        "post-commit. On Qdrant failure, a compensating DETACH DELETE keeps " +
        "the two stores in sync."
      }
    />
  );
}
