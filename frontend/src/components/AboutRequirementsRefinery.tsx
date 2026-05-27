import { useEffect, useRef, useState } from "react";
import {
  Callout,
  Code,
  FactTable,
  Para,
  Section,
  SubHeading,
} from "./whitepaper/primitives";
import {
  DebateGraphFigure,
  EvalPipelineFigure,
  MemoryWriteBackFigure,
  ResearchPipelineFigure,
} from "./whitepaper/refineryDiagrams";
import { exportWhitepaperPdf } from "./whitepaper/exportPdf";

const PDF_EXPORT_BTN_ID = "refinery-pdf-export-btn";
const TOC_NAV_ID = "refinery-whitepaper-toc";

// Section title + anchor list — must mirror the render order below. The
// TOC component reads this. Ids are matched against each Section's ``id``.
const TOC_SECTIONS: ReadonlyArray<{ id: string; label: string }> = [
  { id: "business-value", label: "Business Value" },
  { id: "philosophy", label: "1. Design Philosophy" },
  { id: "system-overview", label: "2. System Overview" },
  { id: "panel", label: "3. The Adversarial Panel" },
  { id: "single-shot-llm", label: "3.5 Single-shot LLM Helper" },
  { id: "debate-graph", label: "4. The Debate Graph" },
  { id: "combined-eval", label: "5. Combined Evaluation Gate" },
  { id: "adversarial-contract", label: "5.5 Adversarial Contract" },
  { id: "research-agent", label: "6. Layered Research Agent" },
  { id: "role-filtered-retrieval", label: "6.5 Role-Filtered Retrieval" },
  { id: "institutional-memory", label: "7. Institutional Memory" },
  { id: "observability", label: "8. Observability" },
  { id: "calibration", label: "9. Continuous Calibration" },
  { id: "l4", label: "10. L4 Agentic Classification" },
  { id: "tradeoffs", label: "11. Tradeoffs" },
];

/**
 * Architectural whitepaper for the Requirements Refinery.
 *
 * Mounted when the URL hash is ``#/about-requirements-refinery``.
 * Reachable from the Refinery tab's input + results headers via
 * the "About" button (``openAboutRequirementsRefinery``).
 *
 * Style mirrors the Dark Factory whitepaper — same primitives,
 * same diagram vocabulary, same decimal section numbering. The
 * body refers to symbols by class / method only, not by file
 * path; the References section at the end is a class index.
 */
export default function AboutRequirementsRefinery() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    document.title = "About Requirements Refinery · AI Dark Factory";
  }, []);

  const exportPDF = async () => {
    if (!containerRef.current) return;
    setExporting(true);
    try {
      await exportWhitepaperPdf({
        container: containerRef.current,
        filename: "requirements-refinery-whitepaper.pdf",
        title: "Requirements Refinery — Architecture Whitepaper",
        hideElementId: PDF_EXPORT_BTN_ID,
      });
    } finally {
      setExporting(false);
    }
  };

  return (
    <>
      <header className="app-header">
        <h1>About Requirements Refinery</h1>
        <span className="badge">popup</span>
      </header>

      <main className="tab-content">
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
            <Hero exporting={exporting} onExport={exportPDF} />
            <BusinessValue />
            <DesignPhilosophy />
            <SystemOverview />
            <AdversarialPanel />
            <SingleShotLLM />
            <DebateGraphSection />
            <CombinedEval />
            <AdversarialContract />
            <ResearchAgent />
            <RoleFilteredRetrieval />
            <InstitutionalMemory />
            <Observability />
            <ContinuousCalibration />
            <L4Classification />
            <Tradeoffs />

            <div
              style={{
                textAlign: "center",
                color: "#57606a",
                fontSize: 11,
                margin: "24px 0 8px",
              }}
            >
              AI Dark Factory — Requirements Refinery
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Table of Contents — sticky left rail. Excluded from PDF export via the
// ``TOC_NAV_ID`` filter at capture time.
// ─────────────────────────────────────────────────────────────────────────────

function TableOfContents() {
  const [activeId, setActiveId] = useState<string>(TOC_SECTIONS[0]?.id || "");

  // Highlight the section currently nearest the top of the viewport. The
  // observer fires whenever a section's top crosses the threshold band.
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

// ─────────────────────────────────────────────────────────────────────────────
// Hero — title card + metadata facts
// ─────────────────────────────────────────────────────────────────────────────

function Hero({
  exporting,
  onExport,
}: {
  exporting: boolean;
  onExport: () => void;
}) {
  return (
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
          gap: 12,
        }}
      >
        <div
          style={{
            fontSize: 20,
            fontWeight: 700,
          }}
        >
          Requirements Refinery — Architecture Whitepaper
        </div>
        <button
          id={PDF_EXPORT_BTN_ID}
          onClick={onExport}
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
        An adversarial-panel system that turns ambiguous requirements
        into evidence-grounded specifications. Six role-specialised
        agents debate each requirement under a fused rules + LLM
        evaluation gate; outputs flow into a five-kind institutional
        memory store that informs every future debate.
      </div>
      <FactTable
        rows={[
          ["Author", "Kevin Quon — linkedin.com/in/kwkwan00"],
          [
            "Architecture pattern",
            "Adversarial multi-agent panel with bounded debate, layered-sourcing research, and fused deterministic + probabilistic evaluation",
          ],
          [
            "Per-requirement orchestration",
            "LangGraph StateGraph subgraph (generator → critic fan-out → synthesize → score → router → finalize / reconcile / research / escalate)",
          ],
          [
            "Panel composition",
            "6 role-specialised seats (Product, Engineering, Security, Operations, Cost, Judge) + 1 search capability (Research) — see Section 3",
          ],
          [
            "Evaluation gate",
            "Three-phase combined pipeline: deterministic rules → LLM judge with rule findings in prompt → rule-to-dimension penalty clamp",
          ],
          [
            "Knowledge stores",
            "Neo4j (graph + memory labels) + Qdrant (dense + BM25 hybrid retrieval) + Postgres (seven refinery_* forensic tables)",
          ],
          [
            "Institutional memory",
            "5 curated kinds — Decision / Incident / Pattern / Constraint / Conflict — with role-filtered recall and atomic write-back on user Apply",
          ],
          [
            "LLM provider routing",
            "Single-shot helper picks Anthropic SDK or OpenAI Responses by model id prefix; emits started / ready / failed events per call to the global broker + the SSE stream",
          ],
          [
            "Streaming protocol",
            "Server-Sent Events for live per-agent activity (Refinery progress timeline + Agent Log tab share the same event stream)",
          ],
          [
            "Operational mode",
            "Assistant — operator reviews and explicitly Applies each requirement and its auto-save-candidate memories",
          ],
          ["Agentic level", "L4 — Fully Autonomous / Explorer (Vellum scale)"],
        ]}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 0 — Business Value
// ─────────────────────────────────────────────────────────────────────────────

function BusinessValue() {
  return (
    <Section
      id="business-value"
      title="Business Value"
      subtitle="What the Requirements Refinery delivers — in plain terms"
    >
      <Para>
        The Requirements Refinery turns ambiguous, underspecified, or
        naïvely-written requirements into evidence-grounded
        specifications the downstream pipeline can act on without
        human re-edit. Operators submit a historical run, a folder
        of business documents, or a single typed requirement, and
        receive back refined requirements with explicit acceptance
        criteria, declared relationships to other requirements,
        suggested spec decompositions, and a documented decision
        trail.
      </Para>
      <Para>
        Rather than relying on a single agent to do everything, the
        refinery composes a six-seat{" "}
        <strong>adversarial panel</strong> — a Product seat that
        drafts, four critic seats (Engineering, Security, Operations,
        Cost) that challenge from their distinct perspectives, and a
        Judge seat that synthesises and scores. The panel debates
        each requirement under an evaluation gate that combines
        deterministic rules with a separate semantic judge, and a
        Research capability fetches external evidence on demand,
        always preferring internal knowledge before reaching for
        the public web.
      </Para>
      <SubHeading>Core features for stakeholders</SubHeading>
      <FactTable
        rows={[
          [
            "Three input modes",
            "Refine the evidence bundle from a historical pipeline run, ingest a folder of business documents in any common format, or submit a single typed requirement directly. The same panel runs in all three; cross-requirement review automatically skips when only one requirement is in play.",
          ],
          [
            "Six-seat adversarial panel",
            "Product drafts; Engineering, Security, Operations, and Cost critique in parallel from their owned perspectives; the Judge writes a revised draft that preserves disagreement in an explicit tradeoffs ledger rather than smoothing it. When two roles propose conflicting fixes, the rationale for picking one over the other is recorded inline so reviewers can audit the decision.",
          ],
          [
            "Bounded debate with circuit breaker",
            "Hard caps on debate rounds, model escalations, and external research excursions guarantee that every refinement terminates. When the panel cannot reach agreement within the budget, a dedicated reconcile step documents what stayed unresolved instead of forcing a synthesis that doesn't exist — operators see the open questions explicitly rather than implicitly.",
          ],
          [
            "Fused evaluation gate",
            "A deterministic rule layer runs first, catching hard-constraint breaks the language model might wave through. The semantic judge then scores against those same findings, and a final per-dimension penalty caps any score the rules have already invalidated. Three independent signals must agree before a refinement is declared converged.",
          ],
          [
            "Layered research capability",
            "When the panel flags missing external information, a four-role research pipeline walks six source tiers in descending trust order: structured internal knowledge first, then internal documents, observability data, official vendor docs, academic sources, and the public web last. Internal knowledge short-circuits external excursions, and any open-web finding requires independent corroboration before it can influence the panel.",
          ],
          [
            "Institutional memory write-back",
            "Validated outputs flow back into a structured memory store covering decisions, incidents, patterns, constraints, and conflicts. The memory write commits atomically alongside the user's Apply, so the next debate begins with the system's prior reasoning loaded under each role's filter policy. The system gets measurably faster and cheaper the more it runs.",
          ],
          [
            "Per-agent observability",
            "Every model call surfaces in real time with the agent's role, the model used, tokens consumed, latency, and a preview of both the prompt and the response. Three stores capture the data: a forensic SQL layer for joinable analysis, a time-series layer for dashboards and alerts, and a per-debate archive for one-shot replays.",
          ],
          [
            "Assistant, not autonomous",
            "The refinery suggests; the user explicitly applies each requirement and any candidate memories proposed by the panel. Requirement identifiers are preserved on edit so existing links between specs and requirements survive every change — applying a refinement never breaks downstream traceability.",
          ],
        ]}
      />
      <Callout tone="success" title="Who this is for">
        Engineering teams that want a structured, defensible answer
        to "is this requirement ready to spec?" — with the rationale,
        tradeoffs, and unresolved tensions written down in a form
        operators can audit and apply selectively.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Design Philosophy
// ─────────────────────────────────────────────────────────────────────────────

function DesignPhilosophy() {
  return (
    <Section title="1. Design Philosophy" id="philosophy">
      <Para>
        The refinery is built on a single premise:{" "}
        <strong>
          a panel of role-specialised agents arguing under a fused
          deterministic + probabilistic evaluation gate produces
          higher-quality refinement than a single agent with a
          larger context window.
        </strong>{" "}
        Every architectural choice flows from that premise.
      </Para>

      <SubHeading>Why specialisation beats scale</SubHeading>
      <Para>
        A single agent operating on an ambiguous requirement tends
        toward two failure modes. It <em>smooths disagreement</em>{" "}
        — one perspective dominates and tradeoffs go unrecorded —
        and it <em>conflates dimensions</em>, papering over
        feasibility or risk-coverage gaps with a "looks fine"
        judgment on clarity. The refinery answers both by giving
        each panel seat one job and one owned dimension, with its
        own model, prompt, and slice of context. Engineering
        challenges feasibility, Security challenges risk coverage,
        Operations challenges completeness, Cost challenges
        cost-shaped feasibility, and the Judge — and only the
        Judge — synthesises.
      </Para>

      <SubHeading>Disagreement as a first-class output</SubHeading>
      <Para>
        Adversariality is enforced structurally, not in the prompt.
        Critics that produce "looks good" output are rejected by a
        schema validator and retried with an explicit "you returned
        a rubber-stamp" follow-up; a second failure becomes a
        placeholder critique that the trace records as such. The
        Judge's synthesis is engineered to <em>preserve</em> tradeoffs
        in an explicit ledger rather than smooth them, and a
        dedicated reconcile step fires on non-convergence to document
        what the panel could not resolve. When the refinery emits a
        converged requirement, it means every evaluation signal
        agreed; when it emits a short-circuited one, operators get a
        precise ledger of what stayed open.
      </Para>

      <SubHeading>Full context for adversaries, limited context for searchers</SubHeading>
      <Para>
        A subtle architectural split: the panel seats receive the
        full shared evidence in one shot. They argue best when every
        role sees the same picture. The Research capability, by
        contrast, walks a sequence of tiered sources with
        intentionally narrow per-tier context — search is a
        different cognitive task than synthesis, and feeding a
        searcher every prior page degrades, rather than improves,
        the next decision. Adversaries get the same map; searchers
        get one quadrant at a time.
      </Para>

      <SubHeading>Three layers of agreement before convergence</SubHeading>
      <Para>
        Convergence is not a single verdict. A deterministic rule
        layer runs first — fast, free, and immune to model drift.
        The semantic judge then scores against those same rule
        findings, instructed to stay consistent with them. Finally
        a penalty clamp caps any dimension a blocker rule has
        already invalidated, so the semantic layer cannot overrule
        a deterministic finding even by accident. Convergence
        requires an overall score above threshold, every dimension
        above its own threshold, and zero blockers — three signals
        that must agree.
      </Para>

      <Callout tone="info" title="Bounded by construction">
        Every loop in the debate increments a counter compared to a
        finite cap, so non-convergence cannot silently loop.
        Belt-and-braces: the orchestrator carries a hard ceiling on
        total step count derived from the round budget, so even a
        pathological branch sequence terminates in a small,
        predictable number of steps.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. System Overview
// ─────────────────────────────────────────────────────────────────────────────

function SystemOverview() {
  return (
    <Section
      title="2. System Overview"
      subtitle="The four-phase SSE pipeline"
      id="system-overview"
    >
      <Para>
        A refinery invocation is a four-phase Server-Sent-Events
        pipeline orchestrated by{" "}
        <Code>run_refinery_stream</Code>. Phase 1 assembles the
        evidence and the requirements list;
        Phase 2 runs the per-requirement debate (this is where the
        panel does its work); Phase 3 reviews the refined set
        cross-requirement; Phase 4 persists results and yields a{" "}
        <Code>done</Code> event with the full <Code>RefineryResponse</Code>{" "}
        payload.
      </Para>

      <FactTable
        rows={[
          [
            "Phase 1 — Gather",
            "Collect requirements from one of three input modes: a historical run (traceability + gaps + episodes + memories + evals + run stats), uploaded documents (parsed via the IngestStage), or a single typed requirement wrapped in a minimal run_context with source_mode='direct'.",
          ],
          [
            "Phase 2 — Refine",
            "Each requirement is refined by the LangGraph debate graph in its own worker thread, up to MAX_CONCURRENT_AGENTS (5) in parallel. The graph's per-agent events stream back to the SSE consumer via a thread-local progress callback installed for each worker, so concurrent debates don't bleed events.",
          ],
          [
            "Phase 3 — Reconcile",
            "Runs only when the refined set has more than one requirement. The Judge's review_set verb (currently a structural pass-through; LLM-backed cross-set review is a follow-up) feeds apply_cross_review_report which patches relationships, priorities, and duplicate-pair edges back onto the set.",
          ],
          [
            "Phase 4 — Persist",
            "The full RefineryResponse is serialised to S3 or local storage at refinery/{result_id}/ alongside metadata.json and a rendered REPORT.md. The SSE stream then yields a single done event with the response payload so the UI can transition from progress to results.",
          ],
        ]}
      />

      <SubHeading>Three input modes</SubHeading>
      <Para>
        Phase 1 accepts <Code>run_id</Code> (refine evidence from a
        historical pipeline run), <Code>input_path</Code> (refine from
        uploaded MD/PDF/DOCX/etc), or <Code>direct</Code> (one typed
        requirement). The direct mode wraps the single requirement in
        a minimal <Code>run_context</Code> with{" "}
        <Code>source_mode="direct"</Code> and skips Phase 3 because
        cross-req review is meaningless on a single-item set.
      </Para>

      <SubHeading>Worker isolation</SubHeading>
      <Para>
        Phase 2 dispatches each requirement to its own worker thread
        via a <Code>ThreadPoolExecutor</Code>. Two pieces of
        per-thread state keep concurrent debates from bleeding into
        each other: the SSE orchestrator installs a thread-local
        progress callback so per-agent LLM events stream back to the
        right consumer, and each debate's evidence bag carries its
        own memory accumulator.
      </Para>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The Adversarial Panel
// ─────────────────────────────────────────────────────────────────────────────

function AdversarialPanel() {
  return (
    <Section
      title="3. The Adversarial Panel"
      subtitle="Six panel seats and one search capability"
      id="panel"
    >
      <Para>
        Every panel seat subclasses the <Code>RoleAgent</Code> ABC.
        The base class declares six verbs —{" "}
        <Code>propose</Code>, <Code>critique</Code>,{" "}
        <Code>defend</Code>, <Code>reconcile_unresolved</Code>,{" "}
        <Code>score</Code>, <Code>review_set</Code> — each defaulting
        to <Code>NotImplementedError</Code>. Each concrete role
        overrides only the verbs it owns, so the orchestrator can't
        accidentally call the wrong one (e.g.{" "}
        <Code>EngineeringRole.defend</Code> raises loudly).
      </Para>

      <FactTable
        rows={[
          [
            "Product (generator)",
            "Drafts the first-pass requirement using the full shared context (other requirements + run evidence) baked into one prompt. Single-shot LLM call; no tools, no iteration.",
          ],
          [
            "Engineering (critic)",
            "Owns the feasibility dimension. Challenges architecture, scalability, and stack-fit assumptions; pulls from Pattern + Mistake memories under a role-filter policy.",
          ],
          [
            "Security (critic)",
            "Owns risk_coverage. Pulls Incident + Constraint memories tagged security/auth/privacy/compliance; identifies abuse cases, vulnerabilities, and compliance gaps.",
          ],
          [
            "Operations (critic)",
            "Owns completeness. Pulls Solution + Incident + Pattern memories tagged ops/observability/runbook/incident; evaluates reliability, observability, and failure-mode coverage.",
          ],
          [
            "Cost (critic)",
            "Owns cost-shaped feasibility. Pulls Constraint + Pattern memories tagged infra/cost/pricing; explicitly excludes security/privacy memories so cost reasoning isn't biased by unrelated context.",
          ],
          [
            "Judge (synthesis + verdict)",
            "Owns all five dimensions. Three verbs: defend (synthesizes the round's critiques into a revised draft, preserving tradeoffs), score (runs the combined evaluation gate), reconcile_unresolved (short-circuit synthesis on non-convergence), review_set (cross-requirement Phase 3).",
          ],
          [
            "Research (capability, not a seat)",
            "Invoked by the router only when the Judge flags missing_external_info AND the per-debate research_call_cap has budget remaining. Runs the four-role Librarian → Analyst → Editor → Historian pipeline across six tiered providers — see Section 6.",
          ],
        ]}
      />

      <SubHeading>Default model tiers</SubHeading>
      <Para>
        Each seat picks its default model from one of three tiers,
        chosen so a false-negative would be most costly where the
        model is strongest. Within a tier, every seat shares the
        same default — flipping that default upgrades or downgrades
        the whole group at once via{" "}
        <Code>refinery_role_models</Code>.
      </Para>
      <FactTable
        rows={[
          [
            "Flagship reasoning tier",
            "Security and Judge. The Security seat owns the highest blast radius (false-negative on a vulnerability beats a noisy false-positive every time); the Judge owns synthesis and the verdict. Both run on the strongest reasoning-tier model the registry ships with.",
          ],
          [
            "High-volume critic tier",
            "Engineering and Operations. Both fire on every round across every requirement, so a mid-cost model with strong long-context behaviour is the right balance of latency, cost, and depth.",
          ],
          [
            "Numeric / retrieval tier",
            "Product, Cost, and Research. Product synthesizes structured JSON from full context; Cost reasons about quantitative tradeoffs; Research orchestrates retrieval across tiered providers. All three share an OpenAI-family default tuned for structured output and tool-style invocations.",
          ],
        ]}
      />

      <SubHeading>Registry + per-role overrides</SubHeading>
      <Para>
        The <Code>RoleRegistry</Code> reads{" "}
        <Code>refinery_role_models</Code> and{" "}
        <Code>refinery_role_reasoning_effort</Code> dictionaries from{" "}
        <Code>PipelineConfig</Code>, instantiates the factory, and
        threads overrides through{" "}
        <Code>configure(model=..., reasoning_effort=...)</Code> — so
        swapping a role's model or escalating its reasoning effort is
        a config-file edit, not a code change. The same
        <Code>configure</Code> method is what the router uses to
        switch to the strong-model tier on escalation.
      </Para>

      <Callout tone="info" title="Information hiding by ABC">
        The orchestrator may see role <em>names</em> (strings) and the
        ABC's six verbs. It may not import concrete role classes, read
        prompt text, or know retrieval strategies. This is the Parnas
        property the ABC exists to protect: every concrete role can be
        rewritten or swapped without touching the graph.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.5 Single-shot LLM helper
// ─────────────────────────────────────────────────────────────────────────────

function SingleShotLLM() {
  return (
    <Section
      title="3.5 Single-shot LLM Helper"
      subtitle="One helper, two providers, full per-call observability"
      id="single-shot-llm"
    >
      <Para>
        Every panel seat's <Code>_call_llm</Code> hook routes through{" "}
        <Code>call_refinery_llm</Code>. The helper picks the SDK from
        the model id prefix: Anthropic-prefixed ids go to the
        Anthropic Messages API with extended-thinking budget on{" "}
        <Code>high</Code> / <Code>xhigh</Code> reasoning effort;
        every other id goes to the OpenAI Responses API with{" "}
        <Code>reasoning.effort</Code> normalised to OpenAI's four
        tiers. The two adversarial seats that share the flagship
        reasoning tier go to Anthropic; the high-volume critics
        share an Anthropic mid-tier; the structured-output seats
        (Product, Cost, Research) go to OpenAI.
      </Para>

      <SubHeading>Per-call event triple</SubHeading>
      <Para>
        Each call emits exactly three events to the observability
        layer. Tests don't reach this helper — <Code>_call_llm</Code>{" "}
        is monkey-patched on the concrete role before the graph runs —
        so production network paths can never fire from a test.
      </Para>

      <FactTable
        rows={[
          [
            "refinery_llm_started",
            "Carries agent, model, provider, reasoning_effort. Renders as a dim 'PRODUCT …' / 'SEC …' / 'JUDGE …' row in both the Agent Log tab and the Refinery progress timeline. Prompt content is intentionally NOT broadcast — it lives in the forensic trace store only.",
          ],
          [
            "refinery_llm_ready",
            "Carries agent, model, provider, latency_ms, tokens_in, tokens_out. Renders as a bright 'PRODUCT ✓' / 'SEC ✓' / 'JUDGE ✓' row with the model + token counts. Response content is intentionally NOT broadcast — it lives in the forensic trace store only.",
          ],
          [
            "refinery_llm_failed",
            "On exception, before re-raising. Carries error text, provider, latency_ms. Renders as a red 'AGENT ✗' row with the error preview so operators can grep for misconfigured model IDs and provider auth issues.",
          ],
        ]}
      />

      <SubHeading>Dual-routing of events</SubHeading>
      <Para>
        Events are dual-routed by an internal <Code>_emit</Code>{" "}
        helper. The first sink is the global progress broker via{" "}
        <Code>emit_progress</Code> — that's how the Agent Log tab sees
        swarm and refinery events in one timeline. The second sink is
        a <Code>threading.local()</Code> slot installed by the SSE
        orchestrator's <Code>install_progress_callback</Code> context
        manager — that's how the Refinery progress view sees the live
        debate inline. Either sink missing degrades gracefully
        (broker-only or callback-only); neither sink blocks.
      </Para>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. The Debate Graph
// ─────────────────────────────────────────────────────────────────────────────

function DebateGraphSection() {
  return (
    <Section
      title="4. The Debate Graph"
      subtitle="Per-requirement LangGraph StateGraph"
      id="debate-graph"
    >
      <Para>
        <Code>build_debate_graph</Code> wires eight nodes plus the{" "}
        <Code>START</Code> /{" "}
        <Code>END</Code> sentinels. The deterministic spine
        (generator → critic fan-out → synthesize → score → finalize)
        runs on static edges; the conditional edges branch on router
        functions that read the most recent score.
      </Para>

      <DebateGraphFigure />

      <SubHeading>State schema</SubHeading>
      <Para>
        <Code>DebateState</Code> is a{" "}
        <Code>TypedDict</Code> with append-only reducers on every
        list-shaped field so the parallel critic Sends can write{" "}
        <Code>critiques_by_round</Code> without clobbering each other.
        The non-obvious entries:
      </Para>

      <FactTable
        rows={[
          [
            "round_number",
            "Incremented only by the synthesizer; bounds the conditional cap. Critics, the score node, and the router all read it but never write it.",
          ],
          [
            "research_calls_used / escalation_level",
            "Monotonic counters capped by refinery_research_cap and refinery_escalation_cap respectively. Once exhausted, the router stops branching to those paths.",
          ],
          [
            "critiques_by_round",
            "Round → list[Critique]. The barrier reducer merges parallel writes from the four critic Sends so synthesize sees a complete list when its node fires.",
          ],
          [
            "research_notes",
            "List[ResearchNote] surfaced into the next synthesize round's evidence bag. The Judge sees ValidatedInsights (with confidence + tier_mix), never raw T5 web snippets.",
          ],
          [
            "aborted_reason",
            "Hard abort signal that fast-paths the router to finalize. Set on generator crash or judge error. Ensures even pathological failures terminate cleanly.",
          ],
        ]}
      />

      <SubHeading>Three terminal outcomes</SubHeading>
      <FactTable
        rows={[
          [
            "Converged",
            "passed=True before max_rounds. finalize picks the best-scoring draft; convergence_status='converged'. The user sees the draft + a passing rubric.",
          ],
          [
            "Short-circuited",
            "round_number ≥ max_rounds without passing. reconcile_node consolidates what the panel HAS, not what it agreed on, and emits a CONFLICT memory tagged cause='non_convergence'. The frontend renders a 'NOT CONVERGED' badge with the unresolved_points + open_questions surfaced before the description.",
          ],
          [
            "Aborted",
            "aborted_reason set (generator crash, judge error). Carry-forward draft; convergence_status='aborted'; no CONFLICT memory because the failure isn't a substantive disagreement, it's a runtime fault.",
          ],
        ]}
      />

      <Callout tone="info" title="Termination invariants">
        Every loop back-edge increments a monotonic counter that is
        compared to a finite cap in <Code>_route_after_score</Code>.
        Once <Code>round_number ≥ max_rounds</Code>, the only
        reachable non-terminal path is <Code>reconcile</Code>, and{" "}
        <Code>reconcile → finalize</Code> is a plain edge with no
        conditional. Belt-and-braces:{" "}
        <Code>recursion_limit = max_rounds * 8 + 4</Code> on the
        compiled graph caps node visits at ≈ 27 even when every
        conditional branch fires.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Combined Evaluation Gate
// ─────────────────────────────────────────────────────────────────────────────

function CombinedEval() {
  return (
    <Section
      title="5. Combined Evaluation Gate"
      subtitle="Rules + LLM judge + penalty clamp, one verdict"
      id="combined-eval"
    >
      <Para>
        <Code>CombinedJudgePipeline.score</Code> runs three phases
        and returns one <Code>EvaluationScore</Code> with both
        pre-penalty and post-penalty dimension dictionaries so the
        trace can audit divergence between rule findings and LLM
        scores.
      </Para>

      <EvalPipelineFigure />

      <SubHeading>Why "fused" rather than "stacked"</SubHeading>
      <Para>
        Phase B sees Phase A's findings inside its prompt — the LLM
        is instructed to score consistently with the rule output
        rather than judging in isolation. Phase C is the backstop:
        if the LLM ignores the preamble and returns clarity = 0.95
        despite a blocker rule fire, Phase C clamps it to 0.5, the
        per-dim threshold of 0.7 rejects, and the gate fails. Rules
        have the final say; the LLM rarely needs the clamp in
        practice.
      </Para>

      <SubHeading>Rule catalog</SubHeading>
      <Para>
        Every <Code>RuleSpec</Code> declares its affected dimension
        as a class-level field, so Phase B can route findings into
        the right scoring criterion and Phase C can clamp the
        matching dimension. Starting catalog:
      </Para>
      <FactTable
        rows={[
          [
            "rule_ids_preserved (blocker / completeness)",
            "Draft.requirement_id must equal input.id. Graph-stability invariant — preserves all existing IMPLEMENTS edges from specs.",
          ],
          [
            "rule_priority_valid (blocker / feasibility)",
            "Priority must be one of {low, medium, high, critical}.",
          ],
          [
            "rule_convergence_consistency (blocker / clarity)",
            "If convergence_status='converged', unresolved_points and open_questions must be empty; if 'short_circuited', at least one must be non-empty.",
          ],
          [
            "rule_acceptance_criteria_min_count (blocker / testability)",
            "Each suggested_spec must have ≥ 3 acceptance criteria.",
          ],
          [
            "rule_acceptance_criteria_measurable (blocker / testability)",
            "Each criterion must match GIVEN/WHEN/THEN or carry a measurable outcome (number + unit, observable condition). Pure exhortation ('should be fast') fails.",
          ],
          [
            "rule_description_specificity (warning / clarity)",
            "Vague modifiers (fast, secure, scalable, robust, simple) without an adjacent concrete measure within ±15 tokens.",
          ],
          [
            "rule_relationships_non_circular (blocker / completeness)",
            "Relationship set must not form a cycle when added to the graph.",
          ],
          [
            "rule_tradeoffs_required (blocker / feasibility)",
            "If Rebuttal.rejected has ≥ 1 entry citing a conflicting fix, Draft.explicit_tradeoffs must be non-empty.",
          ],
          [
            "rule_no_rubber_stamp_rebuttal (blocker / feasibility)",
            "Judge's Rebuttal.rejected entries must cite a value-judgment (rationale ≥ 25 chars). Empty-rationale rejects fail the gate — bridges to the adversarial-contract layer.",
          ],
          [
            "rule_t5_confidence_cap (warning / risk_coverage)",
            "Any ValidatedInsight with source_tier_mix=[T5] only must have confidence ≤ 0.30. Belt-and-braces backstop for the research guardrail layer.",
          ],
        ]}
      />

      <Callout tone="info" title="Operator levers">
        <Code>refinery_rules_disabled</Code> skips named rules
        (logged to <Code>trace.rules_skipped</Code>);{" "}
        <Code>refinery_rules_short_circuit_llm = True</Code> skips
        Phase B when ≥ N blocker violations make LLM scoring
        wasteful (default off because trace value remains useful
        even when the gate will fail);{" "}
        <Code>refinery_rules_penalty_blocker_cap</Code> and{" "}
        <Code>refinery_rules_penalty_warning_delta</Code> tune the
        Phase C clamp.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.5 Adversarial Contract
// ─────────────────────────────────────────────────────────────────────────────

function AdversarialContract() {
  return (
    <Section
      title="5.5 Adversarial Contract"
      subtitle="Disagreement enforced at the data layer, not the prompt"
      id="adversarial-contract"
    >
      <Para>
        The hardest failure mode of multi-agent debate is{" "}
        <strong>convergent rubber-stamping</strong> — every critic
        returns generic approval and the system claims unanimous
        agreement. The refinery rejects this at three layers.
      </Para>

      <SubHeading>Layer 1 — Adversarial preamble</SubHeading>
      <Para>
        Each critic's prompt opens with: "your job is to challenge
        this draft from your role's perspective, not to validate
        it" plus an explicit escape hatch — return INFO-severity
        with a non-empty search-narrative if no concern is found
        after a deliberate search.
      </Para>

      <SubHeading>Layer 2 — Rubber-stamp validator</SubHeading>
      <Para>
        <Code>is_rubber_stamp</Code> rejects schema-conforming but
        vacuous output: <Code>finding</Code>{" "}
        shorter than 20 chars, <Code>proposed_fix</Code> shorter
        than 10 chars, or any of a known-rubber-stamp pattern set
        ("looks good", "no issues", etc.) — unless severity is
        INFO.
      </Para>

      <SubHeading>Layer 3 — Single retry, then placeholder</SubHeading>
      <Para>
        <Code>call_critic_with_retry</Code> retries once with a
        pointed "you returned a rubber-stamp" follow-up. A second
        failure becomes a placeholder Critique with{" "}
        <Code>error</Code> set, the round continues, and the trace
        records <Code>placeholder_rubber_stamp = true</Code> so
        operators can grep for the failure mode.
      </Para>

      <SubHeading>Synthesis preserves disagreement</SubHeading>
      <Para>
        The Judge's <Code>defend</Code> prompt requires every{" "}
        <Code>Rebuttal.rejected</Code> entry to cite the specific
        critique AND the value-judgment behind the rejection (e.g.
        "rejecting Cost's WARNING because Security's BLOCKER
        outweighs a 2× spend delta on a priority=critical auth
        feature"). When two critics raise mutually exclusive fixes,
        the revised <Code>Draft.explicit_tradeoffs</Code> must name
        the choice — and{" "}
        <Code>rule_tradeoffs_required</Code> is a Phase A blocker
        that fails the gate when it doesn't.
      </Para>

      <Callout tone="warn" title="Anti-collusion isn't free">
        A sufficiently-clever prompt could craft "substantive"
        critiques that all four critics align on, leaving real
        issues uncovered. Mitigations: per-role distinct dimensions
        reduce the surface, an anti-collusion test fixture plants
        flaws and asserts each role finds a different one, and the
        rules layer catches deterministic gaps even when the
        critics miss them. The system trusts the rules; it
        instruments the LLM.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Layered Research Agent
// ─────────────────────────────────────────────────────────────────────────────

function ResearchAgent() {
  return (
    <Section
      title="6. Layered Research Agent"
      subtitle="Internal-first triage across six tiers, three guardrails"
      id="research-agent"
    >
      <Para>
        Research is a <em>capability</em>, not a panel seat. The
        router invokes it only when the Judge flags{" "}
        <Code>missing_external_info = True</Code> AND the per-debate{" "}
        <Code>research_call_cap</Code> has budget remaining.
        Internally, <Code>ResearchAgent.run</Code> walks four roles
        across six tiered providers.
      </Para>

      <ResearchPipelineFigure />

      <SubHeading>The four roles</SubHeading>
      <FactTable
        rows={[
          [
            "Librarian",
            "Internal-first triage. Hits T0 (Neo4j + Qdrant + trace) and T1 (PRDs / ADRs / postmortems / wikis) first. If best_trust_score ≥ refinery_research_internal_sufficient_threshold (default 0.75), short-circuits — no external tier fires. Otherwise, T2 (observability), T3 (official vendor docs), T4 (academic), T5 (web) run in descending trust order, each capped by its tier budget.",
          ],
          [
            "Analyst",
            "ONE LLM call over the fetched corpus. Extracts atomic claims into ExtractedClaim records, each citing source.id (Pydantic-validated). Deliberately tool-less — no follow-up URLs, no per-claim browsing.",
          ],
          [
            "Editor",
            "Clusters claims by embedding similarity (≥ 0.85), trust-weights across tiers, and emits ValidatedInsight records. Cross-tier contradictions are preserved in ValidatedInsight.contradicting_claims so the Judge can surface tension instead of picking a side silently.",
          ],
          [
            "Historian",
            "Converts ValidatedInsight to SuggestedMemoryV2 with provenance_source_tier_mix, provenance_confidence, provenance_citations. These flow through the institutional-memory write-back loop on user Apply.",
          ],
        ]}
      />

      <SubHeading>Three enforced guardrails</SubHeading>
      <FactTable
        rows={[
          [
            "T5 raw never propagates",
            "A Pydantic validator on ResearchNote rejects any payload with a T5 citation that isn't wrapped in a ValidatedInsight backed by ≥ 1 non-T5 corroborating source. Web findings cannot reach the Judge unless an internal tier corroborates them.",
          ],
          [
            "No free browsing",
            "Each provider exposes search() + fetch(url) only; fetch URLs must match the provider's allowlist. T3 uses a hardcoded vendor list (AWS / Azure / GCP / Anthropic / OpenAI / FastAPI / Qdrant / Neo4j); T5 limits fetch URLs to those returned by its own search.",
          ],
          [
            "Per-tier budget exhaustion",
            "Each tier has a slot count from refinery_research_tier_budgets. Further calls raise TierBudgetExhausted, forcing the agent to reason with what it has. Default budgets: T0=8, T1=6, T2=4, T3=4, T4=2, T5=6.",
          ],
        ]}
      />

      <Callout tone="success" title="Why internal-first matters">
        When validated insights flow through the write-back loop they
        become T0 / T1 entries on subsequent runs. The next debate
        that hits the same query short-circuits at the Librarian — the
        system <em>doesn't search; it already knows</em>. Each
        external excursion measurably reduces the next one.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.5 Role-Filtered Retrieval
// ─────────────────────────────────────────────────────────────────────────────

function RoleFilteredRetrieval() {
  return (
    <Section
      title="6.5 Role-Filtered Retrieval"
      subtitle="Information hiding enforced at the data layer"
      id="role-filtered-retrieval"
    >
      <Para>
        Every retrieval is filtered server-side by the role's{" "}
        <Code>RoleFilterPolicy</Code>, not post-hoc in Python. The
        policy translator emits a{" "}
        <Code>qdrant_client.models.Filter</Code> with{" "}
        <Code>must</Code> / <Code>must_not</Code> / <Code>should</Code>{" "}
        clauses; a Cypher counterpart constrains the Neo4j graph
        traversal. A role cannot receive rows outside its slice
        even if its prompt malfunctions and tries to ask for them.
      </Para>

      <FactTable
        rows={[
          [
            "Product",
            "Memory kinds: decision, pattern, conflict. Graph: RELATED_TO + DEPENDS_ON. Row cap: 12. Receives prior tradeoffs at propose-time so the generator preempts known disagreements.",
          ],
          [
            "Engineering",
            "Memory kinds: pattern, incident, constraint, conflict. Excludes security-only tags. Graph: DEPENDS_ON + IMPLEMENTS. Row cap: 15.",
          ],
          [
            "Security",
            "Memory kinds: incident, constraint, conflict. Includes security/auth/privacy/compliance/incident tags. Graph: DEPENDS_ON + RELATED_TO. Row cap: 15.",
          ],
          [
            "Operations",
            "Memory kinds: incident, constraint, pattern. Includes ops/observability/runbook/incident tags. Graph: IMPLEMENTS + RELATED_TO. Row cap: 12.",
          ],
          [
            "Cost",
            "Memory kinds: constraint, pattern. Includes infra/cost/pricing tags; excludes security/privacy. Graph: DEPENDS_ON. Row cap: 10.",
          ],
          [
            "Research",
            "empty=True (no Qdrant or Neo4j access). Fetches externally through its own provider stack; nothing the role does is constrained by the local store.",
          ],
          [
            "Judge",
            "Reads the full DebateTrace for per-req scoring + the full refined set + all traces for cross-req review_set. No store retrieval; trace-only access by design.",
          ],
        ]}
      />

      <SubHeading>Frozen context + audit fingerprint</SubHeading>
      <Para>
        A built <Code>RoleContext</Code> is{" "}
        <Code>model_config = ConfigDict(frozen=True)</Code> so a role
        cannot mutate it to smuggle in extra rows. Each retrieved row
        is wrapped in a <Code>SourceRef</Code> (collection, point_id,
        score, matched_filter_fields) before it enters{" "}
        <Code>RoleContext.evidence</Code>; the policy + query are
        hashed into <Code>context_fingerprint</Code> so traces can
        recover which filter produced which context for every role
        call.
      </Para>

      <Callout tone="info" title="Operator overrides">
        <Code>refinery_role_filter_overrides</Code> is a{" "}
        <Code>dict[str, dict]</Code> of partial{" "}
        <Code>RoleFilterPolicy</Code> overrides applied at registry
        construction time. Example: bump Security's row_cap to 20
        and add "threat-model" to its include-tags list without
        editing code or restarting the rest of the panel.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Institutional Memory
// ─────────────────────────────────────────────────────────────────────────────

function InstitutionalMemory() {
  return (
    <Section
      title="7. Institutional Memory"
      subtitle="Five curated kinds, kind-scoped dedup, atomic write-back"
      id="institutional-memory"
    >
      <Para>
        Memory is the substrate the refinery both produces and
        consumes. Five kinds are emitted at specific debate events
        by per-event producer functions:
      </Para>
      <FactTable
        rows={[
          [
            "Decision",
            "Synthesizer accepts/rejects a critique with a generalisable rationale; Judge assigns a verdict at threshold boundary. Consumed by Product (propose) and Judge (defend / score / review_set).",
          ],
          [
            "Incident",
            "Critic emits severity=blocker citing a runtime/incident pattern (:Mistake label in Neo4j). Consumed by Security, Operations, Engineering.",
          ],
          [
            "Pattern",
            "Generated draft structure matches a reusable template; cross-req Judge sees the same structure across ≥ 2 reqs. Consumed by Product, Engineering.",
          ],
          [
            "Constraint",
            "Critic blocker cites a system or business limitation (stack pin, API rate limit, compliance, budget). Consumed by Product, Judge, Cost, Engineering.",
          ],
          [
            "Conflict",
            "disagreement_score ≥ refinery_conflict_emit_threshold at finalize OR reconcile_node fires unconditionally on short-circuit (cause='non_convergence'). Consumed by Product (propose), Judge (review_set), all critics (heads-up).",
          ],
        ]}
      />

      <MemoryWriteBackFigure />

      <SubHeading>Atomic write-back contract</SubHeading>
      <Para>
        On user Apply, the request to{" "}
        <Code>PATCH /api/graph/requirements/{"{req_id}"}</Code>{" "}
        carries an optional <Code>apply_memories</Code> list. The
        handler routes through{" "}
        <Code>MemoryRepository.record_refinery_memories_atomic</Code>.
        Dedup runs read-only
        outside the transaction (Qdrant similarity ≥{" "}
        <Code>memory_dedup_threshold</Code>, kind-scoped — a Pattern
        and a Decision with identical text are NOT considered
        duplicates of each other). New nodes commit in one Neo4j
        transaction; Qdrant upserts happen post-commit. On Qdrant
        failure, a compensating <Code>DETACH DELETE</Code> keeps
        the two stores in sync.
      </Para>

      <SubHeading>Validation status governs auto-save</SubHeading>
      <Para>
        Each suggested memory carries a <Code>validation_status</Code>{" "}
        field. <Code>validated</Code> (Judge cited it in synthesis)
        and <Code>produced</Code> (generated during a converged debate)
        auto-save by default; <Code>unvalidated</Code> and{" "}
        <Code>user_dismissed</Code> stay suggested for explicit
        review. Operators flip{" "}
        <Code>refinery_auto_save_on_apply = False</Code> to restore
        strict assistant-mode (every memory reviewed individually).
      </Para>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Observability
// ─────────────────────────────────────────────────────────────────────────────

function Observability() {
  return (
    <Section
      title="8. Observability"
      subtitle="Three stores, one trace, dual-write to legacy llm_calls"
      id="observability"
    >
      <Para>
        Each LLM call, tool call, and routing decision becomes a
        typed record in <Code>DebateTrace</Code>. The{" "}
        <Code>ObservabilityHub</Code> fans events to three sinks
        with intentionally distinct retention and query shapes.
      </Para>

      <FactTable
        rows={[
          [
            "Postgres — forensic + joinable",
            "Ten refinery_* tables FK'd back to pipeline_runs: the seven core forensic tables (refinery_runs, refinery_debates with priority+tags columns for archetype bucketing, refinery_debate_rounds, refinery_llm_calls, refinery_rule_violations, refinery_research_sources, refinery_memory_audits), the resume cache (refinery_debate_completions), plus the three operator-decision tables (refinery_memory_feedback, refinery_critique_dispositions, refinery_requirement_decisions). The seven section-9 loops join these forensic + decision tables in different combinations — adaptive max_rounds reads refinery_debates, self-tuning rule severity joins rule_violations to requirement_decisions, critic-dimension credibility groups dispositions by (role, dimension), etc. For SQL questions like 'which rules fire most on Security-tagged requirements this month' or 'is the Judge actually right about what operators apply'.",
          ],
          [
            "Prometheus — time-series aggregates",
            "≈ 25 dark_factory_refinery_* series covering convergence outcomes, rounds histogram, judge scores (pre/post-penalty), rule violations by rule_id/severity/dimension, research tier mix, internal-first hit ratio, memory write-back failures, cost + tokens per role, dual-write failures. A pre-built Grafana dashboard at deploy/grafana-dashboards/refinery.json ships 10 starter panels.",
          ],
          [
            "Trace JSON — full-fidelity archive",
            "Per-debate file under refinery/{result_id}/trace.json. Captures every prompt, response preview, tool call, routing decision, rule violation. Persistent for one-shot 'show me what happened in this debate' replays.",
          ],
        ]}
      />

      <SubHeading>Dual-write to llm_calls</SubHeading>
      <Para>
        Every refinery LLM call writes to <em>both</em>{" "}
        <Code>llm_calls</Code> (legacy system-wide cost ledger) and{" "}
        <Code>refinery_llm_calls</Code> (refinery-specific role /
        round / req_id detail) inside one transaction. Existing
        FinOps queries against <Code>llm_calls</Code> automatically
        include refinery cost; the dashboards don't need a regex
        tweak to see it. The <Code>phase</Code> column is set to{" "}
        <Code>refinery.&lt;role&gt;.&lt;kind&gt;</Code> (e.g.{" "}
        <Code>refinery.security.critique</Code>) so a single{" "}
        <Code>GROUP BY</Code> reveals per-role spend.
      </Para>

      <Callout tone="info" title="Per-agent live view">
        The LLM helper's three-event triple (started / ready /
        failed) flows to both the global progress broker and the
        refinery SSE callback, so the Agent Log tab and the Refinery
        progress timeline both show the same per-agent activity in
        real time. Filter by role name ("security", "judge") or
        event ("refinery_llm_ready") to audit a single agent's
        activity across an entire run.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Continuous Calibration — seven learning loops + active-learning feedback
// ─────────────────────────────────────────────────────────────────────────────

function ContinuousCalibration() {
  return (
    <Section
      title="9. Continuous Calibration"
      subtitle="Seven learning loops + an active-learning feedback path turn operator and panel signal into bounded multipliers on the next run"
      id="calibration"
    >
      <Para>
        Seven cooperating subsystems — each owning a distinct signal —
        nudge the next debate's behaviour based on what happened in
        prior runs. All seven follow the same architectural pattern:
        Postgres telemetry table → bounded aggregator → identity
        default on missing data → multiplier injected into the
        evidence bag (or runner kwarg, for orchestration parameters)
        at run start. None can hard-fail the pipeline; a Postgres
        outage means every loop falls back to <Code>×1.0</Code>
        (configured base for max_rounds; configured severity for
        rules) and the system behaves exactly as it did before the
        loop existed. This is what graduates the refinery from a
        stateful debate to genuinely L4 — the system observes its own
        predictions vs. operator follow-through and adjusts.
      </Para>

      <FactTable
        rows={[
          [
            "Provider trust learning",
            "Per-(tier, provider) propagation rate from refinery_research_sources feeds the Editor's confidence weighting. A T5 web provider whose claims keep getting rejected drifts down within its tier envelope; a T3 vendor doc that consistently survives cross-referencing drifts up. Bounds are per-tier so a runaway loop can't turn T5 into T0 or vice versa.",
          ],
          [
            "Adaptive role weighting",
            "The synthesize node records every (role, severity, action) triple from the Judge's rebuttal ledger into refinery_critique_dispositions. The aggregator computes per-role acceptance rates over the trailing 30-day window; the resulting multiplier (bounded [0.60, 1.40]) is rendered into format_defend_prompt as a calibration block so the LLM weighs each critic by historical accuracy. Below 8 disposition rows in the window, every role gets identity — early signal isn't trustworthy.",
          ],
          [
            "Critic-dimension credibility weighting",
            "Refines the flat per-role multiplier by splitting on the dimension a critique actually cited. The same dispositions table is grouped by (role, dimension); per-cell bounds are [0.50, 1.50] (slightly wider on the high side because a role's findings on its native dimension should outweigh the flat band). Per-role and per-dim multipliers compose multiplicatively, with a hard product clamp [0.50, 1.60] so two factors can't compose into ranges neither was tuned for. The synthesis prompt receives a 2D table of pre-combined values, rendered once per run because role × dim is invariant across rounds of the same debate.",
          ],
          [
            "Judge confidence calibration",
            "Operator apply / dismiss / edit decisions on refined requirements land in refinery_requirement_decisions, joined against the Judge's predicted overall score (or convergence_status as a fallback when scores haven't been plumbed end-to-end). The aggregator computes a four-quadrant miscalibration ratio (high-applied, high-dismissed, low-applied, low-dismissed) and emits a bounded [0.85, 1.15] multiplier that scales the effective threshold inside CombinedJudgePipeline.score. Tighter bounds than role-weighting because moving the threshold has cascading effects on rounds-per-debate and cost.",
          ],
          [
            "Adaptive max_rounds",
            "Per-archetype (priority, source_mode, primary_tag) convergence history from refinery_debates determines whether each requirement should run with a tighter or looser per-debate cap. Archetypes that consistently converge fast (avg rounds < base × 0.55) get the cap reduced by 1; archetypes whose debates short-circuit ≥30% of the time get +1, ≥55% get +2. Hard floor 2, hard ceiling 6. Plumbed as a runner kwarg into DebateConfig — orchestration parameters (recursion-limit, fanout cost) stay off the evidence bag.",
          ],
          [
            "Self-tuning rule severity",
            "Per-rule operator override rate (operator applied a requirement despite a rule blocker) on refinery_rule_violations joined to refinery_requirement_decisions. When ≥50% of operators override a particular rule's blockers (with at least 8 decided firings in the window), the rule is auto-demoted from blocker to warning at runtime — its findings still surface in the trace and apply Phase C's warning-delta penalty, but no longer hard-cap dimensions or gate convergence. Demote-only; warnings are never auto-promoted because they don't generate the override signal.",
          ],
          [
            "Active-learning feedback",
            "When an operator dismisses or accepts a suggested memory in the UI, three best-effort sinks fire: telemetry to refinery_memory_feedback (joins the calibration tables on cross-run analysis), Neo4j boost_relevance / demote_relevance on the memory node (re-ranks future role retrieval), and on a reasoned dismissal a Conflict memory tagged cause=\"user_override\" so the next debate sees the prior judgement as institutional context. Per-sink success flags come back so the UI can surface partial success without blocking.",
          ],
        ]}
      />

      <SubHeading>Why the same pattern, seven times</SubHeading>
      <Para>
        Every loop is a Postgres aggregation followed by a bounded
        multiplier (or bounded integer, for max_rounds) and an
        identity default. Repetition is the point — once the pattern
        is in the codebase, adding an eighth loop (e.g. systemic-
        issue detection across runs, or per-run goal-conditioned
        routing) is a copy-and-adapt of an existing module rather
        than a fresh design exercise. Each loop's bounds are tuned
        independently because the cost of being wrong differs: a
        noisy provider can't break a debate (per-tier envelope), a
        misjudged role still gets heard (multiplier never reaches
        zero), a miscalibrated Judge threshold cascades into rounds
        and cost so its bounds are tightest, and a runaway max_rounds
        bumper is hard-capped at 6 because every extra round
        multiplies critic-fanout cost.
      </Para>

      <Callout
        tone="info"
        title="The fingerprint that ties the loops together"
      >
        Five of the seven loops stamp into the same{" "}
        <Code>evidence_bag</Code> the Judge reads off{" "}
        <Code>RoleContext.evidence</Code> (
        <Code>role_weights</Code>, <Code>role_dim_weights</Code>,{" "}
        <Code>calibration_block</Code>,{" "}
        <Code>judge_threshold_multiplier</Code>,{" "}
        <Code>rule_severity_overrides</Code>). Provider trust feeds
        the Editor inside the research pipeline. Adaptive max_rounds
        is plumbed as a runner kwarg because it controls graph
        topology (recursion-limit, fanout) rather than role
        synthesis. The orchestrator computes every loop's snapshot
        once at run start (parallel debates share one snapshot so
        behaviour is stable across the run), promotes the in-bag
        signals from <Code>run_context</Code> into the evidence bag
        inside <Code>run_debate</Code>, and the composition pipeline
        + prompt formatter pick them up by name.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. L4 Agentic Classification
// ─────────────────────────────────────────────────────────────────────────────

function L4Classification() {
  return (
    <Section
      title="10. L4 Agentic Classification"
      subtitle="Where this sits on the Vellum scale"
      id="l4"
    >
      <Para>
        The Requirements Refinery operates at{" "}
        <strong>L4 — Fully Autonomous / Explorer</strong> on the{" "}
        <a
          href="https://www.vellum.ai/blog/levels-of-agentic-behavior"
          target="_blank"
          rel="noreferrer"
          style={{ color: "#0969da" }}
        >
          Vellum agentic behavior scale
        </a>
        , by the same definition the rest of the Dark Factory
        platform uses. All five L4 traits are implemented; the
        deliberately-bounded scope at the Apply boundary makes the
        system <em>operate</em> as an assistant without changing
        its agentic architecture.
      </Para>
      <FactTable
        rows={[
          [
            "Persistent state across sessions",
            "The five-kind institutional memory store (Decision / Incident / Pattern / Constraint / Conflict) survives across debates and runs. Per-debate ContextCache is scoped to one invocation and cleared between requirements; trace JSON is archived per debate; the seven refinery_* Postgres forensic tables persist forever and dual-write to llm_calls so existing FinOps queries auto-include refinery cost.",
          ],
          [
            "Refines execution from feedback",
            "Two surfaces. Inside one debate: the combined evaluation gate fuses three signals — deterministic rules → LLM judge with rule findings injected into the prompt → Phase C dimension clamp — and the router branches to next_round / research / escalate / reconcile / finalize accordingly. Across runs: the four calibration loops in section 9 turn operator and panel signal into bounded multipliers on the next debate (provider trust, role weighting, Judge threshold, plus active-learning feedback that re-ranks memory retrieval). The system measures its own predictions against operator follow-through and adjusts.",
          ],
          [
            "Parallel execution",
            "Within a debate round, the four adversarial critics run in parallel via a LangGraph Send fan-out and merge through a barrier reducer on critiques_by_round. Across the requirement set, MAX_CONCURRENT_AGENTS workers run debates concurrently via a ThreadPoolExecutor; a thread-local progress callback installed per worker keeps per-agent SSE events from bleeding between concurrent debates.",
          ],
          [
            "Real-time adaptation",
            "Mid-debate escalation upgrades the panel to a stronger reasoning tier when disagreement_score crosses a configurable threshold. Mid-debate research excursions invoke the layered-sourcing agent when the Judge flags missing_external_info AND the per-debate budget has slots. Mid-debate short-circuit to reconcile_node fires when rounds exhaust without convergence — the panel pivots from synthesis to documenting the unresolved tensions instead of forcing a consensus.",
          ],
          [
            "Cross-requirement reconciliation",
            "Phase 3 invokes JudgeRole.review_set across the full refined set + every per-requirement DebateTrace, emitting a CrossReviewReport. The structural patcher applies duplicate-pair edges, coherence-issue notes, missing or invalid relationships, priority inversions, and spec overlaps back onto the requirement set in one pass.",
          ],
        ]}
      />
      <Callout tone="info" title="Assistant-mode discipline ≠ lower autonomy">
        The Refinery is operated as an assistant — the user reviews
        and explicitly Applies each refined requirement, and the
        suggested-memories checklist gives them per-memory veto
        before write-back commits. That gate is a{" "}
        <em>scope choice</em> at the boundary between the panel and
        the persistent graph; the panel itself debates, escalates,
        researches, reconciles, and writes its own trace
        autonomously inside the loop. The Apply boundary is the
        difference between L4-as-assistant and L4-as-autonomous-loop;
        same architecture, different operational mode.
      </Callout>
    </Section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Tradeoffs + Open Questions
// ─────────────────────────────────────────────────────────────────────────────

function Tradeoffs() {
  return (
    <Section
      title="11. Tradeoffs + Open Questions"
      subtitle="Where the design pays a price, and what's left to build"
      id="tradeoffs"
    >
      <SubHeading>Cost vs depth</SubHeading>
      <Para>
        Six panel seats × up to three rounds × ~2k token prompts is
        substantially more expensive than a single-agent refinery.
        The Librarian's internal-first triage and the rules
        short-circuit amortise this — most debates terminate in
        round 1 with two Phase A blockers caught before any LLM
        scoring fires. For cost-constrained operators, the levers
        are <Code>refinery_debate_max_rounds = 1</Code> +{" "}
        <Code>refinery_rules_short_circuit_llm = True</Code> (catches
        hard-constraint breaks deterministically, skips the LLM
        gate when the verdict is already obvious).
      </Para>

      <SubHeading>Speed vs adversariality</SubHeading>
      <Para>
        Critic fan-out runs in parallel via LangGraph{" "}
        <Code>Send</Code> primitives — a round's wall-clock latency
        is bounded by the slowest critic, not the sum. The Judge
        synthesis + score serialise on top, so a typical converged
        debate runs four parallel critic calls plus two sequential
        Judge calls per round (~6× single-agent latency for ~4× the
        evaluation depth). Acceptable for an assistant-mode
        refinery where humans Apply; less so for an autonomous
        closed-loop pipeline.
      </Para>

      <SubHeading>Adversarial collusion at the prompt level</SubHeading>
      <Para>
        The rubber-stamp validator catches schema-conforming
        approval but a clever prompt could craft "substantive"
        critiques that all four agree on, leaving real issues
        uncovered. Mitigations: per-role distinct dimensions reduce
        the surface; an anti-collusion test fixture plants flaws
        and asserts each role finds a different one; and the rules
        layer catches deterministic gaps even when the critics
        miss them.
      </Para>

      <SubHeading>Open: cross-set review compute envelope</SubHeading>
      <Para>
        Phase 3 runs <Code>JudgeRole.review_set</Code> as a bounded
        critique-and-ratify loop (default 1 pass; raise{" "}
        <Code>refinery_set_review_max_rounds</Code> for stricter
        adjudication). The Judge sees the full refined set + every
        per-requirement <Code>DebateTrace</Code> in one prompt, which
        is fine for tens of requirements but scales poorly past a few
        hundred. The clean follow-up is a hierarchical pass —
        cluster the set first, run review_set per cluster, then a
        meta-review across cluster reports — gated on the same
        combined-evaluation primitives so the dimension-scoring
        contract stays consistent at every scale.
      </Para>

      <SubHeading>Open: research provider breadth</SubHeading>
      <Para>
        The current T3 allowlist hardcodes the major cloud + AI
        SDK vendor docs. Adding a new T3 source is a config edit
        plus a provider implementation. T4 is arXiv-only; T5 uses
        OpenAI Responses' built-in web_search tool. Tavily / Serper
        / Brave alternates are deliberately deferred until the
        existing T5 provider hits a documented limit.
      </Para>
    </Section>
  );
}
