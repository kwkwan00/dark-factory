# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tech stack

- **Backend:** Python 3.12+ via `uv`, FastAPI, structlog, Pydantic, LangGraph (per-feature swarms + refinery debate subgraph), Anthropic SDK, OpenAI SDK, DeepEval (GEval).
- **Datastores:** Neo4j (knowledge graph + procedural memory), Qdrant (vector search, dense + BM25 sparse), Postgres (forensic metrics).
- **Frontend:** React + Vite + TypeScript, AG-UI protocol over SSE, ReactFlow for diagrams.
- **Infra:** Docker Compose (dark-factory, neo4j, qdrant, postgres, prometheus, grafana, adminer); pluggable storage backend (`local` / `s3` / replicated local+s3).

## Common commands

```bash
# Backend
uv sync
uv run uvicorn dark_factory.api.app:app --host 0.0.0.0 --port 8000 --reload

# Frontend dev (proxies /api to :8000)
cd frontend && npm install && npm run dev

# Frontend production build (output copied into src/dark_factory/api/static/)
cd frontend && npm run build && npm run copy-dist

# Tests
uv run pytest tests/ -v                                # full suite (~1000+ tests)
uv run pytest tests/refinery/ -q                       # one suite
uv run pytest tests/refinery/test_resume.py::test_resume_replays_cached_and_runs_only_pending -v  # single test
uv run pytest tests/ -m "not integration and not slow" # fast subset

# Lint / format / typecheck
uv run ruff check src/ tests/
uv run ruff format src/ tests/
cd frontend && npx tsc --noEmit

# Docker stack
make build && make up    # full stack on :8000 (Grafana :3000, Adminer :8080)
make logs                # tail dark-factory container
```

Pytest markers: `@pytest.mark.slow` (>1s), `@pytest.mark.integration` (needs Neo4j / Qdrant / Postgres).

## High-level architecture

This is a **two-pipeline system** sharing one FastAPI app, one React SPA, and one set of datastores. Both pipelines stream progress over SSE and persist forensic state.

### Pipeline 1 — Manufacture (autonomous code generation)

Six phases driven by `dark_factory.api.routes_agent` and orchestrated through `agents/orchestrator.py`:

1. **Ingest** (`stages/ingest.py`, `stages/doc_extraction.py`, `stages/dedup.py`) — native parsers for `.md/.json/.yaml/OpenSpec`; rich docs (`.docx/.xlsx/.pdf/.html/...`) routed to per-file Claude Agent SDK invocations with isolated context. Semantic dedup pass collapses near-duplicate requirements (cosine ≥ `requirement_dedup_threshold`).
2. **Spec** (`stages/spec.py`) — early-exit preflight against Neo4j `IMPLEMENTS` edges; otherwise decompose → generate → DeepEval → refine swarm (capped by `max_spec_handoffs`); auto-index into Qdrant.
3. **Spec Reconciliation** (`stages/spec_reconciliation.py`) — deterministic pass (strip phantoms, break cycles via DFS) + LLM pass (implicit deps, coverage gaps, capability grouping); LLM patches re-validated.
4. **Graph** (`stages/graph.py`) — persists to Neo4j; orchestrator groups by capability and computes a Tarjan-SCC topological execution order.
5. **Per-feature swarms** (`agents/swarm.py`) — parallel within layers (`max_parallel_features`). Each feature is a LangGraph `create_swarm` of **Planner / Coder / Reviewer / Tester** rotating via handoffs. Cross-feature learning briefs subsequent features in the same run.
6. **Reconciliation + E2E** (`stages/reconciliation.py`, `stages/e2e_validation.py`) — single extended Claude Agent SDK pass over the full output tree, then a Playwright cross-browser smoke test. Both are **best-effort**: failures become incidents, never block delivery. E2E only runs when reconciliation status is `clean`.

### Pipeline 2 — Refinery (adversarial panel)

`api/refinery/` is a separate SSE pipeline that converts raw requirements into refined, debate-validated specs. It is **assistant-mode** (suggests; user decides), not autonomous. Feature-flag-gated by `refinery_debate_enabled` in `PipelineConfig`.

Per-requirement, the refinery runs a **LangGraph debate subgraph** (`api/refinery/debate/graph.py`):

```
generator → critic-fanout → synthesize → score → router →
  finalize | research | escalate | reconcile | next round
```

Six panel seats (`api/refinery/roles/`): **Product** (generator), **Engineering / Security / Operations / Cost** (parallel adversarial critics), **Judge** (synthesis + score + cross-set review). Research is a *capability* (`api/refinery/research/`) the Judge invokes via a 4-role layered-sourcing pipeline (Librarian → Analyst → Editor → Historian) across 6 source tiers (T0 structured → T5 web), with internal-first triage and trust-weighted cross-referencing.

Three terminal outcomes per requirement: `converged` (threshold met), `short_circuited` (max_rounds hit; reconcile node documents what couldn't be resolved + emits a `Conflict` memory), `aborted` (generator crash). Cross-requirement Phase 3 (`set_review.py`) runs a multi-pass critique-and-ratify loop after all per-req debates complete.

The **combined evaluation pipeline** in `api/refinery/judge/composition.py` is load-bearing: `Phase A` deterministic rules → `Phase B` LLM judge (with rule findings injected into its prompt) → `Phase C` rule→dimension penalty clamps. One unified `EvaluationScore` carries both signals; rules have the final say via Phase C clamps regardless of LLM output.

**Per-requirement debate episode.** Every debate produces a `DebateEpisode` (`api/refinery/episode.py`) — a deterministic, no-LLM projection of the final `DebateTrace` into a structured artifact (key events, dimension scores, accepted/rejected rebuttals, terminal outcome). The runner stamps `refined.debate.episode` + `episode_markdown` onto the refined requirement; storage writes per-req `refinery/{result_id}/episodes/{req_id}.{md,json}`; `stream.py` projects each episode onto the swarm `Episode` shape and writes it through `EpisodeWriter` to Neo4j `:Episode` + Qdrant `episodes` (best-effort). The refinery is the second consumer of the swarm Episode store — the id prefix `refinery-episode-…` is the only discriminator today.

**Convergence score** (`compute_convergence_score` in `episode.py`) is a deterministic 0.0–1.0 value derived from the trace: `1.0` if `convergence_status == converged`, `0.0` if no scoring rounds, else `best_overall / overall_threshold` clamped to `[0, 0.99]`. Stamped onto `RefinedRequirement.convergence_score` and rendered by the frontend `ConvergenceBar`. Operator triage depends on the formula; keep it pinned via contract test if `EvaluationScore.overall_threshold` semantics ever change.

### Three independent learning loops (L4 agentic)

All three follow the same architectural pattern — Postgres telemetry table + bounded multiplier aggregator + identity-default + evidence-bag injection. Each is no-op when Postgres is unavailable.

| Loop | Signal source | Where applied |
|------|--------------|---------------|
| **Provider trust learning** (`research/learning.py`) | `refinery_research_sources.propagated` aggregated per (tier, provider) | Editor's confidence scoring weights per-tier |
| **Adaptive role weighting** (`role_weighting.py`) | `refinery_critique_dispositions` accept/reject rate per role | Multiplier injected into `format_defend_prompt` so LLM weighs critics by historical accuracy |
| **Judge confidence calibration** (`judge_calibration.py`) | `refinery_requirement_decisions` (operator apply/dismiss/edit vs. Judge score) | Scales `effective_threshold = base * multiplier` in `CombinedJudgePipeline.score` |

Active-learning feedback (`feedback.py`) closes the loop: when an operator dismisses a memory or requirement in the UI, three sinks fire best-effort — Postgres telemetry, Neo4j relevance boost/demote on the memory node, and a `Conflict` memory tagged `cause="user_override"` on reasoned dismissals.

### Five-kind institutional memory

`memory/repository.py` owns five Neo4j labels — **Pattern, Mistake (alias: Incident), Solution, Strategy** (legacy swarm types) plus **Decision, Constraint, Conflict** (refinery types). Memory write-back is **kind-scoped** (a Decision and a Pattern with identical text are not duplicates). Every kind carries `relevance_score`, a usage counter (`times_applied` / `times_seen` / `times_recalled`), and `last_feedback_at`. Recall fuses Neo4j + Qdrant via relevance-weighted RRF.

Refinery debates also project onto the **swarm `:Episode` label** (Neo4j + Qdrant `episodes` collection) via `EpisodeWriter`, with `episode_id = "refinery-episode-{run_id}-{requirement_id}"` and the swarm outcome enum collapsed (`converged → success`, `short_circuited → partial`, `aborted → failed`). The id prefix is currently the only discriminator between refinery-origin and swarm-origin episodes.

### Observability triple-store

| Store | What goes here | Retention |
|-------|---------------|-----------|
| **Postgres** (`metrics/refinery_repository.py`) | Joinable forensic rows: `refinery_runs`, `refinery_debates`, `refinery_debate_rounds`, `refinery_llm_calls`, `refinery_rule_violations`, `refinery_research_sources`, `refinery_memory_audits`, `refinery_memory_feedback`, `refinery_critique_dispositions`, `refinery_requirement_decisions` | Long |
| **Prometheus** (`metrics/prometheus.py`) | Counters/histograms for dashboards + alerts | 2 weeks default |
| **Trace JSON** | Full-fidelity per-debate archive at `output/refinery/{result_id}/trace.json` | Long (S3 if enabled) |
| **Episode artifacts** | Per-requirement `output/refinery/{result_id}/episodes/{req_id}.{md,json}` + Neo4j `:Episode` + Qdrant `episodes` | Long (S3 if enabled) |

Refinery LLM calls **dual-write** to both `llm_calls` (canonical, for cost dashboards — `phase = "refinery.{role}.{kind}"`) and `refinery_llm_calls` (refinery-specific columns) inside one transaction. Dual-write failures roll back together and bump `refinery_dual_write_failures_total`; the debate continues regardless.

## Conventions

- **Information hiding (Parnas):** roles, retrievers, judges, and research providers are ABCs with stable contracts. The orchestrator imports only contracts (`api/refinery/contracts.py`) and never reads prompt text or knows model IDs.
- **Role-filtered retrieval at the data layer**, not post-hoc. `api/refinery/context/role_slices.py` declares `RoleFilterPolicy` per role; `VectorRepository.search_*_hybrid` requires the policy as a non-optional argument so a role *cannot receive* rows outside its slice.
- **Best-effort sinks never block.** Postgres / Neo4j / Qdrant outages must log + continue. A telemetry failure is always preferable to a telemetry-driven outage.
- **Identity defaults** for every learning loop. No data, missing client, or insufficient signal → multiplier = 1.0 (no change). Bounds are enforced even when signal is strong.
- **All progress flows through SSE.** Background work (thread pool, deep-agent subprocesses) emits via `_on_progress` callback into a per-run `asyncio.Queue` that the orchestrator drains on the event loop. Thread-local `install_progress_callback` lets the shared LLM helper inside roles echo `refinery_llm_*` events back into the stream alongside the broker.
- **SSE event payloads carry semantics, not content.** `refinery_llm_started`/`refinery_llm_ready` emit model + token count + reasoning_effort only — no prompt or response text. Full prompt/response bodies live in the forensic `trace.json` and Postgres `refinery_llm_calls`. Critic/synthesis events carry structured fields (`finding`, `proposed_fix`, `dimensions`, `severity`, `confidence`, `cited_evidence_count`) — bounded previews, never full text.
- **Critic fan-out must return `list[Send]`, never a string.** LangGraph conditional edges that re-dispatch to the panel (`generator → critic-fanout`, `score → critic-fanout` on non-convergent rounds) build per-role state overlays via `Send(node, {..., "_critic_role": role})`. Routing back via a path_map string key (e.g. `"next_round" → "critic"`) silently degenerates the round to a single un-overlayed critic node. See `_critic_sends` / `_route_after_score` in `debate/graph.py`; regression covered by `tests/refinery/test_router.py`.
- **JSON extraction shared.** LLM JSON responses go through `api/refinery/roles/_shared/json_utils.extract_json_object` (handles fenced blocks, prose-wrapped objects). Don't reimplement the regex.
- **Per-provider effort enum translation.** Anthropic Opus 4.6 rejects `xhigh` (supports only `low`/`medium`/`high`/`max`) and uses `output_config.effort` + adaptive thinking. OpenAI uses `reasoning_effort` directly. `roles/_shared/llm.py` keeps separate `_ANTHROPIC_REASONING_MAP` and `_OPENAI_REASONING_MAP` translations — when adding a provider, add its own map; don't share. Judge + Security currently default to `claude-opus-4-6` for the adaptive-thinking + `effort=max` path; other critics default elsewhere.
- **Refinery v2 only.** The legacy single-agent refinery path was removed; `run_phase2_generator` always routes through the LangGraph debate. The `refinery_debate_enabled` flag mentioned in REFINERY.md is historical (it was the soak gate).
- **No emojis in code or commits unless explicitly requested.**

## Storage layout

Per-run files live under `{run_id}/{input,requirements,specs,output}/`. Local backend → `./output/{run_id}/...`. S3 backend → `s3://{bucket}/{run_id}/...`. Replicated mode (`STORAGE_BACKEND=local` + `S3_BUCKET` set) writes to both, falls back to S3 on local read miss (handles container restarts), and stores MD5 manifests at write time for cross-run diffing.

Deep agents need real filesystem tools, so for S3 backends files are downloaded to a scratch directory before each phase and synced back after. For local, scratch *is* the storage directory.

## Where to look first

- **Adding a refinery role** — extend `api/refinery/roles/base.py` ABC; register in `roles/registry.py`; add `RoleFilterPolicy` entry in `context/role_slices.py`.
- **Adding a rule to the judge gate** — `api/refinery/judge/rules.py`. Every rule must declare its `dimension` (one of the 5 eval dimensions); Phase B's prompt and Phase C's clamp use that mapping.
- **Adding a research source tier** — implement `ResearchProvider` ABC in `api/refinery/research/providers/`; register in the `ProviderRegistry` with tier + URL allowlist; add bounds to `_BOUNDS_PER_TIER` in `research/learning.py`.
- **New forensic table** — add to `metrics/schema.py` (idempotent `CREATE TABLE IF NOT EXISTS` + indexes), then a write method on `RefineryMetricsRepository` wrapped in `_safe`-style try/except logging.
- **New SSE event type** — emit through `_on_progress` callback; the AG-UI bridge passes anything through. Frontend handlers live in `frontend/src/lib/agentLogFormat.ts` (badges) and `frontend/src/components/RefineryTab.tsx` (refinery-specific phases). Payloads must carry semantics only — no raw prompts or responses.
- **Episode shape change** — `DebateEpisode`/`EpisodeKeyEvent` live in `api/refinery/episode.py`; the swarm-side write goes through `dark_factory.memory.episodes.EpisodeWriter` via `write_debate_episode_to_memory`. Frontend mirror is the inline `episode` type on `RefinedRequirement.debate` in `frontend/src/api/client.ts`; render path is `EpisodesPanel` / `EpisodeCard` in `RefineryTab.tsx`.
- **Adjusting convergence-score bands** — formula lives in `episode.py:compute_convergence_score`; UI bands + labels in `RefineryTab.tsx` (`convergenceColor` / `convergenceLabel`). Both must move together — the formula's clamp to `0.99` means only a true `converged` outcome ever shows 1.0.
- **Adding an LLM provider or model with a new effort enum** — extend the per-provider translation map in `roles/_shared/llm.py` (e.g. `_ANTHROPIC_REASONING_MAP`, `_OPENAI_REASONING_MAP`); do not share maps across providers.

## Notes on REFINERY.md and README.md

`README.md` is the canonical Manufacture pipeline doc with the full phase-by-phase walkthrough, REST API table, and frontend tab matrix. `REFINERY.md` is the canonical refinery v2 doc — read it before making non-trivial changes to `api/refinery/`. The Apr-2026 version of REFINERY.md still references the `refinery_debate_enabled` feature flag; the soak completed and the legacy path was removed, but the document's surface description of the panel + research + judge + memory contracts is current.
