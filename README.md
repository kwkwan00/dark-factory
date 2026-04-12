# AI Dark Factory

**Author:** [Kevin Quon](https://www.linkedin.com/in/kwkwan00/)

An autonomous code generation platform that converts requirements into specs, populates a knowledge graph, and generates production-quality application code through a multi-agent swarm pipeline with cross-feature reconciliation, procedural memory, and real-time observability.

Built as a **FastAPI backend + React SPA** with the [AG-UI protocol](https://docs.ag-ui.com) for real-time agent event streaming.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          React SPA (Vite)                                    │
│  Manufacture · Agent Logs · Gap Finder · Memory · Metrics · Settings · About │
└──────────────────────┬───────────────────────────────────────────────┘
                       │  AG-UI SSE events  +  REST
┌──────────────────────▼───────────────────────────────────────────────┐
│                  FastAPI (dark_factory.api.app)                      │
│  /api/agent/run  /api/agent/events  /api/agent/cancel                │
│  /api/history  /api/metrics/*  /api/graph/gaps  /api/settings        │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────────────────────────────┐
     │                 │                                          │
     ▼                 ▼                                          ▼
┌────────────────┐    ┌────────────────────────────┐    ┌──────────────┐
│ Phase 1        │    │ Phase 2: Spec (decompose + │    │  Phase 5:    │
│ Ingest + doc-  │──▶ │  refine swarm + evaluate)  │─┐  │ Reconcile    │
│ extract +      │    └────────────────────────────┘ │  │ (Claude SDK) │
│ semantic dedup │                                    ▼  └──────┬───────┘
└────────────────┘                              ┌───────────┐   │
                                                 │  Phase 3  │   │
                                                 │   Graph   │   │
                                                 └─────┬─────┘   │
                                                       ▼         ▼
                            ┌─────────────────────────────────────────┐
                            │  Phase 4: Per-feature LangGraph swarms  │
                            │  Planner ↔ Coder ↔ Reviewer ↔ Tester    │
                            │  (parallel within dependency layers)    │
                            └─────────────────────────────────────────┘
                                                  │
                                                  ▼
                            ┌─────────────────────────────────────────┐
                            │  Phase 6: E2E Validation (Playwright)   │
                            │  chromium · firefox · webkit smoke      │
                            │  tests against the reconciled output    │
                            └─────────────────────────────────────────┘

  Neo4j (graph + memory)   Qdrant (vectors)   Postgres (metrics)
  Prometheus + Grafana     Claude Agent SDK   DeepEval (GPT judge)
```

---

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Node.js 20+** with npm (for the frontend)
- **Docker + Docker Compose** (recommended) or local Neo4j + Qdrant
- **Anthropic API key** — code generation, reconciliation agent, Phase 6 E2E agent
- **OpenAI API key** — evaluation judge + embeddings (also powers semantic requirement dedup)

---

## Quick Start (Docker Compose)

The fastest way to get a full stack running:

```bash
# 1. Configure environment
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, NEO4J_PASSWORD, ...

# 2. Bring up the stack
docker compose up -d

# 3. Open the app
open http://localhost:8000
```

The first `docker compose up` (or `make build && make up`) takes
roughly 5–10 minutes because the Dockerfile installs Python 3.12
(Debian bookworm), Node 20, uv, and `@playwright/test` with all three
browser engines (chromium, firefox, webkit) for the Phase 6 E2E
validation stage. Subsequent builds are fast — Docker caches the
Playwright layer until the pinned version changes.

This starts seven services:

| Service       | Port          | Purpose                                        |
|---------------|---------------|------------------------------------------------|
| `dark-factory`| 8000          | FastAPI + React SPA (single container)         |
| `neo4j`       | 7474, 7687    | Knowledge graph + procedural memory            |
| `qdrant`      | 6333          | Vector database for semantic search            |
| `postgres`    | 5432          | Forensic metrics store (optional)              |
| `adminer`     | 8080          | Web UI for browsing / troubleshooting Postgres |
| `prometheus`  | 9090          | In-process metrics scrape                      |
| `grafana`     | 3000          | Dashboards (default `admin` / `admin`)         |

Check `GET http://localhost:8000/api/health` to verify all dependencies are healthy.

To troubleshoot the Postgres metrics store, open [http://localhost:8080](http://localhost:8080) and log in with system `PostgreSQL`, server `postgres` (pre-filled), and the `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` values from your `.env`.

---

## Local Development

Run the backend and frontend as separate processes for hot-reload:

```bash
# Backend
uv sync
uv run uvicorn dark_factory.api.app:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev          # Vite dev server on :5173, proxies /api to :8000
```

For a production frontend build served from FastAPI:

```bash
cd frontend && npm run build
# Build output is copied into src/dark_factory/api/static/ and served at /
```

---

## Configuration

Settings are loaded from `config.toml` with environment variable overrides. Most fields are also mutable at runtime via `PATCH /api/settings` (Settings tab in the UI).

```toml
[neo4j]
uri = "bolt://localhost:7687"
database = "neo4j"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-6"

[pipeline]
output_dir = "./output"
max_parallel_specs = 4             # spec generation workers
max_parallel_features = 4          # codegen swarm workers per layer
max_spec_handoffs = 5              # generate → evaluate → refine iterations
max_codegen_handoffs = 50          # planner↔coder↔reviewer↔tester transitions
spec_eval_threshold = 0.8

[openspec]
root_dir = "./openspec"

[memory]
database = "neo4j"                 # "memory" for Neo4j Enterprise multi-db
enabled = true

[watch]
enabled = false
paths = ["./openspec/specs"]
debounce_seconds = 5
auto_run = true

[qdrant]
url = "http://localhost:6333"
collection_prefix = "dark_factory"
embedding_model = "text-embedding-3-large"
enabled = true

[evaluation]
base_threshold = 0.5
adaptive = true
decay_factor = 0.95

[logging]
level = "INFO"
format = "console"
```

All pipeline fields can be tuned live from the Settings tab without a restart. Additional pipeline settings (`max_codegen_handoffs`, `max_specs_per_requirement`, `enable_spec_decomposition`, `reuse_existing_specs`, `max_reconciliation_turns`, `reconciliation_timeout_seconds`, `requirement_dedup_threshold`, `enable_e2e_validation`, `max_e2e_turns`, `e2e_timeout_seconds`, `e2e_browsers`, `enable_episodic_memory`, `memory_dedup_threshold`) are set via environment variables or the Settings tab — see `.env.example` for the full list.

---

## Pipeline Phases

The pipeline runs in six phases, streamed to the frontend as AG-UI events.

### Phase 1 — Ingest

`IngestStage` parses a requirements directory/file into `Requirement` models. Two tiers of input are supported:

1. **Native formats** (`.md`, `.txt`, `.json`, `.yaml`, `.yml`, OpenSpec `specs/` tree) — parsed directly. Large text documents go through an LLM splitter that extracts each discrete testable requirement as its own entry. OpenSpec directories are auto-detected and WHEN/THEN scenarios extracted.
2. **Rich business documents** (`.docx`, `.xlsx`, `.pptx`, `.pdf`, `.rtf`, `.html`, `.htm`, `.xml`, `.csv`, `.vtt`, `.srt`, `.log`) — each file is routed through a **clean-context Claude Agent SDK invocation** that reads the document via the appropriate Python library (`python-docx`, `openpyxl`, `pypdf`, `beautifulsoup4`, `striprtf`, `lxml`), extracts discrete testable requirements, and writes a staging JSON file that's loaded back into `Requirement` models. Each document gets its own fresh agent context so raw meeting-transcript noise never pollutes the main pipeline. See `src/dark_factory/stages/doc_extraction.py`.

After all files are parsed, the stage runs a **semantic deduplication pass** (`src/dark_factory/stages/dedup.py`). A real corpus assembled from multiple uploaded documents (meeting notes + Word brief + spreadsheet) routinely contains the same underlying requirement expressed multiple ways. The dedup pass embeds each requirement with `text-embedding-3-large`, clusters near-duplicates at cosine similarity ≥ `requirement_dedup_threshold` (default 0.90), and collapses each cluster into a single canonical entry — preferring the highest-priority + most-detailed member and unioning tags across all merged requirements so no source-document attribution is lost. Dedup failures (transient OpenAI outages) fall back to the un-deduped list rather than blocking the pipeline.

### Phase 2 — Spec Generation

`SpecStage` converts each requirement into one or more `Spec` objects:

1. **Preflight skip** — if `reuse_existing_specs=true`, specs whose target id already exists in Neo4j are passed through unchanged (free re-runs).
2. **Decomposition** — an LLM planner optionally splits each requirement into multiple granular sub-specs.
3. **Refinement swarm** — each sub-spec runs through a generate → evaluate → refine loop (capped by `max_spec_handoffs`).
4. **Evaluation** — DeepEval GEval metrics (GPT judge) score every spec on Correctness, Coherence, Instruction Following, and Safety.
5. **Auto-index** — all passing specs are upserted into Qdrant for semantic retrieval by downstream agents.

### Phase 3 — Knowledge Graph

`GraphStage` persists specs + requirements to Neo4j with `IMPLEMENTS` and `DEPENDS_ON` relationships. The orchestrator later uses this graph to group specs into features and compute a cycle-tolerant topological execution order (via Tarjan's SCC).

### Phase 4 — Per-feature Swarms

`run_orchestrator()` groups specs by `capability`, layers them by dependency order, then dispatches features in **parallel within each layer** (bounded by `max_parallel_features`). Each feature runs an isolated LangGraph swarm with four agents rotating via `create_swarm` handoffs:

- **Planner** — evaluates the spec, queries eval history, recalls strategies, picks the next action
- **Coder** — searches for similar specs/code via Qdrant RAG, recalls patterns and past mistakes, generates code directly or delegates to the Claude Agent SDK
- **Reviewer** — runs DeepEval on the generated code, compares against historical scores, records mistakes/solutions
- **Tester** — writes tests, evaluates them, records failures, hands back to Planner

Cross-feature learning: after each feature completes, its patterns/mistakes/solutions are briefed to subsequent features in the same run. If the layer pass rate drops below threshold, the orchestrator forces subsequent coders onto the SDK path and tightens the handoff budget — when performance recovers, overrides relax.

### Phase 5 — Reconciliation

`ReconciliationStage` runs a **single extended Claude Agent SDK invocation** over the full run output directory with file I/O tools (`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`). It follows a six-step checklist:

1. **Inventory** the generated tree
2. **Review** for cross-feature issues (broken imports, inconsistent APIs, missing glue, security)
3. **Fix** with minimal targeted edits
4. **Validate** via language-appropriate commands (`py_compile`, `tsc --noEmit`, `pytest`, `npm test`, ...)
5. **Iterate** on validation failures (bounded by `max_reconciliation_turns` and `reconciliation_timeout_seconds`)
6. **Report** to `RECONCILIATION_REPORT.md` at the output root

This phase is **best-effort**: failures, timeouts, or crashes never fail the pipeline — the feature output is still delivered as-is. The report is surfaced in the Run Detail popup's Output screen.

### Phase 6 — End-to-End Validation

`E2EValidationStage` runs a second clean-context Claude Agent SDK invocation after reconciliation completes with a `clean` status. The gate is strict: `partial`, `error`, or `skipped` reconciliation statuses all skip E2E validation, and a skip-reason text event is emitted to the Agent Log so operators see exactly why. Its job is to verify the generated application actually runs in a real browser across a matrix of engines. The agent follows another six-step checklist:

1. **Detect** whether the output is a web app (package manifests, Dockerfile ports, HTML entry points, framework markers). If not, write `Overall status: skipped` and stop.
2. **Install** `@playwright/test`. Browser binaries for `chromium`, `firefox`, and `webkit` are **pre-installed in the Docker image** at `/ms-playwright`, so no browser download is needed at runtime.
3. **Plan** 3–8 user-facing acceptance criteria from the specs that are reachable through the UI.
4. **Write** `e2e/smoke.spec.ts` and a `playwright.config.ts` that enables every browser in `settings.pipeline.e2e_browsers` (default all three).
5. **Run** the server in the background with a shell trap for clean teardown, poll the health endpoint for readiness, then `npx playwright test --reporter=line,html` across the matrix.
6. **Report** to `E2E_REPORT.md` at the output root with a per-test / per-browser result table, failure reasons, and an `Overall status` of `pass`, `partial`, or `broken`. The Playwright HTML report lands at `e2e_artifacts/html-report/` and any failure screenshots under `e2e_artifacts/`.

Per-browser test counts are fanned out to `dark_factory_e2e_tests_total{browser, status}` so Grafana dashboards can answer "is WebKit the flaky one?". Phase 6 inherits the same best-effort policy as reconciliation — agent crashes, timeouts, server startup failures, and flaky tests are all logged, recorded as incidents, and then swallowed. A broken E2E pass never fails the run.

---

## Frontend Tabs

| Tab              | What it shows                                                      |
|------------------|--------------------------------------------------------------------|
| **Manufacture**  | Run launcher (path input or drag-and-drop upload), cancel button, always-visible run history (10 most recent with datetime started, status, pass rate, duration), per-run detail popup |
| **Agent Logs**   | Real-time ring buffer of ~2000 AG-UI progress events, color-coded badges by layer / feature / agent / decision / handoff / tool call / spec / eval, with pause/resume/clear, auto-scroll, and text filter |
| **Gap Finder**   | Neo4j-powered gap detection — unplanned requirements, stale specs, specs without artifacts, failing evaluations, with priority badges |
| **Agent Memory** | Browse procedural memory (Pattern / Mistake / Solution / Strategy) with search + filters |
| **Metrics**      | 17 dashboards: summary KPIs, eval trends, LLM cost breakdown, per-run stats, quality, throughput, incidents, tool calls, memory activity, decomposition, artifacts, background loop sampler, episodic memory, memory graph |
| **Settings**     | Live-mutable pipeline config — parallelism, handoff limits, reconciliation, spec decomposition, E2E validation, model selection + API key overrides, service health, file watcher control, danger-zone clear-all |
| **About**        | Architecture whitepaper with interactive React Flow diagrams (system topology, pipeline, swarm mechanics, memory, observability, cancellation), design philosophy, business value, data model reference, extensibility guide |

**Run Detail popup** (opens from run history) has five tabs:

| Tab              | What it shows                                                      |
|------------------|--------------------------------------------------------------------|
| **Metrics**      | Status, pass rate, duration, spec/feature counts, LLM cost, incidents, eval metrics (with spec ID, requirement, type, reason), tool calls, artifacts, decomposition |
| **Agent Log**    | Historical progress events for the run — same color-coded badge layout as the main Agent Logs tab, with text filter and expandable JSON payload detail per event |
| **Evaluations**  | Per-spec evaluation tree with requirements, metric scores, attempt history |
| **Output**       | File explorer for generated code/artifacts with syntax highlighting |
| **Episodes**     | Episodic memory timeline — feature narratives, outcomes, key turning-point events, eval scores |

---

## REST API

### Agent pipeline

| Method | Path                  | Purpose                                           |
|--------|-----------------------|---------------------------------------------------|
| POST   | `/api/agent/run`      | Start a pipeline run (requirements path + optional key overrides) |
| POST   | `/api/agent/cancel`   | Cooperative cancel — sets the kill-switch event   |
| GET    | `/api/agent/events`   | SSE stream of AG-UI events for an active run     |

### Dashboard

| Method | Path                     | Purpose                                           |
|--------|--------------------------|---------------------------------------------------|
| GET    | `/api/health`            | Service liveness (Neo4j + Qdrant + Postgres)     |
| GET    | `/api/history`           | Paginated run history                             |
| GET    | `/api/memory/list`       | Browse procedural memories                        |
| GET    | `/api/memory/search`     | Hybrid RRF search over Neo4j + Qdrant             |
| GET    | `/api/eval`              | All spec evaluations                              |
| GET    | `/api/eval/{spec_id}`    | Eval history for a single spec                    |
| GET    | `/api/graph/gaps`        | Gap finder output                                 |
| GET    | `/api/settings`          | Current pipeline settings                         |
| PATCH  | `/api/settings`          | Update pipeline settings at runtime               |
| POST   | `/api/watch/start`       | Start file watcher                                |
| POST   | `/api/watch/stop`        | Stop file watcher                                 |
| GET    | `/api/watch/status`      | Current watcher status                            |
| GET    | `/api/watch/events`      | SSE stream of file system events                  |
| POST   | `/api/upload`            | Drag-and-drop file upload — native + rich formats, 25 MB/file, 150 MB/upload, 24h TTL |

### Metrics

17 endpoints under `/api/metrics/`:

| Method | Path                               | Purpose                                    |
|--------|------------------------------------|--------------------------------------------|
| GET    | `/api/metrics/summary`             | Overview KPIs (runs, LLM calls, evals, incidents) |
| GET    | `/api/metrics/runs`                | Recent runs (paginated)                    |
| GET    | `/api/metrics/runs/{run_id}`       | Full metrics detail for a specific run (includes eval metrics, progress log, tool calls, artifacts, decomposition, incidents) |
| GET    | `/api/metrics/eval_trend`          | Evaluation metric trends over time         |
| GET    | `/api/metrics/llm_usage`           | LLM usage grouped by model/phase/client    |
| GET    | `/api/metrics/swarm_features`      | Feature swarm statistics                   |
| GET    | `/api/metrics/cost_rollup`         | Cost aggregation                           |
| GET    | `/api/metrics/throughput`          | Throughput over N days                     |
| GET    | `/api/metrics/quality`             | Quality metrics                            |
| GET    | `/api/metrics/incidents`           | Incident log (filterable by category)      |
| GET    | `/api/metrics/agent_stats`         | Per-agent statistics for a run             |
| GET    | `/api/metrics/tool_calls`          | Tool invocation stats (by tool/agent/feature) |
| GET    | `/api/metrics/memory_activity`     | Memory node creation/update activity       |
| GET    | `/api/metrics/decomposition`       | Spec decomposition metrics                 |
| GET    | `/api/metrics/artifacts`           | Generated artifact summary by language     |
| GET    | `/api/metrics/memory`              | Procedural memory graph observability      |
| GET    | `/api/metrics/episodes/{run_id}`   | Episodic memory timeline for a run         |
| GET    | `/api/metrics/background_loop`     | Background loop health metrics             |

### Models

| Method | Path                   | Purpose                                            |
|--------|------------------------|----------------------------------------------------|
| POST   | `/api/models/anthropic`| Configure Anthropic LLM provider                   |
| POST   | `/api/models/openai`   | Configure OpenAI LLM provider                      |

### Runs

| Method | Path                          | Purpose                                    |
|--------|-------------------------------|--------------------------------------------|
| GET    | `/api/runs/{run_id}/files`    | File tree of the run's output directory   |
| GET    | `/api/runs/{run_id}/file`     | Fetch a single file by path query          |

### Admin

| Method | Path                   | Purpose                                            |
|--------|------------------------|----------------------------------------------------|
| POST   | `/api/admin/clear-all` | Wipe Neo4j + Qdrant + Postgres + output dir (requires `?confirm=yes` and no active run) |

---

## Environment Variables

Required:

- `ANTHROPIC_API_KEY` — code generation + Phase 5 reconciliation
- `OPENAI_API_KEY` — DeepEval judge + embeddings
- `NEO4J_PASSWORD` — graph + memory auth

Optional:

- `QDRANT_URL`, `QDRANT_API_KEY` — vector database (fallback to Neo4j if unreachable)
- `POSTGRES_ENABLED`, `POSTGRES_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — forensic metrics store
- `PROMETHEUS_ENABLED` — in-process metrics (default on)
- `GRAFANA_USER`, `GRAFANA_PASSWORD` — dashboard auth (change from admin/admin in production)
- `DEEP_AGENT_TIMEOUT_SECONDS` — default ceiling for Claude Agent SDK calls (default 600s)
- `DEEP_AGENT_DEBUG_STDERR` — enable `--debug-to-stderr` on the Claude Agent SDK Node CLI subprocess for verbose diagnostics when investigating silent crashes (default off)
- `EVAL_MODEL` — override the DeepEval judge model (default `gpt-5.4`)
- `MAX_PARALLEL_FEATURES`, `MAX_PARALLEL_SPECS`, `MAX_SPEC_HANDOFFS`, `MAX_CODEGEN_HANDOFFS`, `SPEC_EVAL_THRESHOLD`, `ENABLE_SPEC_DECOMPOSITION`, `MAX_SPECS_PER_REQUIREMENT`, `MAX_RECONCILIATION_TURNS`, `RECONCILIATION_TIMEOUT_SECONDS`, `REQUIREMENT_DEDUP_THRESHOLD`, `ENABLE_E2E_VALIDATION`, `MAX_E2E_TURNS`, `E2E_TIMEOUT_SECONDS`, `E2E_BROWSERS`, `ENABLE_EPISODIC_MEMORY`, `MEMORY_DEDUP_THRESHOLD` — pipeline tuning overrides

See `.env.example` for the full list with descriptions.

---

## Key Features

### Rich-document ingestion

Drop any mix of **Markdown, JSON, YAML, OpenSpec, Word, Excel, PowerPoint, PDF, HTML, XML, RTF, CSV, or transcript files** (`.vtt` / `.srt` / `.log`) into the upload dropzone. Native formats parse directly; business documents are routed through a **clean-context Claude Agent SDK invocation** per file that reads the document via the appropriate Python library (`python-docx`, `openpyxl`, `pypdf`, `beautifulsoup4`, `striprtf`, `lxml`) and extracts discrete testable requirements. Each document gets its own fresh agent context so raw meeting-transcript noise never pollutes the main swarm. Per-file size limit is 25 MB; per-upload total is 150 MB; uploads auto-expire after 24h.

### Semantic requirement dedup

After all input files are parsed, `IngestStage` runs a **semantic deduplication pass** before the Spec stage sees anything. Each requirement is embedded with `text-embedding-3-large`; near-duplicates (cosine similarity ≥ `requirement_dedup_threshold`, default 0.90) are clustered and collapsed into a single canonical entry. Canonical selection prefers the highest-priority member, breaking ties by description length then original position. Tags from every merged requirement are unioned onto the canonical so source-document attribution isn't lost. Merges are surfaced in the Agent Logs tab (`• kept X (merged: Y, Z)`) and recorded as a `requirements_deduped` progress event. Embedding-service outages fall back to the un-deduped list — dedup is a correctness guarantee but never a hard blocker.

### Cross-browser E2E validation (Phase 6)

After reconciliation finishes with a `clean` status, `E2EValidationStage` spawns a dedicated Claude Agent SDK session that runs **Playwright smoke tests across `chromium`, `firefox`, and `webkit`** by default. The agent detects whether the run produced a web app, generates minimal tests from the specs' acceptance criteria, starts the server in the background, runs the suite across the full browser matrix, and writes `E2E_REPORT.md` plus a browsable Playwright HTML report + failure screenshots under `e2e_artifacts/`. Browser binaries are **bundled in the Docker image** (via `npx playwright install --with-deps chromium firefox webkit`) so no runtime download is needed. Per-browser test counts feed the `dark_factory_e2e_tests_total{browser, status}` Prometheus counter for precise flake attribution. Like reconciliation, E2E is best-effort — crashes are recorded as incidents and swallowed.

### Procedural memory (Neo4j + Qdrant)

Agents learn from past runs via a dedicated memory database with both **semantic** and **episodic** tiers:

| Type         | Tier      | Written by         | Used by     | Encodes                                |
|--------------|-----------|--------------------|-------------|----------------------------------------|
| **Pattern**  | semantic  | Coder              | Coder       | "use this code structure"              |
| **Mistake**  | semantic  | Reviewer, Tester   | All agents  | "this failure mode + root cause"       |
| **Solution** | semantic  | Reviewer, Tester   | All agents  | "this fix resolved the mistake"        |
| **Strategy** | semantic  | Planner            | Planner     | "this planning approach worked"        |
| **Episode**  | episodic  | Orchestrator (auto) | Planner    | "what happened last time for feature X" |

**Semantic memory** (Pattern / Mistake / Solution / Strategy) answers *"what should I do?"* with generalised lessons. Feedback loop: eval pass → boost recalled memories; eval fail → demote. Relevance decays 5% each run. Cross-feature learnings are briefed to subsequent features **within the same run**.

**Episodic memory** answers *"what happened last time I was in this exact situation?"* After every feature swarm completes, the orchestrator spins up a small LLM call to synthesise a 200-word narrative summary plus 3–8 key turning-point events, embeds the result with `text-embedding-3-large`, and writes it to both Neo4j (`Episode` node with `PRODUCED_IN` edge to its Run) and Qdrant (`dark_factory_episodes` collection). Planners call `recall_episodes` at the start of every feature to retrieve ranked past trajectories — if an earlier run succeeded with a specific approach, the current run biases toward it; if it failed a particular way, the current run avoids the same mode. Hybrid RRF merge between Neo4j keyword match and Qdrant vector match mirrors the existing `recall_memories` architecture. Episodes are surfaced in the Run Detail popup's Episodes tab alongside Metrics, Agent Log, Evaluations, and Output.

Episodic memory costs ~1k LLM tokens per feature for the summarisation pass — toggle off via `enable_episodic_memory` if you're running single-shot features that never recur.

#### Memory hygiene (Tier A)

Three improvements keep the memory graph clean and the recall path sharp:

1. **Write-time dedup** — before creating a new Pattern / Mistake / Solution / Strategy, the repository embeds the candidate text and cosine-matches against existing same-type same-feature memories. Matches above `memory_dedup_threshold` (default 0.92) get their `relevance_score` boosted and their `times_applied` counter bumped instead of being duplicated. A Coder that learns "use parameterised queries for SQL" across five features ends up with one high-relevance Pattern instead of five near-identical low-relevance ones. Set to `0.0` to disable.
2. **Relevance-weighted RRF recall** — the hybrid Neo4j + Qdrant merge now multiplies each rank contribution by the memory's relevance_score. Memories boosted by successful eval feedback outrank memories at the same semantic similarity that have been demoted by failures. The floor (0.1) keeps fully-demoted memories visible for cleanup but pushes them to the back.
3. **Memory observability dashboard** — the Metrics tab now has a "Memory graph" section showing per-type node counts, relevance distribution histograms, the 10 most-recalled memories (with their `times_recalled` counter), and 7-day boost/demote effectiveness. Powered by `GET /api/metrics/memory` and backed by new `MemoryRepository.get_memory_stats()`, `get_top_recalled_memories()`, and `get_recall_effectiveness()` methods. Can't improve what you can't measure.

### Observability

- **Prometheus** counters/histograms for every phase, tool call, LLM invocation, reconciliation status, incident, and the BackgroundLoop sampler — always-on and zero-cost.
- **Postgres** (optional) forensic rows for LLM calls, eval results, tool calls, incidents, and progress events — high-cardinality debugging.
- **Grafana** ships with provisioned datasources and dashboards for pipeline throughput, cost rollups, and error budgets.
- **Incident table** surfaces errors, warnings, and reconciliation issues in the Run Detail popup with stack traces.

### Deep agent resilience

The Claude Agent SDK subprocess can crash silently (OOM, Node segfault, network timeout) with no useful stderr. Three layers of defense:

1. **`_safe_tool_deep_agent`** — a wrapper used by all 9 `@tool`-decorated deep-agent functions (codegen, dependency analysis, risk assessment, security review, performance review, spec compliance review, unit/integration/edge-case test gen). Catches `Exception` and returns a structured error string instead of crashing the feature swarm. Non-tool callers (reconciliation, doc extraction, E2E validation) still see the exception for their own error handling.
2. **Stderr capture** — every SDK invocation buffers up to 200 lines / 16 KiB of subprocess stderr via a callback. When a crash occurs, the tail is logged alongside the incident for diagnostics.
3. **`DEEP_AGENT_DEBUG_STDERR`** — when set to `1`/`true`, the underlying Node CLI is spawned with `--debug-to-stderr` so it emits startup, transport, and protocol-state lines. Operators flip this on while investigating silent exits to see what the subprocess was doing before it died.

A source-level regression test (`test_all_tool_decorated_deep_agents_use_safe_wrapper`) verifies that every `@tool` function using `_run_deep_agent` actually calls `_safe_tool_deep_agent`, preventing future tools from bypassing the safety wrapper.

### Cooperative cancellation

A module-level `threading.Event` is polled at hot-path checkpoints across all six phases. `POST /api/agent/cancel` sets the flag; the pipeline raises `PipelineCancelled` at the next checkpoint, runs the `finally` cleanup, and emits a clean `cancelled` status (not a generic error). The flag auto-resets at the start of every run to prevent bleed.

### File watcher

Optional: monitors `./openspec/specs` (configurable) and emits SSE events to the Monitor UI. With `auto_run=true` it will kick off a pipeline run on debounced file changes.

### Adaptive evaluation

DeepEval thresholds adjust automatically based on score trends across runs. Strategy overrides kick in mid-layer when performance drops below a floor and relax when it recovers.

### AG-UI protocol

All pipeline progress is streamed using [AG-UI](https://docs.ag-ui.com): `RunStartedEvent`, `StepStartedEvent`/`StepFinishedEvent` per phase (including nested per-feature steps in Phase 4), `TextMessageContentEvent` for progress text, `StateSnapshotEvent` after each layer, and `RunFinishedEvent` with the final payload.

---

## Project Structure

```
src/dark_factory/
├── api/                   # FastAPI app + 7 route modules + AG-UI bridge
│   ├── app.py             # FastAPI application + static SPA mount
│   ├── ag_ui_bridge.py    # Pipeline → AG-UI event stream adapter
│   ├── routes_agent.py    # /api/agent/{run,cancel,events}
│   ├── routes_dashboard.py
│   ├── routes_metrics.py
│   ├── routes_runs.py
│   ├── routes_admin.py
│   ├── routes_models.py
│   └── routes_upload.py
├── stages/                # Pipeline phases
│   ├── ingest.py          # Phase 1 — native + rich parsing, dedup dispatch
│   ├── doc_extraction.py  # Rich-doc deep-agent extractor (.docx/.xlsx/.pdf/...)
│   ├── dedup.py           # Semantic requirement dedup (cosine similarity)
│   ├── spec.py            # Phase 2
│   ├── graph.py           # Phase 3
│   ├── reconciliation.py  # Phase 5
│   └── e2e_validation.py  # Phase 6 — Playwright cross-browser smoke tests
├── agents/
│   ├── orchestrator.py    # Parent: layer dispatch + strategy adjustment
│   ├── swarm.py           # Per-feature LangGraph swarm
│   ├── tools.py           # LangChain tools (file, graph, memory, RAG, SDK)
│   ├── background_loop.py # Singleton daemon asyncio loop for SDK calls
│   └── progress.py        # ProgressBroker for AG-UI event fan-out
├── graph/                 # Neo4j client, schema, repository
├── memory/                # Procedural memory schema + repository
│   ├── schema.py          # Neo4j constraints + indexes (Pattern / Mistake / Solution / Strategy / Episode)
│   ├── repository.py      # MemoryRepository: CRUD + recall + eval feedback + stats
│   ├── dedup_writer.py    # Tier A: write-time dedup via cosine similarity
│   └── episodes.py        # Episodic memory: synthesizer + EpisodeWriter
├── vector/                # Qdrant client, embeddings, hybrid RRF merge
├── metrics/               # Prometheus + Postgres recorder + helpers
├── evaluation/            # DeepEval GEval metrics + adaptive thresholds
├── llm/                   # Anthropic / OpenAI / LangChain clients
├── models/domain.py       # Pydantic domain models
├── openspec/              # OpenSpec parser + Jinja2 writer
├── config.py              # Settings (TOML + env overrides)
└── log.py                 # Structlog setup

frontend/
├── src/
│   ├── App.tsx              # Tab routing + layout (7 tabs)
│   ├── api/client.ts        # AG-UI HttpAgent + REST client
│   ├── components/
│   │   ├── ManufactureTab.tsx    # Run launcher + history + detail popup
│   │   ├── AgentLogsTab.tsx      # Real-time SSE event stream
│   │   ├── GapFinderTab.tsx      # Neo4j gap detection
│   │   ├── AgentMemoryTab.tsx    # Procedural memory browser
│   │   ├── MetricsTab.tsx        # 17 metrics dashboards
│   │   ├── SettingsTab.tsx       # Live config + health + admin
│   │   ├── AboutTab.tsx          # Architecture whitepaper
│   │   └── RunDetailWindow.tsx   # Per-run popup (5 tabs)
│   ├── contexts/
│   │   └── ManufactureContext.tsx # State preserved across tab switches
│   ├── hooks/               # useAgentRun, useDashboard, ...
│   ├── lib/
│   │   ├── agentLogFormat.ts     # Shared event badge/formatting (live + historical)
│   │   └── openRunDetail.ts      # Run detail window opener
│   └── main.tsx
└── vite.config.ts

tests/                        # 635 tests across 38 files
```

---

## Development

```bash
# Run the full test suite (635 tests)
uv run pytest tests/ -v

# Fast subset (skip integration + slow tests)
uv run pytest tests/ -m "not integration and not slow"

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# TypeScript check
cd frontend && npx tsc --noEmit
```

Pytest markers:

- `@pytest.mark.slow` — > 1s
- `@pytest.mark.integration` — requires external services (Neo4j, Qdrant, Postgres)

---

## Agentic Behavior Level

This system operates at **L4 (Fully Autonomous / Explorer)** on the [Vellum agentic behavior scale](https://www.vellum.ai/blog/levels-of-agentic-behavior):

| L4 Trait                             | Implementation                                                         |
|--------------------------------------|------------------------------------------------------------------------|
| Persist state across sessions        | Neo4j procedural memory + Qdrant embeddings + eval/run history         |
| Refine execution based on feedback   | Eval → memory feedback loop, adaptive thresholds, cross-feature learning, mid-run strategy adjustment |
| Parallel execution                   | Concurrent feature swarms within dependency layers                     |
| Real-time adaptation                 | Strategy overrides triggered by layer pass rate; cross-feature briefing within same run |
| Cross-feature reconciliation         | Phase 5 extended Claude Agent SDK pass over the full run output        |

---

## Author

**[Kevin Quon](https://www.linkedin.com/in/kwkwan00/)**

---

## License

Proprietary. © Kevin Quon.
