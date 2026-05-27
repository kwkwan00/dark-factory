# AI Dark Factory

**Author:** [Kevin Quon](https://www.linkedin.com/in/kwkwan00/)

An autonomous code generation platform that converts requirements into specs, populates a knowledge graph, and generates production-quality application code through a multi-agent swarm pipeline with cross-feature reconciliation, procedural memory, and real-time observability.

Built as a **FastAPI backend + React SPA** with the [AG-UI protocol](https://docs.ag-ui.com) for real-time agent event streaming.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          React SPA (Vite)                                    │
│  Manufacture · Agent Logs · Gap Finder · Memory · Metrics · Settings · About │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │  AG-UI SSE events  +  REST
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  FastAPI (dark_factory.api.app)                      │
│  /api/agent/run  /api/agent/events  /api/agent/cancel                │
│  /api/history  /api/metrics/*  /api/graph/gaps  /api/settings        │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
     ┌─────────────────────────────┼───────────────────────────┐
     │                             │                           │
     ▼                             ▼                           ▼
┌────────────────┐    ┌────────────────────────────┐    ┌──────────────┐
│ Phase 1        │    │ Phase 2: Spec (decompose + │    │  Phase 5:    │
│ Ingest + doc-  │──▶ │  refine swarm + evaluate)  │    │ Reconcile    │
│ extract +      │    └────────────┬───────────────┘    │ (Claude SDK) │
│ semantic dedup │                 ▼                    └──────┬───────┘
└────────────────┘    ┌────────────────────────────┐           │
                      │ Phase 2b: Spec Reconcile   │           │
                      │ (validate deps, coverage,  │           │
                      │  break cycles, fix links)  │           │
                      └────────────┬───────────────┘           │
                                   ▼                           │
                             ┌───────────┐                     │
                             │  Phase 3  │                     │
                             │   Graph   │                     │
                             └─────┬─────┘                     │
                                   ▼                           ▼
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
  Local + S3 replicated storage
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
max_parallel_features = 4
max_parallel_specs = 4
max_spec_handoffs = 5
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

The pipeline runs in seven phases, streamed to the frontend as AG-UI events. The Manufacture tab shows phase-level steps with feature sub-steps (showing agent handoffs and key decisions, with "Show all" to expand verbose messages). Every phase transition and completion emits a progress event to the Agent Logs broker for real-time observability.

### Phase 1 — Ingest

`IngestStage` parses a requirements directory/file into `Requirement` models. Two tiers of input are supported:

1. **Native formats** (`.md`, `.txt`, `.json`, `.yaml`, `.yml`, OpenSpec `specs/` tree) — parsed directly. Large text documents go through an LLM splitter that extracts each discrete testable requirement as its own entry. OpenSpec directories are auto-detected and WHEN/THEN scenarios extracted.
2. **Rich business documents** (`.docx`, `.xlsx`, `.pptx`, `.pdf`, `.rtf`, `.html`, `.htm`, `.xml`, `.csv`, `.vtt`, `.srt`, `.log`) — each file is routed through a **clean-context Claude Agent SDK invocation** that reads the document via the appropriate Python library (`python-docx`, `openpyxl`, `pypdf`, `beautifulsoup4`, `striprtf`, `lxml`), extracts discrete testable requirements, and writes a staging JSON file that's loaded back into `Requirement` models. Each document gets its own fresh agent context so raw meeting-transcript noise never pollutes the main pipeline. See `src/dark_factory/stages/doc_extraction.py`.

After all files are parsed, the stage runs a **semantic deduplication pass** (`src/dark_factory/stages/dedup.py`). A real corpus assembled from multiple uploaded documents (meeting notes + Word brief + spreadsheet) routinely contains the same underlying requirement expressed multiple ways. The dedup pass embeds each requirement with `text-embedding-3-large`, clusters near-duplicates at cosine similarity ≥ `requirement_dedup_threshold` (default 0.90), and collapses each cluster into a single canonical entry — preferring the highest-priority + most-detailed member and unioning tags across all merged requirements so no source-document attribution is lost. Dedup failures (transient OpenAI outages) fall back to the un-deduped list rather than blocking the pipeline.

### Phase 2 — Spec Generation

`SpecStage` converts each requirement into one or more `Spec` objects:

1. **Early-exit preflight** — queries Neo4j via `IMPLEMENTS` edges to check if every requirement already has specs. If so, the entire planner + refinement loop is skipped — re-runs on unchanged inputs complete in a single Neo4j query with zero LLM spend.
2. **Preflight skip** — if `reuse_existing_specs=true` and only some specs exist, those are loaded from Neo4j and passed through unchanged. Only missing specs enter the refinement loop.
3. **Decomposition** — an LLM planner optionally splits each requirement into multiple granular sub-specs.
4. **Refinement swarm** — each sub-spec runs through a generate → evaluate → refine loop (capped by `max_spec_handoffs`).
5. **Evaluation** — DeepEval GEval metrics (GPT judge) score every spec on Correctness, Coherence, Instruction Following, and Safety.
6. **Auto-index** — all passing specs are upserted into Qdrant with enriched metadata (eval scores, attempts, scenarios, dependencies) for semantic retrieval by downstream agents.

### Phase 2b — Spec Reconciliation

`SpecReconciliationStage` runs between spec generation and the knowledge graph write to ensure specs form a coherent, correctly-linked dependency graph. Two passes:

**Deterministic (always runs):**
- Strip phantom `requirement_ids` (refs to nonexistent requirements)
- Strip phantom `dependencies` (refs to nonexistent specs)
- Detect and break circular dependencies (iterative DFS, deterministic back-edge removal)
- Flag uncovered requirements (requirements with no implementing spec)
- Detect capability islands (specs sharing a capability but with no dependency path between them)

**LLM-assisted (best-effort):**
- Analyze spec descriptions, acceptance criteria, and WHEN/THEN scenarios to surface **implicit dependencies** the planner missed
- Fix **requirement coverage gaps** — assign uncovered requirements to specs that clearly implement them
- Correct **capability grouping** — merge or split capability assignments when the current grouping doesn't match the actual dependency structure
- Priority-weighted analysis — high/critical uncovered requirements are flagged more urgently

Controlled by `ENABLE_SPEC_RECONCILIATION` (default `true`). The LLM pass is sandboxed: cycles and phantoms are re-checked after LLM patches are applied, so the LLM cannot introduce graph corruption.

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

| Tab              | What it shows                                                                                     |
|------------------|---------------------------------------------------------------------------------------------------|
| **Manufacture**  | "New Run" modal (drag-and-drop upload, path input, run/cancel), live-polling run history (status, |
|                  | pass rate, duration), per-run actions menu with delete, per-run detail popup                      |
| **Agent Logs**   | Real-time AG-UI progress event stream, color-coded badges by layer / feature / agent / decision / |
|                  | handoff / tool call / spec / eval / deep agent turn, with pause/resume/clear, auto-scroll, and    |
|                  | text filter                                                                                       |
| **Gap Finder**   | Neo4j-powered gap detection — unplanned requirements, stale specs, specs without artifacts,       |
|                  | failing evaluations (all failing metrics per spec), broken dependencies, disconnected capability  |
|                  | islands, missing episodes, with priority badges                                                   |
| **Agent Memory** | Browse procedural memory (Pattern / Mistake / Solution / Strategy) with type filters, sort (most  |
|                  | recalled / relevance / newest / oldest), Mistake→Solution pairing, recall counts, "···" delete    |
|                  | menu, expandable detail rows with run_id and timestamps                                           |
| **Metrics**      | Summary KPIs, eval trends, LLM cost breakdown (by phase), per-run stats, quality, throughput,     |
|                  | incidents, tool calls, memory activity, decomposition, artifacts, background loop sampler,        |
|                  | episodic memory, memory graph                                                                     |
| **Settings**     | Live-mutable pipeline config — parallelism, handoff limits, reconciliation, spec decomposition,   |
|                  | E2E validation, model selection + API key overrides, service health, file watcher control,        |
|                  | danger-zone clear-all                                                                             |
| **About**        | Architecture whitepaper with interactive React Flow diagrams (system topology, pipeline, swarm    |
|                  | mechanics, memory, observability, cancellation), design philosophy, business value, data model    |
|                  | reference, extensibility guide                                                                    |

**Run Detail popup** (opens from run history):

| Tab              | What it shows                                                                                     |
|------------------|---------------------------------------------------------------------------------------------------|
| **Agent Log**    | Historical progress events for the run — same color-coded badge layout as the main Agent Logs     |
|                  | tab, with text filter and expandable JSON payload detail per event                                |
| **Metrics**      | Status, pass rate, duration, spec/feature counts, LLM cost, incidents, eval metrics (with spec    |
|                  | ID, requirement, type, reason), tool calls, artifacts, decomposition                              |
| **Evaluations**  | Per-spec evaluation tree with requirements, metric scores, attempt history                        |
| **Episodes**     | Episodic memory timeline — feature narratives, outcomes, key turning-point events, eval scores,   |
|                  | recalled memory IDs                                                                               |
| **Output**       | File explorer for generated code/artifacts with syntax highlighting and **Download ZIP** button   |
| **Compare**      | Side-by-side comparison against another run — select from dropdown, see pass rate / duration /    |
|                  | feature status deltas, per-feature status transitions (error→success), "View File Diffs" for      |
|                  | unified diff viewer with line-level coloring                                                      |
| **Traceability** | Requirements → specs → files → tests → eval scores matrix with Table/Graph toggle. Table view:    |
|                  | expandable rows with bulleted evaluations, files (linked to S3 presigned URLs), tests, status     |
|                  | info tooltips. Graph view: interactive dependency graph (React Flow) with status-colored nodes    |

---

## REST API

### Agent pipeline

| Method | Path                | Purpose                                                                               |
|--------|---------------------|---------------------------------------------------------------------------------------|
| POST   | `/api/agent/run`    | Start a pipeline run (requirements path + optional key overrides)                     |
| POST   | `/api/agent/cancel` | Cooperative cancel — sets the kill-switch event                                       |
| GET    | `/api/agent/events` | SSE stream of AG-UI events for an active run                                          |

### Dashboard

| Method | Path                  | Purpose                                                                             |
|--------|-----------------------|-------------------------------------------------------------------------------------|
| GET    | `/api/health`         | Service liveness (Neo4j + Qdrant + Postgres)                                        |
| GET    | `/api/history`        | Paginated run history                                                               |
| GET    | `/api/memory/list`    | Browse procedural memories                                                          |
| GET    | `/api/memory/search`  | Hybrid Reciprocal Rank Fusion search over Neo4j + Qdrant                            |
| GET    | `/api/eval`           | All spec evaluations                                                                |
| GET    | `/api/eval/{spec_id}` | Eval history for a single spec                                                      |
| GET    | `/api/graph/gaps`     | Gap finder output                                                                   |
| GET    | `/api/settings`       | Current pipeline settings                                                           |
| PATCH  | `/api/settings`       | Update pipeline settings at runtime                                                 |
| POST   | `/api/watch/start`    | Start file watcher                                                                  |
| POST   | `/api/watch/stop`     | Stop file watcher                                                                   |
| GET    | `/api/watch/status`   | Current watcher status                                                              |
| GET    | `/api/watch/events`   | SSE stream of file system events                                                    |
| POST   | `/api/upload`         | Drag-and-drop file upload — native + rich formats, 25 MB/file, 150 MB/upload, 24h   |
|        |                       | TTL                                                                                 |

### Metrics

Endpoints under `/api/metrics/`:

| Method | Path                             | Purpose                                                                  |
|--------|----------------------------------|--------------------------------------------------------------------------|
| GET    | `/api/metrics/summary`           | Overview KPIs (runs, LLM calls, evals, incidents)                        |
| GET    | `/api/metrics/runs`              | Recent runs (paginated)                                                  |
| GET    | `/api/metrics/runs/{run_id}`     | Full metrics detail for a specific run (includes eval metrics, progress  |
|        |                                  | log, tool calls, artifacts, decomposition, incidents)                    |
| GET    | `/api/metrics/eval_trend`        | Evaluation metric trends over time                                       |
| GET    | `/api/metrics/llm_usage`         | LLM usage grouped by model/phase/client                                  |
| GET    | `/api/metrics/swarm_features`    | Feature swarm statistics                                                 |
| GET    | `/api/metrics/cost_rollup`       | Cost aggregation                                                         |
| GET    | `/api/metrics/throughput`        | Throughput over N days                                                   |
| GET    | `/api/metrics/quality`           | Quality metrics                                                          |
| GET    | `/api/metrics/incidents`         | Incident log (filterable by category)                                    |
| GET    | `/api/metrics/agent_stats`       | Per-agent statistics for a run                                           |
| GET    | `/api/metrics/tool_calls`        | Tool invocation stats (by tool/agent/feature)                            |
| GET    | `/api/metrics/memory_activity`   | Memory node creation/update activity                                     |
| GET    | `/api/metrics/decomposition`     | Spec decomposition metrics                                               |
| GET    | `/api/metrics/artifacts`         | Generated artifact summary by language                                   |
| GET    | `/api/metrics/memory`            | Procedural memory graph observability                                    |
| GET    | `/api/metrics/episodes/{run_id}` | Episodic memory timeline for a run                                       |
| GET    | `/api/metrics/background_loop`   | Background loop health metrics                                           |

### Models

| Method | Path                    | Purpose                                                                           |
|--------|-------------------------|-----------------------------------------------------------------------------------|
| POST   | `/api/models/anthropic` | Configure Anthropic LLM provider                                                  |
| POST   | `/api/models/openai`    | Configure OpenAI LLM provider                                                     |

### Runs

| Method | Path                                | Purpose                                                               |
|--------|-------------------------------------|-----------------------------------------------------------------------|
| GET    | `/api/runs/{run_id}/files`          | File tree of the run's output directory (falls back to S3)            |
| GET    | `/api/runs/{run_id}/file`           | Fetch a single file by path query (falls back to S3)                  |
| GET    | `/api/runs/{run_id}/download`       | Stream the run's output as a zip archive                              |
| GET    | `/api/runs/diff?run_a=X&run_b=Y`    | Unified file diffs between two runs (uses pre-computed MD5 manifests) |
| GET    | `/api/runs/compare?run_a=X&run_b=Y` | Side-by-side metrics comparison of two runs                           |
| DELETE | `/api/history/{run_id}`             | Delete a run and all linked data (Neo4j, Qdrant, Postgres, storage)   |
| DELETE | `/api/memory/{memory_id}`           | Delete a single memory node and its vector                            |

### Traceability & Graph

| Method | Path                           | Purpose                                                                    |
|--------|--------------------------------|----------------------------------------------------------------------------|
| GET    | `/api/traceability/{run_id}`   | Requirements → specs → files → tests → evals matrix with S3 presigned URLs |
| GET    | `/api/graph/topology/{run_id}` | Run-scoped dependency graph (nodes with status, edges with types)          |

### Admin

| Method | Path                   | Purpose                                                                            |
|--------|------------------------|------------------------------------------------------------------------------------|
| POST   | `/api/admin/clear-all` | Wipe Neo4j + Qdrant + Postgres + output dir (requires `?confirm=yes` and no active |
|        |                        | run)                                                                               |

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
- `STORAGE_BACKEND` — `local` (default) or `s3`. When set to `local` and `S3_BUCKET` is also set, a `ReplicatedStorage` backend is auto-created that writes to both local disk (fast) and S3 (durable) — S3 failures are best-effort and never block the pipeline. Read operations fall back to S3 when local is empty (handles container restarts). MD5 hashes are computed at write time and stored in `.md5-manifest.json` for efficient cross-run diffing
- `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL` — S3 bucket config (required when `STORAGE_BACKEND=s3`, optional for replication when `STORAGE_BACKEND=local`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — AWS credentials (or use IAM role)
- `ENABLE_SPEC_RECONCILIATION` — enable/disable Phase 2b spec reconciliation (default `true`)
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

| Type         | Tier      | Written by         | Used by     | Encodes                                 |
|--------------|-----------|--------------------|-------------|-----------------------------------------|
| **Pattern**  | semantic  | Coder              | Coder       | "use this code structure"               |
| **Mistake**  | semantic  | Reviewer, Tester   | All agents  | "this failure mode + root cause"        |
| **Solution** | semantic  | Reviewer, Tester   | All agents  | "this fix resolved the mistake"         |
| **Strategy** | semantic  | Planner            | Planner     | "this planning approach worked"         |
| **Episode**  | episodic  | Orchestrator (auto) | Planner    | "what happened last time for feature X" |

**Semantic memory** (Pattern / Mistake / Solution / Strategy) answers *"what should I do?"* with generalised lessons. Feedback loop: eval pass → boost recalled memories; eval fail → demote (without inflating `times_applied` — demote only adjusts relevance). Relevance decays 5% each run. Memories that decay below 0.05 are automatically pruned from both Neo4j and Qdrant (garbage collection runs at pipeline start). Relevance scores are synced to Qdrant after every boost, demote, and decay so vector search ranking stays consistent with Neo4j keyword search.

**Cross-feature recall** — `recall_memories` runs two Qdrant passes: one feature-scoped (high precision) and one unscoped (cross-feature). A pattern recorded by feature "auth" about parameterized SQL now surfaces when working on "user-profile." Write-time dedup is also cross-feature for Patterns and Strategies (the most reusable types), while Mistakes and Solutions stay feature-scoped to avoid conflating distinct failure modes.

**Mistake → Solution pairing** — when the reviewer recalls a Mistake, the associated Solution (via `RESOLVED_BY` edge) is returned inline as a `resolved_by` dict. The agent sees the problem AND the fix in a single recall hit.

**Episodic memory** answers *"what happened last time I was in this exact situation?"* After every feature swarm completes, the orchestrator synthesises a narrative summary (up to 300 words) plus key turning-point events (up to 15), embeds the result, and writes to Neo4j + Qdrant. Key enhancements:

- **Episodes capture recalled memory IDs** — the synthesis prompt includes which patterns, strategies, and prior episodes were recalled during this feature's lifecycle, so future Planners can see "the last run succeeded *because it applied pattern-abc*." The embedded text includes memory IDs for semantic discovery.
- **Episode-to-memory graph edges** — `(Episode)-[:APPLIED]->(Pattern/Mistake/Solution/Strategy)` edges enable traversals like "which episodes used this pattern?" and "which patterns come from successful runs?"
- **Per-feature ID tracking** — recalled memory IDs are accumulated across the entire feature lifecycle (planner + coder + reviewer + tester) via a per-feature set that survives per-eval clears. The per-eval set (used for boost/demote feedback) is separate.
- **Cross-feature briefing exclusion** — `run_learnings` now excludes the current feature's own memories so retried features don't see redundant data.

Episodic memory costs ~1k LLM tokens per feature for the summarisation pass — toggle off via `enable_episodic_memory` if you're running single-shot features that never recur.

#### Memory hygiene

Five mechanisms keep the memory graph clean and the recall path sharp:

1. **Write-time dedup** — before creating a new memory, the repository embeds the candidate text and cosine-matches against existing same-type memories. Matches above `memory_dedup_threshold` (default 0.92) get boosted instead of duplicated. For Patterns and Strategies, dedup is **cross-feature** — the same pattern discovered by different features consolidates into one high-relevance node. Set threshold to `0.0` to disable.
2. **Relevance-weighted Reciprocal Rank Fusion recall** — the hybrid Neo4j + Qdrant merge multiplies each rank contribution by the memory's relevance_score. Memories boosted by successful eval feedback outrank demoted ones. The floor (0.1) keeps demoted memories visible but pushes them to the back.
3. **Garbage collection** — `prune_low_relevance(threshold=0.05)` runs alongside `decay_all_relevance` at every pipeline start. Memories that have decayed below the threshold are deleted from both Neo4j and Qdrant, preventing unbounded graph growth.
4. **Qdrant payload sync** — `relevance_score`, `run_id`, `created_at`, `times_recalled`, and `last_recalled_at` are kept in sync between Neo4j and Qdrant payloads. Every boost, demote, and decay operation updates both stores.
5. **Memory observability dashboard** — the Metrics tab shows per-type node counts, relevance distribution, the 10 most-recalled memories, and 7-day boost/demote effectiveness. Prometheus metrics cover all four memory types plus episode writes, recalls, and garbage collection events.

### Observability

- **Prometheus** counters/histograms for every phase, tool call, LLM invocation, reconciliation status, incident, and the BackgroundLoop sampler — always-on and zero-cost.
- **Postgres** (optional) forensic rows for LLM calls, eval results, tool calls, incidents, and progress events — high-cardinality debugging.
- **Grafana** ships with provisioned datasources and dashboards for pipeline throughput, cost rollups, and error budgets.
- **Incident table** surfaces errors, warnings, and reconciliation issues in the Run Detail popup with stack traces.

### Deep agent architecture

Dark Factory makes extensive use of the **Claude Agent SDK** to spawn isolated, clean-context subprocess agents at several points in the pipeline. Unlike the LangGraph swarm agents — which share a persistent graph state across handoffs — deep agents run in a completely fresh context, with their own working directory and a full file-system tool set: `Read / Write / Edit / Glob / Grep / Bash`. Each invocation starts with zero memory of previous runs and exits cleanly when its task is complete.

Each deep agent runs as a Node.js subprocess managed by the `BackgroundLoop` singleton — a daemon asyncio event loop that ensures subprocess cleanup callbacks always have a valid loop to land on.

| Phase | Deep agent role |
|-------|----------------|
| **Phase 1 — Doc extraction** | Rich business documents (Word, Excel, PDF, HTML, XML, RTF, CSV, transcripts) are routed to a per-file deep agent with its cwd set to the upload directory. Each document gets its own isolated context so raw meeting-transcript noise never pollutes the main pipeline context. |
| **Phase 4 — Code generation** | The Coder swarm agent can delegate to a `claude_agent_codegen` deep agent for complex implementation tasks. The deep agent has full filesystem access to the run output directory, iterates with the linter, and returns a structured result to the swarm. |
| **Phase 4 — Review & test gen** | Specialised deep agents handle dependency analysis, risk review, security review, performance review, compliance review, and unit / integration / edge-case test generation. Each is a separate `@tool`-decorated function backed by its own SDK invocation. |
| **Phase 5 — Reconciliation** | A single extended deep agent runs over the full run output directory after all feature swarms complete. It follows a rigid six-step checklist (inventory → review → fix → validate → iterate → report) and is the only agent that can see every feature's output simultaneously. |
| **Phase 6 — E2E validation** | A second extended deep agent executes a Playwright cross-browser smoke test suite. It detects whether the output is a web application, writes a `smoke.spec.ts` against the acceptance criteria, starts the server, and runs the suite across chromium, firefox, and webkit. |

Deep agent failures (crashes, timeouts, SDK errors) are caught, recorded as incidents, and returned as structured error strings so the LangGraph swarm treats them as soft tool errors rather than feature-killing crashes. Each subprocess turn emits a `deep_agent_turn` progress event to the Agent Logs tab in real-time. `DEEP_AGENT_DEBUG_STDERR=1` enables verbose Node CLI diagnostics, and stderr is buffered (200 lines / 16 KiB) for crash forensics.

### Storage backend (local / S3)

Pipeline inputs and outputs are persisted through a pluggable storage backend (`STORAGE_BACKEND=local` or `s3`). Each run is stored under a run-ID-scoped layout:

```
{run_id}/
  input/              ← uploaded raw files
  requirements/       ← parsed Requirement models (JSON, post-ingest)
  specs/              ← generated Spec models (JSON, post-spec-stage)
  output/             ← generated code, reports, artifacts
```

For **local storage** (default), this maps to `./output/{run_id}/...`. For **S3**, objects go directly into the bucket: `s3://my-bucket/{run_id}/input/meeting.docx`.

Data is synced at pipeline checkpoints:
- **Input files** → synced after Phase 1 Ingest
- **Requirements** → serialized to JSON after Phase 1 Ingest
- **Specs** → serialized to JSON after Phase 2 Spec Generation
- **Output** → dual-written on every `write_file` call + bulk synced after Phase 4 (Swarm), Phase 5 (Reconciliation), and Phase 6 (E2E)

Deep agents operate on a local scratch directory (they need real filesystem tools). For S3, files are downloaded to scratch before the phase and synced back after. For local storage, the scratch directory *is* the storage directory — download/sync are no-ops.

The Output tab in the Run Detail popup reads from the storage backend, so it works identically for both local and S3.

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
├── storage/               # Pluggable storage backend (local / S3)
│   └── backend.py         # LocalStorage, S3Storage, RunStorage, factory
├── llm/                   # Anthropic / OpenAI / LangChain clients
│   ├── anthropic.py       # Direct Anthropic SDK client with observability
│   ├── agentic.py         # Multi-turn tool-use loop (replaces SDK subprocess)
│   └── tool_handlers.py   # Sandboxed Read/Write/Edit/Glob/Grep/Bash handlers
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

tests/                        # 715 tests across 41 files
```

---

## Development

```bash
# Run the full test suite (715 tests)
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

| L4 Trait                           | Implementation                                                               |
|------------------------------------|------------------------------------------------------------------------------|
| Persist state across sessions      | Neo4j procedural memory + Qdrant embeddings + eval/run history               |
| Refine execution based on feedback | Eval → memory feedback loop, adaptive thresholds, cross-feature learning,    |
|                                    | mid-run strategy adjustment                                                  |
| Parallel execution                 | Concurrent feature swarms within dependency layers                           |
| Real-time adaptation               | Strategy overrides triggered by layer pass rate; cross-feature briefing      |
|                                    | within same run                                                              |
| Cross-feature reconciliation       | Phase 5 extended Claude Agent SDK pass over the full run output              |

---

## Author

**[Kevin Quon](https://www.linkedin.com/in/kwkwan00/)**

---

## License

[MIT](LICENSE)
