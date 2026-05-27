# AI Requirements Refinery

The Requirements Refinery is an adversarial-panel system that transforms raw or underspecified requirements into a detailed, structured, and actionable set. It operates as an assistant — analyzing and suggesting changes for the user to review — not as an autonomous generator.

The refinery runs an **adversarial panel of 6 role-specialized agents** (Product, Engineering, Security, Operations, Cost, Judge) with a layered-sourcing Research capability, a combined evaluation gate fusing deterministic rules with DeepEval LLM scoring, bounded debate with short-circuit reconcile on non-convergence, an institutional-memory write-back loop across 5 memory kinds, and seven cooperating L4 calibration loops (provider trust, role weighting, critic-dimension credibility, Judge confidence, adaptive max_rounds, self-tuning rule severity, active-learning feedback) that turn operator and panel signal into bounded multipliers on the next run.

---

## How It Works

The refinery runs as a 4-phase SSE (Server-Sent Events) pipeline:

```
Phase 1: Gather        Collect requirements from a run, uploaded documents, or a single typed entry
    |
Phase 2: Refine        Run up to 5 concurrent debates (one per requirement) through the LangGraph panel
    |
Phase 3: Cross-set     JudgeRole.review_set adjudicates the full refined set + every DebateTrace
    |
Phase 4: Persist       Save results to S3/local storage and stream to the UI
```

### Phase 1 — Gather

Three input modes:

**From a historical run** — The refinery collects all available evidence:
- Traceability matrix (requirement → spec → files → eval scores)
- Gap analysis (unimplemented requirements, failing specs)
- Episodic memories (what agents tried and failed)
- Procedural memories (patterns, mistakes, solutions, strategies)
- Eval critiques (why specs failed, with LLM-generated reasons)
- Run statistics (pass rate, worst features, duration)

**From uploaded documents** — The refinery runs the `IngestStage` standalone to parse requirement files (MD, TXT, JSON, YAML, DOCX, XLSX, PDF, CSV, HTML) into structured requirements without launching a full pipeline.

**From a single typed requirement** — `POST /api/refinery` with a `direct` body (title / description / priority / tags) wraps one requirement in a minimal run context with `source_mode="direct"` and runs the full panel on it. The cross-set Phase 3 is short-circuited when there's only one requirement.

### Phase 2 — Refine

Each requirement runs through a **LangGraph debate subgraph** (`generator → critic-fanout → synthesize → score → router → finalize | research | escalate | reconcile | next round`). Up to 5 debates run concurrently via a thread pool. The router's three terminal outcomes:

| Outcome           | Trigger                             | Effect                                                     |
|-------------------|-------------------------------------|------------------------------------------------------------|
| `converged`       | `EvaluationScore.passed` before     | Best-scoring draft is finalized                            |
|                   | max_rounds                          |                                                            |
| `short_circuited` | `round_number >= max_rounds`        | `reconcile_node` runs a dedicated synthesis pass that      |
|                   | without passing                     | documents what the panel could not resolve and emits a     |
|                   |                                     | `Conflict` memory                                          |
| `aborted`         | Generator crash or unrecoverable    | Carry-forward original requirement; debate trace records   |
|                   | error                               | the abort reason                                           |

Each debate produces a `RefinedRequirement` plus a full `DebateTrace` with every prompt, response, tool call, and routing decision. The Judge's synthesis ledger declares `accepted` / `rejected` / `deferred` for every critique so disagreement is preserved in `explicit_tradeoffs` rather than smoothed away.

### Phase 3 — Cross-set review

After all per-requirement debates complete, `JudgeRole.review_set` adjudicates the full refined set + every `DebateTrace`. A bounded critique-and-ratify loop (`refinery_set_review_max_rounds`, default 1) iterates the Judge against the four critic seats at set-level scope until no blockers remain or the round budget is exhausted.

The resulting `CrossReviewReport` carries:
- **Duplicates** — semantically identical requirements
- **Coherence issues** — contradictory constraints, inconsistent terminology, priority mismatches
- **Relationship fixes** — missing or invalid edges, circular dependencies, priority inversions
- **Spec overlaps** — suggested specs that duplicate each other across requirements
- **Set-level dimension scores** + risk areas + unresolved debates surfaced from per-req traces

A pure structural patcher (`apply_cross_review_report`) applies the report's mutations onto the refined set in one pass.

### Phase 4 — Persist

The final `RefineryResponse` is saved to the storage backend:

```
refinery/{result_id}/
    metadata.json      Stats, timestamps, source info
    response.json      Full RefineryResponse
    REPORT.md          Rendered markdown report
```

The result ID follows the pattern `refinery-YYYYMMDD-HHMMSS-xxxx`.

---

## Data Models

### RefinedRequirement

The core output per requirement:

| Field                  | Type                      | Description                                             |
|------------------------|---------------------------|---------------------------------------------------------|
| `id`                   | string                    | Original requirement ID (preserved for graph stability) |
| `original_title`       | string                    | Title before refinement                                 |
| `original_description` | string                    | Description before refinement                           |
| `title`                | string                    | Refined title                                           |
| `description`          | string                    | Expanded description with constraints and edge cases    |
| `priority`             | string                    | `low` / `medium` / `high` / `critical`                  |
| `tags`                 | string[]                  | Enriched tags (capability area, complexity, etc.)       |
| `relationships`        | RequirementRelationship[] | Discovered inter-requirement links                      |
| `suggested_specs`      | SuggestedSpec[]           | Preview of how this should decompose into specs         |
| `changes`              | string[]                  | Human-readable list of what changed and why             |
| `pass_context`         | string                    | Reasoning behind the changes                            |

### RequirementRelationship

| Field       | Type   | Description                                                             |
|-------------|--------|-------------------------------------------------------------------------|
| `target_id` | string | ID of the related requirement                                           |
| `type`      | string | `depends_on` / `conflicts_with` / `extends` / `replaces` / `related_to` |
| `rationale` | string | Why this relationship exists                                            |

### SuggestedSpec

| Field                 | Type     | Description              |
|-----------------------|----------|--------------------------|
| `title`               | string   | Proposed spec title      |
| `capability`          | string   | kebab-case feature group |
| `description`         | string   | What this spec covers    |
| `acceptance_criteria` | string[] | 3-5 testable criteria    |

### SuggestedMemory

Reusable insights for future planner agents. Memory descriptions must be **general and reusable** — never reference specific spec IDs, requirement IDs, or implementation details.

| Field            | Type   | Description                   |
|------------------|--------|-------------------------------|
| `type`           | string | `pattern` or `strategy`       |
| `description`    | string | General reusable insight      |
| `context`        | string | When this applies             |
| `applicability`  | string | Scope of applicability        |
| `source_feature` | string | Feature that inspired this    |
| `rationale`      | string | Why this should be remembered |

### RefineryResponse

The full result payload:

| Field                          | Type                 | Description                               |
|--------------------------------|----------------------|-------------------------------------------|
| `summary`                      | string               | Overview of the refinement                |
| `pass_summaries`               | string[]             | What each phase accomplished              |
| `refined_requirements`         | RefinedRequirement[] | All requirements (modified and unchanged) |
| `suggested_memories`           | SuggestedMemory[]    | Patterns/strategies to save               |
| `new_relationships_count`      | int                  | Total relationship edges discovered       |
| `requirements_modified_count`  | int                  | How many requirements were changed        |
| `requirements_unchanged_count` | int                  | How many were left as-is                  |
| `source_run_id`                | string?              | Run ID if refined from a historical run   |
| `methodology`                  | string               | How the analysis was structured           |
| `evidence_summary`             | string               | Key evidence that drove changes           |
| `risk_areas`                   | string[]             | Issues that need attention                |

---

## API Endpoints

### POST /api/refinery

Start a refinery run. Returns an SSE stream.

**Request body:**
```json
{
  "run_id": "run-20260417-053205-8b7d",
  "input_path": null
}
```
Provide either `run_id` (refine from a historical run) or `input_path` (refine from uploaded documents).

**SSE events:**
```
data: {"phase": "gathering", "step": "traceability"}
data: {"phase": "gathering", "step": "complete", "requirement_count": 12}
data: {"phase": "refining", "message": "Completed 3/12 — req-abc123"}
data: {"phase": "reconciling", "message": "Checking for duplicates..."}
data: {"phase": "done", "result_id": "refinery-20260423-120000-ab12", "data": {...}, "duration_seconds": 245.3}
```

### GET /api/refinery/history?limit=20

List past refinery results, newest first.

**Response:**
```json
{
  "results": [
    {
      "id": "refinery-20260423-120000-ab12",
      "timestamp": "2026-04-23T12:00:00Z",
      "source_run_id": "run-20260417-053205-8b7d",
      "source_mode": "run",
      "requirements_count": 12,
      "requirements_modified": 8,
      "requirements_unchanged": 4,
      "relationships_count": 15,
      "suggested_memories_count": 3,
      "duration_seconds": 245.3
    }
  ]
}
```

### GET /api/refinery/{result_id}

Load a previously saved refinery result.

**Response:** Full `RefineryResponse` object.

### DELETE /api/refinery/{result_id}

Delete a saved result from storage.

### POST /api/refinery/export

Export results as a ZIP file containing the markdown report and individual requirement files.

**Request body:** The `RefineryResponse` object from the frontend.

**Response:** ZIP file:
```
refinery-{run_id}/
    REPORT.md
    requirements/
        req-abc123.md
        req-def456.md
```

### PATCH /api/graph/requirements/{req_id}

Apply a refined requirement to the Neo4j knowledge graph. Partial update — only the fields present in the body are changed. The requirement ID is preserved so all existing `IMPLEMENTS` edges from specs are maintained.

**Request body:**
```json
{
  "title": "Refined title",
  "description": "Expanded description",
  "priority": "high",
  "tags": ["auth", "security"]
}
```

### POST /api/memory/check-duplicate

Check if a suggested memory is a semantic duplicate of an existing one before saving. Read-only — does not write or boost anything.

**Request body:**
```json
{
  "type": "strategy",
  "description": "Auth features need explicit session timeout criteria",
  "context": ""
}
```

**Response:**
```json
{
  "is_duplicate": true,
  "existing_id": "strategy-a1b2c3d4",
  "existing_description": "Authentication modules should specify session timeouts",
  "similarity": 0.94
}
```

### POST /api/refinery/memory/{memory_id}/feedback

Operator decision on a suggested memory. Drives the active-learning
loop: telemetry to Postgres, Neo4j relevance boost (`accepted`) or
demote (`dismissed`), and a `Conflict` memory tagged
`cause="user_override"` when a dismissal carries a reason. All three
sinks are best-effort; per-sink success flags are returned so the UI
can surface partial success without blocking.

**Request body:**
```json
{
  "decision": "accepted",
  "memory_kind": "pattern",
  "refinery_run_id": "refinery-20260425-153012-abcd",
  "source_role": "security",
  "reason": "optional free-text explanation"
}
```

### POST /api/refinery/{run_id}/requirement/{req_id}/decision

Operator decision on a refined requirement. Drives the Judge
confidence-calibration loop: the next run's effective threshold is
nudged based on whether the Judge's predictions are tracking
operator follow-through. Best-effort write to Postgres; UI action
proceeds regardless.

**Request body:**
```json
{
  "decision": "accepted",
  "judge_overall_score": 0.86,
  "convergence_status": "converged"
}
```

`judge_overall_score` and `convergence_status` are both optional —
the aggregator coalesces missing scores by treating
`convergence_status="converged"` as the high bucket.

---

## Configuration

All settings are in `PipelineConfig` and can be changed at runtime via the Settings tab or `PATCH /api/settings`.

| Setting                     | Default   | Range                               | Description                          |
|-----------------------------|-----------|-------------------------------------|--------------------------------------|
| `refinery_model`            | `gpt-5.4` | any OpenAI model                    | Model for the deep agents            |
| `refinery_reasoning_effort` | `xhigh`   | `low` / `medium` / `high` / `xhigh` | Reasoning depth                      |
| `refinery_max_turns`        | `40`      | 5-100                               | Total turn budget (split across      |
|                             |           |                                     | agents)                              |
| `refinery_timeout_seconds`  | `900`     | 60-3600                             | Wall-clock timeout per agent         |

Per-requirement debates receive `max(15, max_turns // 2)` turns each. The cross-set Judge pass receives the same budget.

---

## UI Guide

The Refinery is a top-level tab in the dashboard ("Refine"), next to Manufacture.

### Input Selection

Three modes via a segmented toggle:

- **From a Run** — Select a historical pipeline run from a table showing run ID, status, and pass rate. The refinery uses the run's traceability data, eval failures, and agent memories as evidence.
- **From Documents** — Drag-and-drop requirement files. Supports MD, TXT, JSON, YAML, DOCX, XLSX, PDF, CSV, HTML. Files are parsed via the ingest stage.
- **From a Requirement** — A title / description / priority / tags form runs the panel on a single typed requirement.

Below the input selector, a **Recent Refinements** table shows past results with View and Delete actions.

### Progress

The refinery streams real-time progress:
- **Gathering** — checkmarks for each data source collected
- **Refining** — per-debate events (generator started, critic ready, judge ready, round complete) as each requirement progresses
- **Reconciling** — events from the cross-set Judge pass

### Results

A two-panel layout with three sub-tabs:

**Requirements tab** (default):
- **Left sidebar** — scrollable requirement list with filter buttons (All / Modified / Unchanged). Each item shows priority badge and status (MODIFIED / APPLIED / DISMISSED).
- **Right panel** — selected requirement's full detail:
  - Title (with "was: original" if changed)
  - Priority, tags, ID
  - Changes list (what was modified and why)
  - Description with side-by-side diff toggle (original vs refined)
  - Relationships table (target, type, rationale)
  - Suggested specs (expandable)
  - Action buttons: **Apply to Graph** (saves to Neo4j), **Edit** (opens inline editor), **Dismiss**

**Insights tab**:
- Summary, evidence summary, methodology
- Risk areas
- Pass-by-pass analysis

**Memories tab**:
- Suggested strategy/pattern memories with "Save to Memory" and "Dismiss" buttons
- Before saving, the system checks for semantic duplicates — if a similar memory already exists (92%+ match), the save is blocked with a warning showing the existing memory

### Export

**Export ZIP** downloads a ZIP containing:
- `REPORT.md` — full markdown report with methodology, evidence, per-requirement rationale
- `requirements/` — individual markdown files per requirement with description, relationships, and suggested specs

---

## Architecture

### Package Structure

```
src/dark_factory/api/refinery/
    __init__.py           Re-exports for routing layer
    models.py             Pydantic models (RefinedRequirement, RefineryResponse, ...)
    contracts.py          Debate contracts (Draft, Critique, Rebuttal, EvaluationScore, RoleContext, ...)
    markdown.py           SnakeMD report rendering
    storage.py            Persistence CRUD (S3/local)
    gather.py             Data gathering (run evidence + document ingest)
    stream.py             SSE pipeline orchestrator
    resume.py             Resume registry adapter (Postgres-backed)
    set_review.py         Phase 3 cross-set critique-and-ratify loop
    feedback.py           Active-learning feedback orchestrator
    role_weighting.py     Per-role + per-(role, dim) critique-acceptance aggregator
    judge_calibration.py  Judge threshold-multiplier aggregator
    archetype_rounds.py   Per-archetype effective max_rounds aggregator
    rule_severity.py      Per-rule blocker→warning demote-only aggregator
    debate/               LangGraph subgraph: state, graph, runner, cross-debate bus
    roles/                Role agents: product, engineering, security, operations, cost, judge, planner
    judge/                Combined evaluation pipeline: rules + LLM + composition
    research/             4-role layered-sourcing pipeline + 6 tier providers
    context/              HybridContextBuilder + RoleFilterPolicy (server-side filter)
    memory/               Producer hooks + 5-kind write-back contract
    observability/        DebateTrace + ObservabilityHub + structured event emission
```

### Key Design Decisions

| Decision                       | Rationale                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------|
| **Adversarial panel over       | Six role-specialized seats with mechanically-enforced disagreement (rubber-stamp    |
| single-agent**                 | validator, explicit_tradeoffs, disagreement_score) catch flaws a single agent's     |
|                                | reasoning trace would smooth over                                                   |
| **Information hiding           | Roles, retrievers, judges, and research providers are ABCs with stable contracts.   |
| (Parnas)**                     | The orchestrator imports only contracts and never reads prompt text or knows model  |
|                                | IDs                                                                                 |
| **Role-filtered retrieval at   | `RoleFilterPolicy` translates to Qdrant `Filter` + Cypher `WHERE` so a role *cannot |
| the data layer**               | receive* rows outside its slice even if its prompt malfunctions                     |
| **Combined evaluation gate**   | Phase A rules → Phase B LLM judge with rule findings injected → Phase C dimension   |
|                                | clamps. Rules have the final say via Phase C regardless of LLM output               |
| **Bounded debate with three    | converged / short_circuited / aborted. Non-convergence is itself a signal —         |
| terminal outcomes**            | `reconcile_node` documents what couldn't be resolved instead of forcing a synthesis |
| **5 concurrent debates**       | Balances panel parallelism against LLM rate limits; thread-local progress callback  |
|                                | prevents per-agent SSE event bleed across workers                                   |
| **Calibration via bounded      | Seven learning loops (provider trust, role weighting, critic-dimension credibility, |
| multipliers**                  | Judge threshold, adaptive max_rounds, self-tuning rule severity, active-learning    |
|                                | feedback) all stamp into the evidence bag (or runner kwarg for orchestration        |
|                                | parameters) with identity defaults; no prompt rewrites required                     |
| **Assistant-mode at the Apply  | The panel debates, escalates, researches, reconciles, and writes its own trace      |
| boundary**                     | autonomously; the user reviews and explicitly Applies each refined requirement      |
| **ID stability on edit**       | `upsert_requirement()` matches on ID, so editing a requirement preserves all        |
|                                | existing `IMPLEMENTS` edges from specs                                              |
| **Memory dedup check before    | Kind-scoped: a Pattern and a Decision with identical text are not considered        |
| save**                         | duplicates                                                                          |

### Error Handling

| Scenario               | Behavior                                                                                    |
|------------------------|---------------------------------------------------------------------------------------------|
| Critic role raises     | Caught and replaced with a placeholder INFO critique; the round continues with the          |
|                        | surviving critics                                                                           |
| Judge synthesis raises | Logged; deterministic pass-through Rebuttal (no body mutation, all critiques recorded as    |
|                        | deferred) keeps the debate runnable                                                         |
| Generator crash        | Debate aborts cleanly with `convergence_status="aborted"`; carry-forward requirement        |
|                        | returned to the orchestrator                                                                |
| Max rounds without     | `reconcile_node` produces a `short_circuited` Draft with populated `unresolved_points` +    |
| convergence            | `open_questions` and emits a `Conflict` memory tagged `cause="non_convergence"`             |
| Phase 3 `review_set`   | Returns the un-patched refined set with an error note in the report summary; the run still  |
| raises                 | completes                                                                                   |
| API timeout (429, 500, | Single automatic retry with backoff at the LLM helper layer                                 |
| 502, 503)              |                                                                                             |
| Wall-clock timeout     | Loop breaks early; partial results preserved                                                |
| Memory duplicate       | Save blocked; user sees existing memory + similarity score                                  |
| detected               |                                                                                             |
| Postgres outage        | All seven calibration loops + resume registry + forensics writer fall back to no-op with    |
|                        | structured warnings; debates continue under identity defaults (configured base for          |
|                        | max_rounds; configured severity for rule violations)                                        |

---

## Architecture deep dive

### Canonical 9-step loop

| Step                                   | Subsystem                                                                   |
|----------------------------------------|-----------------------------------------------------------------------------|
| 1. User submits a requirement          | Phase 1 Gather — now 3 input modes: `run_id` / `input_path` / `direct`      |
|                                        | (typed)                                                                     |
| 2. Role-filtered hybrid retrieval      | `HybridContextBuilder` + `RoleFilterPolicy` (memory + graph, server-side    |
|                                        | filtered)                                                                   |
| 3. Product agent drafts v0             | `ProductRole.propose` — generator only                                      |
| 4. Adversarial critics run in parallel | Eng/Sec/Ops/Cost via LangGraph `Send` fan-out with anti-rubber-stamp        |
|                                        | validator                                                                   |
| 5. Judge synthesizes                   | `JudgeRole.defend` — preserves disagreement in `explicit_tradeoffs`         |
| 6. Combined evaluation gate            | Rules → LLM-with-rule-findings → rule→dim penalties → unified               |
|                                        | `EvaluationScore`                                                           |
| 7a. Converged                          | `finalize` terminates with `convergence_status="converged"`                 |
| 7b. Gap → refine or research           | Next round (critic fan-out), or research node invokes the layered-sourcing  |
|                                        | agent                                                                       |
| 7c. Circuit breaker                    | `reconcile_node` on `max_rounds` → `convergence_status="short_circuited"` + |
|                                        | CONFLICT memory                                                             |
| 8. Write-back to memory                | On user Apply, validated memories auto-save atomically (Neo4j + Qdrant, one |
|                                        | transaction)                                                                |
| 9. Cross-req Judge                     | `Judge.review_set` adjudicates the full refined set + every per-req trace   |

### Adversarial panel (6 seats + 1 capability)

| Role                                 | Verbs                                   | Default model       | Dimension     |
|--------------------------------------|-----------------------------------------|---------------------|---------------|
| Product                              | propose                                 | `claude-opus-4-7`   | — (generator) |
| Engineering                          | critique                                | `claude-sonnet-4-6` | feasibility   |
| Security                             | critique                                | `claude-opus-4-7`   | risk_coverage |
| Operations                           | critique                                | `claude-sonnet-4-6` | completeness  |
| Cost                                 | critique                                | `gpt-5.4`           | feasibility   |
| Judge                                | defend, reconcile_unresolved, score,    | `claude-opus-4-7`   | (all five)    |
|                                      | review_set                              |                     |               |
| **Research** (capability, not a      | investigate                             | `gpt-5.4`           | —             |
| seat)                                |                                         |                     |               |

Research is invoked by the router when the Judge flags
`missing_external_info`. It runs the 4-role Librarian → Analyst →
Editor → Historian pipeline across 6 tier providers (T0 structured / T1
internal / T2 observability / T3 official / T4 academic / T5 web) with
3 enforced guardrails: internal-first triage, URL-allowlisted fetch,
and per-tier budget exhaustion.

### Combined evaluation gate

One `Judge.score` call produces one `EvaluationScore` by running three
phases:

1. **Phase A — Rules.** Deterministic catalog (`rule_ids_preserved`,
   `rule_priority_valid`, `rule_convergence_consistency`,
   `rule_acceptance_criteria_min_count`,
   `rule_acceptance_criteria_measurable`,
   `rule_description_specificity`, `rule_relationships_non_circular`,
   `rule_tradeoffs_required`). Every rule declares its affected
   dimension. ~ms, cost-free.
2. **Phase B — LLM judge with rule context.** DeepEval GEval metrics
   for the 5 dimensions, each prompt-instructed to factor in Phase-A
   findings. Falls back to `FallbackJudge` (heuristic) on
   DeepEval/provider failure.
3. **Phase C — Rule→dimension penalties.** Every blocker caps its
   affected dimension at 0.5; every warning subtracts 0.1. Hard
   backstop even if the LLM ignores Phase B's preamble.

Convergence requires `overall >= threshold` **AND** every dimension
`>= per_dim_threshold` **AND** zero rule blockers.

### Institutional memory (5 curated kinds)

- **Decision** — why a choice was made over alternatives (synthesizer rebuttals)
- **Incident** — production failures + blast radius (`:Mistake` label)
- **Pattern** — reusable requirement / code structures
- **Constraint** — system or business limitations
- **Conflict** — recurring disagreements (especially on short-circuit)

Producers fire on specific debate events (see
`refinery/memory/producers.py`). Dedup is **kind-scoped** — a Pattern
and a Decision with identical text are not considered duplicates.

### Write-back on Apply

`PATCH /api/graph/requirements/{req_id}` accepts an optional
`apply_memories: list[...]` field. When the user applies a converged
requirement via the UI, the handler writes the requirement and all
validated / produced memories in one transaction (Neo4j commit
followed by Qdrant upserts with compensating-delete rollback). Per-
memory failures are recorded in
`dark_factory_refinery_memory_write_back_atomic_failures_total` but
never fail the requirement update.

Controlled by `refinery_auto_save_on_apply` (default `True`).
`refinery_auto_save_on_apply=False` requires the user to review and
save each suggested memory individually.

### Continuous calibration — seven learning loops + feedback

Seven cooperating subsystems turn operator + panel signal into bounded
multipliers that nudge the next debate's behaviour. All seven follow the
same architectural pattern: Postgres telemetry table → bounded aggregator
→ identity default on missing data → multiplier injected via the
evidence bag (or runner kwarg, for orchestration parameters) at run start.
None can hard-fail the pipeline; a Postgres outage means every loop falls
back to identity (`×1.0` or configured base) and the system behaves
exactly as it did pre-feature.

| Loop                     | Signal source              | Where applied               | Bounds                         |
|--------------------------|----------------------------|-----------------------------|--------------------------------|
| Provider trust learning  | Per-(tier, provider)       | Editor's confidence         | per-tier `(min, max)`          |
|                          | propagation rate from the  | weighting at the per-tier   | envelopes                      |
|                          | research-sources table     | level                       |                                |
| Adaptive role weighting  | Per-(role, severity)       | Multiplier rendered into    | `[0.60, 1.40]`,                |
|                          | accept/reject rate from    | the Judge's defend prompt   | `_MIN_TOTAL_FOR_LEARNING=8`    |
|                          | the dispositions table     | so the LLM weighs critics   |                                |
|                          | (recorded by the           | by historical accuracy      |                                |
|                          | synthesize node)           |                             |                                |
| Critic-dimension         | Per-(role, dimension)      | Combined (product-clamped)  | per-cell `[0.50, 1.50]`,       |
| credibility weighting    | acceptance rate from the   | multiplier rendered as a    | product `[0.50, 1.60]`,        |
|                          | dispositions table; refines| 2D table inside the         | `_MIN_TOTAL_FOR_LEARNING=6`    |
|                          | the flat per-role weight   | calibration block, pre-     |                                |
|                          | by splitting on the        | rendered once per run and   |                                |
|                          | dimension a critique cited | reused across rounds        |                                |
| Judge confidence         | Operator apply / dismiss / | Scales `effective_threshold | `[0.85, 1.15]`,                |
| calibration              | edit decisions joined      | = base × multiplier` inside | `_MIN_TOTAL_FOR_LEARNING=10`,  |
|                          | against the Judge's        | the combined pipeline; the  | `_MISCALIBRATION_TRIGGER=0.30` |
|                          | predicted overall score    | EvaluationScore reflects    |                                |
|                          | (or convergence_status as  | the effective threshold     |                                |
|                          | fallback)                  |                             |                                |
| Adaptive max_rounds      | Per-archetype              | Replaces                    | hard `[2, 6]`,                 |
|                          | (priority, source_mode,    | `DebateConfig.max_rounds`   | reduce when `avg_rounds <      |
|                          | primary_tag) convergence   | per-debate via runner kwarg | base × 0.55`; bump when        |
|                          | history from the debates   | (orchestration parameter,   | `short_circuit_rate ≥ 0.30`    |
|                          | table; reduce when         | not on the evidence bag —   | (+1) or `≥ 0.55` (+2);         |
|                          | converging fast, bump when | recursion-limit + fanout    | `_MIN_TOTAL_FOR_LEARNING=8`    |
|                          | short-circuiting often     | cost are graph-topology     |                                |
|                          |                            | concerns)                   |                                |
| Self-tuning rule         | Per-rule operator          | Demoted blockers reach      | demote-only,                   |
| severity                 | override rate (operator    | Phase B's prompt as         | `_DEMOTE_TRIGGER=0.50`,        |
|                          | applied a requirement      | warnings, hit Phase C's     | `_MIN_TOTAL_FOR_LEARNING=8`,   |
|                          | despite a rule blocker) on | `warning_delta` instead of  | warnings never promoted        |
|                          | the rule_violations and    | the blocker cap, and stay   |                                |
|                          | requirement_decisions      | in the trace as the loop's  |                                |
|                          | tables                     | audit                       |                                |
| Active-learning feedback | UI dismiss / accept / edit | Three best-effort sinks:    | `_BOOST_DELTA=0.10`,           |
|                          | on suggested memories      | Postgres telemetry, Neo4j   | `_DEMOTE_DELTA=0.05`           |
|                          |                            | relevance boost/demote, and |                                |
|                          |                            | a Conflict memory tagged    |                                |
|                          |                            | `cause="user_override"` on  |                                |
|                          |                            | reasoned dismissals         |                                |

Modules: `research/learning.py`, `role_weighting.py` (carries both flat
and per-dimension variants plus `combined_weight_for`),
`judge_calibration.py`, `archetype_rounds.py`, `rule_severity.py`,
`feedback.py`.

The calibration block (combined role × dimension multipliers, identity
cells dropped) is rendered **once per run** by `format_calibration_block`
in `roles/judge/prompt.py`, stamped on `run_context["calibration_block"]`,
and inlined verbatim by `JudgeRole.defend` on every round — the
role × dim Cartesian doesn't change between rounds of the same debate, so
re-rendering it per synthesis call would be wasted work.

Endpoints driving the feedback signals:

- `POST /api/refinery/memory/{memory_id}/feedback` — operator decision
  on a suggested memory (kind / decision / optional reason).
- `POST /api/refinery/{run_id}/requirement/{req_id}/decision` —
  operator decision on a refined requirement (decision / optional
  Judge score / optional convergence_status).

Both endpoints return per-sink success flags (`telemetry_recorded`,
`relevance_adjusted`, `conflict_emitted`) so the UI can surface partial
success without blocking the operator action; the frontend fires-and-
forgets via `.catch(() => undefined)` so a 503 from telemetry never
prevents an apply.

### Observability

Three stores, each with a distinct role:

- **Postgres** — 10 forensic tables FK'd back to `refinery_runs`:
  `refinery_runs`, `refinery_debates`, `refinery_debate_rounds`,
  `refinery_debate_completions` (resume cache), `refinery_llm_calls`,
  `refinery_rule_violations`, `refinery_research_sources`,
  `refinery_memory_audits`, plus the three calibration tables
  (`refinery_memory_feedback`, `refinery_critique_dispositions`,
  `refinery_requirement_decisions`) — the latter two are joined back
  against `refinery_rule_violations` and `refinery_debates` to drive
  the self-tuning rule-severity, critic-dimension, and adaptive-
  max_rounds loops. `refinery_debates` carries `priority` and `tags`
  columns so the archetype aggregator can bucket convergence
  behaviour by `(priority, source_mode, primary_tag)`. Every refinery
  LLM call **dual-
  writes** to the legacy `llm_calls` table with
  `phase='refinery.{role}.{kind}'` so existing cost dashboards auto-
  include refinery without a regex tweak.
- **Prometheus** — 25+ series under `dark_factory_refinery_*`,
  covering convergence outcomes, rounds histogram, judge scores
  (pre/post-penalty), rule violations by rule_id/severity/dimension,
  research tier mix, internal-first hit ratio, memory audit outcomes,
  cost + tokens per role, dual-write failures.
- **Trace JSON** — full-fidelity per-debate archive under
  `refinery/{result_id}/trace.json`.

Pre-built Grafana dashboard at `deploy/grafana-dashboards/refinery.json`
(10 starter panels).

### Feature flags

| Flag                                  | Default                                  | Controls                          |
|---------------------------------------|------------------------------------------|-----------------------------------|
| `refinery_auto_save_on_apply`         | `True`                                   | Auto-save validated memories when |
|                                       |                                          | the user Applies; `False`         |
|                                       |                                          | requires per-memory review.       |
| `refinery_rules_enabled`              | `True`                                   | Runs Phase A + Phase C of the     |
|                                       |                                          | evaluation pipeline. `False` =    |
|                                       |                                          | LLM-only scoring (logged          |
|                                       |                                          | warning).                         |
| `refinery_postgres_forensics_enabled` | `True`                                   | Writes the `refinery_*` forensic  |
|                                       |                                          | tables; `False` skips Postgres    |
|                                       |                                          | without losing Prometheus.        |
| `swarm_memory_kinds_enabled`          | `[pattern, mistake, solution, strategy]` | Scopes swarm-pipeline memory      |
|                                       |                                          | recall to the four legacy kinds   |
|                                       |                                          | so refinery-only kinds (Decision  |
|                                       |                                          | / Constraint / Conflict) don't    |
|                                       |                                          | leak into Planner / Coder /       |
|                                       |                                          | Reviewer / Tester contexts.       |
