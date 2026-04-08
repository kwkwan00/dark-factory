# AI Dark Factory

An autonomous code generator that converts requirements into specs, populates a knowledge graph, and generates application code with evaluation tests using a per-feature swarm of AI agents. Features cross-session procedural memory, adaptive eval thresholds, real-time strategy adjustment, semantic vector search, and parallel feature execution.

## Architecture

```
Requirements  -->  Specs  -->  Knowledge Graph  -->  Code  -->  Tests
     |               |              |                  |           |
   ingest        spec + eval      graph           swarm agents   eval
                                                 (per feature,
                                                  parallel)

Neo4j (structured)              Qdrant (semantic)
├── Spec/Requirement nodes      ├── Memory embeddings
├── DEPENDS_ON/IMPLEMENTS       ├── Spec embeddings
├── Procedural memory           ├── Code artifact embeddings
├── EvalResult/Run history      └── Hybrid RRF search
└── Feature groups + topo sort
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Neo4j (local or remote)
- Qdrant (local or cloud, optional -- graceful fallback to Neo4j)
- Anthropic API key (code generation)
- OpenAI API key (evaluation + embeddings)

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your credentials

# Source the env (or use direnv / dotenv)
export $(cat .env | xargs)

# Initialize the Neo4j schema
dark-factory graph init
```

## Configuration

Edit `config.toml`:

```toml
[neo4j]
uri = "bolt://localhost:7687"
database = "neo4j"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-6"

[pipeline]
output_dir = "./output"
max_parallel_features = 4     # concurrent features per dependency layer

[openspec]
root_dir = "./openspec"

[qdrant]
url = "http://localhost:6333"
collection_prefix = "dark_factory"
embedding_model = "text-embedding-3-large"
enabled = true

[memory]
database = "memory"
enabled = true

[evaluation]
base_threshold = 0.5
adaptive = true               # thresholds adjust based on score trends
decay_factor = 0.95           # memory relevance decays each run

[logging]
level = "INFO"
format = "console"
```

## Usage

### Run the pipeline

```bash
dark-factory run requirements/
dark-factory run openspec/
```

### OpenSpec workflow

```bash
dark-factory openspec init
dark-factory openspec propose "Add user authentication"
dark-factory openspec apply --change <change-name>
dark-factory openspec archive <change-name>
```

### Graph and memory management

```bash
dark-factory graph init    # Create schema constraints and indexes
dark-factory graph clear   # Delete all graph data
dark-factory memory clear  # Wipe all procedural memory
```

## How It Works

When you run `dark-factory run <path>`, the pipeline goes through two phases:

### Phase 1: Planning (requirements to swarm-ready features)

1. **Ingest** -- Parses raw input into `Requirement` models. Supports `.json` (structured), `.md`/`.txt` (one requirement per file), and OpenSpec `specs/` directories (auto-detected, extracts WHEN/THEN scenarios).

2. **Spec** -- An LLM converts each Requirement into a `Spec` with capability, scenarios, acceptance criteria, and dependencies. Each spec is evaluated by deepeval (GPT-5.4 as judge) for correctness, coherence, instruction following, and safety/ethics. Specs are auto-indexed in Qdrant for semantic retrieval.

3. **Graph** -- Persists specs and requirements to Neo4j with `IMPLEMENTS` and `DEPENDS_ON` relationships.

### Phase 2: Execution (per-feature swarm orchestration)

4. **Orchestrator** -- Queries the knowledge graph, groups specs by `capability`, computes topological execution order. Decays all memory relevance scores (0.95x) and creates a Run tracking node.

5. **Feature swarms** -- Each feature gets its own isolated swarm with a fresh 50-handoff budget. Features in the same dependency layer run **in parallel** (configurable via `max_parallel_features`). Four agents rotate:
   - **Planner** evaluates the spec, queries eval history for past performance, recalls strategies from memory, picks the next spec
   - **Coder** searches for similar specs/code via Qdrant RAG, recalls patterns and past mistakes, generates code (directly or via Claude Agent SDK)
   - **Reviewer** evaluates code with deepeval, compares against past eval scores, records mistakes and solutions
   - **Tester** writes tests, evaluates them for correctness/coherence/completeness, records failures

6. **Cross-feature learning** -- After each feature completes, its patterns/mistakes/solutions are briefed to subsequent features in the same run. Agents don't wait for the next run to benefit from new learnings.

7. **Strategy adjustment** -- After each dependency layer, the orchestrator reviews the pass rate. If performance drops below threshold, it forces the Coder to use `claude_agent_codegen` for all remaining specs and reduces the handoff budget. When performance recovers, overrides relax.

8. **Aggregate** -- Merges all artifacts and tests, computes pass rate, mean eval scores, identifies worst-performing features. Completes the Run node with final stats.

## AI Evaluation (deepeval + GPT-5.4)

All generated artifacts are evaluated using deepeval GEval metrics with OpenAI GPT-5.4 as the LLM judge:

### Spec evaluation

| Metric | What it checks |
|--------|---------------|
| **Spec Correctness** | Accurately captures the requirement's intent |
| **Spec Coherence** | Well-structured, unambiguous, internally consistent |
| **Spec Instruction Following** | Follows OpenSpec format with all required fields |
| **Spec Safety & Ethics** | No unsafe/unethical functionality, includes security considerations |

### Test evaluation

| Metric | What it checks |
|--------|---------------|
| **Test Correctness** | Tests validate the right conditions against acceptance criteria |
| **Test Coherence** | Well-structured, readable, follows testing conventions |
| **Test Completeness** | Covers all acceptance criteria from the spec |

Eval results are auto-persisted to the memory graph. Recalled memories are boosted when evals pass and demoted when they fail. Thresholds adapt based on score trends across runs.

## Procedural Memory (Neo4j)

Agents learn from past runs via a separate Neo4j database. Memory is auto-initialized on startup and persists across sessions.

| Memory type | Written by | Used by | Semantic search |
|-------------|-----------|---------|-----------------|
| **Pattern** | Coder | Coder | Qdrant embeddings |
| **Mistake** | Reviewer, Tester | All agents | Qdrant embeddings |
| **Solution** | Reviewer, Tester | All agents | Qdrant embeddings |
| **Strategy** | Planner | Planner | Qdrant embeddings |

**Feedback loop**: eval pass → boost recalled memories; eval fail → demote recalled memories. Memory decays 5% each run. Cross-feature learnings are briefed to subsequent features within the same run.

**Run history**: each pipeline run is tracked with pass rate, mean eval scores, worst features, and duration.

## Vector Search (Qdrant)

Qdrant provides semantic search where Neo4j keyword matching falls short. Three collections:

| Collection | Contents | Used by |
|-----------|---------|---------|
| `dark_factory_memories` | Pattern/Mistake/Solution/Strategy embeddings | `recall_memories` (hybrid RRF merge with Neo4j) |
| `dark_factory_specs` | Spec description embeddings | `search_similar_specs` (RAG for Coder) |
| `dark_factory_code` | Code artifact embeddings | `search_similar_code` (RAG for Coder) |

Embeddings generated via OpenAI `text-embedding-3-large` (3072 dims). All Qdrant operations gracefully degrade to Neo4j if unavailable.

## Swarm Architecture

```
Orchestrator
  |-- plan: query graph -> group by capability -> topo-sort by deps
  |-- execute_layer (parallel within layer):
  |     |-- [cross-feature briefing from prior features]
  |     |-- Planner: evaluate spec, query eval history, recall strategies
  |     |-- Coder: RAG similar specs/code, recall patterns, generate code
  |     |-- Reviewer: evaluate code, query eval history, approve/reject
  |     |-- Tester: write tests, evaluate tests, hand back to Planner
  |     `-- (up to 50 handoffs per feature)
  |-- adjust_strategy: review layer pass rate, adjust Coder strategy if needed
  `-- aggregate: merge artifacts + tests, compute scores, complete Run
```

## Project Structure

```
src/dark_factory/
  cli.py                    # Click CLI entry point
  config.py                 # Settings from config.toml + env vars
  models/domain.py          # Pydantic models (Requirement, Spec, Scenario, etc.)
  stages/                   # Pipeline stages (ingest, spec, graph, codegen, testgen)
  llm/                      # LLM clients (Anthropic, LangChain, Claude Agent SDK)
  graph/                    # Neo4j client, schema, repository
  agents/
    swarm.py                # Per-feature swarm (Planner/Coder/Reviewer/Tester)
    orchestrator.py         # Parent orchestrator (topo-sort, parallel dispatch, strategy)
    tools.py                # LangChain tools (graph, file, OpenSpec, memory, eval, RAG)
  evaluation/
    metrics.py              # DeepEval GEval metrics (GPT-5.4 judge)
    adaptive.py             # Adaptive threshold computation from score trends
  memory/
    schema.py               # Memory + EvalResult + Run graph schema
    repository.py           # MemoryRepository (CRUD, search, eval history, run tracking)
  vector/
    client.py               # Qdrant client wrapper
    embeddings.py           # OpenAI text-embedding-3-large service
    collections.py          # Collection creation and indexing
    repository.py           # VectorRepository (upsert/search memories, specs, code)
    merge.py                # Reciprocal Rank Fusion for hybrid search
  openspec/
    parser.py               # Parse OpenSpec spec.md files
    writer.py               # Write OpenSpec artifacts via Jinja2
    templates/              # Jinja2 templates (spec, proposal, design, tasks)
```

## Agentic Behavior Level

This system operates at **L4 (Fully Autonomous / Explorer)** on the [Vellum agentic behavior scale](https://www.vellum.ai/blog/levels-of-agentic-behavior):

| L4 Trait | Implementation |
|----------|---------------|
| **Persist state across sessions** | Neo4j procedural memory + Qdrant embeddings + eval/run history |
| **Refine execution based on feedback** | Eval→memory feedback loop, adaptive thresholds, cross-feature learning, mid-run strategy adjustment |
| **Parallel execution** | Concurrent feature swarms within dependency layers |
| **Real-time adaptation** | Strategy overrides triggered by layer pass rate; cross-feature briefing within same run |

## Development

```bash
# Run tests (98 tests)
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## License

Proprietary.
