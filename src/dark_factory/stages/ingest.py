"""Stage 1: Parse requirements documents into Requirement models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import structlog
from pydantic import BaseModel, Field

from dark_factory.models.domain import PipelineContext, Priority, Requirement
from dark_factory.stages.base import Stage
from dark_factory.stages.dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    DedupeResult,
    semantically_dedupe,
)

if TYPE_CHECKING:
    from dark_factory.llm.base import LLMClient

log = structlog.get_logger()


# Threshold above which we try to split a document into multiple requirements
SPLIT_THRESHOLD_CHARS = 1500

# Maximum characters to send to the LLM splitter. Documents larger than this
# are truncated with a warning — true map-reduce chunking is future work.
MAX_SPLIT_INPUT_CHARS = 40_000


# File extensions the ingest stage parses natively. Everything else
# that the upload endpoint accepts is routed through the clean-context
# deep agent in :mod:`dark_factory.stages.doc_extraction`.
NATIVE_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}
RICH_EXTENSIONS = {
    ".docx",
    ".xlsx",
    ".pptx",
    ".pdf",
    ".rtf",
    ".html",
    ".htm",
    ".xml",
    ".csv",
    ".vtt",
    ".srt",
    ".log",
}
SUPPORTED_EXTENSIONS = NATIVE_EXTENSIONS | RICH_EXTENSIONS


class _ExtractedRequirement(BaseModel):
    """A single requirement extracted by the LLM splitter."""

    title: str = Field(description="Short, specific feature name")
    description: str = Field(
        description="What the system shall do, including acceptance criteria"
    )
    priority: str = Field(
        default="medium",
        description="Priority: 'critical', 'high', 'medium', or 'low'",
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags such as feature area, priority, etc."
    )


class _RequirementList(BaseModel):
    """Structured output: a list of extracted requirements."""

    requirements: list[_ExtractedRequirement]


_SPLITTER_SYSTEM_PROMPT = """\
You are a senior requirements analyst. You extract DISCRETE, TESTABLE features \
from product requirements documents.

Each extracted requirement must be:
- Atomic: a single feature, user story, or capability (not a group)
- Testable: concrete enough that acceptance criteria can be written
- Independent: can be understood and implemented without reference to other items
- Scoped: focused on one behavior, not a whole subsystem

When a document contains tables, numbered lists, or user stories, extract each \
row/item as its own requirement. When a document describes a single large \
feature, split it into its logical sub-components.

Do NOT extract:
- Section headers that are just labels
- Metadata (version, author, date)
- Goals or non-goals (these are context, not requirements)
- Success metrics (these are measurements, not requirements)
- Open questions or future considerations

Return ALL discrete requirements in the document."""


class IngestStage(Stage):
    name = "ingest"

    def __init__(
        self,
        llm: "LLMClient | None" = None,
        *,
        strict_split: bool = False,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        """
        :param llm: optional LLM client used to split large text
            documents into multiple granular requirements.
        :param strict_split: if True (M8), a splitter failure re-raises
            instead of silently falling back to a single-requirement
            output.
        :param embed_fn: callable that takes ``list[str]`` and returns
            the matching ``list[list[float]]`` of embeddings. When
            provided, the stage runs a semantic dedup pass over the
            final requirement list before returning — collapsing
            near-duplicates across uploaded documents into one
            canonical entry per cluster. When ``None`` dedup is
            skipped (useful for unit tests that don't care about it
            and for environments without an embedding service).
        :param dedup_threshold: cosine similarity >= this is treated
            as "same requirement". Defaults to the module-level
            ``DEFAULT_DEDUP_THRESHOLD`` (0.90).
        """
        self.llm = llm
        self.strict_split = strict_split
        self.embed_fn = embed_fn
        self.dedup_threshold = dedup_threshold
        # Most recent dedup result, exposed so callers (the AG-UI
        # bridge) can emit progress events describing what was merged
        # without re-running the dedup pass. Cleared at the start of
        # every run() so stale data can't leak between invocations.
        self.last_dedup_result: DedupeResult | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        self.last_dedup_result = None
        input_path = Path(context.input_path)

        # Auto-detect OpenSpec directory structure
        if input_path.is_dir() and (input_path / "specs").is_dir():
            from dark_factory.openspec.parser import parse_openspec_dir

            log.info("detected_openspec", path=str(input_path))
            requirements = parse_openspec_dir(input_path)
        else:
            if input_path.is_file():
                files = [input_path]
            elif input_path.is_dir():
                files = sorted(
                    p
                    for p in input_path.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS
                    # Skip the deep-agent staging files we write ourselves
                    # (they start with a dot and end in
                    # .requirements.json); re-ingesting them would
                    # double-count requirements.
                    and not (
                        p.name.startswith(".")
                        and p.name.endswith(".requirements.json")
                    )
                )
            else:
                raise FileNotFoundError(f"Input path not found: {input_path}")

            requirements = []
            for f in files:
                log.info("ingesting", file=str(f))
                reqs = self._parse_file(f)
                requirements.extend(reqs)

        # ── Semantic dedup ──────────────────────────────────────────
        # Runs before the requirements list is handed to the Spec
        # stage so downstream consumers (spec generation, Neo4j
        # writes, swarm execution) all see the already-deduped list.
        # The dedup pass is a correctness guarantee, not an
        # optimisation — without it, two uploaded documents that
        # describe the same feature produce duplicate specs, burn LLM
        # budget, and pollute the knowledge graph.
        if self.embed_fn is not None and len(requirements) > 1:
            try:
                self.last_dedup_result = semantically_dedupe(
                    requirements,
                    self.embed_fn,
                    threshold=self.dedup_threshold,
                )
                requirements = self.last_dedup_result.requirements
            except Exception as exc:
                # Embedding service outage — log + continue with the
                # un-deduped list. A transient OpenAI hiccup should
                # not take the whole pipeline down.
                log.warning(
                    "requirement_dedup_failed_falling_back",
                    error=str(exc),
                    count=len(requirements),
                )
                self.last_dedup_result = None

        log.info("ingest_complete", count=len(requirements))
        context.requirements = requirements
        return context

    def _parse_file(self, path: Path) -> list[Requirement]:
        """Parse a file into requirements.

        Native formats (.md / .txt / .json / .yaml / .yml) go through
        the existing fast path. Everything else is a "rich" business
        document (Word, Excel, PDF, HTML, XML, RTF, transcript,
        CSV, ...) and is dispatched to the clean-context deep agent
        for extraction — each document gets its own fresh Claude Agent
        SDK invocation so raw meeting transcripts and spreadsheet noise
        never leak into the main pipeline's LLM context.
        """
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._parse_json(path)
        # M2 fix: route YAML through dedicated parser instead of treating as text
        if suffix in (".yaml", ".yml"):
            return self._parse_yaml(path)
        if suffix in (".md", ".txt"):
            return self._parse_text(path)
        if suffix in RICH_EXTENSIONS:
            return self._parse_rich_document(path)
        # Unknown extension — shouldn't happen because the directory
        # walker filters to SUPPORTED_EXTENSIONS, but log + skip
        # defensively if a caller passes a single file with an
        # unexpected suffix.
        log.warning("ingest_unsupported_extension", file=str(path), suffix=suffix)
        return []

    def _parse_rich_document(self, path: Path) -> list[Requirement]:
        """Route a rich business document through the deep-agent extractor.

        Lazy import so unit tests that stub out the deep-agent helper
        don't pull the Claude Agent SDK at module load time.
        """
        from dark_factory.stages.doc_extraction import extract_with_deep_agent

        return extract_with_deep_agent(path)

    def _parse_json(self, path: Path) -> list[Requirement]:
        return self._parse_structured(json.loads(path.read_text(encoding="utf-8")), path)

    def _parse_yaml(self, path: Path) -> list[Requirement]:
        try:
            import yaml  # type: ignore
        except ImportError:
            log.warning("yaml_not_installed_falling_back_to_text", file=str(path))
            return self._parse_text(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            # Log the actual parse error before falling back — without this,
            # "my YAML didn't load" debugging becomes pure guesswork.
            log.warning(
                "yaml_parse_failed_falling_back_to_text",
                file=str(path),
                error=str(exc),
            )
            return self._parse_text(path)
        return self._parse_structured(data, path)

    def _parse_structured(self, data: object, path: Path) -> list[Requirement]:
        """Convert a parsed JSON/YAML payload into Requirement models.

        M3 fix: per-item error handling so a single bad entry doesn't kill
        the whole file. Bad items are logged and skipped.
        """
        if data is None:
            return []
        items = data if isinstance(data, list) else [data]
        reqs: list[Requirement] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                log.warning(
                    "ingest_skipping_non_dict_item",
                    file=str(path),
                    index=i,
                    type=type(item).__name__,
                )
                continue
            item.setdefault("source_file", str(path))
            try:
                reqs.append(Requirement(**item))
            except Exception as exc:
                log.warning(
                    "ingest_item_invalid",
                    file=str(path),
                    index=i,
                    error=str(exc),
                )
        return reqs

    def _parse_text(self, path: Path) -> list[Requirement]:
        """Parse a text/markdown file into one or more requirements.

        For large documents with an LLM available, the content is split into
        multiple granular features. Otherwise the file becomes a single
        requirement.
        """
        content = path.read_text().strip()
        if not content:
            return []

        # Try LLM-powered splitting for larger documents
        if self.llm is not None and len(content) >= SPLIT_THRESHOLD_CHARS:
            try:
                split = self._split_with_llm(content, path)
                if split:
                    log.info(
                        "document_split",
                        file=str(path),
                        chars=len(content),
                        requirements=len(split),
                    )
                    return split
                # Empty result: fall through to single-requirement fallback
                log.warning("llm_split_empty_result", file=str(path))
            except Exception as exc:
                # M8: strict mode re-raises so callers can detect splitter failure
                if self.strict_split:
                    raise
                log.warning(
                    "llm_split_failed_fallback",
                    file=str(path),
                    error=str(exc),
                    exc_info=True,
                )

        # Fallback: treat the whole file as a single requirement
        req_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return [
            Requirement(
                id=req_id,
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                description=content,
                source_file=str(path),
            )
        ]

    def _split_with_llm(self, content: str, path: Path) -> list[Requirement]:
        """Use the LLM to extract discrete requirements from a document."""
        assert self.llm is not None

        # M10 fix: truncate very large documents with a warning. True map-reduce
        # chunking is out of scope for now but can be added here if needed.
        truncated = False
        if len(content) > MAX_SPLIT_INPUT_CHARS:
            log.warning(
                "llm_split_truncating",
                file=str(path),
                original_chars=len(content),
                truncated_chars=MAX_SPLIT_INPUT_CHARS,
            )
            content = content[:MAX_SPLIT_INPUT_CHARS]
            truncated = True

        prompt = (
            f"Extract every discrete, testable requirement from the following "
            f"document. Each requirement must be atomic (one feature per entry). "
            f"If the document contains a requirements table or numbered list, "
            f"each row/item becomes its own requirement.\n\n"
            f"Document:\n{content}"
        )
        if truncated:
            prompt += "\n\n(Note: document was truncated; extract what is visible.)"

        result = self.llm.complete_structured(
            prompt=prompt,
            response_model=_RequirementList,
            system=_SPLITTER_SYSTEM_PROMPT,
        )

        # M9 fix: filter out blank titles and dedupe by (title, description)
        seen: set[tuple[str, str]] = set()
        requirements: list[Requirement] = []
        stem = path.stem
        for i, extracted in enumerate(result.requirements):
            title = (extracted.title or "").strip()
            description = (extracted.description or "").strip()
            if not title or not description:
                log.warning(
                    "llm_split_skipped_blank",
                    index=i,
                    has_title=bool(title),
                    has_description=bool(description),
                )
                continue

            key = (title.lower(), description.lower())
            if key in seen:
                log.debug("llm_split_skipped_duplicate", title=title)
                continue
            seen.add(key)

            # Priority parsing with fallback
            try:
                priority = Priority(extracted.priority.lower())
            except (ValueError, AttributeError):
                # M11 fix: log the invalid value so bad LLM output or enum
                # drift is visible in diagnostics instead of silently
                # collapsing every requirement to MEDIUM.
                log.warning(
                    "llm_split_invalid_priority",
                    file=str(path),
                    index=i,
                    raw_priority=getattr(extracted, "priority", None),
                    fallback="medium",
                )
                priority = Priority.MEDIUM

            # Content-based, position-independent ID. Previously this
            # hashed ``path.name + i + title``, where ``i`` was the LLM
            # splitter's output index. LLM ordering is not bit-stable
            # across calls, so a second run on an unchanged document
            # produced different req_ids and downstream ``spec-{req_id}``
            # ids — causing the spec generation swarm to redo work on
            # logically identical requirements and Neo4j to accumulate
            # duplicate Spec nodes on every re-run. Hashing only the
            # normalised (title, description) tuple means re-runs on
            # unchanged content are idempotent, while edits to either
            # field produce a fresh id (new spec on purpose).
            hash_input = f"{title.strip().lower()}\n{description.strip().lower()}".encode()
            req_id = hashlib.sha256(hash_input).hexdigest()[:16]

            tags = list(extracted.tags)
            # L10: file stem is added as a tag so downstream filtering can
            # distinguish requirements by source document. Weak but useful.
            if stem and stem not in tags:
                tags.append(stem)

            requirements.append(
                Requirement(
                    id=req_id,
                    title=title,
                    description=description,
                    source_file=str(path),
                    priority=priority,
                    tags=tags,
                )
            )
        return requirements
