"""Clean-context deep-agent extraction of requirements from rich documents.

Any file the frontend lets the user upload that the existing
IngestStage cannot natively parse (Word, Excel, PDF, HTML, XML, RTF,
transcript logs, CSV, ...) is routed through a **multi-turn agentic
tool-use loop** (direct Anthropic API) per document. Each invocation
runs in total isolation from the main swarm — a fresh conversation
with its own sandbox directory, no shared memory — so the noise of
raw meeting transcripts and Excel rows never pollutes the
Planner/Coder/Reviewer/Tester agents that produce the actual code.

The agentic loop has access to ``Read``, ``Write``, ``Edit``,
``Glob``, ``Grep``, and ``Bash``, sandboxed to the upload directory.
For binary formats it uses short ``python -c`` one-liners that call
the bundled extractor libraries (``python-docx``, ``openpyxl``,
``pypdf``, ``beautifulsoup4``, ``striprtf``, ``lxml``). It writes
the extracted requirements as a JSON array to a staging file next to
the source document, and this module loads that staging file back
into :class:`Requirement` models.

Failures are best-effort: if the agent crashes, times out, or
produces unparseable output, the document is logged and skipped — the
pipeline continues with whatever could be extracted from the other
uploaded documents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import structlog

from dark_factory.agents.cancellation import PipelineCancelled
from dark_factory.models.domain import Priority, Requirement

log = structlog.get_logger()


# Per-document ceilings. The deep agent runs serially per file, so a
# tight turn budget is important to keep a 20-document upload from
# taking an hour. 25 turns is enough for the inventory + extraction +
# write cycle even on a complex multi-sheet spreadsheet, and 5 minutes
# hard-caps the worst case.
DEEP_EXTRACTION_MAX_TURNS = 25
DEEP_EXTRACTION_TIMEOUT_SECONDS = 300.0


from dark_factory.prompts import get_prompt

_EXTRACTION_PROMPT_TEMPLATE = get_prompt("doc_extraction", "system")


def _safe_filename_for_prompt(source: Path) -> str:
    """Return a sanitised, prompt-safe version of the source filename.

    The uploaded filename is untrusted input — an attacker who
    controls the filename can inject instructions into the extraction
    agent's prompt context. We generate a deterministic synthetic
    name based on the content hash of the original path so the
    prompt never carries the user-supplied characters.

    The synthetic name preserves the original extension (which is
    already validated by the upload allowlist) so the extension-based
    dispatch hints in the prompt still work.
    """
    import hashlib

    hashed = hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:12]
    ext = source.suffix.lower()
    return f"doc_{hashed}{ext}"


def _staging_filename_for(source: Path) -> str:
    """Return the conventional staging filename used by the deep agent.

    We keep the extracted output next to the original document in the
    same upload directory so that (a) cleanup via ``shutil.rmtree`` on
    the upload dir removes it automatically and (b) the ingest stage
    can find it by convention rather than by parsing agent output.

    Uses the sanitised synthetic filename so a malicious original
    filename like ``../../../etc/passwd`` can't end up written back
    through the staging filename path.
    """
    return f".{_safe_filename_for_prompt(source)}.requirements.json"


def _build_prompt(
    source: Path,
    staging_filename: str,
    *,
    safe_filename: str | None = None,
) -> str:
    """Build the extraction prompt with a sanitised filename.

    The prompt no longer carries the absolute path or the original
    filename — both are untrusted and were a prompt-injection vector.
    Instead it uses a content-addressed synthetic filename; the
    agent accesses the document via the synthetic name through a
    symlink / rename done by the caller before invoking the agent.
    """
    effective_name = safe_filename or _safe_filename_for_prompt(source)
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        safe_filename=effective_name,
        extension=source.suffix.lower(),
        staging_filename=staging_filename,
    )


def _parse_staging_file(staging_path: Path, source: Path) -> list[Requirement]:
    """Load and validate the JSON array the deep agent wrote.

    Malformed JSON, missing fields, and blank entries are logged and
    skipped individually so one bad row cannot kill the whole file.
    """
    try:
        raw = staging_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        log.warning(
            "doc_extraction_staging_read_failed",
            file=str(source),
            staging=str(staging_path),
            error=str(exc),
        )
        return []

    # Some agents insist on wrapping JSON in a code fence even when
    # told not to — try to strip one fence layer before giving up.
    stripped = raw.strip()
    if stripped.startswith("```"):
        # Drop the first line and the trailing fence if present.
        parts = stripped.split("\n", 1)
        if len(parts) == 2:
            stripped = parts[1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")].rstrip()

    try:
        data: Any = json.loads(stripped)
    except json.JSONDecodeError as exc:
        log.warning(
            "doc_extraction_staging_invalid_json",
            file=str(source),
            staging=str(staging_path),
            error=str(exc),
            preview=stripped[:200],
        )
        return []

    if not isinstance(data, list):
        log.warning(
            "doc_extraction_staging_not_a_list",
            file=str(source),
            staging=str(staging_path),
            type=type(data).__name__,
        )
        return []

    requirements: list[Requirement] = []
    seen: set[tuple[str, str]] = set()
    stem = source.stem
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            log.warning(
                "doc_extraction_item_not_a_dict",
                file=str(source),
                index=i,
                type=type(item).__name__,
            )
            continue

        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if not title or not description:
            log.warning(
                "doc_extraction_item_blank",
                file=str(source),
                index=i,
                has_title=bool(title),
                has_description=bool(description),
            )
            continue

        # Dedupe within a single document on normalised (title,
        # description). Cross-document dedup is intentionally NOT
        # done here — that's the graph stage's job.
        key = (title.lower(), description.lower())
        if key in seen:
            log.debug("doc_extraction_item_duplicate", file=str(source), title=title)
            continue
        seen.add(key)

        # Priority with graceful fallback.
        raw_priority = str(item.get("priority") or "medium").strip().lower()
        try:
            priority = Priority(raw_priority)
        except ValueError:
            log.warning(
                "doc_extraction_invalid_priority",
                file=str(source),
                index=i,
                raw_priority=raw_priority,
                fallback="medium",
            )
            priority = Priority.MEDIUM

        # Tag list — coerce stringly, then add the file stem so
        # downstream filtering can distinguish requirements by source.
        tags_raw = item.get("tags") or []
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        if stem and stem not in tags:
            tags.append(stem)

        # Content-addressed id — matches the ingest LLM-splitter
        # convention so re-uploading an unchanged document produces the
        # same requirement ids and downstream spec generation is
        # idempotent.
        hash_input = f"{title.lower()}\n{description.lower()}".encode()
        req_id = hashlib.sha256(hash_input).hexdigest()[:16]

        requirements.append(
            Requirement(
                id=req_id,
                title=title,
                description=description,
                source_file=str(source),
                priority=priority,
                tags=tags,
            )
        )

    return requirements


def extract_with_deep_agent(source: Path) -> list[Requirement]:
    """Extract requirements from a single rich document via a deep agent.

    Runs a multi-turn agentic tool-use loop (direct Anthropic API)
    with the sandbox set to the directory containing ``source``. The
    agent reads the document, extracts requirements, and writes a
    staging JSON file that this function then parses back into
    :class:`Requirement` models.

    Returns an empty list on any failure (agent crash, timeout,
    missing staging file, invalid JSON) after logging the details —
    the pipeline treats rich-document extraction as best-effort.
    """
    source = source.resolve()
    if not source.is_file():
        log.warning("doc_extraction_source_missing", file=str(source))
        return []

    # H2 hardening: the uploaded filename is untrusted input — a
    # malicious user could embed prompt-injection instructions in
    # the filename itself (e.g. ``ignore_previous_instructions.txt``).
    # We copy the document to a content-addressed synthetic filename
    # so the agent's prompt + working directory never reference the
    # original name. The synthetic copy lives alongside the original
    # in the same upload dir, gets parsed into the staging file, and
    # is cleaned up at the end.
    safe_filename = _safe_filename_for_prompt(source)
    safe_copy_path = source.parent / safe_filename
    if source.name != safe_filename:
        try:
            # Use a hardlink when possible (same filesystem) so we
            # don't double-count disk usage; fall back to a copy.
            import shutil

            if safe_copy_path.exists():
                safe_copy_path.unlink()
            try:
                _os = __import__("os")
                _os.link(str(source), str(safe_copy_path))
            except Exception:
                shutil.copy2(str(source), str(safe_copy_path))
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "doc_extraction_safe_copy_failed",
                file=str(source),
                error=str(exc),
            )
            # Fall back to using the original name, but log the
            # security-relevant failure so operators can investigate.
            safe_filename = source.name
            safe_copy_path = source

    staging_filename = _staging_filename_for(source)
    staging_path = source.parent / staging_filename

    # Clean any stale staging file from a previous (failed) run so we
    # can't accidentally parse yesterday's output.
    if staging_path.exists():
        try:
            staging_path.unlink()
        except Exception:  # pragma: no cover — defensive
            pass

    prompt = _build_prompt(
        source, staging_filename, safe_filename=safe_filename
    )

    log.info(
        "doc_extraction_starting",
        file=str(source),
        extension=source.suffix.lower(),
        max_turns=DEEP_EXTRACTION_MAX_TURNS,
        timeout_seconds=DEEP_EXTRACTION_TIMEOUT_SECONDS,
    )

    # Run the extraction via direct Anthropic API agentic loop.
    # The sandbox_root is the upload directory so Read/Write/Bash
    # paths resolve correctly against the uploaded document.
    from dark_factory.llm.agentic import run_agentic_loop

    try:
        try:
            agent_output = run_agentic_loop(
                prompt=prompt,
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
                sandbox_root=source.parent,
                max_turns=DEEP_EXTRACTION_MAX_TURNS,
                timeout_seconds=DEEP_EXTRACTION_TIMEOUT_SECONDS,
            )
        except PipelineCancelled:
            # B2 fix: cooperative cancellation must propagate, not be
            # swallowed by the best-effort ``except Exception`` below.
            raise
        except Exception as exc:
            log.warning(
                "doc_extraction_agent_failed",
                file=str(source),
                error=str(exc),
            )
            return []
    finally:
        # Clean up the synthetic safe-copy so we don't leave
        # duplicates next to the originals. The staging JSON
        # (written by the agent) is left in place — the ingest
        # stage's directory walker filters it out via the
        # ``.requirements.json`` suffix.
        if safe_copy_path != source and safe_copy_path.exists():
            try:
                safe_copy_path.unlink()
            except Exception:  # pragma: no cover — defensive
                pass

    if not staging_path.exists():
        log.warning(
            "doc_extraction_no_staging_file",
            file=str(source),
            staging=str(staging_path),
            agent_output_preview=(agent_output or "")[:300],
        )
        return []

    requirements = _parse_staging_file(staging_path, source)
    log.info(
        "doc_extraction_complete",
        file=str(source),
        extracted=len(requirements),
    )
    return requirements
