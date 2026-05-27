"""Persistence CRUD for refinery results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import structlog

from dark_factory.api.refinery.markdown import render_report_markdown
from dark_factory.api.refinery.models import RefineryResponse

log = structlog.get_logger()


def generate_refinery_id() -> str:
    """Generate a unique refinery result ID."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"refinery-{ts}-{uuid4().hex[:4]}"


def save_refinery_result(
    result_id: str,
    response: RefineryResponse,
    duration_seconds: float,
    source_mode: str = "run",
) -> None:
    """Persist a refinery result to the storage backend (local + S3)."""
    from dark_factory.storage.backend import get_storage

    storage = get_storage()
    prefix = f"refinery/{result_id}"

    metadata = {
        "id": result_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "source_run_id": response.source_run_id,
        "source_mode": source_mode,
        "requirements_count": len(response.refined_requirements),
        "requirements_modified": response.requirements_modified_count,
        "requirements_unchanged": response.requirements_unchanged_count,
        "relationships_count": response.new_relationships_count,
        "suggested_memories_count": len(response.suggested_memories),
        "duration_seconds": round(duration_seconds, 1),
    }

    try:
        storage.write_text(f"{prefix}/metadata.json", json.dumps(metadata, indent=2))
        storage.write_text(f"{prefix}/response.json", json.dumps(response.model_dump(), indent=2, default=str))
        storage.write_text(f"{prefix}/REPORT.md", render_report_markdown(response))
        log.info("refinery_result_saved", result_id=result_id)
    except Exception as exc:
        log.warning("refinery_result_save_failed", result_id=result_id, error=str(exc))

    # Per-requirement Debate Episodes — written as exportable documentation
    # artifacts (.md) and structured records (.json). Best-effort: a failure
    # here must not block the response.json write above.
    episodes_written = 0
    for refined in response.refined_requirements:
        debate = refined.debate or {}
        episode = debate.get("episode") if isinstance(debate, dict) else None
        markdown = debate.get("episode_markdown") if isinstance(debate, dict) else None
        if not isinstance(episode, dict):
            continue
        try:
            storage.write_text(
                f"{prefix}/episodes/{refined.id}.json",
                json.dumps(episode, indent=2, default=str),
            )
            if isinstance(markdown, str) and markdown:
                storage.write_text(
                    f"{prefix}/episodes/{refined.id}.md", markdown,
                )
            episodes_written += 1
        except Exception as exc:  # pragma: no cover — best-effort
            log.warning(
                "refinery_episode_save_failed",
                result_id=result_id, requirement_id=refined.id, error=str(exc),
            )
    if episodes_written:
        log.info(
            "refinery_episodes_saved",
            result_id=result_id, count=episodes_written,
        )


def list_refinery_results(limit: int = 20) -> list[dict]:
    """List historical refinery results from storage, newest first."""
    from dark_factory.storage.backend import get_storage

    storage = get_storage()
    try:
        keys = storage.list_keys("refinery/")
    except Exception:
        return []

    results: list[dict] = []
    seen: set[str] = set()
    for key in keys:
        if not key.endswith("/metadata.json"):
            continue
        result_id = key.split("/")[1]
        if result_id in seen:
            continue
        seen.add(result_id)
        try:
            raw = storage.read_text(key)
            results.append(json.loads(raw))
        except Exception:
            continue

    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results[:limit]


def load_refinery_result(result_id: str) -> RefineryResponse | None:
    """Load a saved refinery result from storage."""
    from dark_factory.storage.backend import get_storage

    storage = get_storage()
    try:
        raw = storage.read_text(f"refinery/{result_id}/response.json")
        return RefineryResponse(**json.loads(raw))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("refinery_result_load_failed", result_id=result_id, error=str(exc))
        return None


def delete_refinery_result(result_id: str) -> bool:
    """Delete a refinery result from storage."""
    from dark_factory.storage.backend import get_storage

    storage = get_storage()
    try:
        storage.delete_prefix(f"refinery/{result_id}/")
        log.info("refinery_result_deleted", result_id=result_id)
        return True
    except Exception as exc:
        log.warning("refinery_result_delete_failed", result_id=result_id, error=str(exc))
        return False
