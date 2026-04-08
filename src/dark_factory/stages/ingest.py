"""Stage 1: Parse requirements documents into Requirement models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import structlog

from dark_factory.models.domain import PipelineContext, Requirement
from dark_factory.stages.base import Stage

log = structlog.get_logger()


class IngestStage(Stage):
    name = "ingest"

    def run(self, context: PipelineContext) -> PipelineContext:
        input_path = Path(context.input_path)

        # Auto-detect OpenSpec directory structure
        if input_path.is_dir() and (input_path / "specs").is_dir():
            from dark_factory.openspec.parser import parse_openspec_dir

            log.info("detected_openspec", path=str(input_path))
            context.requirements = parse_openspec_dir(input_path)
            log.info("ingest_complete", count=len(context.requirements))
            return context

        if input_path.is_file():
            files = [input_path]
        elif input_path.is_dir():
            files = sorted(
                p
                for p in input_path.iterdir()
                if p.suffix in (".md", ".txt", ".json")
            )
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        requirements: list[Requirement] = []
        for f in files:
            log.info("ingesting", file=str(f))
            reqs = self._parse_file(f)
            requirements.extend(reqs)

        log.info("ingest_complete", count=len(requirements))
        context.requirements = requirements
        return context

    def _parse_file(self, path: Path) -> list[Requirement]:
        """Parse a file into requirements. JSON files are parsed structurally; others as single requirements."""
        if path.suffix == ".json":
            return self._parse_json(path)
        return self._parse_text(path)

    def _parse_json(self, path: Path) -> list[Requirement]:
        data = json.loads(path.read_text())
        items = data if isinstance(data, list) else [data]
        reqs = []
        for item in items:
            item.setdefault("source_file", str(path))
            reqs.append(Requirement(**item))
        return reqs

    def _parse_text(self, path: Path) -> list[Requirement]:
        content = path.read_text().strip()
        if not content:
            return []
        req_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        return [
            Requirement(
                id=req_id,
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                description=content,
                source_file=str(path),
            )
        ]
