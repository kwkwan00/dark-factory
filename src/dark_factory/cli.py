"""CLI entry point for dark-factory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import structlog

from dark_factory.config import load_settings
from dark_factory.log import setup_logging


@click.group()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="Path to config.toml")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """AI Dark Factory - Requirements to code via knowledge graph."""
    settings = load_settings(Path(config_path) if config_path else None)
    setup_logging(level=settings.logging.level, fmt=settings.logging.format)
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


@main.command()
@click.argument("requirements_path", type=click.Path(exists=True))
@click.pass_context
def run(ctx: click.Context, requirements_path: str) -> None:
    """Run the per-feature swarm pipeline on a requirements document or directory."""
    _run_swarm(ctx.obj["settings"], requirements_path)


def _run_swarm(settings, requirements_path):
    """Run the swarm pipeline: ingest → spec → graph, then per-feature orchestrator."""
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.graph.repository import GraphRepository
    from dark_factory.models.domain import PipelineContext
    from dark_factory.stages.graph import GraphStage
    from dark_factory.stages.ingest import IngestStage
    from dark_factory.stages.spec import SpecStage

    log = structlog.get_logger()
    llm = _build_llm(settings)
    neo4j_client = Neo4jClient(settings.neo4j)
    repo = GraphRepository(neo4j_client)

    try:
        # Run ingest → spec → graph to produce specs
        ctx_pipeline = PipelineContext(input_path=requirements_path)
        ctx_pipeline = IngestStage().run(ctx_pipeline)
        ctx_pipeline = SpecStage(llm=llm).run(ctx_pipeline)
        ctx_pipeline = GraphStage(repo=repo).run(ctx_pipeline)

        spec_ids = [s.id for s in ctx_pipeline.specs]
        log.info("orchestrator_starting", specs=len(spec_ids))

        # Hand off to the per-feature orchestrator
        from dark_factory.agents.orchestrator import run_orchestrator

        result = run_orchestrator(settings, spec_ids=spec_ids)
        completed = result.get("completed_features", [])
        click.echo(
            f"Orchestrator complete: {len(completed)} features, "
            f"{len(result.get('all_artifacts', []))} artifacts, "
            f"{len(result.get('all_tests', []))} tests"
        )
    finally:
        neo4j_client.close()


@main.group()
def graph() -> None:
    """Manage the Neo4j knowledge graph."""


@graph.command("init")
@click.pass_context
def graph_init(ctx: click.Context) -> None:
    """Create graph schema constraints and indexes."""
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.graph.schema import init_schema

    settings = ctx.obj["settings"]
    client = Neo4jClient(settings.neo4j)
    try:
        init_schema(client)
        click.echo("Graph schema initialized.")
    finally:
        client.close()


@graph.command("clear")
@click.confirmation_option(prompt="This will delete all graph data. Continue?")
@click.pass_context
def graph_clear(ctx: click.Context) -> None:
    """Delete all nodes and relationships from the graph."""
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.graph.schema import clear_graph

    settings = ctx.obj["settings"]
    client = Neo4jClient(settings.neo4j)
    try:
        clear_graph(client)
        click.echo("Graph cleared.")
    finally:
        client.close()


@main.group("memory")
@click.pass_context
def memory_group(ctx: click.Context) -> None:
    """Manage procedural memory."""


@memory_group.command("clear")
@click.confirmation_option(prompt="This will delete all procedural memory. Continue?")
@click.pass_context
def memory_clear(ctx: click.Context) -> None:
    """Delete all procedural memory nodes and relationships."""
    from dark_factory.config import Neo4jConfig
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.memory.schema import clear_memory

    settings = ctx.obj["settings"]
    mem_config = Neo4jConfig(
        uri=settings.neo4j.uri,
        database=settings.memory.database,
        user=settings.neo4j.user,
        password=settings.neo4j.password,
    )
    client = Neo4jClient(mem_config)
    try:
        clear_memory(client)
        click.echo("Memory cleared.")
    finally:
        client.close()


@main.group("openspec")
@click.pass_context
def openspec_group(ctx: click.Context) -> None:
    """Manage OpenSpec specifications."""


@openspec_group.command("init")
@click.pass_context
def openspec_init(ctx: click.Context) -> None:
    """Initialize the OpenSpec directory structure."""
    from dark_factory.openspec.writer import init_openspec_dir

    settings = ctx.obj["settings"]
    root = Path(settings.openspec.root_dir)
    init_openspec_dir(root)
    click.echo(f"OpenSpec directory initialized at {root}/")


@openspec_group.command("propose")
@click.argument("description")
@click.option("--input", "input_path", default=None, type=click.Path(exists=True), help="Input requirements path (default: openspec root)")
@click.option("--change-name", default=None, help="Name for the change (auto-generated if omitted)")
@click.pass_context
def openspec_propose(ctx: click.Context, description: str, input_path: str | None, change_name: str | None) -> None:
    """Generate an OpenSpec change proposal from requirements."""
    import re
    from datetime import datetime, timezone

    from dark_factory.graph.client import Neo4jClient
    from dark_factory.graph.repository import GraphRepository
    from dark_factory.models.domain import PipelineContext
    from dark_factory.openspec import writer
    from dark_factory.stages.graph import GraphStage
    from dark_factory.stages.ingest import IngestStage
    from dark_factory.stages.spec import SpecStage

    settings = ctx.obj["settings"]
    root = Path(settings.openspec.root_dir)

    if input_path is None:
        input_path = str(root)

    if change_name is None:
        slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:40]
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        change_name = f"{ts}-{slug}"

    log = structlog.get_logger()
    llm = _build_llm(settings)
    neo4j_client = Neo4jClient(settings.neo4j)
    repo = GraphRepository(neo4j_client)

    try:
        # Run ingest → spec → graph stages
        ctx_pipeline = PipelineContext(input_path=input_path)
        ctx_pipeline = IngestStage().run(ctx_pipeline)
        ctx_pipeline = SpecStage(llm=llm).run(ctx_pipeline)
        ctx_pipeline = GraphStage(repo=repo).run(ctx_pipeline)

        specs = ctx_pipeline.specs

        # Write OpenSpec artifacts
        for spec in specs:
            path = writer.write_spec_md(spec, root)
            click.echo(f"  spec: {path}")

        writer.write_proposal(change_name, specs, description, root)
        writer.write_design(change_name, specs, root)
        writer.write_tasks(change_name, specs, root)
        writer.write_change_specs(change_name, specs, root)

        click.echo(f"\nChange proposal created: {root}/changes/{change_name}/")
        log.info("propose_complete", change=change_name, specs=len(specs))
    finally:
        neo4j_client.close()


@openspec_group.command("apply")
@click.option("--change", default=None, help="Change name to apply (default: most recent)")
@click.pass_context
def openspec_apply(ctx: click.Context, change: str | None) -> None:
    """Apply a change proposal — generate code and tests from specs."""
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.graph.repository import GraphRepository
    from dark_factory.models.domain import PipelineContext
    from dark_factory.openspec.parser import parse_openspec_specs
    from dark_factory.stages.codegen import CodegenStage
    from dark_factory.stages.testgen import TestgenStage

    settings = ctx.obj["settings"]
    root = Path(settings.openspec.root_dir)

    # Find the change to apply
    changes_dir = root / "changes"
    if change is None:
        if not changes_dir.is_dir():
            click.echo("No changes directory found. Run 'openspec propose' first.", err=True)
            sys.exit(1)
        candidates = sorted(changes_dir.iterdir(), reverse=True)
        if not candidates:
            click.echo("No pending changes found.", err=True)
            sys.exit(1)
        change = candidates[0].name

    change_specs_dir = changes_dir / change / "specs"
    if not change_specs_dir.is_dir():
        # Fall back to reading from the main specs directory
        specs = parse_openspec_specs(root)
    else:
        specs = parse_openspec_specs(changes_dir / change)

    if not specs:
        click.echo(f"No specs found for change '{change}'.", err=True)
        sys.exit(1)

    log = structlog.get_logger()
    llm = _build_llm(settings)
    neo4j_client = Neo4jClient(settings.neo4j)
    repo = GraphRepository(neo4j_client)
    output_dir = settings.pipeline.output_dir

    try:
        ctx_pipeline = PipelineContext(input_path=str(root))
        ctx_pipeline.specs = specs

        ctx_pipeline = CodegenStage(llm=llm, repo=repo, output_dir=output_dir).run(ctx_pipeline)
        ctx_pipeline = TestgenStage(llm=llm, output_dir=output_dir).run(ctx_pipeline)

        click.echo(f"\nApplied change '{change}': {len(ctx_pipeline.artifacts)} artifacts, {len(ctx_pipeline.tests)} tests")
        log.info("apply_complete", change=change, artifacts=len(ctx_pipeline.artifacts), tests=len(ctx_pipeline.tests))
    finally:
        neo4j_client.close()


@openspec_group.command("archive")
@click.argument("change")
@click.pass_context
def openspec_archive(ctx: click.Context, change: str) -> None:
    """Archive a completed change proposal."""
    from dark_factory.openspec.writer import archive_change

    settings = ctx.obj["settings"]
    root = Path(settings.openspec.root_dir)
    dest = archive_change(change, root)
    click.echo(f"Archived: {dest}")


def _build_llm(settings):
    """Instantiate the configured LLM client."""
    if settings.llm.provider == "anthropic":
        from dark_factory.llm.anthropic import AnthropicClient

        return AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=settings.llm.model,
        )
    elif settings.llm.provider == "langchain":
        from dark_factory.llm.langchain import LangChainClient

        return LangChainClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=settings.llm.model,
        )
    elif settings.llm.provider == "claude-agent":
        from dark_factory.llm.claude_code import ClaudeAgentClient

        return ClaudeAgentClient(model=settings.llm.model)
    else:
        click.echo(f"Unknown LLM provider: {settings.llm.provider}. Use 'anthropic', 'langchain', or 'claude-agent'.", err=True)
        sys.exit(1)
