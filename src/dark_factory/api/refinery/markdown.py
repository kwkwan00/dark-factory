"""SnakeMD-based markdown rendering for refinery reports and requirements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import snakemd

if TYPE_CHECKING:
    from dark_factory.api.refinery.models import (
        RefinedRequirement,
        RefineryResponse,
        RequirementRelationship,
        SuggestedSpec,
    )


def _req_metadata_paragraph(rr: RefinedRequirement) -> snakemd.Paragraph:
    return snakemd.Paragraph([
        snakemd.Inline("ID: ", bold=True),
        snakemd.Inline(rr.id, code=True),
        " | ",
        snakemd.Inline("Priority: ", bold=True),
        rr.priority,
        " | ",
        snakemd.Inline("Tags: ", bold=True),
        ", ".join(rr.tags) or "—",
    ])


def _add_relationships_table(
    doc: snakemd.Document,
    relationships: list[RequirementRelationship],
) -> None:
    doc.add_table(
        header=["Target", "Type", "Rationale"],
        data=[
            [snakemd.Inline(rel.target_id, code=True), rel.type, rel.rationale]
            for rel in relationships
        ],
    )


def _add_suggested_specs(
    doc: snakemd.Document,
    specs: list[SuggestedSpec],
    heading_level: int = 3,
) -> None:
    for spec in specs:
        doc.add_heading(f"{spec.title} ({spec.capability})", level=heading_level)
        doc.add_paragraph(spec.description)
        if spec.acceptance_criteria:
            doc.add_block(snakemd.Paragraph([snakemd.Inline("Acceptance Criteria:", bold=True)]))
            doc.add_unordered_list(spec.acceptance_criteria)


def render_report_markdown(response: RefineryResponse) -> str:
    """Render the full refinery report as markdown."""
    doc = snakemd.new_doc()
    run_label = response.source_run_id or "uploaded documents"

    doc.add_heading("Requirements Refinery Report", level=1)
    doc.add_block(snakemd.Paragraph([
        snakemd.Inline("Source: ", bold=True), run_label,
    ]))
    doc.add_block(snakemd.Paragraph([
        snakemd.Inline("Stats: ", bold=True),
        f"{response.requirements_modified_count} modified, "
        f"{response.requirements_unchanged_count} unchanged, "
        f"{response.new_relationships_count} relationships discovered",
    ]))

    if response.summary:
        doc.add_heading("Summary", level=2)
        doc.add_paragraph(response.summary)

    if response.methodology:
        doc.add_heading("Methodology", level=2)
        doc.add_paragraph(response.methodology)

    if response.evidence_summary:
        doc.add_heading("Evidence Summary", level=2)
        doc.add_paragraph(response.evidence_summary)

    if response.pass_summaries:
        doc.add_heading("Pass-by-Pass Analysis", level=2)
        doc.add_ordered_list(response.pass_summaries)

    if response.risk_areas:
        doc.add_heading("Risk Areas", level=2)
        doc.add_unordered_list(response.risk_areas)

    if response.refined_requirements:
        doc.add_heading("Refined Requirements", level=2)
        for rr in response.refined_requirements:
            title = rr.title
            if rr.original_title != rr.title:
                title += f"  (was: {rr.original_title})"
            doc.add_heading(title, level=3)
            doc.add_block(_req_metadata_paragraph(rr))
            if rr.changes:
                doc.add_block(snakemd.Paragraph([snakemd.Inline("What changed:", bold=True)]))
                doc.add_unordered_list(rr.changes)
            if rr.pass_context:
                doc.add_block(snakemd.Paragraph([
                    snakemd.Inline("Rationale: ", bold=True),
                    rr.pass_context,
                ]))
            doc.add_block(snakemd.Paragraph([snakemd.Inline("Description:", bold=True)]))
            doc.add_paragraph(rr.description)
            if rr.original_description != rr.description:
                doc.add_raw("<details><summary>Original description</summary>")
                doc.add_raw("")
                doc.add_paragraph(rr.original_description)
                doc.add_raw("")
                doc.add_raw("</details>")
            if rr.relationships:
                doc.add_block(snakemd.Paragraph([snakemd.Inline("Relationships:", bold=True)]))
                _add_relationships_table(doc, rr.relationships)
            if rr.suggested_specs:
                doc.add_block(snakemd.Paragraph([snakemd.Inline("Suggested Specs:", bold=True)]))
                _add_suggested_specs(doc, rr.suggested_specs, heading_level=4)
            doc.add_horizontal_rule()

    if response.suggested_memories:
        doc.add_heading("Suggested Memories", level=2)
        doc.add_paragraph(
            "These insights were identified during refinement and can be saved as "
            "procedural memories for future planner agents."
        )
        for mem in response.suggested_memories:
            doc.add_block(snakemd.Paragraph([
                snakemd.Inline(f"[{mem.type}]", bold=True),
                f" {mem.description}",
            ]))
            details: list[str] = []
            if mem.rationale:
                details.append(f"Rationale: {mem.rationale}")
            if mem.applicability:
                details.append(f"Applicability: {mem.applicability}")
            if details:
                doc.add_unordered_list(details)

    return str(doc)


def render_requirement_markdown(rr: RefinedRequirement) -> str:
    """Render a single refined requirement as a standalone markdown file."""
    doc = snakemd.new_doc()
    doc.add_heading(rr.title, level=1)
    doc.add_block(_req_metadata_paragraph(rr))
    doc.add_paragraph(rr.description)
    if rr.relationships:
        doc.add_heading("Relationships", level=2)
        _add_relationships_table(doc, rr.relationships)
    if rr.suggested_specs:
        doc.add_heading("Suggested Specs", level=2)
        _add_suggested_specs(doc, rr.suggested_specs, heading_level=3)
    return str(doc)
