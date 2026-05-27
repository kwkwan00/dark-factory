"""Requirements Refinery package.

Re-exports all public names so existing imports from
``dark_factory.api.refinery`` continue to work after the
monolith was split into submodules.
"""

from dark_factory.api.refinery.gather import gather_run_context, ingest_requirements
from dark_factory.api.refinery.markdown import render_report_markdown, render_requirement_markdown
from dark_factory.api.refinery.models import (
    ReconciliationReport,
    RefinedRequirement,
    RefineryResponse,
    RequirementPatchRequest,
    RequirementRelationship,
    SuggestedMemory,
    SuggestedSpec,
)
from dark_factory.api.refinery.storage import (
    delete_refinery_result,
    generate_refinery_id,
    list_refinery_results,
    load_refinery_result,
    save_refinery_result,
)
from dark_factory.api.refinery.stream import run_refinery_stream

__all__ = [
    "ReconciliationReport",
    "RefinedRequirement",
    "RefineryResponse",
    "RequirementPatchRequest",
    "RequirementRelationship",
    "SuggestedMemory",
    "SuggestedSpec",
    "delete_refinery_result",
    "gather_run_context",
    "generate_refinery_id",
    "ingest_requirements",
    "list_refinery_results",
    "load_refinery_result",
    "render_report_markdown",
    "render_requirement_markdown",
    "run_refinery_stream",
    "save_refinery_result",
]
