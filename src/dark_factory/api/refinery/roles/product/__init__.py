"""Product role — generator (and only generator) in the adversarial panel.

``propose`` is the only verb Product overrides. The synthesis verb
(``defend``) lives on the Judge role; Product's job is to draft a
first-pass specification that the critics will challenge.

Implementation: single-shot LLM call with full shared context, the
same pattern the Critics + Judge use. The prompt lives in
``prompt.py``; the LLM call is delegated through the shared
``_call_llm`` hook (with the production helper at
``roles/_shared/llm.py``) so tests can monkey-patch ``_call_llm`` to
return canned JSON without exercising any network code.
"""

from __future__ import annotations

from typing import Any, ClassVar

import structlog

from dark_factory.api.refinery.roles._shared.json_utils import extract_json_object
from dark_factory.api.refinery.contracts import (
    Draft,
    DraftRelationship,
    DraftSpec,
    MemoryKind,
    RawRequirement,
    RoleContext,
    SuggestedMemoryV2,
)
from dark_factory.api.refinery.models import RefinedRequirement, SuggestedMemory
from dark_factory.api.refinery.roles.base import RoleAgent
from dark_factory.api.refinery.roles.product.prompt import format_product_prompt
from dark_factory.log import trace_methods

log = structlog.get_logger()


@trace_methods
class ProductRole(RoleAgent):
    """Generator role. ``propose`` emits the first draft; no other verbs."""

    role_name: ClassVar[str] = "product"
    default_model: ClassVar[str] = "gpt-5.4"
    default_reasoning_effort: ClassVar[str] = "xhigh"

    @property
    def model(self) -> str:
        return self._model or self.default_model

    @property
    def reasoning_effort(self) -> str:
        return self._reasoning_effort or self.default_reasoning_effort

    def _call_llm(self, prompt: str) -> str:
        """Single-shot LLM call. Tests override this; production routes
        through the shared helper that emits per-call agent log events."""

        from dark_factory.api.refinery.roles._shared.llm import call_refinery_llm

        return call_refinery_llm(
            prompt,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            agent=self.role_name,
        )

    def propose(
        self,
        requirement: RawRequirement,
        context: RoleContext,
    ) -> Draft:
        """Generate a first-pass Draft via one LLM call.

        Reads ``all_requirements`` and ``run_context`` from the
        evidence bag. Any legacy ``SuggestedMemory`` entries the
        generator returns are appended to the
        ``suggested_memories_accumulator`` list when the orchestrator
        provides one (matching the legacy contract).
        """

        ev = context.evidence or {}
        all_requirements: list[dict] = ev.get("all_requirements", []) or []
        run_context: dict | None = ev.get("run_context")

        req_dict = {
            "id": requirement.id,
            "title": requirement.title,
            "description": requirement.description,
            "priority": requirement.priority,
            "tags": list(requirement.tags),
            "source_file": requirement.source_file,
        }

        prompt = format_product_prompt(
            req=req_dict,
            all_requirements=all_requirements,
            run_context=run_context,
        )

        try:
            raw_response = self._call_llm(prompt)
        except Exception as exc:
            log.warning(
                "refinery_product_llm_failed",
                req_id=requirement.id, error=str(exc),
            )
            return _carry_forward_draft(requirement, role_name=self.role_name)

        payload = extract_json_object(raw_response)
        if payload is None:
            log.warning(
                "refinery_product_unparseable",
                req_id=requirement.id, response_preview=raw_response[:200],
            )
            return _carry_forward_draft(requirement, role_name=self.role_name)

        # Pull out memories before mapping to RefinedRequirement (the
        # legacy SuggestedMemory schema doesn't live on RefinedRequirement).
        legacy_memories = _extract_legacy_memories(payload.pop("suggested_memories", None))
        if legacy_memories:
            buf = ev.get("suggested_memories_accumulator")
            if isinstance(buf, list):
                buf.extend(legacy_memories)

        try:
            refined = RefinedRequirement(**payload)
        except Exception as exc:
            log.warning(
                "refinery_product_invalid_payload",
                req_id=requirement.id, error=str(exc),
            )
            return _carry_forward_draft(requirement, role_name=self.role_name)

        return _refined_to_draft(refined, produced_by=self.role_name)




def _extract_legacy_memories(raw: Any) -> list[SuggestedMemory]:
    """Coerce the LLM's ``suggested_memories`` field into a list of
    legacy ``SuggestedMemory`` records, dropping any entry that fails
    validation rather than blowing up the whole draft."""

    if not isinstance(raw, list):
        return []
    out: list[SuggestedMemory] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(SuggestedMemory(**entry))
        except Exception:
            continue
    return out


def _refined_to_draft(refined: RefinedRequirement, *, produced_by: str) -> Draft:
    """Map the RefinedRequirement shape onto the Draft contract."""

    return Draft(
        requirement_id=refined.id,
        title=refined.title,
        description=refined.description,
        priority=refined.priority,
        tags=list(refined.tags),
        suggested_specs=[
            DraftSpec(
                title=s.title,
                capability=s.capability,
                description=s.description,
                acceptance_criteria=list(s.acceptance_criteria),
            )
            for s in refined.suggested_specs
        ],
        relationships=[
            DraftRelationship(
                target_id=r.target_id,
                type=r.type,
                rationale=r.rationale,
            )
            for r in refined.relationships
        ],
        produced_by=produced_by,
        iteration=0,
    )


def _carry_forward_draft(requirement: RawRequirement, *, role_name: str) -> Draft:
    """Produce a no-op Draft when the LLM call fails or the response
    can't be parsed, so the debate graph still has something to score
    instead of crashing the round."""

    return Draft(
        requirement_id=requirement.id,
        title=requirement.title,
        description=requirement.description,
        priority=requirement.priority,
        tags=list(requirement.tags),
        produced_by=role_name,
        iteration=0,
    )


def legacy_memory_to_v2(legacy: SuggestedMemory, *, source_role: str) -> SuggestedMemoryV2:
    """Upgrade a legacy ``SuggestedMemory`` (type=pattern|strategy) to the
    v2 contract with ``kind`` populated. Both legacy types are reusable
    reasoning patterns, hence map to ``MemoryKind.PATTERN``."""

    return SuggestedMemoryV2(
        kind=MemoryKind.PATTERN,
        summary=(legacy.description or "")[:120],
        body=legacy.description or "",
        context=legacy.context or "",
        applicability=legacy.applicability or "",
        source_role=source_role,
        rationale=legacy.rationale or "",
    )
