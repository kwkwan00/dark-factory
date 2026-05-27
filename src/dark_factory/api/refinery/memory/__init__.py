"""Refinery institutional memory layer.

Wire-up:

- ``producers`` — hooks that fire on each debate event per the plan's
  lifecycle table (rebuttal → Decision; blocker critique → Constraint
  or Incident; finalize with high disagreement → Conflict; short-
  circuit reconcile → Conflict unconditionally).
- ``writers`` — thin wrappers around ``MemoryRepository.record_*`` so
  the debate graph nodes have a single place to call.
"""

from dark_factory.api.refinery.memory.producers import (
    MemoryProducerResult,
    produce_anti_pattern_from_blocker,
    produce_conflict_on_short_circuit,
    produce_conflict_on_disagreement,
    produce_constraint_from_blocker,
    produce_decision_from_rebuttal,
    produce_hypotheses_from_open_questions,
    produce_incident_from_blocker,
)

__all__ = [
    "MemoryProducerResult",
    "produce_anti_pattern_from_blocker",
    "produce_conflict_on_short_circuit",
    "produce_conflict_on_disagreement",
    "produce_constraint_from_blocker",
    "produce_decision_from_rebuttal",
    "produce_hypotheses_from_open_questions",
    "produce_incident_from_blocker",
]
