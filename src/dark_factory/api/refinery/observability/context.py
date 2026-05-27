"""TraceContext — threads the active DebateTrace through role-agent calls.

Explicitly passed (not thread-local / not contextvar) because async tasks
share contextvars in ways that would let one feature's trace leak into
another. The orchestrator constructs a TraceContext per requirement and
hands it to every role invocation via ``RoleContext`` or via the
LangGraph config object in Phase 4+.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dark_factory.api.refinery.observability.trace import (
    CallRecord,
    DebateTrace,
    DecisionRecord,
    MemoryAuditEntry,
)


@dataclass
class TraceContext:
    """Active trace for the current requirement's debate. Role modules use
    ``record_call()`` as a context manager around each LLM invocation:

        with run_context.trace.record_call(role="security", kind="critique", model=m) as rec:
            result = llm.invoke(prompt)
            rec.tokens_in = result.usage.input_tokens
            rec.tokens_out = result.usage.output_tokens
    """

    trace: DebateTrace
    _call_counter: int = field(default=0, init=False)

    @contextmanager
    def record_call(
        self,
        *,
        role: str,
        kind: str,
        model: str = "",
        reasoning_effort: str | None = None,
    ) -> Iterator[CallRecord]:
        """Yield a mutable CallRecord scoped to this block. On exit, the
        record is appended to the current round's call list — actual
        round wiring is done in Phase 4's debate graph nodes."""

        self._call_counter += 1
        record = CallRecord(
            call_id=f"call-{self._call_counter}",
            role=role,
            kind=kind,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        try:
            yield record
        except Exception as exc:
            record.error = str(exc)
            raise
        finally:
            record.ended_at = datetime.now(timezone.utc)

    def record_decision(self, decision: DecisionRecord) -> None:
        self.trace.decisions.append(decision)

    def record_memory_audit(self, entry: MemoryAuditEntry) -> None:
        self.trace.memory_audits.append(entry)

    def record_error(self, *, node: str, round_number: int, message: str) -> None:
        self.trace.errors.append(
            {"node": node, "round": round_number, "message": message}
        )
