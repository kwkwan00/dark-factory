"""ContextCache — per-debate LRU cache of RoleContext by (req, role, round).

Scoped per-debate so requirement A's cache doesn't leak into requirement
B's context. Cleared between requirements by the orchestrator.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from dark_factory.api.refinery.contracts import RoleContext
from dark_factory.log import trace_methods


@dataclass(frozen=True)
class CacheKey:
    requirement_id: str
    role: str
    round_number: int


@trace_methods
class ContextCache:
    """Simple LRU keyed on (requirement_id, role, round_number)."""

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[CacheKey, RoleContext] = OrderedDict()

    def get(self, key: CacheKey) -> RoleContext | None:
        ctx = self._store.get(key)
        if ctx is not None:
            self._store.move_to_end(key)
        return ctx

    def set(self, key: CacheKey, ctx: RoleContext) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = ctx
            return
        if len(self._store) >= self._maxsize:
            self._store.popitem(last=False)
        self._store[key] = ctx

    def clear_for_requirement(self, requirement_id: str) -> None:
        # Copy keys out so we don't mutate while iterating.
        victims = [k for k in self._store if k.requirement_id == requirement_id]
        for k in victims:
            del self._store[k]

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
