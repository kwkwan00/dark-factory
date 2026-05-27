"""Role-filtered hybrid retrieval for the adversarial refinery."""

from dark_factory.api.refinery.context.base import (
    ContextBuilder,
    RoleFilterPolicy,
    RunContext,
)
from dark_factory.api.refinery.context.cache import CacheKey, ContextCache
from dark_factory.api.refinery.context.hybrid import HybridContextBuilder
from dark_factory.api.refinery.context.rewriter import IdentityRewriter, QueryRewriter
from dark_factory.api.refinery.context.role_slices import (
    ROLE_FILTERS,
    get_role_filter_policy,
)

__all__ = [
    "ContextBuilder",
    "RoleFilterPolicy",
    "RunContext",
    "CacheKey",
    "ContextCache",
    "HybridContextBuilder",
    "IdentityRewriter",
    "QueryRewriter",
    "ROLE_FILTERS",
    "get_role_filter_policy",
]
