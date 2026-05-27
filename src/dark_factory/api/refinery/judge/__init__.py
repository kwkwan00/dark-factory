"""Combined evaluation pipeline — rules gate fused with DeepEval LLM judge."""

from dark_factory.api.refinery.judge.base import Judge
from dark_factory.api.refinery.judge.composition import CombinedJudgePipeline
from dark_factory.api.refinery.judge.cross_review_patcher import (
    apply_cross_review_report,
)
from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
from dark_factory.api.refinery.judge.rules import RULES, RuleSpec
from dark_factory.api.refinery.judge.rules_judge import RuleResult, RulesJudge

__all__ = [
    "Judge",
    "CombinedJudgePipeline",
    "FallbackJudge",
    "RuleResult",
    "RuleSpec",
    "RULES",
    "RulesJudge",
    "apply_cross_review_report",
]
