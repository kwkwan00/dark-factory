"""``DebateConfig`` — consolidated debate-budget + models + timeouts.

Replaces the 18-kwarg signature on ``run_debate`` with a single typed
config object. Constructed explicitly by callers so tests can parameterize
just the knobs they care about; ``from_settings()`` is the production
factory that builds one from ``PipelineConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DebateConfig:
    """All dials that control one per-requirement debate.

    ``frozen=True`` so the state snapshot doesn't drift mid-debate —
    e.g. escalation swaps the active model tier via state mutation,
    not by mutating the config.
    """

    # Debate shape
    max_rounds: int = 3
    score_threshold: float = 0.8
    research_call_cap: int = 1
    escalation_cap: int = 1

    # Model tiers (strong_model kicks in after escalation)
    base_model: str = "gpt-5.4"
    strong_model: str = "gpt-5.4"
    reasoning_effort: str = "xhigh"

    # Legacy-runner budgets preserved for role.propose delegations
    max_turns: int = 15
    timeout_seconds: float = 900.0

    # Sandbox + opaque extras (the generator writes JSON here)
    tmpdir: str = "/tmp/refinery"

    # Opaque evidence extras threaded into the graph's evidence_bag.
    evidence_extras: dict = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        pipeline_config: "Any",  # PipelineConfig — avoid circular import
        *,
        base_model: str,
        reasoning_effort: str,
        tmpdir: str = "/tmp/refinery",
        max_turns: int = 15,
        timeout_seconds: float = 900.0,
        evidence_extras: dict | None = None,
    ) -> "DebateConfig":
        """Build a DebateConfig from ``PipelineConfig`` + overrides."""

        p = pipeline_config
        return cls(
            max_rounds=p.refinery_debate_max_rounds,
            score_threshold=p.refinery_debate_threshold,
            research_call_cap=p.refinery_research_cap,
            escalation_cap=p.refinery_escalation_cap,
            base_model=base_model,
            strong_model=p.refinery_model_strong or base_model,
            reasoning_effort=reasoning_effort,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            tmpdir=tmpdir,
            evidence_extras=dict(evidence_extras or {}),
        )
