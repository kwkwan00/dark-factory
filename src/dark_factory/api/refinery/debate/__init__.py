"""Debate subgraph — LangGraph StateGraph for the adversarial refinery.

The outer ``stream.py`` calls ``run_phase2_generator`` to run one
requirement through the per-requirement debate.
"""

from dark_factory.api.refinery.debate.config import DebateConfig
from dark_factory.api.refinery.debate.graph import build_debate_graph, run_debate
from dark_factory.api.refinery.debate.runner import run_phase2_generator

__all__ = ["DebateConfig", "build_debate_graph", "run_debate", "run_phase2_generator"]
