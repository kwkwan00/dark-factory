"""Single source of truth for the stack description used across prompts.

Engineering critic, cost critic, and the DeepEval feasibility criterion
all need to name the same stack. Centralising it here prevents the
trio from drifting (which masks "this requirement breaks our stack"
critiques when one prompt forgets a dependency).
"""

from __future__ import annotations

STACK_DESCRIPTION: str = (
    "Python 3.12+, FastAPI, Neo4j, Qdrant, Postgres, "
    "OpenAI/Anthropic SDKs, LangGraph"
)
