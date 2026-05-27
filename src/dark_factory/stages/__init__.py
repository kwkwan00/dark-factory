"""Pipeline stages: ingest → spec → spec_reconciliation → graph.

Code generation no longer lives here — Phase 5 routes through the
LangGraph swarm in ``dark_factory.agents.swarm``. Reconciliation and
E2E validation are standalone modules (``reconciliation.py``,
``e2e_validation.py``) invoked by the AG-UI bridge after the swarm
completes.
"""
