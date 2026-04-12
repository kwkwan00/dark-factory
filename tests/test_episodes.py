"""Tests for the episodic memory subsystem.

Covers:
- ``synthesize_episode`` happy path with stubbed LLM
- LLM failure → deterministic fallback summary
- Content-addressed id stability across re-syntheses
- Outcome classification (success / partial / failed) from status +
  eval scores
- Eval score flattening across per-spec nesting
- Tool call counts extraction + top-10 cap
- ``EpisodeWriter`` happy path (Neo4j + Qdrant both succeed)
- EpisodeWriter tolerates Neo4j failure
- EpisodeWriter tolerates Qdrant failure
- EpisodeWriter tolerates embedding failure
- EpisodeWriter handles missing repos gracefully
- ``recall_episodes`` hybrid merge with stubbed repos
- ``recall_episodes`` returns empty-state message when no episodes exist
- ``recall_episodes`` expands JSON fields for LLM consumption
- Episode model JSON round-trip (for StateSnapshot compatibility)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from dark_factory.memory.episodes import (
    Episode,
    EpisodeKeyEvent,
    EpisodeWriter,
    _any_eval_failed,
    _flatten_eval_scores,
    _normalise_outcome,
    _extract_tool_call_counts,
    episode_from_feature_result,
    synthesize_episode,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _feature_result(
    status: str = "success",
    eval_scores: dict | None = None,
    stats: dict | None = None,
) -> dict:
    """Build a FeatureResult-shaped dict for tests."""
    return {
        "feature": "auth",
        "spec_ids": ["spec-1", "spec-2"],
        "status": status,
        "artifacts": [],
        "tests": [],
        "eval_scores": eval_scores
        or {"spec-1": {"correctness": {"score": 0.9}, "coherence": {"score": 0.85}}},
        "stats": stats
        or {
            "agent_transitions": 6,
            "duration_seconds": 14.2,
            "unique_agents_visited": ["planner", "coder", "reviewer", "tester"],
            "tool_call_counts": {"read_file": 4, "write_file": 3, "evaluate_spec": 2},
        },
    }


def _fake_llm_returning(
    summary: str = "Planner picked JWT. Coder generated middleware. Reviewer approved on turn 4. Tester passed all 3 tests. Outcome: success.",
    key_events: list | None = None,
):
    """Build a stubbed LLM that returns a canned _EpisodeSynthesis."""
    from dark_factory.memory.episodes import _EpisodeSynthesis

    fake = MagicMock()
    fake.complete_structured.return_value = _EpisodeSynthesis(
        summary=summary,
        key_events=key_events
        or [
            EpisodeKeyEvent(
                order=1,
                agent="planner",
                event="strategy_picked",
                description="Chose JWT over OAuth",
            ),
            EpisodeKeyEvent(
                order=2,
                agent="reviewer",
                event="approval",
                description="Code approved on turn 4",
            ),
        ],
    )
    return fake


# ── synthesize_episode happy path ────────────────────────────────────────────


def test_synthesize_episode_happy_path_calls_llm():
    fake_llm = _fake_llm_returning()
    started = datetime.now(timezone.utc)
    ended = datetime.now(timezone.utc)

    episode = synthesize_episode(
        run_id="run-test-001",
        feature="auth",
        spec_ids=["spec-1"],
        outcome="success",
        turns_used=6,
        duration_seconds=14.2,
        started_at=started,
        ended_at=ended,
        final_eval_scores={"correctness": 0.9},
        agents_visited=["planner", "coder", "reviewer", "tester"],
        tool_calls_summary={"read_file": 4},
        progress_events=[
            {"event": "feature_started", "feature": "auth"},
            {"event": "handoff", "from": "planner", "to": "coder"},
        ],
        llm=fake_llm,
    )

    assert fake_llm.complete_structured.called
    assert episode.run_id == "run-test-001"
    assert episode.feature == "auth"
    assert episode.outcome == "success"
    assert "JWT" in episode.summary
    assert len(episode.key_events) == 2
    assert episode.key_events[0].agent == "planner"
    assert episode.turns_used == 6
    assert episode.duration_seconds == 14.2
    # Content-addressed id is 16 hex chars + ep- prefix
    assert episode.id.startswith("ep-")
    assert len(episode.id) == len("ep-") + 16


def test_synthesize_episode_id_stable_across_runs():
    """Re-synthesising an episode with the same run_id + feature
    produces the same id, so a retry overwrites rather than
    duplicates."""
    started = datetime.now(timezone.utc)
    fake = _fake_llm_returning()
    ep1 = synthesize_episode(
        run_id="run-stable",
        feature="auth",
        spec_ids=[],
        outcome="success",
        turns_used=1,
        duration_seconds=1.0,
        started_at=started,
        ended_at=started,
        final_eval_scores={},
        agents_visited=[],
        tool_calls_summary={},
        progress_events=[],
        llm=fake,
    )
    ep2 = synthesize_episode(
        run_id="run-stable",
        feature="auth",
        spec_ids=[],
        outcome="success",
        turns_used=99,  # different metadata — id should be unchanged
        duration_seconds=999.0,
        started_at=started,
        ended_at=started,
        final_eval_scores={},
        agents_visited=[],
        tool_calls_summary={},
        progress_events=[],
        llm=fake,
    )
    assert ep1.id == ep2.id


def test_synthesize_episode_different_run_ids_produce_different_ids():
    started = datetime.now(timezone.utc)
    fake = _fake_llm_returning()
    ep1 = synthesize_episode(
        run_id="run-1", feature="auth", spec_ids=[], outcome="success",
        turns_used=1, duration_seconds=1.0, started_at=started, ended_at=started,
        final_eval_scores={}, agents_visited=[], tool_calls_summary={},
        progress_events=[], llm=fake,
    )
    ep2 = synthesize_episode(
        run_id="run-2", feature="auth", spec_ids=[], outcome="success",
        turns_used=1, duration_seconds=1.0, started_at=started, ended_at=started,
        final_eval_scores={}, agents_visited=[], tool_calls_summary={},
        progress_events=[], llm=fake,
    )
    assert ep1.id != ep2.id


# ── Fallback path (LLM unavailable / crashes) ───────────────────────────────


def test_synthesize_episode_falls_back_when_llm_is_none():
    """No LLM provided → deterministic fallback summary, no crash."""
    started = datetime.now(timezone.utc)
    ep = synthesize_episode(
        run_id="r-1", feature="api", spec_ids=["spec-a"], outcome="failed",
        turns_used=12, duration_seconds=120.0, started_at=started, ended_at=started,
        final_eval_scores={}, agents_visited=[], tool_calls_summary={},
        progress_events=[], llm=None, error="timeout waiting for reviewer",
    )
    assert ep.outcome == "failed"
    assert "fallback" in ep.summary.lower()
    assert "timeout" in ep.summary
    assert ep.key_events == []


def test_synthesize_episode_falls_back_when_llm_raises():
    """LLM crashes mid-call → fallback summary, no propagation."""
    fake = MagicMock()
    fake.complete_structured.side_effect = RuntimeError("API quota exceeded")
    started = datetime.now(timezone.utc)
    ep = synthesize_episode(
        run_id="r-1", feature="api", spec_ids=[], outcome="success",
        turns_used=3, duration_seconds=5.0, started_at=started, ended_at=started,
        final_eval_scores={}, agents_visited=[], tool_calls_summary={},
        progress_events=[], llm=fake, error=None,
    )
    assert "fallback" in ep.summary.lower()
    assert ep.outcome == "success"


# ── Outcome classification ──────────────────────────────────────────────────


def test_normalise_outcome_success_with_passing_evals():
    eval_scores = {"spec-1": {"correctness": {"score": 0.9}}}
    assert _normalise_outcome("success", eval_scores) == "success"


def test_normalise_outcome_success_with_failing_evals_is_partial():
    eval_scores = {"spec-1": {"correctness": {"score": 0.3}}}
    assert _normalise_outcome("success", eval_scores) == "partial"


def test_normalise_outcome_error_and_skipped_collapse_to_failed():
    assert _normalise_outcome("error", None) == "failed"
    assert _normalise_outcome("skipped", None) == "failed"
    assert _normalise_outcome("cancelled", None) == "failed"


def test_any_eval_failed_handles_nested_and_flat_shapes():
    nested = {"spec": {"c": {"score": 0.3}}}
    flat = {"spec": {"c": 0.3}}
    assert _any_eval_failed(nested) is True
    assert _any_eval_failed(flat) is True
    assert _any_eval_failed({"spec": {"c": {"score": 0.9}}}) is False
    assert _any_eval_failed(None) is False


# ── Score flattening ─────────────────────────────────────────────────────────


def test_flatten_eval_scores_averages_across_specs():
    scores = {
        "spec-1": {"correctness": {"score": 0.8}, "coherence": {"score": 0.6}},
        "spec-2": {"correctness": {"score": 0.4}},
    }
    flat = _flatten_eval_scores(scores)
    # correctness: (0.8 + 0.4) / 2 = 0.6
    # coherence: only on spec-1 → 0.6
    assert flat["correctness"] == pytest.approx(0.6)
    assert flat["coherence"] == pytest.approx(0.6)


def test_flatten_eval_scores_handles_flat_values():
    scores = {"spec-1": {"correctness": 0.75}}
    flat = _flatten_eval_scores(scores)
    assert flat["correctness"] == pytest.approx(0.75)


# ── Tool call extraction ────────────────────────────────────────────────────


def test_extract_tool_call_counts_finds_dict():
    stats = {"tool_call_counts": {"read_file": 4, "write_file": 2}}
    assert _extract_tool_call_counts(stats) == {"read_file": 4, "write_file": 2}


def test_extract_tool_call_counts_returns_empty_when_missing():
    assert _extract_tool_call_counts({}) == {}


def test_synthesize_episode_caps_tool_calls_to_top_10():
    fake = _fake_llm_returning()
    started = datetime.now(timezone.utc)
    huge = {f"tool_{i}": (20 - i) for i in range(30)}
    ep = synthesize_episode(
        run_id="r-1", feature="f", spec_ids=[], outcome="success",
        turns_used=1, duration_seconds=1.0, started_at=started, ended_at=started,
        final_eval_scores={}, agents_visited=[], tool_calls_summary=huge,
        progress_events=[], llm=fake,
    )
    assert len(ep.tool_calls_summary) == 10
    # Top 10 should be the highest-count tools
    assert "tool_0" in ep.tool_calls_summary  # count=20
    assert "tool_9" in ep.tool_calls_summary  # count=11
    assert "tool_10" not in ep.tool_calls_summary  # count=10, cut


# ── EpisodeWriter ───────────────────────────────────────────────────────────


def _make_episode() -> Episode:
    now = datetime.now(timezone.utc)
    return Episode(
        id="ep-abcdef1234567890",
        run_id="run-test",
        feature="auth",
        outcome="success",
        summary="Planner picked JWT. Success.",
        key_events=[
            EpisodeKeyEvent(
                order=1, agent="planner", event="strategy_picked", description="JWT"
            )
        ],
        turns_used=6,
        duration_seconds=14.2,
        spec_ids=["spec-1"],
        final_eval_scores={"correctness": 0.9},
        agents_visited=["planner", "coder"],
        tool_calls_summary={"read_file": 4},
        started_at=now,
        ended_at=now,
    )


def test_episode_writer_happy_path_writes_both_stores():
    memory_repo = MagicMock()
    vector_repo = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_batch.return_value = [[0.1] * 3072]

    writer = EpisodeWriter(
        memory_repo=memory_repo,
        vector_repo=vector_repo,
        embeddings=embeddings,
    )
    episode = _make_episode()
    ok = writer.write(episode)

    assert ok is True
    memory_repo.write_episode.assert_called_once_with(episode)
    embeddings.embed_batch.assert_called_once()
    vector_repo.upsert_episode.assert_called_once()
    call_kwargs = vector_repo.upsert_episode.call_args.kwargs
    assert call_kwargs["episode_id"] == episode.id
    assert call_kwargs["feature"] == "auth"
    assert call_kwargs["outcome"] == "success"
    assert len(call_kwargs["vector"]) == 3072


def test_episode_writer_tolerates_neo4j_failure():
    """If Neo4j is down, the writer still attempts the Qdrant upsert
    and returns False — but does NOT raise."""
    memory_repo = MagicMock()
    memory_repo.write_episode.side_effect = RuntimeError("neo4j unreachable")
    vector_repo = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_batch.return_value = [[0.1] * 3072]

    writer = EpisodeWriter(
        memory_repo=memory_repo,
        vector_repo=vector_repo,
        embeddings=embeddings,
    )
    ok = writer.write(_make_episode())
    assert ok is False
    # Vector side still attempted — loss is partial not total
    vector_repo.upsert_episode.assert_called_once()


def test_episode_writer_tolerates_qdrant_failure():
    memory_repo = MagicMock()
    vector_repo = MagicMock()
    vector_repo.upsert_episode.side_effect = RuntimeError("qdrant timeout")
    embeddings = MagicMock()
    embeddings.embed_batch.return_value = [[0.1] * 3072]

    writer = EpisodeWriter(
        memory_repo=memory_repo,
        vector_repo=vector_repo,
        embeddings=embeddings,
    )
    ok = writer.write(_make_episode())
    # Neo4j succeeded → overall success
    assert ok is True
    memory_repo.write_episode.assert_called_once()


def test_episode_writer_tolerates_embedding_failure():
    memory_repo = MagicMock()
    vector_repo = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_batch.side_effect = RuntimeError("openai 429")

    writer = EpisodeWriter(
        memory_repo=memory_repo,
        vector_repo=vector_repo,
        embeddings=embeddings,
    )
    ok = writer.write(_make_episode())
    assert ok is True  # Neo4j still succeeded
    memory_repo.write_episode.assert_called_once()
    # Qdrant never reached because embedding blew up
    vector_repo.upsert_episode.assert_not_called()


def test_episode_writer_handles_missing_repos_gracefully():
    """All-None writer is a silent no-op."""
    writer = EpisodeWriter(memory_repo=None, vector_repo=None, embeddings=None)
    ok = writer.write(_make_episode())
    assert ok is False  # Nothing got written, but also no crash


# ── episode_from_feature_result shortcut ─────────────────────────────────────


def test_episode_from_feature_result_maps_status_to_outcome():
    fake = _fake_llm_returning()
    ep = episode_from_feature_result(
        run_id="run-1",
        feature_result=_feature_result(status="success"),
        started_at=datetime.now(timezone.utc),
        progress_events=[],
        llm=fake,
    )
    assert ep.feature == "auth"
    assert ep.outcome == "success"
    assert ep.turns_used == 6
    assert ep.duration_seconds == pytest.approx(14.2)
    assert "correctness" in ep.final_eval_scores


def test_episode_from_feature_result_error_becomes_failed():
    result = _feature_result(status="error")
    result["error"] = "swarm crashed"
    ep = episode_from_feature_result(
        run_id="run-1",
        feature_result=result,
        started_at=datetime.now(timezone.utc),
        progress_events=[],
        llm=None,  # fallback path
    )
    assert ep.outcome == "failed"
    assert "swarm crashed" in ep.summary


# ── Episode model JSON round-trip ────────────────────────────────────────────


def test_episode_model_dump_is_json_serialisable():
    """The frontend receives episodes via JSON, and the
    StateSnapshot payload needs to round-trip cleanly."""
    ep = _make_episode()
    dump = ep.model_dump(mode="json")
    blob = json.dumps(dump)
    loaded = json.loads(blob)
    assert loaded["id"] == ep.id
    assert loaded["feature"] == "auth"
    assert len(loaded["key_events"]) == 1
    assert loaded["key_events"][0]["agent"] == "planner"


# ── recall_episodes agent tool ──────────────────────────────────────────────


def test_recall_episodes_returns_disabled_when_no_repo():
    """When memory is disabled, the tool returns the canonical
    disabled message rather than crashing."""
    from dark_factory.agents import tools as tools_mod

    previous = tools_mod._memory_repo
    tools_mod._memory_repo = None
    try:
        result = tools_mod.recall_episodes.invoke(
            {"keywords": "auth", "feature_name": "auth", "limit": 5}
        )
        assert "disabled" in result.lower()
    finally:
        tools_mod._memory_repo = previous


def test_recall_episodes_hybrid_merge_uses_neo4j_and_qdrant():
    """When both Neo4j and Qdrant return hits, the tool merges them
    via RRF and returns the top ``limit`` as JSON."""
    from dark_factory.agents import tools as tools_mod

    neo4j_rows = [
        {
            "id": "ep-a",
            "feature": "auth",
            "outcome": "success",
            "summary": "Planner picked JWT. Success.",
            "turns_used": 6,
            "duration_seconds": 14.0,
            "run_id": "run-1",
            "key_events_json": json.dumps(
                [{"order": 1, "agent": "planner", "event": "pick", "description": "JWT"}]
            ),
        },
    ]
    vector_rows = [
        {
            "id": "ep-b",
            "feature": "auth",
            "outcome": "success",
            "summary": "Session cookies worked.",
            "turns_used": 8,
            "duration_seconds": 20.0,
            "run_id": "run-2",
            "score": 0.85,
        },
    ]

    fake_memory = MagicMock()
    fake_memory.search_episodes_keyword.return_value = neo4j_rows
    fake_vector = MagicMock()
    fake_vector.search_episodes.return_value = vector_rows

    prev_m = tools_mod._memory_repo
    prev_v = tools_mod._vector_repo
    tools_mod._memory_repo = fake_memory
    tools_mod._vector_repo = fake_vector
    try:
        raw = tools_mod.recall_episodes.invoke(
            {
                "keywords": "auth JWT",
                "feature_name": "auth",
                "outcome": "success",
                "limit": 5,
            }
        )
        parsed = json.loads(raw)
        # Both results should be present
        ids = {ep["id"] for ep in parsed}
        assert "ep-a" in ids
        assert "ep-b" in ids
        # The Neo4j row's key_events_json should be expanded into
        # structured form for LLM consumption.
        ep_a = next(ep for ep in parsed if ep["id"] == "ep-a")
        assert "key_events" in ep_a
        assert ep_a["key_events"][0]["agent"] == "planner"
    finally:
        tools_mod._memory_repo = prev_m
        tools_mod._vector_repo = prev_v


def test_recall_episodes_returns_empty_state_when_no_hits():
    from dark_factory.agents import tools as tools_mod

    fake_memory = MagicMock()
    fake_memory.search_episodes_keyword.return_value = []

    prev_m = tools_mod._memory_repo
    prev_v = tools_mod._vector_repo
    tools_mod._memory_repo = fake_memory
    tools_mod._vector_repo = None
    try:
        result = tools_mod.recall_episodes.invoke(
            {"keywords": "never-seen", "feature_name": "", "limit": 5}
        )
        assert "no past episodes" in result.lower()
    finally:
        tools_mod._memory_repo = prev_m
        tools_mod._vector_repo = prev_v


def test_recall_episodes_tolerates_qdrant_failure():
    """Vector search blowing up must not kill the tool — fall back
    to Neo4j-only results."""
    from dark_factory.agents import tools as tools_mod

    fake_memory = MagicMock()
    fake_memory.search_episodes_keyword.return_value = [
        {
            "id": "ep-1",
            "feature": "auth",
            "outcome": "success",
            "summary": "ok",
            "turns_used": 1,
            "duration_seconds": 1.0,
            "run_id": "run-x",
        }
    ]
    fake_vector = MagicMock()
    fake_vector.search_episodes.side_effect = RuntimeError("qdrant down")

    prev_m = tools_mod._memory_repo
    prev_v = tools_mod._vector_repo
    tools_mod._memory_repo = fake_memory
    tools_mod._vector_repo = fake_vector
    try:
        raw = tools_mod.recall_episodes.invoke(
            {"keywords": "auth", "feature_name": "auth", "limit": 5}
        )
        parsed = json.loads(raw)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "ep-1"
    finally:
        tools_mod._memory_repo = prev_m
        tools_mod._vector_repo = prev_v
