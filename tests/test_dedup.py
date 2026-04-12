"""Tests for semantic deduplication of requirements.

Covers:
- Cosine similarity edge cases (zero-length, norm-zero, mismatched dims)
- Greedy clustering: singletons pass through; duplicates collapse
- Canonical selection: priority → description length → original index
- Tag union across merged members
- DedupeResult / DedupeGroup shape + dropped_count
- Threshold tuning (lower → more merges, higher → fewer merges)
- Embedding function errors propagate to the caller
- Stable ordering across re-runs (determinism)
- IngestStage wiring: dedup runs when embed_fn is provided,
  skipped when not, fall back gracefully on embed_fn failure
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

from dark_factory.models.domain import PipelineContext, Priority, Requirement
from dark_factory.stages.dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    DedupeGroup,
    DedupeResult,
    _cosine,
    semantically_dedupe,
)
from dark_factory.stages.ingest import IngestStage


# ── Helpers ──────────────────────────────────────────────────────────────────


def _req(
    rid: str,
    title: str = "t",
    description: str = "d",
    priority: Priority = Priority.MEDIUM,
    tags: list[str] | None = None,
) -> Requirement:
    return Requirement(
        id=rid,
        title=title,
        description=description,
        source_file="test",
        priority=priority,
        tags=tags or [],
    )


def _make_embed_fn(vectors: dict[str, list[float]]) -> Callable[[list[str]], list[list[float]]]:
    """Build a deterministic embed_fn that maps text blobs to fixed
    vectors. The key is the exact text we know ``semantically_dedupe``
    will build via ``_requirement_text`` (title + "\\n\\n" + description).
    """

    def _fn(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if t not in vectors:
                raise KeyError(f"test embed_fn saw unexpected text: {t!r}")
            out.append(vectors[t])
        return out

    return _fn


def _text_for(req: Requirement) -> str:
    return f"{req.title}\n\n{req.description}".strip()


# ── _cosine ──────────────────────────────────────────────────────────────────


def test_cosine_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_empty_returns_zero():
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([1.0], []) == 0.0


def test_cosine_mismatched_dims_returns_zero():
    assert _cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_zero_norm_returns_zero():
    """A zero vector has no direction — cosine should be 0 not NaN."""
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ── semantically_dedupe: trivial cases ───────────────────────────────────────


def test_dedupe_empty_list():
    result = semantically_dedupe([], lambda _: [])
    assert result.requirements == []
    assert result.groups == []
    assert result.dropped_count == 0


def test_dedupe_single_requirement_passes_through():
    reqs = [_req("a", "Only one")]
    # embed_fn should not even be called for a single-element input
    embed_fn = MagicMock(side_effect=RuntimeError("should not run"))
    result = semantically_dedupe(reqs, embed_fn)
    assert result.requirements == reqs
    assert result.groups == []
    assert result.dropped_count == 0
    embed_fn.assert_not_called()


# ── Clustering / dedup happy paths ───────────────────────────────────────────


def test_dedupe_collapses_identical_embeddings():
    a = _req("a", "User login", "Allow users to log in")
    b = _req("b", "User sign-in", "Allow users to sign in")
    c = _req("c", "Password reset", "Email-based password reset")
    vec_login = [1.0, 0.0]
    vec_reset = [0.0, 1.0]
    embed_fn = _make_embed_fn(
        {
            _text_for(a): vec_login,
            _text_for(b): vec_login,  # identical to a
            _text_for(c): vec_reset,  # orthogonal to both
        }
    )
    result = semantically_dedupe(reqs := [a, b, c], embed_fn, threshold=0.9)

    assert len(result.requirements) == 2
    assert result.dropped_count == 1
    assert len(result.groups) == 1
    group = result.groups[0]
    # Canonical is one of a/b (tie-broken by priority → desc length → index)
    assert group.canonical_id in {"a", "b"}
    assert group.merged_ids == ["b"] if group.canonical_id == "a" else ["a"]
    assert group.max_similarity == pytest.approx(1.0)


def test_dedupe_distinct_requirements_pass_through():
    a = _req("a", "Login", "Log in")
    b = _req("b", "Logout", "Log out")
    embed_fn = _make_embed_fn(
        {
            _text_for(a): [1.0, 0.0],
            _text_for(b): [0.0, 1.0],
        }
    )
    result = semantically_dedupe([a, b], embed_fn, threshold=0.9)
    assert len(result.requirements) == 2
    assert result.dropped_count == 0
    assert result.groups == []


def test_dedupe_threshold_is_respected():
    """At threshold 0.99 a near-match (sim=0.95) passes through; at
    threshold 0.90 the same pair is collapsed."""
    a = _req("a", "A", "a")
    b = _req("b", "B", "b")
    # Vectors chosen so cosine ≈ 0.95
    import math

    theta = math.acos(0.95)
    va = [1.0, 0.0]
    vb = [math.cos(theta), math.sin(theta)]
    embed_fn = _make_embed_fn({_text_for(a): va, _text_for(b): vb})

    high = semantically_dedupe([a, b], embed_fn, threshold=0.99)
    assert high.dropped_count == 0

    low = semantically_dedupe([a, b], embed_fn, threshold=0.90)
    assert low.dropped_count == 1


# ── Canonical selection ─────────────────────────────────────────────────────


def test_canonical_prefers_higher_priority():
    lo = _req("lo", "Login", "short", priority=Priority.LOW)
    hi = _req("hi", "Login", "short", priority=Priority.HIGH)
    embed_fn = _make_embed_fn(
        {
            _text_for(lo): [1.0, 0.0],
            _text_for(hi): [1.0, 0.0],
        }
    )
    result = semantically_dedupe([lo, hi], embed_fn)
    assert len(result.requirements) == 1
    assert result.requirements[0].id == "hi"
    assert result.groups[0].canonical_id == "hi"


def test_canonical_prefers_longer_description_when_priority_ties():
    short = _req("short", "Login", "short")
    long = _req("long", "Login", "a much longer and more detailed description")
    embed_fn = _make_embed_fn(
        {
            _text_for(short): [1.0, 0.0],
            _text_for(long): [1.0, 0.0],
        }
    )
    result = semantically_dedupe([short, long], embed_fn)
    assert result.requirements[0].id == "long"


def test_canonical_prefers_earlier_when_everything_ties():
    """All else equal, the original position of the requirement in the
    input list wins. This makes re-runs deterministic."""
    a = _req("a", "Login", "desc")
    b = _req("b", "Login", "desc")
    embed_fn = _make_embed_fn(
        {
            _text_for(a): [1.0, 0.0],
            _text_for(b): [1.0, 0.0],
        }
    )
    result = semantically_dedupe([a, b], embed_fn)
    assert result.requirements[0].id == "a"


# ── Tag union ───────────────────────────────────────────────────────────────


def test_dedupe_unions_tags_across_cluster():
    a = _req("a", "Login", "x", tags=["auth", "meeting-notes"])
    b = _req("b", "Sign in", "x", tags=["auth", "word-doc"])
    c = _req("c", "Log in", "x", tags=["AUTH", "spreadsheet"])  # case-insensitive dedupe
    embed_fn = _make_embed_fn(
        {
            _text_for(a): [1.0, 0.0],
            _text_for(b): [1.0, 0.0],
            _text_for(c): [1.0, 0.0],
        }
    )
    result = semantically_dedupe([a, b, c], embed_fn)
    assert len(result.requirements) == 1
    tags = result.requirements[0].tags
    # Canonical's tags preserve their original order first, then
    # merged members' additional tags appended in order.
    assert tags[0] == "auth"
    assert "meeting-notes" in tags
    assert "word-doc" in tags
    assert "spreadsheet" in tags
    # Case-insensitive dedup kept the first "auth" variant; "AUTH" is dropped
    assert sum(1 for t in tags if t.lower() == "auth") == 1


# ── DedupeGroup shape ────────────────────────────────────────────────────────


def test_dedupe_group_records_merged_titles_for_display():
    a = _req("a", "Canonical one", "x", priority=Priority.HIGH)
    b = _req("b", "Variant B", "x")
    c = _req("c", "Variant C", "x")
    embed_fn = _make_embed_fn(
        {
            _text_for(a): [1.0, 0.0],
            _text_for(b): [1.0, 0.0],
            _text_for(c): [1.0, 0.0],
        }
    )
    result = semantically_dedupe([a, b, c], embed_fn)
    assert len(result.groups) == 1
    g = result.groups[0]
    assert g.canonical_id == "a"
    assert g.canonical_title == "Canonical one"
    assert set(g.merged_ids) == {"b", "c"}
    assert set(g.merged_titles) == {"Variant B", "Variant C"}


# ── Error propagation ───────────────────────────────────────────────────────


def test_embed_fn_exception_propagates_to_caller():
    """The caller decides how to handle embedding-service outages;
    ``semantically_dedupe`` must not swallow those errors."""
    reqs = [_req("a"), _req("b")]
    with pytest.raises(RuntimeError, match="boom"):
        semantically_dedupe(reqs, lambda _: (_ for _ in ()).throw(RuntimeError("boom")))


def test_embed_fn_vector_count_mismatch_raises():
    reqs = [_req("a"), _req("b"), _req("c")]
    # Only returns 2 vectors for 3 requirements
    with pytest.raises(ValueError, match="3 requirements"):
        semantically_dedupe(reqs, lambda _: [[1.0], [1.0]])


# ── Determinism ─────────────────────────────────────────────────────────────


def test_dedupe_is_deterministic_across_reruns():
    reqs = [_req(f"r{i}", f"title{i}", f"desc{i}") for i in range(5)]
    # All 5 are identical in embedding space → one canonical
    embed_fn = _make_embed_fn({_text_for(r): [1.0, 0.0] for r in reqs})

    first = semantically_dedupe(reqs, embed_fn)
    second = semantically_dedupe(reqs, embed_fn)

    assert [r.id for r in first.requirements] == [r.id for r in second.requirements]
    assert first.groups[0].canonical_id == second.groups[0].canonical_id
    assert first.groups[0].merged_ids == second.groups[0].merged_ids


# ── IngestStage wiring ──────────────────────────────────────────────────────


def test_ingest_stage_skips_dedup_when_no_embed_fn(tmp_path: Path):
    (tmp_path / "a.md").write_text("user login\n")
    (tmp_path / "b.md").write_text("user sign-in\n")
    stage = IngestStage()  # no embed_fn
    ctx = PipelineContext(input_path=str(tmp_path))
    stage.run(ctx)
    # Without dedup, both files become separate single-requirement
    # entries. last_dedup_result must be None so the bridge doesn't
    # emit a spurious dedup progress event.
    assert len(ctx.requirements) == 2
    assert stage.last_dedup_result is None


def test_ingest_stage_runs_dedup_when_embed_fn_provided(tmp_path: Path):
    (tmp_path / "a.md").write_text("user login\n")
    (tmp_path / "b.md").write_text("user sign-in\n")

    # Stub embed_fn: both files embed to the same vector → collapse
    def _stub(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]] * len(texts)

    stage = IngestStage(embed_fn=_stub, dedup_threshold=0.9)
    ctx = PipelineContext(input_path=str(tmp_path))
    stage.run(ctx)

    assert len(ctx.requirements) == 1
    assert isinstance(stage.last_dedup_result, DedupeResult)
    assert stage.last_dedup_result.dropped_count == 1
    assert len(stage.last_dedup_result.groups) == 1


def test_ingest_stage_falls_back_on_embed_fn_failure(tmp_path: Path):
    """A transient embedding outage must NOT break the pipeline — the
    stage logs a warning, clears last_dedup_result, and returns the
    un-deduped list."""
    (tmp_path / "a.md").write_text("req a\n")
    (tmp_path / "b.md").write_text("req b\n")

    def _broken(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("openai 429")

    stage = IngestStage(embed_fn=_broken)
    ctx = PipelineContext(input_path=str(tmp_path))
    stage.run(ctx)

    # Requirements still produced — dedup failure isn't fatal
    assert len(ctx.requirements) == 2
    assert stage.last_dedup_result is None


def test_ingest_stage_skips_dedup_for_single_requirement(tmp_path: Path):
    """Dedup on a 1-element list is a no-op; the embed_fn must not
    even be called."""
    (tmp_path / "only.md").write_text("just one thing\n")

    embed_called = False

    def _stub(_texts: list[str]) -> list[list[float]]:
        nonlocal embed_called
        embed_called = True
        return [[1.0, 0.0]]

    stage = IngestStage(embed_fn=_stub)
    ctx = PipelineContext(input_path=str(tmp_path))
    stage.run(ctx)

    assert len(ctx.requirements) == 1
    assert embed_called is False
    # A single-requirement dedup is a degenerate no-op — we skip
    # setting last_dedup_result since there's nothing interesting
    # to surface.
    assert stage.last_dedup_result is None


def test_ingest_stage_default_dedup_threshold():
    stage = IngestStage()
    assert stage.dedup_threshold == DEFAULT_DEDUP_THRESHOLD


def test_dedupe_group_model_roundtrips_to_dict():
    """DedupeGroup must be JSON-safe so it can be attached to a
    StateSnapshot payload without a custom encoder."""
    import json

    g = DedupeGroup(
        canonical_id="abc",
        canonical_title="A thing",
        merged_ids=["def", "ghi"],
        merged_titles=["Variant", "Another"],
        max_similarity=0.97,
    )
    payload = g.model_dump()
    json.dumps(payload)  # must not raise
    assert payload["canonical_id"] == "abc"
    assert payload["max_similarity"] == pytest.approx(0.97)
