"""Semantic deduplication of requirements before spec generation.

A real requirements corpus — especially one assembled from a mix of
uploaded documents (meeting transcripts, Word briefs, spreadsheets) —
routinely contains the same underlying requirement expressed multiple
ways. The existing :class:`IngestStage` already handles exact-string
dedup within a single document, but it does NOT catch:

- Two Word docs that both describe "user login" in different words.
- A meeting transcript that captures the same action item mentioned by
  two speakers.
- A PDF brief + a spreadsheet row that point at the same feature.

Without a dedup pass, the Spec stage would spend LLM budget generating
duplicate specs, Neo4j would accumulate redundant nodes, and the
downstream swarm would implement the same feature multiple times.

This module provides a **pure function** :func:`semantically_dedupe`
that takes a list of :class:`Requirement` objects, an embedding
callable, and a cosine similarity threshold, and returns a
:class:`DedupeResult` containing:

- The deduped list of canonical requirements (one per cluster).
- A list of :class:`DedupeGroup` objects documenting which requirements
  were collapsed into each canonical — surfaced in progress events and
  logs so the operator can audit the merges.

The function is synchronous, deterministic, and has no I/O beyond the
embedding callable itself — making it trivial to test with a stubbed
embedding function that returns fixed vectors.

**Clustering algorithm**: greedy single-link with a configurable cosine
similarity threshold. For N requirements the cost is O(N²) pairwise
comparisons, which is fine for any realistic corpus (<1000 items — a
typical run has 10–100). A more scalable approach (locality-sensitive
hashing, HDBSCAN, etc.) is future work if corpus sizes ever grow.

**Canonical selection**: within each cluster we pick the requirement
with the highest priority, breaking ties by longest description (more
detail is usually more useful for the Spec stage) and then by earliest
position in the input (stable for re-runs). All tags from merged
requirements are unioned onto the canonical so no source-file
attribution is lost.
"""

from __future__ import annotations

import math
from typing import Callable

import structlog
from pydantic import BaseModel, Field

from dark_factory.models.domain import Priority, Requirement

log = structlog.get_logger()


# Default cosine similarity threshold used when the caller doesn't
# specify one. 0.90 is conservative: text-embedding-3-large puts
# paraphrases of the same requirement around 0.92–0.97 and unrelated
# requirements well below 0.85, so 0.90 catches paraphrases without
# collapsing genuinely distinct requirements into one cluster.
DEFAULT_DEDUP_THRESHOLD = 0.90


# Priority ordering for canonical selection. Higher value = more
# important, so argmax picks the highest-priority requirement in a
# cluster as the canonical representative.
_PRIORITY_RANK: dict[Priority, int] = {
    Priority.CRITICAL: 4,
    Priority.HIGH: 3,
    Priority.MEDIUM: 2,
    Priority.LOW: 1,
}


class DedupeGroup(BaseModel):
    """Documents one cluster of semantically-duplicate requirements."""

    canonical_id: str
    """The id of the requirement that was kept."""

    canonical_title: str
    """Convenience: title of the canonical, for log / event display."""

    merged_ids: list[str] = Field(default_factory=list)
    """Ids of the other requirements in the cluster that were dropped
    in favour of the canonical. Empty for singleton clusters (which
    are omitted from the final result — a group with no merges isn't
    a group worth reporting)."""

    merged_titles: list[str] = Field(default_factory=list)
    """Parallel to ``merged_ids`` for display-only purposes."""

    max_similarity: float = 0.0
    """The highest cosine similarity seen between any two members of
    this cluster. Useful for threshold tuning: if real duplicates
    routinely come in below the active threshold, the operator can
    lower it; if distinct requirements keep colliding, raise it."""


class DedupeResult(BaseModel):
    """Return payload of :func:`semantically_dedupe`."""

    requirements: list[Requirement]
    """The deduped list. Each entry is the canonical from its cluster.
    Singletons (clusters of size 1) pass through unchanged."""

    groups: list[DedupeGroup] = Field(default_factory=list)
    """One entry per cluster that actually merged something. Omits
    singletons so the list stays short and audit-friendly."""

    dropped_count: int = 0
    """Total number of requirements removed as duplicates. Equal to
    ``sum(len(g.merged_ids) for g in groups)``. Surfaced in the
    ingest progress event so the Agent Logs tab can show a running
    count without re-counting on the frontend."""

    threshold: float = DEFAULT_DEDUP_THRESHOLD
    """The threshold that was active for this pass. Recorded for
    forensics so a surprising result can be correlated with a recent
    settings change."""


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 for zero-length inputs (defensive — an empty embedding
    is almost certainly a failed API call, not a legitimate zero
    vector) so nothing in a batch with a blank requirement can
    accidentally get collapsed into everything else via a degenerate
    norm-zero comparison.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _requirement_text(req: Requirement) -> str:
    """Build the text blob we embed for a single requirement.

    Includes both title and description so paraphrases of the same
    feature cluster together even when one source document uses a
    different title. We deliberately do NOT include tags or
    source_file — those are metadata that shouldn't influence
    semantic similarity.
    """
    return f"{req.title}\n\n{req.description}".strip()


def _canonical_index(cluster: list[int], requirements: list[Requirement]) -> int:
    """Pick the canonical member of a cluster.

    Ranking: (priority desc, description length desc, original index
    asc). The original-index tiebreaker is what makes the pick
    deterministic and stable across re-runs on the same input.
    """
    return max(
        cluster,
        key=lambda i: (
            _PRIORITY_RANK.get(requirements[i].priority, 0),
            len(requirements[i].description),
            -i,  # negated so earlier indices win when everything else ties
        ),
    )


def semantically_dedupe(
    requirements: list[Requirement],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> DedupeResult:
    """Cluster semantically-duplicate requirements and keep one per cluster.

    :param requirements: the full list of requirements from the ingest
        stage. Order is preserved for the canonical tiebreaker.
    :param embed_fn: a callable that takes ``list[str]`` and returns
        the matching ``list[list[float]]`` of embeddings. In
        production this is
        :meth:`dark_factory.vector.embeddings.EmbeddingService.embed_batch`;
        in tests it's a deterministic stub.
    :param threshold: cosine similarity >= this means "same
        requirement". Clamped to [0.0, 1.0] by the caller (the
        Pydantic config field enforces the range).

    Failures in ``embed_fn`` are NOT caught here — the caller is
    responsible for deciding what to do when the embedding service
    is down (typically: skip dedup entirely and log a warning, so
    one outage doesn't block the whole pipeline).
    """
    n = len(requirements)
    if n <= 1:
        return DedupeResult(
            requirements=list(requirements),
            groups=[],
            dropped_count=0,
            threshold=threshold,
        )

    texts = [_requirement_text(r) for r in requirements]
    embeddings = embed_fn(texts)

    # Defensive: embed_fn should return exactly one vector per input,
    # but if it doesn't we bail out rather than producing a subtly
    # wrong cluster assignment. The caller's except block will log
    # and skip.
    if len(embeddings) != n:
        raise ValueError(
            f"embed_fn returned {len(embeddings)} vectors for {n} requirements"
        )

    # Greedy single-link clustering. ``cluster_of[i]`` is the index of
    # the cluster that requirement ``i`` belongs to; clusters are
    # built up left-to-right, assigning each new requirement to the
    # FIRST existing cluster whose representative scores above the
    # threshold. This is O(N²) in the worst case but the inner loop
    # is just a cosine over in-memory floats — fast for any realistic
    # corpus.
    cluster_of: list[int] = [-1] * n
    clusters: list[list[int]] = []
    # Track the max similarity observed inside each cluster so the
    # DedupeGroup can surface it for threshold tuning.
    cluster_max_sim: list[float] = []

    for i in range(n):
        best_cluster = -1
        best_sim = 0.0
        for c_idx, cluster in enumerate(clusters):
            # Compare to the cluster's first (seed) member. Single-link
            # with seed comparison is simpler than full-link and
            # produces stable clusters for a threshold as high as ours.
            rep = cluster[0]
            sim = _cosine(embeddings[i], embeddings[rep])
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_cluster = c_idx
        if best_cluster >= 0:
            clusters[best_cluster].append(i)
            cluster_of[i] = best_cluster
            if best_sim > cluster_max_sim[best_cluster]:
                cluster_max_sim[best_cluster] = best_sim
        else:
            clusters.append([i])
            cluster_of[i] = len(clusters) - 1
            cluster_max_sim.append(0.0)

    # Build the output list. For each cluster pick the canonical and
    # union tags from the merged members onto it. Singletons pass
    # through untouched. Multi-member clusters get a DedupeGroup
    # recorded for observability.
    deduped: list[Requirement] = []
    groups: list[DedupeGroup] = []
    for c_idx, cluster in enumerate(clusters):
        if len(cluster) == 1:
            deduped.append(requirements[cluster[0]])
            continue

        canonical_i = _canonical_index(cluster, requirements)
        canonical = requirements[canonical_i]

        # Union tags (canonical tags first, preserving order) so no
        # source-document attribution is lost when we collapse the
        # cluster. Dedup tags case-insensitively.
        seen_tags: set[str] = set()
        merged_tags: list[str] = []
        for t in canonical.tags:
            if t.lower() not in seen_tags:
                seen_tags.add(t.lower())
                merged_tags.append(t)
        for i in cluster:
            if i == canonical_i:
                continue
            for t in requirements[i].tags:
                if t.lower() not in seen_tags:
                    seen_tags.add(t.lower())
                    merged_tags.append(t)

        # Emit an updated copy of the canonical with the unioned tags.
        # We use model_copy so all other fields (id, title, etc.)
        # pass through unchanged.
        merged = canonical.model_copy(update={"tags": merged_tags})
        deduped.append(merged)

        merged_ids = [
            requirements[i].id for i in cluster if i != canonical_i
        ]
        merged_titles = [
            requirements[i].title for i in cluster if i != canonical_i
        ]
        groups.append(
            DedupeGroup(
                canonical_id=canonical.id,
                canonical_title=canonical.title,
                merged_ids=merged_ids,
                merged_titles=merged_titles,
                max_similarity=cluster_max_sim[c_idx],
            )
        )

    dropped = sum(len(g.merged_ids) for g in groups)

    if groups:
        log.info(
            "requirements_semantically_deduped",
            input_count=n,
            output_count=len(deduped),
            dropped=dropped,
            groups=len(groups),
            threshold=threshold,
        )
    else:
        log.info(
            "requirements_semantically_deduped",
            input_count=n,
            output_count=len(deduped),
            dropped=0,
            groups=0,
            threshold=threshold,
        )

    return DedupeResult(
        requirements=deduped,
        groups=groups,
        dropped_count=dropped,
        threshold=threshold,
    )
