"""Regression tests for the hybrid/reranker confidence & score-contract bugs.

Covers four distinct defects that were reported and fixed:

1. `_as_relevance` fallback produced NEGATIVE relevance for l2 distances > 1
   (Bug 7 - "worse than worst" scores that inverted the ranking).
2. `EmbeddingReranker._fallback` could emit an out-of-contract similarity.
3. `EmbeddingReranker._align_scores` used a magic 1.0 pseudo-distance.
4. `KnowledgeBaseService._calculate_confidence_score` was fed hybrid
   pseudo-distances but used a logistic calibrated for raw L2 distances
   (0.0 .. ~3.5), saturating every result to ~[0.92, 1.0] so the confidence
   gate never fired.
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain.schema import Document  # noqa: E402

from app.services.hybrid_retrieval import (  # noqa: E402
    DEFAULT_PSEUDO_DISTANCE,
    HybridRetrievalService,
)
from app.services.reranker import DEFAULT_PSEUDO_DISTANCE as RERANKER_PSEUDO_DISTANCE  # noqa: E402


def _make_service(vector_store) -> HybridRetrievalService:
    dp = MagicMock()
    dp.get_vector_store.return_value = vector_store
    dp.embeddings = None
    return HybridRetrievalService(dp)


class _NoConverterChroma:
    """Vector store WITHOUT `_select_relevance_score_fn` (forces the fallback)."""

    def __init__(self, scored):
        self._scored = scored

    def similarity_search_by_vector_with_relevance_scores(self, embedding, k=4, filter=None):
        return self._scored[:k]


@pytest.mark.asyncio
async def test_l2_fallback_relevance_is_clamped_non_negative():
    """Regression (Bug 7): l2 distances > 1 must not yield negative relevance.

    With the old `1 - distance` fallback, a doc at distance 2.0 (very far in l2
    space) produced relevance -1.0. Min-max normalization then ranked a
    mediocre doc ABOVE a perfect one. Distances beyond 1.0 now clamp to 0.0.
    """
    near = Document(page_content="near", metadata={"id": "near"})
    far = Document(page_content="far", metadata={"id": "far"})
    # NOTE: distances > 1.0 are ordinary for l2 (max is sqrt(2) for unit vectors).
    vs = _NoConverterChroma([(near, 0.2), (far, 2.0)])
    svc = _make_service(vs)

    pairs = await svc._dense_similarity_search(vs, [0.0], k=2, filter_dict={})
    scores = {svc._make_doc_key(d): s for d, s in pairs}

    assert scores["far"] >= 0.0, "REGRESSION: fallback relevance must never be negative"
    assert scores["near"] > scores["far"]

    # Direction must survive normalization (this is the user-visible symptom).
    norm = svc._normalize_dense(pairs)
    assert norm["near"] == pytest.approx(1.0)
    assert norm["far"] == pytest.approx(0.0)


def test_pseudo_distance_contract_is_shared():
    """Both modules agree on the single pseudo-distance constant."""
    assert DEFAULT_PSEUDO_DISTANCE == RERANKER_PSEUDO_DISTANCE == 1.0


def test_reranker_fallback_similarity_is_in_range():
    """`_fallback` must emit similarities within [0, 1] even for bad inputs."""
    from app.services.reranker import EmbeddingReranker

    reranker = EmbeddingReranker(embeddings_model=None)
    docs = [Document(page_content="a"), Document(page_content="b")]
    # 2.5 is out of contract for a pseudo-distance but must not break the contract.
    out = reranker._fallback(docs, [0.0, 2.5], top_k=2, reason="test")

    for _, sim, meta in out:
        assert 0.0 <= sim <= 1.0
        assert 0.0 <= meta["l2_similarity"] <= 1.0
    # descending similarity (bigger = better)
    assert out[0][1] > out[1][1]


def test_reranker_align_scores_pads_with_worst_pseudo_distance():
    """Missing scores are padded with the shared worst-case pseudo-distance."""
    from app.services.reranker import EmbeddingReranker

    aligned = EmbeddingReranker._align_scores([0.1], 3)
    assert aligned == [0.1, DEFAULT_PSEUDO_DISTANCE, DEFAULT_PSEUDO_DISTANCE]


def _make_kb_service():
    """Build a KnowledgeBaseService with heavy deps stubbed out."""
    from unittest.mock import patch

    with patch("app.services.chat_service.get_llm"):
        with patch("app.services.document_processor.get_document_processor"):
            with patch("app.services.excel_processor.get_excel_qa_processor"):
                from app.services.knowledge_base import KnowledgeBaseService

                return KnowledgeBaseService()


def test_confidence_is_calibrated_across_the_pseudo_distance_range():
    """Regression (Bug 4): confidence must span the gate range, not saturate.

    The previous logistic (midpoint 1.5, scale 5) mapped every possible hybrid
    pseudo-distance in [0, 1] to ~[0.92, 1.0], so a completely irrelevant
    document still cleared a 0.5 (or even 0.9) gate and the strict handoff never
    triggered. The calibrated curve now puts the worst document below the default
    0.5 gate while a strong hit stays above it.
    """
    kb = _make_kb_service()
    doc = Document(page_content="d", metadata={})

    perfect = kb._calculate_confidence_score([(doc, 0.0)], top_n=1)
    strong = kb._calculate_confidence_score([(doc, 0.2)], top_n=1)
    weak = kb._calculate_confidence_score([(doc, 0.8)], top_n=1)
    irrelevant = kb._calculate_confidence_score([(doc, 1.0)], top_n=1)

    # Ordering is preserved (lower pseudo-distance = higher confidence).
    assert perfect > strong > weak > irrelevant

    # Old behaviour: worst document scored ~0.924 (and 1-doc gave ~0.924 too).
    assert perfect == pytest.approx(1.0)
    assert irrelevant < 0.5, f"worst doc still clears the gate: {irrelevant}"

    # The gate is now discriminative at the default 0.5 threshold.
    assert strong > 0.5 > irrelevant

    # The usable spread is real, not the old ~0.08 band.
    assert (perfect - irrelevant) > 0.4


def test_confidence_multi_factor_weights_are_preserved():
    """Consistency (30%) and coverage (10%) weights are unchanged by the fix."""
    kb = _make_kb_service()
    docs = [(Document(page_content=str(i), metadata={}), 0.0) for i in range(3)]

    # All three docs identical + perfect distance => best 1.0, coverage 1.0.
    assert kb._calculate_confidence_score(docs, top_n=3) == pytest.approx(1.0)

    # Perfect best score but a single (low-coverage) doc => 0.6 + 0.3 + ~0.1.
    single = kb._calculate_confidence_score(docs[:1], top_n=3)
    assert single < 1.0
    assert single == pytest.approx(0.6 + 0.3 + (1 / 3) * 0.1)

    # The structural factors are relative weights, not a floor: with the best
    # document at zero relevance they must not manufacture any confidence.
    perfect_structural = [(Document(page_content="x", metadata={}), 1.0)] * 3
    assert kb._calculate_confidence_score(perfect_structural, top_n=3) == 0.0


def test_confidence_single_score_helper_delegates():
    """`_calculate_single_score_confidence` shares the calibrated curve."""
    kb = _make_kb_service()
    for distance in (0.0, 0.3, 1.5, 2.5, 5.0):
        assert kb._calculate_single_score_confidence(distance) == pytest.approx(
            kb._similarity_to_confidence(distance)
        )


def test_confidence_score_scale_parameterizes_the_distance_domain():
    """A raw-L2 caller must not be normalized over the [0, 1] hybrid domain.

    The non-hybrid (vector-only) path yields raw Chroma L2 distances in
    [0, ~3.5]. Normalizing those over the hybrid pseudo-distance range would
    clamp every distance > 1 to zero relevance, so the caller-supplied
    `score_scale` must be honoured.
    """
    from app.services.knowledge_base import (
        HYBRID_PSEUDO_DISTANCE_RANGE,
        RAW_L2_DISTANCE_RANGE,
    )

    kb = _make_kb_service()
    doc = Document(page_content="d", metadata={})

    # Same raw distance, two different declared domains -> different confidence.
    hybrid_view = kb._calculate_confidence_score(
        [(doc, 0.8)], top_n=1, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE
    )
    l2_view = kb._calculate_confidence_score(
        [(doc, 0.8)], top_n=1, score_scale=RAW_L2_DISTANCE_RANGE
    )
    assert l2_view > hybrid_view, "raw-L2 domain must not be judged on the hybrid scale"

    # Default behaviour is the hybrid domain (unchanged for existing callers).
    assert kb._calculate_confidence_score([(doc, 0.8)], top_n=1) == pytest.approx(
        hybrid_view
    )

    # An in-range raw L2 distance must not be clamped to zero relevance.
    assert l2_view > 0.5, f"raw L2 distance 0.8 was unfairly punished: {l2_view}"

    # The raw-L2 best-document factor equals the explicitly-scaled curve.
    assert kb._similarity_to_confidence(0.8) == pytest.approx(
        kb._similarity_to_confidence(0.8, max_distance=RAW_L2_DISTANCE_RANGE)
    )
    assert kb._similarity_to_confidence(
        0.8, max_distance=RAW_L2_DISTANCE_RANGE
    ) > kb._similarity_to_confidence(
        0.8, max_distance=HYBRID_PSEUDO_DISTANCE_RANGE
    )


def test_irrelevant_best_document_cannot_ride_the_consistency_floor():
    """Regression (weak gate): the 0.4 consistency+coverage floor is capped.

    A single document has std=0, so consistency=1.0 unconditionally and the OLD
    additive blend floored at 0.3 + 0.1 = 0.4. A document with a pseudo-distance
    up to 0.92 (best_confidence ~= 0.17) therefore still produced a blended score
    of ~0.50 and cleared the default 0.5 gate. The best-document factor now
    multiplies the structural factors, so `final <= best_confidence` always.
    """
    from app.services.knowledge_base import (
        BEST_DOCUMENT_RELEVANCE_FLOOR,
        HYBRID_PSEUDO_DISTANCE_RANGE,
    )

    kb = _make_kb_service()
    doc = Document(page_content="d", metadata={})

    # Worst-case single-document retrieval riding the consistency floor.
    irrelevant = kb._calculate_confidence_score(
        [(doc, 0.92)], top_n=1, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE
    )
    best_confidence = kb._similarity_to_confidence(
        0.92, max_distance=HYBRID_PSEUDO_DISTANCE_RANGE
    )
    assert best_confidence < BEST_DOCUMENT_RELEVANCE_FLOOR
    assert irrelevant == pytest.approx(best_confidence)
    assert irrelevant < 0.5, (
        f"irrelevant best document cleared the 0.5 gate via the floor: {irrelevant}"
    )

    # A genuinely relevant document is unaffected by the floor: the blend is
    # best * (0.6 + 0.3*consistency + 0.1*coverage) with best = 1.0 and
    # coverage = 1/3 for 1 of 3.
    relevant = kb._calculate_confidence_score(
        [(doc, 0.0)], top_n=3, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE
    )
    assert relevant == pytest.approx(0.6 + 0.3 + (1 / 3) * 0.1)

    # INVARIANT: the structural factors may only scale a real match down. Every
    # score in the domain must stay at or below its best-document factor, and in
    # particular a zero-relevance retrieval must yield exactly zero.
    for distance in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        final = kb._calculate_confidence_score(
            [(doc, distance)], top_n=3, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE
        )
        assert final <= kb._similarity_to_confidence(
            distance, max_distance=HYBRID_PSEUDO_DISTANCE_RANGE
        ) + 1e-12, distance
    assert kb._calculate_confidence_score(
        [(doc, 1.0)], top_n=3, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE
    ) == 0.0

    # Monotonicity and continuity across the whole domain, including the former
    # relevance-floor boundary. A threshold-based cap used to jump from ~0.58 to
    # ~0.30 there. The multiplicative form must not add a cliff of its own, so
    # every adjacent step is bounded by the underlying best-document curve's own
    # step (the base curve is legitimately steep near distance 1.0).
    distances = [i / 500 for i in range(501)]
    base_scores = [
        kb._similarity_to_confidence(d, max_distance=HYBRID_PSEUDO_DISTANCE_RANGE)
        for d in distances
    ]
    scores = [
        kb._calculate_confidence_score(
            [(doc, d)], top_n=3, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE
        )
        for d in distances
    ]
    assert scores == sorted(scores, reverse=True), "confidence must be monotonic"
    base_step = max(abs(b - a) for a, b in zip(base_scores, base_scores[1:]))
    final_step = max(abs(b - a) for a, b in zip(scores, scores[1:]))
    assert final_step <= base_step + 1e-12, (
        f"structural scaling added a discontinuity: {final_step} > {base_step}"
    )


def test_kb_reuses_a_single_hybrid_service():
    """Regression (Bug 6): the hybrid service must not be rebuilt per call.

    Each rebuild discarded the BM25/doc caches, forcing a full Chroma scan and
    BM25 index build on every query.
    """
    kb = _make_kb_service()
    assert kb._hybrid_service is None  # built lazily

    first = kb._get_hybrid_service()
    second = kb._get_hybrid_service()

    assert first is second
    assert kb._hybrid_service is first

    # Replacing the document processor must invalidate the cached instance.
    replacement = MagicMock()
    replacement.get_vector_store.return_value = None
    replacement.embeddings = None
    kb.document_processor = replacement

    third = kb._get_hybrid_service()
    assert third is not first
    assert third.document_processor is replacement


def test_hybrid_cache_is_bounded_and_ttl_aware():
    """Regression (Bug 5): cache writes are capped and TTL-checked thread-safely."""
    svc = _make_service(MagicMock())
    cache: dict = {}

    for i in range(svc._MAX_CACHE_ENTRIES + 20):
        svc._store_cache(cache, f"k{i}", (i, float(i)))

    assert len(cache) == svc._MAX_CACHE_ENTRIES
    # Oldest timestamps evicted first, newest retained.
    assert f"k{svc._MAX_CACHE_ENTRIES + 19}" in cache
    assert "k0" not in cache

    # Fresh entry is returned, expired entry is not.
    svc._store_cache(cache, "fresh", ("v", time.time()))
    assert svc._get_cached_entry(cache, "fresh")[0] == "v"

    svc._store_cache(cache, "expired", ("v", time.time() - svc._cache_ttl_seconds - 1))
    assert svc._get_cached_entry(cache, "expired") is None

    # Negative-cache entries (value None) stay meaningful.
    svc._store_cache(cache, "negative", (None, time.time()))
    entry = svc._get_cached_entry(cache, "negative")
    assert entry is not None and entry[0] is None


def test_invalidate_caches_clears_both_caches():
    """`invalidate_caches` remains the explicit cache-drop hook."""
    svc = _make_service(MagicMock())
    svc._store_cache(svc._bm25_cache, "a", ("retriever", time.time()))
    svc._store_cache(svc._docs_cache, "b", ([], time.time()))

    svc.invalidate_caches()

    assert svc._bm25_cache == {}
    assert svc._docs_cache == {}
