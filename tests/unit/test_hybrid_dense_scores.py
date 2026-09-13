"""Regression tests for the Chroma "relevance score" inversion bug.

`similarity_search_by_vector_with_relevance_scores` in langchain_community's
Chroma returns raw DISTANCES (lower = better) despite the misleading name.
`HybridRetrievalService._dense_similarity_search` must convert them into
REAL relevance scores (higher = better) before `_normalize_dense` min-max
normalizes them, otherwise the closest document receives the lowest weight.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain.schema import Document  # noqa: E402

from app.services.hybrid_retrieval import HybridRetrievalService  # noqa: E402


def _make_service(vector_store) -> HybridRetrievalService:
    """Build the service with a stubbed document processor (no embeddings/reranker)."""
    dp = MagicMock()
    dp.get_vector_store.return_value = vector_store
    dp.embeddings = None  # disables the optional reranker
    return HybridRetrievalService(dp)


class _FakeChroma:
    """Minimal stand-in whose relevance API returns DISTANCES like real Chroma."""

    def __init__(self, scored, relevance_fn):
        self._scored = scored
        self._relevance_fn = relevance_fn

    def _select_relevance_score_fn(self):
        return self._relevance_fn

    def similarity_search_by_vector_with_relevance_scores(self, embedding, k=4, filter=None):
        return self._scored[:k]


@pytest.mark.asyncio
async def test_l2_distances_are_inverted_to_relevance():
    """A smaller L2 distance must yield a HIGHER relevance score (regression)."""
    from langchain_core.vectorstores import VectorStore

    near = Document(page_content="near", metadata={"id": "near"})
    far = Document(page_content="far", metadata={"id": "far"})
    # NOTE: real API returns DISTANCES, closest first.
    vs = _FakeChroma(
        [(near, 0.1), (far, 0.9)],
        relevance_fn=VectorStore._euclidean_relevance_score_fn,
    )
    svc = _make_service(vs)

    pairs = await svc._dense_similarity_search(vs, [0.0, 0.0], k=2, filter_dict={})

    scores = {svc._make_doc_key(d): s for d, s in pairs}
    assert scores["near"] > scores["far"], "closest document must score higher"

    # End-to-end normalization: closest -> 1.0, farthest -> 0.0
    norm = svc._normalize_dense(pairs)
    assert norm["near"] == pytest.approx(1.0)
    assert norm["far"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cosine_distance_conversion_matches_one_minus_distance():
    """For cosine space the converter is 1 - distance."""
    from langchain_core.vectorstores import VectorStore

    a = Document(page_content="a", metadata={"id": "a"})
    b = Document(page_content="b", metadata={"id": "b"})
    vs = _FakeChroma(
        [(a, 0.2), (b, 0.8)],
        relevance_fn=VectorStore._cosine_relevance_score_fn,
    )
    svc = _make_service(vs)

    pairs = await svc._dense_similarity_search(vs, [0.0, 0.0], k=2, filter_dict={})
    scores = {svc._make_doc_key(d): s for d, s in pairs}

    assert scores["a"] == pytest.approx(0.8)
    assert scores["b"] == pytest.approx(0.2)
    assert scores["a"] > scores["b"]


@pytest.mark.asyncio
async def test_rank_fallback_is_higher_is_better():
    """When no score-returning API exists, the first (best) rank scores highest."""

    class _RankOnlyChroma:
        async def asimilarity_search_by_vector(self, embedding, k=4, filter=None):
            return [
                Document(page_content="first", metadata={"id": "first"}),
                Document(page_content="last", metadata={"id": "last"}),
            ]

    svc = _make_service(_RankOnlyChroma())
    pairs = await svc._dense_similarity_search(
        _RankOnlyChroma(), [0.0], k=2, filter_dict={}
    )

    assert pairs[0][1] > pairs[1][1]


class _DeterministicEmbeddings:
    """Local embedding function: no network, deterministic known geometry."""

    _VECTORS = {
        "query": [1.0, 0.0, 0.0],
        "near": [0.99, 0.05, 0.0],
        "far": [0.0, 0.0, 1.0],
    }

    def embed_documents(self, texts):
        return [self._VECTORS[t] for t in texts]

    def embed_query(self, text):
        return self._VECTORS[text]


@pytest.mark.asyncio
async def test_real_chroma_l2_collection_direction():
    """Validate direction against real Chroma with the production (l2) default."""
    chromadb = pytest.importorskip("chromadb")
    from langchain_community.vectorstores import Chroma

    client = chromadb.EphemeralClient()
    # Production calls get_or_create_collection(name=...) with NO metadata,
    # which makes Chroma use its default l2 space.
    collection = client.get_or_create_collection(name="hybrid_dense_test")
    assert collection.metadata is None  # same as production

    embeddings = _DeterministicEmbeddings()
    vs = Chroma(
        client=client,
        collection_name="hybrid_dense_test",
        embedding_function=embeddings,
    )
    vs.add_documents(
        [
            Document(page_content="near", metadata={"id": "near"}),
            Document(page_content="far", metadata={"id": "far"}),
        ]
    )

    svc = _make_service(vs)
    query_embedding = embeddings.embed_query("query")
    pairs = await svc._dense_similarity_search(vs, query_embedding, k=2, filter_dict=None)

    scores = {svc._make_doc_key(d): s for d, s in pairs}
    assert set(scores) == {"near", "far"}
    assert scores["near"] > scores["far"], (
        "REGRESSION: closest doc must have the higher dense relevance score"
    )
