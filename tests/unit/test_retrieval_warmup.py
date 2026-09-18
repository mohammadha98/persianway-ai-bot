"""Tests for the retrieval warm-up and the off-loop blocking-I/O fixes.

Production symptom these guard against: the first chat request served by a fresh
worker did the Chroma/BM25 cold start inline, on the event loop. The loop froze,
gunicorn stopped receiving heartbeats and killed the worker (`WORKER TIMEOUT ...
SIGABRT`) while the client was still waiting for its first byte.

Covered here:
  * `HybridRetrievalService.prewarm()` builds and caches the BM25 branch indexes
    (so the first query reuses them instead of paying the cost itself),
  * `_bm25_parallel_async` reuses that cache and does not re-read Chroma,
  * the collection-count probe runs in a worker thread, never on the loop thread,
  * a failed embedding probe backs off instead of blocking every request for the
    full 30s timeout.
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.document_processor import (  # noqa: E402
    EMBEDDING_PROBE_RETRY_INTERVAL_SECONDS,
    DocumentProcessor,
)
from app.services.hybrid_retrieval import HybridRetrievalService  # noqa: E402


class _FakeCollection:
    """Minimal `chromadb` collection: `get(where=...)` + `count()`."""

    def __init__(self, docs_by_entry_type):
        self._docs = docs_by_entry_type
        self.get_calls = []
        self.count_thread_ids = []

    def count(self):
        self.count_thread_ids.append(threading.get_ident())
        return sum(len(docs) for docs in self._docs.values())

    def get(self, where=None):
        self.get_calls.append(where)
        entry_types = where["$and"][0]["entry_type"]["$in"]
        ids, texts, metas = [], [], []
        for entry_type in entry_types:
            for i, text in enumerate(self._docs.get(entry_type, [])):
                ids.append(f"{entry_type}-{i}")
                texts.append(text)
                metas.append({"entry_type": entry_type, "is_public": False})
        return {"ids": ids, "documents": texts, "metadatas": metas}


class _FakeVectorStore:
    def __init__(self, collection):
        self._collection = collection


def _make_hybrid_service(docs_by_entry_type):
    collection = _FakeCollection(docs_by_entry_type)
    processor = type("FakeProcessor", (), {})()
    processor.get_vector_store = lambda: _FakeVectorStore(collection)
    processor.embeddings = None  # disables the optional reranker
    return HybridRetrievalService(processor), collection


_DOCS = {
    "user_contribution": ["wheat needs nitrogen", "urea is 46% nitrogen"],
    "user_contribution_docx": ["potassium for fruit quality"],
    "user_contribution_excel": ["fertilizer price per ton"],
}
@pytest.mark.asyncio
async def test_prewarm_builds_and_caches_every_bm25_branch():
    """`warm_up` pays the cold cost at boot: all three branches end up cached."""
    service, collection = _make_hybrid_service(_DOCS)

    ready = await service.prewarm(is_public=False, k=15)

    assert ready == 3
    # One Chroma read per branch ...
    assert len(collection.get_calls) == 3
    # ... and each branch is now served from the cache.
    for key in ("contrib", "docx", "excel"):
        assert service._bm25_cache[f"{key}_k15_public_False"][0] is not None


@pytest.mark.asyncio
async def test_bm25_retrieval_reuses_the_warmed_cache():
    """After the warm-up, a query must not touch Chroma again."""
    service, collection = _make_hybrid_service(_DOCS)
    await service.prewarm()
    reads_after_warmup = len(collection.get_calls)

    pairs = await service._bm25_parallel_async("wheat nitrogen", k=15, is_public=False)

    assert pairs, "the warmed BM25 indexes must still return documents"
    assert len(collection.get_calls) == reads_after_warmup


@pytest.mark.asyncio
async def test_collection_count_probe_runs_off_the_event_loop():
    """`coll.count()` is blocking, so it must never run on the loop thread."""
    service, collection = _make_hybrid_service(_DOCS)
    loop_thread = threading.get_ident()

    await service.refresh_caches_if_collection_changed_async()

    assert collection.count_thread_ids, "the probe must have run"
    assert all(tid != loop_thread for tid in collection.count_thread_ids)


class _FailingEmbeddings:
    """Embeddings client whose probe always fails; counts the attempts."""

    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        raise RuntimeError("embeddings endpoint unavailable")


def _bare_processor(embeddings):
    """A `DocumentProcessor` without `__init__` (no dirs/settings/network)."""
    processor = DocumentProcessor.__new__(DocumentProcessor)
    processor.embeddings = embeddings
    processor.embeddings_available = False
    processor._provider = "test"
    processor._model_name = "test"
    processor._dim = 0
    processor._last_probe_failure_at = 0.0
    return processor


def test_embedding_probe_backs_off_after_a_failure():
    """A broken endpoint must not cost the 30s probe on *every* request."""
    embeddings = _FailingEmbeddings()
    processor = _bare_processor(embeddings)

    assert processor._ensure_embeddings() is False
    assert embeddings.calls == 1

    # Requests within the backoff window fail fast, without probing again.
    assert processor._ensure_embeddings() is False
    assert processor._ensure_embeddings() is False
    assert embeddings.calls == 1


def test_embedding_probe_retries_once_the_backoff_expired():
    """The retry still happens - just not on the critical path of every request."""
    embeddings = _FailingEmbeddings()
    processor = _bare_processor(embeddings)

    assert processor._ensure_embeddings() is False
    assert embeddings.calls == 1

    processor._last_probe_failure_at = time.monotonic() - EMBEDDING_PROBE_RETRY_INTERVAL_SECONDS - 1

    assert processor._ensure_embeddings() is False
    assert embeddings.calls == 2

