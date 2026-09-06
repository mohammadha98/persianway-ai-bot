import asyncio
import json
import logging
import math
import re
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
try:
    import nltk as _nltk
except Exception:
    _nltk = None


logger = logging.getLogger(__name__)

def _ensure_nltk():
    if _nltk is None:
        return
    try:
        _nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            _nltk.download("punkt")
        except Exception:
            pass
    try:
        _nltk.data.find("tokenizers/punkt_tab")
    except Exception:
        try:
            _nltk.download("punkt_tab")
        except Exception:
            pass

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    try:
        _ensure_nltk()
        if _nltk is None:
            raise RuntimeError("nltk not available")
        return _nltk.word_tokenize(text)
    except Exception:
        return re.findall(r"[\w\u0600-\u06FF]+", text)

class HybridRetrievalService:
    _MAX_CACHE_ENTRIES = 128
    def __init__(self, document_processor):
        self.document_processor = document_processor
        self.vector_store = self.document_processor.get_vector_store()
        self._bm25_cache: Dict[str, Tuple[Optional[BM25Retriever], float]] = {}
        self._docs_cache: Dict[str, Tuple[List[Document], float]] = {}
        self._cache_ttl_seconds = 3600
        self.reranker = None

        try:
            from app.services.reranker import EmbeddingReranker

            embeddings = getattr(self.document_processor, "embeddings", None)
            if embeddings is not None:
                self.reranker = EmbeddingReranker(embeddings)
        except Exception as e:
            logger.warning(f"[HYBRID] Failed to initialize reranker: {e}")


    def _store_cache(self, cache: dict, key: str, value) -> None:
        """درج در کش با سقف اندازه (حذف قدیمی‌ترین ورودی‌ها به‌صورت FIFO ساده)."""
        cache[key] = value
        if len(cache) > self._MAX_CACHE_ENTRIES:
            # قدیمی‌ترین‌ها بر اساس timestamp حذف می‌شوند (value = (obj, ts))
            for old_key in sorted(cache, key=lambda k: cache[k][1])[: len(cache) - self._MAX_CACHE_ENTRIES]:
                cache.pop(old_key, None)

    def invalidate_caches(self) -> None:
        """پس از ingestion داده‌های جدید فراخوانی شود تا کش BM25/docs بازسازی شود."""
        self._bm25_cache.clear()
        self._docs_cache.clear()
        logger.info("[HYBRID] Caches invalidated (call this after new-data ingestion)")

    def _is_cache_valid(self, timestamp: float) -> bool:
        return (time.time() - timestamp) <= self._cache_ttl_seconds

    def _make_doc_key(self, doc: Document) -> str:
        return (
            doc.metadata.get("chroma_id")
            or doc.metadata.get("id")
            or doc.metadata.get("source")
            or str(hash(doc.page_content[:300]))
        )

    def _get_docs_for_filter(self, filt: Dict) -> List[Document]:
        cache_key = json.dumps(filt, sort_keys=True, ensure_ascii=False)
        cached = self._docs_cache.get(cache_key)
        if cached and self._is_cache_valid(cached[1]):
            return cached[0]

        coll = getattr(self.vector_store, "_collection", None)
        if coll is None:
            return []

        try:
            data = coll.get(where=filt)
            docs = []
            ids = data.get("ids") or []
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            for i in range(len(texts)):
                meta = metas[i] if i < len(metas) else {}
                meta = dict(meta or {})
                if i < len(ids):
                    meta["chroma_id"] = ids[i]
                docs.append(Document(page_content=texts[i], metadata=meta))

            self._store_cache(self._docs_cache, cache_key, (docs, time.time()))
            return docs
        except Exception as e:
            logger.error(f"[HYBRID] Error getting docs for filter: {e}")
            return []

    def _get_bm25(self, key: str, filt: Dict, is_public: bool = False, k: int = 15) -> Optional[BM25Retriever]:
        cache_key = f"{key}_k{k}_public_{is_public}"
        cached = self._bm25_cache.get(cache_key)
        if cached and self._is_cache_valid(cached[1]):
            return cached[0]

        docs = self._get_docs_for_filter(filt)
        if not docs:
            self._store_cache(self._bm25_cache, cache_key, (None, time.time()))
            return None

        retr = BM25Retriever.from_documents(docs, preprocess_func=_tokenize)
        retr.k = k
        self._store_cache(self._bm25_cache, cache_key, (retr, time.time()))
        return retr

    async def _dense_similarity_search(
        self,
        vs,
        query_embedding,
        k: int,
        filter_dict: Dict,
    ) -> List[Tuple[Document, float]]:
        """Run a dense branch and return REAL relevance scores (bigger = better).

        Critical fix: `asimilarity_search_by_vector` returns no score, so we used to
        assign a flat 1.0 to every doc, which made dense ranking completely flat.
        Prefer score-returning APIs; fall back to rank-based scoring.
        """
        # 1) Async score-returning API (higher = more relevant)
        scorer = getattr(vs, "asimilarity_search_by_vector_with_relevance_scores", None)
        if scorer is not None:
            try:
                scored = await scorer(query_embedding, k=k, filter=filter_dict)
                return [(doc, float(s)) for doc, s in scored]
            except Exception as e:
                logger.warning(f"[DENSE] async relevance-score API failed, trying sync: {e}")

        # 2) Sync score-returning API
        sync_scorer = getattr(vs, "similarity_search_by_vector_with_relevance_scores", None)
        if sync_scorer is not None:
            try:
                scored = await asyncio.to_thread(sync_scorer, query_embedding, k=k, filter=filter_dict)
                return [(doc, float(s)) for doc, s in scored]
            except Exception as e:
                logger.warning(f"[DENSE] sync relevance-score API failed, falling back to rank scoring: {e}")

        # 3) Fallback: rank-based scoring (higher = better)
        results = await vs.asimilarity_search_by_vector(query_embedding, k=k, filter=filter_dict)
        return [(doc, 1.0 - (i / max(k - 1, 1))) for i, doc in enumerate(results)]

    async def _dense_parallel(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
        """Dense vector search across 3 branches with SINGLE shared embedding.
        
        BEFORE FIX:
        - 3 branches × embed_query() = 3 API calls (~5.4s cold)
        
        AFTER FIX:
        - 1 embed_query() + 3 vector searches = 1 API call (~1.3s cold)
        """
        if not query.strip() or not self.vector_store:
            return []

        vs = self.vector_store

        # ===== FIX: Create embedding ONCE (not 3x) =====
        t_embed_start = time.perf_counter()
        try:
            query_embedding = await asyncio.to_thread(vs.embeddings.embed_query, query)
            embed_elapsed = time.perf_counter() - t_embed_start
            logger.info(f"[EMBED_SHARED] query={query[:50]}... elapsed={embed_elapsed:.3f}s dims={len(query_embedding) if query_embedding else 0}")
        except Exception as e:
            logger.error(f"[EMBED_SHARED] FAILED: {e}")
            return []

        # Define filters for 3 branches (same logic as before)
        filters = {
            "contrib": {"$and": [
                {"entry_type": {"$in": ["user_contribution"]}},
                {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}}
            ]},
            "docx": {"$and": [
                {"entry_type": {"$in": ["user_contribution_docx"]}},
                {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}}
            ]},
            "excel": {"$and": [
                {"entry_type": {"$in": ["user_contribution_excel"]}},
                {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}}
            ]},
        }

        # ===== FIX: Use asimilarity_search_by_vector (no additional embedding) =====
        async def search_branch(filter_name: str, filter_dict: Dict) -> Tuple[str, List[Tuple[Document, float]]]:
            """Search one branch using pre-computed embedding."""
            branch_t0 = time.perf_counter()
            try:
                scored_results = await self._dense_similarity_search(
                    vs,
                    query_embedding,
                    k,
                    filter_dict,
                )
                elapsed = time.perf_counter() - branch_t0
                logger.info(f"[DENSE_BRANCH] branch={filter_name} elapsed={elapsed:.3f}s docs={len(scored_results)}")
                return (filter_name, scored_results)
            except Exception as e:
                elapsed = time.perf_counter() - branch_t0
                logger.error(f"[DENSE_BRANCH] branch={filter_name} failed after {elapsed:.3f}s: {e}")
                return (filter_name, [])

        # Execute 3 branches in parallel (only vector search, no additional embedding)
        tasks = [search_branch(name, filt) for name, filt in filters.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined: List[Tuple[Document, float]] = []
        branch_results: Dict[str, List[Tuple[Document, float]]] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[DENSE_PARALLEL] Exception: {result}")
                continue
            name, docs = result
            branch_results[name] = docs
            combined.extend(docs)

        total_elapsed = time.perf_counter() - t_embed_start
        logger.info(
            f"[DENSE_TOTAL] elapsed={total_elapsed:.3f}s embed={embed_elapsed:.3f}s "
            f"branches={len(branch_results)} docs={len(combined)}"
        )
        logger.info(
            f"[DENSE_BREAKDOWN] contrib={len(branch_results.get('contrib', []))} docs, "
            f"docx={len(branch_results.get('docx', []))} docs, "
            f"excel={len(branch_results.get('excel', []))} docs"
        )
        return combined

    def _bm25_parallel_old(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
        if not query.strip():
            return []
        # Create filters with is_public metadata filtering
        filters = {
            "contrib": {"$and": [
                {"entry_type": {"$in": ["user_contribution"]}},
                {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}}
            ]},
            "docx": {"$and": [
                {"entry_type": {"$in": ["user_contribution_docx"]}},
                {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}}
            ]},
            "excel": {"$and": [
                {"entry_type": {"$in": ["user_contribution_excel"]}},
                {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}}
            ]},
        }
        combined: List[Tuple[Document, float]] = []
        for key, f in filters.items():
            retr = self._get_bm25(key, f, is_public=is_public, k=k)
            if retr is None:
                continue
            docs = retr.get_relevant_documents(query)
            top = docs[:k]
            for i, d in enumerate(top):
                score = 1.0 - (i / max(k - 1, 1))
                combined.append((d, float(score)))
        return combined

    async def _bm25_parallel_async(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
        """Parallel BM25 retrieval for 3 branches with async thread offloading."""
        if not query or not query.strip():
            return []

        filters = {
            "contrib": {
                "$and": [
                    {"entry_type": {"$in": ["user_contribution"]}},
                    {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}},
                ]
            },
            "docx": {
                "$and": [
                    {"entry_type": {"$in": ["user_contribution_docx"]}},
                    {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}},
                ]
            },
            "excel": {
                "$and": [
                    {"entry_type": {"$in": ["user_contribution_excel"]}},
                    {"is_public": {"$eq": True}} if is_public else {"is_public": {"$ne": True}},
                ]
            },
        }

        async def run_one_bm25(key: str, filt: Dict[str, Any]) -> List[Tuple[Document, float]]:
            try:
                cache_key = f"{key}_k{k}_public_{is_public}"
                cached = self._bm25_cache.get(cache_key)
                retriever = cached[0] if (cached and self._is_cache_valid(cached[1])) else None

                if retriever is None:
                    docs = await asyncio.to_thread(self._get_docs_for_filter, filt)
                    if not docs:
                        self._store_cache(self._bm25_cache, cache_key, (None, time.time()))
                        return []

                    retriever = await asyncio.to_thread(BM25Retriever.from_documents, docs, preprocess_func=_tokenize)
                    retriever.k = k
                    self._store_cache(self._bm25_cache, cache_key, (retriever, time.time()))

                results = await asyncio.to_thread(retriever.get_relevant_documents, query)
                scored_results: List[Tuple[Document, float]] = []
                for i, doc in enumerate((results or [])[:k]):
                    score = 1.0 - (i / max(k - 1, 1))
                    scored_results.append((doc, float(score)))
                return scored_results
            except Exception as e:
                logger.error(f"[HYBRID] BM25 error for {key}: {e}")
                return []

        tasks = [run_one_bm25(key, filt) for key, filt in filters.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined: List[Tuple[Document, float]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            combined.extend(result or [])
        return combined

    def _bm25_parallel(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
        """Backward-compatible sync wrapper."""
        return self._bm25_parallel_old(query, k, is_public)

    def _normalize_dense(self, pairs: List[Tuple[Document, float]]) -> Dict[str, float]:
        # NOTE: dense scores are now relevance scores (bigger = better), matching
        # BM25 semantics, so we min-max normalize WITHOUT the old distance inversion.
        if not pairs:
            return {}
        scores = [s for _, s in pairs]
        mx = max(scores)
        mn = min(scores)
        if mx == mn:
            norm = [1.0 for _ in scores]
        else:
            norm = [(s - mn) / (mx - mn) for s in scores]
        res: Dict[str, float] = {}
        for (doc, _), n in zip(pairs, norm):
            key = self._make_doc_key(doc)
            res[key] = n
        return res

    def _normalize_bm25(self, pairs: List[Tuple[Document, float]]) -> Dict[str, float]:
        if not pairs:
            return {}
        scores = [s for _, s in pairs]
        mx = max(scores)
        mn = min(scores)
        if mx == mn:
            norm = [1.0 for _ in scores]
        else:
            norm = [(s - mn) / (mx - mn) for s in scores]
        res: Dict[str, float] = {}
        for (doc, _), n in zip(pairs, norm):
            key = self._make_doc_key(doc)
            res[key] = n
        return res

    async def _rerank_async(
        self,
        query: str,
        doc_score_pairs: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Document]:
        """Async wrapper for sync reranker with graceful fallback."""
        if not doc_score_pairs:
            return []

        if self.reranker is None:
            return [doc for doc, _ in doc_score_pairs[:top_k]]

        try:
            docs = [doc for doc, _ in doc_score_pairs]
            original_scores = [1.0 - max(0.0, min(1.0, score)) for _, score in doc_score_pairs]
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                query,
                docs,
                original_scores,
                top_k,
                0.7,
            )
            if not reranked:
                return [doc for doc, _ in doc_score_pairs[:top_k]]
            return [doc for doc, _, _ in reranked[:top_k]]
        except Exception as e:
            logger.error(f"[HYBRID] Rerank error: {e}")
            return [doc for doc, _ in doc_score_pairs[:top_k]]

    async def hybrid_retrieve(self, query: str, is_public: bool = False) -> List[Document]:
        """Hybrid retrieval combining dense vector search and BM25 keyword search.
        
        PERF: This method includes detailed timing instrumentation for performance analysis.
        """
        k = 15
        prefilter_k = 20
        overall_start = time.perf_counter()
        
        # === PERF: Detailed Timing Tracking ===
        timings = {}

        # === Branch 1: Dense Vector Search (all 3 branches in parallel) ===
        dense_start = time.perf_counter()
        dense_pairs = await self._dense_parallel(query, k, is_public)
        dense_elapsed = time.perf_counter() - dense_start
        timings['dense'] = dense_elapsed
        logger.info(f"[PERF_HYBRID] step=dense_search elapsed={dense_elapsed:.3f}s docs={len(dense_pairs)}")

        # === Branch 2-3: BM25 Search (3 branches in parallel) ===
        bm25_start = time.perf_counter()
        try:
            bm25_pairs = await self._bm25_parallel_async(query, k, is_public)
        except Exception as e:
            logger.warning(f"[HYBRID] Async BM25 failed, falling back to sync (offloaded): {e}")
            bm25_pairs = await asyncio.to_thread(self._bm25_parallel_old, query, k, is_public)
        bm25_elapsed = time.perf_counter() - bm25_start
        timings['bm25'] = bm25_elapsed
        logger.info(f"[PERF_HYBRID] step=bm25_search elapsed={bm25_elapsed:.3f}s docs={len(bm25_pairs)}")

        # === Merge & Normalize ===
        merge_start = time.perf_counter()
        dense_norm = self._normalize_dense(dense_pairs)
        bm25_norm = self._normalize_bm25(bm25_pairs)

        doc_map: Dict[str, Document] = {}
        combined_scores: Dict[str, float] = defaultdict(float)

        for doc, _ in dense_pairs:
            key = self._make_doc_key(doc)
            doc_map[key] = doc
            combined_scores[key] += 0.6 * dense_norm.get(key, 0.0)

        for doc, _ in bm25_pairs:
            key = self._make_doc_key(doc)
            doc_map[key] = doc
            combined_scores[key] += 0.4 * bm25_norm.get(key, 0.0)

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        top_doc_pairs: List[Tuple[Document, float]] = []
        for doc_id, score in sorted_docs[:prefilter_k]:
            doc = doc_map.get(doc_id)
            if doc is not None:
                top_doc_pairs.append((doc, score))
        merge_elapsed = time.perf_counter() - merge_start
        timings['merge'] = merge_elapsed
        logger.info(f"[PERF_HYBRID] step=merge_normalize elapsed={merge_elapsed:.3f}s combined_docs={len(top_doc_pairs)}")

        # === Reranking ===
        rerank_start = time.perf_counter()
        reranked_docs = await self._rerank_async(query, top_doc_pairs, top_k=k)
        rerank_elapsed = time.perf_counter() - rerank_start
        timings['reranking'] = rerank_elapsed
        logger.info(f"[PERF_HYBRID] step=reranking elapsed={rerank_elapsed:.3f}s")

        # === Sort & Format Output ===
        reranked_by_key = {self._make_doc_key(d): idx for idx, d in enumerate(reranked_docs)}

        out: List[Document] = []
        for doc, hs in top_doc_pairs:
            key = self._make_doc_key(doc)
            if key not in reranked_by_key:
                continue
            ds = dense_norm.get(key, 0.0)
            bs = bm25_norm.get(key, 0.0)
            meta = dict(doc.metadata or {})
            meta["dense_score_norm"] = ds
            meta["bm25_score_norm"] = bs
            meta["hybrid_score"] = hs
            meta["rerank_position"] = reranked_by_key[key]
            out.append(Document(page_content=doc.page_content, metadata=meta))

        out.sort(key=lambda d: d.metadata.get("rerank_position", 999999))
        
        timings['total_hybrid'] = time.perf_counter() - overall_start
        logger.info(
            f"[PERF_HYBRID] HYBRID_TIMING: {timings}"
        )
        return out[:k]
