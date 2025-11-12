import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


class EmbeddingReranker:
    """Embedding-based re-ranking using cosine similarity and L2 distance normalization.

    - Uses the provided embeddings model to embed the query and documents.
    - Computes cosine similarity between query and document embeddings.
    - Normalizes L2 distance scores so that higher is better.
    - Combines cosine and normalized L2 with a tunable alpha.
    """

    def __init__(self, embeddings_model: Any):
        self.embeddings = embeddings_model
        self.logger = logging.getLogger(__name__)
        self._query_cache: Dict[str, np.ndarray] = {}

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        try:
            if query in self._query_cache:
                return self._query_cache[query]
            vec = np.array(self.embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
            self._query_cache[query] = vec
            return vec
        except Exception as e:
            self.logger.error(f"[RERANKER] Query embedding failed: {e}")
            return None

    def _embed_documents(self, documents: List[Any]) -> Optional[np.ndarray]:
        try:
            texts = [doc.page_content for doc in documents]
            vecs = np.array(self.embeddings.embed_documents(texts), dtype=np.float32)
            return vecs
        except Exception as e:
            self.logger.error(f"[RERANKER] Document embedding failed: {e}")
            return None

    def _cosine_similarity(self, query_emb: np.ndarray, docs_emb: np.ndarray) -> np.ndarray:
        if _HAS_SKLEARN:
            return sk_cosine_similarity(query_emb, docs_emb)[0]
        # Fallback: manual cosine similarity with numpy
        q = query_emb.astype(np.float32)
        d = docs_emb.astype(np.float32)
        # Normalize
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        sim = np.dot(q_norm, d_norm.T)[0]
        return sim

    def rerank(
        self,
        query: str,
        documents: List[Any],
        original_scores: List[float],
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> List[Tuple[Any, float, Dict[str, float]]]:
        """
        Re-rank documents using cosine similarity and normalized L2 distance.

        Args:
            query: User query text
            documents: List of LangChain Document objects
            original_scores: L2 distance scores (lower is better)
            top_k: Number of top results to return
            alpha: Weight for cosine similarity (1-alpha for normalized L2)

        Returns:
            List of tuples: (document, combined_score, metadata)
        """
        if not documents:
            return []

        if self.embeddings is None:
            self.logger.warning("[RERANKER] Embeddings not available. Using original ordering.")
            return [(doc, 1.0, {"fallback": True}) for doc in documents][:top_k]

        try:
            # 1) Embed query and documents
            query_emb = self._embed_query(query)
            docs_emb = self._embed_documents(documents)
            if query_emb is None or docs_emb is None:
                raise RuntimeError("Embeddings could not be computed")

            # 2) Cosine similarity
            cosine_scores = self._cosine_similarity(query_emb, docs_emb)

            # 3) Normalize L2 distance scores (higher is better)
            max_l2 = max(original_scores) if original_scores else 1.0
            if max_l2 <= 0:
                max_l2 = 1.0
            normalized_l2 = [1.0 - (s / max_l2) for s in original_scores]

            # 4) Combine scores
            combined: List[Tuple[Any, float, Dict[str, float]]] = []
            for i, (doc, cos, l2) in enumerate(zip(documents, cosine_scores, normalized_l2)):
                score = float(alpha * cos + (1.0 - alpha) * l2)
                combined.append(
                    (
                        doc,
                        score,
                        {
                            "cosine_similarity": float(cos),
                            "l2_similarity": float(l2),
                            "combined_score": float(score),
                            "original_l2_distance": float(original_scores[i]),
                        },
                    )
                )

            # 5) Sort by combined score (descending)
            reranked = sorted(combined, key=lambda x: x[1], reverse=True)
            if reranked:
                top_meta = reranked[0][2]
                self.logger.info(
                    f"[RERANKER] Top: Cosine={top_meta['cosine_similarity']:.3f}, Combined={reranked[0][1]:.3f}"
                )
            return reranked[:top_k]

        except Exception as e:
            self.logger.error(f"[RERANKER] Error: {e}")
            # Fallback to original scores if available, else equal scores
            fallback = []
            for i, doc in enumerate(documents):
                orig = original_scores[i] if i < len(original_scores) else 0.0
                fallback.append((doc, orig, {"fallback": True}))
            # Sort by inverse of L2 (lower distance = better score)
            fallback_sorted = sorted(fallback, key=lambda x: x[1])
            return fallback_sorted[:top_k]