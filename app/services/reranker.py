import hashlib
import logging
import threading
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# =====================================================================
# NumPy 2.x backwards compatibility shim
# NOTE(arch): بهتر است این shim در entry point مرکزی برنامه لود شود، نه اینجا،
# تا رفتار وابسته به ترتیب import نباشد. فعلاً برای عدم شکستن رفتار حفظ شده.
# =====================================================================


try:
    from numpy.core.multiarray import scalar as _np_scalar  # noqa: F401
except ImportError:
    pass
# =====================================================================

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# Worst possible pseudo-distance (smaller = better). Shared with
# `app.services.hybrid_retrieval` so the score contract stays in one place.
DEFAULT_PSEUDO_DISTANCE = 1.0


class _BoundedCache:
    """Thread-safe LRU cache with a hard size cap (prevents unbounded growth)."""

    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)


class EmbeddingReranker:
    """Embedding-based re-ranking using cosine similarity and L2 distance normalization.

    قرارداد امتیاز (score contract) — یکسان در تمام مسیرها (اصلی و fallback):
        امتیاز خروجی SIMILARITY است: بزرگ‌تر = بهتر، و مرتب‌سازی نزولی است.

    ورودی `original_scores` یک pseudo-distance است (کوچک‌تر = بهتر)، همان‌طور که
    caller در hybrid_retrieve با `1.0 - clamp(hybrid_score)` می‌سازد.
    """

    _MAX_QUERY_CACHE = 512
    _MAX_DOC_CACHE = 4096

    def __init__(self, embeddings_model: Any):
        self.embeddings = embeddings_model
        self.logger = logging.getLogger(__name__)
        self._query_cache = _BoundedCache(self._MAX_QUERY_CACHE)
        self._doc_cache = _BoundedCache(self._MAX_DOC_CACHE)

    # ---------- helpers ----------
    @staticmethod
    def _align_scores(scores: List[float], n: int) -> List[float]:
        """طول scores را با تعداد اسناد هم‌تراز می‌کند تا zip چیزی را بی‌صدا حذف نکند.

        اسناد اضافی pseudo-distance بدترین حالت (1.0) می‌گیرند.
        """
        scores = list(scores or [])
        if len(scores) < n:
            scores = scores + [DEFAULT_PSEUDO_DISTANCE] * (n - len(scores))
        elif len(scores) > n:
            scores = scores[:n]
        return scores

    @staticmethod
    def _doc_key(text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        try:
            cached = self._query_cache.get(query)
            if cached is not None:
                return cached
            vec = np.array(self.embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
            self._query_cache.put(query, vec)
            return vec
        except Exception as e:
            self.logger.error(f"[RERANKER] Query embedding failed: {e}")
            return None

    def _embed_documents(self, documents: List[Any]) -> Optional[np.ndarray]:
        """embed اسناد با کش بر اساس hash محتوا؛ فقط اسناد جدید به مدل ارسال می‌شوند."""
        try:
            texts = [doc.page_content for doc in documents]
            keys = [self._doc_key(t) for t in texts]

            cached_vecs: Dict[str, np.ndarray] = {}
            missing_idx: List[int] = []
            for i, k in enumerate(keys):
                v = self._doc_cache.get(k)
                if v is None:
                    missing_idx.append(i)
                else:
                    cached_vecs[k] = v

            if missing_idx:
                missing_texts = [texts[i] for i in missing_idx]
                new_vecs = np.array(
                    self.embeddings.embed_documents(missing_texts), dtype=np.float32
                )
                for j, i in enumerate(missing_idx):
                    vec = new_vecs[j]
                    self._doc_cache.put(keys[i], vec)
                    cached_vecs[keys[i]] = vec

            return np.array([cached_vecs[k] for k in keys], dtype=np.float32)
        except Exception as e:
            self.logger.error(f"[RERANKER] Document embedding failed: {e}")
            return None

    def _cosine_similarity(self, query_emb: np.ndarray, docs_emb: np.ndarray) -> np.ndarray:
        if _HAS_SKLEARN:
           return sk_cosine_similarity(query_emb, docs_emb)[0]
        q = query_emb.astype(np.float32)
        d = docs_emb.astype(np.float32)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        return np.dot(q_norm, d_norm.T)[0]

    def _fallback(
        self,
        documents: List[Any],
        original_scores: List[float],
        top_k: int,
        reason: str,
    ) -> List[Tuple[Any, float, Dict[str, Any]]]:
        """مسیر fallback با همان قرارداد مسیر اصلی: similarity (بزرگ‌تر = بهتر)."""
        scores = self._align_scores(original_scores, len(documents))
        out: List[Tuple[Any, float, Dict[str, Any]]] = []
        for i, doc in enumerate(documents):
            dist = float(scores[i])
            # pseudo-distance -> similarity. Clamped so an out-of-contract input
            # (distance > 1.0) can never yield a negative, contract-breaking score.
            sim = DEFAULT_PSEUDO_DISTANCE - max(0.0, min(1.0, dist))
            out.append(
                (
                    doc,
                    sim,
                    {
                        # کلیدهای متادیتا کامل پر می‌شوند تا مصرف‌کننده‌های پایین‌دستی KeyError نگیرند
                        "cosine_similarity": None,
                        "l2_similarity": float(sim),
                        "combined_score": float(sim),
                        "original_l2_distance": dist,
                        "fallback": True,
                        "fallback_reason": reason,
                    },
                )
            )
        # همان جهت مسیر اصلی: نزولی (بزرگ‌تر = بهتر)
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    # ---------- main ----------
    def rerank(
        self,
        query: str,
        documents: List[Any],
        original_scores: List[float],
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> List[Tuple[Any, float, Dict[str, Any]]]:
        """
        Re-rank با ترکیب cosine similarity و L2 نرمال‌شده (min-max).

        NOTE(arch): چون سیگنال L2 از همان مدل embedding می‌آید، این reranker عملاً
        رتبه‌بندی dense را تکرار می‌کند. برای بهبود واقعی کیفیت، یک Cross-Encoder
        (مثل BAAI/bge-reranker-v2-m3) پیشنهاد می‌شود.
        """
        if not documents:
            return []

        if self.embeddings is None:
            self.logger.warning("[RERANKER] Embeddings not available. Using consistent fallback.")
            return self._fallback(documents, original_scores, top_k, reason="no_embeddings")

        try:
            # طول scores را قبل از استفاده هم‌تراز کن
            scores = self._align_scores(original_scores, len(documents))

            # 1) Embed
            query_emb = self._embed_query(query)
            docs_emb = self._embed_documents(documents)
            if query_emb is None or docs_emb is None:
                raise RuntimeError("Embeddings could not be computed")

            # 2) Cosine similarity (بزرگ‌تر = بهتر)
            cosine_scores = self._cosine_similarity(query_emb, docs_emb)

            # 3) نرمال‌سازی min-max واقعی روی L2 pseudo-distance (کوچک‌تر = بهتر)
            #    similarity = (max - s) / (max - min) تا کل بازه استفاده شود.
            mn = min(scores)
            mx = max(scores)
            if mx == mn:
                normalized_l2 = [1.0 for _ in scores]
            else:
                span = mx - mn
                normalized_l2 = [(mx - s) / span for s in scores]

            # 4) Combine
            combined: List[Tuple[Any, float, Dict[str, Any]]] = []
            for i, (doc, cos, l2) in enumerate(zip(documents, cosine_scores, normalized_l2)):
                score = float(alpha * float(cos) + (1.0 - alpha) * float(l2))
                combined.append(
                    (
                        doc,
                        score,
                        {
                            "cosine_similarity": float(cos),
                            "l2_similarity": float(l2),
                            "combined_score": float(score),
                            "original_l2_distance": float(scores[i]),
                            "fallback": False,
                        },
                    )
                )

            # 5) Sort نزولی (بزرگ‌تر = بهتر)
            reranked = sorted(combined, key=lambda x: x[1], reverse=True)
            if reranked:
                top_meta = reranked[0][2]
                self.logger.info(
                    f"[RERANKER] Top: Cosine={top_meta['cosine_similarity']:.3f}, "
                    f"Combined={reranked[0][1]:.3f}"
                )
            return reranked[:top_k]

        except Exception as e:
            self.logger.error(f"[RERANKER] Error: {e}")
            # قرارداد یکسان با مسیر اصلی حفظ می‌شود (similarity، نزولی)
            return self._fallback(documents, original_scores, top_k, reason=str(e))
