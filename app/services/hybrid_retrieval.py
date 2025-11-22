import asyncio
import math
import re
from typing import List, Dict, Tuple
import numpy as np
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
try:
    import nltk as _nltk
except Exception:
    _nltk = None

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
    def __init__(self, document_processor):
        self.document_processor = document_processor
        self.vector_store = self.document_processor.get_vector_store()
        self._bm25_cache: Dict[str, BM25Retriever] = {}

    def _get_docs_for_filter(self, filt: Dict) -> List[Document]:
        coll = getattr(self.vector_store, "_collection", None)
        if coll is None:
            return []
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
        return docs

    def _get_bm25(self, key: str, filt: Dict):
        if key in self._bm25_cache:
            return self._bm25_cache[key]
        docs = self._get_docs_for_filter(filt)
        if not docs:
            self._bm25_cache[key] = None
            return None
        retr = BM25Retriever.from_documents(docs, preprocess_func=_tokenize)
        self._bm25_cache[key] = retr
        return retr

    async def _dense_parallel(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
        if not query.strip():
            return []
        vs = self.vector_store
        if vs is None:
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
        async def run(f):
            return await asyncio.to_thread(vs.similarity_search_with_score, query, k=k, filter=f)
        results = await asyncio.gather(*[run(f) for f in filters.values()], return_exceptions=True)
        combined: List[Tuple[Document, float]] = []
        for r in results:
            if isinstance(r, Exception):
                continue
            combined.extend(r or [])
        return combined

    def _bm25_parallel(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
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
            retr = self._get_bm25(key, f)
            if retr is None:
                continue
            docs = retr.get_relevant_documents(query)
            top = docs[:k]
            for i, d in enumerate(top):
                score = 1.0 - (i / max(k - 1, 1))
                combined.append((d, float(score)))
        return combined

    def _normalize_dense(self, pairs: List[Tuple[Document, float]]) -> Dict[str, float]:
        if not pairs:
            return {}
        scores = [s for _, s in pairs]
        mx = max(scores)
        mn = min(scores)
        inv = [mx - s for s in scores]
        mx2 = max(inv)
        mn2 = min(inv)
        if mx2 == mn2:
            norm = [1.0 for _ in inv]
        else:
            norm = [(v - mn2) / (mx2 - mn2) for v in inv]
        res: Dict[str, float] = {}
        for (doc, _), n in zip(pairs, norm):
            key = doc.metadata.get("chroma_id") or doc.metadata.get("source") or str(hash(doc.page_content[:300]))
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
            key = doc.metadata.get("chroma_id") or doc.metadata.get("source") or str(hash(doc.page_content[:300]))
            res[key] = n
        return res

    async def hybrid_retrieve(self, query: str, is_public: bool = False) -> List[Document]:
        k = 15
        dense_pairs = await self._dense_parallel(query, k, is_public)
        bm25_pairs = self._bm25_parallel(query, k, is_public)
        dense_norm = self._normalize_dense(dense_pairs)
        bm25_norm = self._normalize_bm25(bm25_pairs)
        keys = set(dense_norm.keys()) | set(bm25_norm.keys())
        scored: List[Tuple[Document, float, float, float]] = []
        for key in keys:
            d = next((doc for doc, _ in dense_pairs if (doc.metadata.get("chroma_id") or doc.metadata.get("source") or str(hash(doc.page_content[:300]))) == key), None)
            if d is None:
                d = next((doc for doc, _ in bm25_pairs if (doc.metadata.get("chroma_id") or doc.metadata.get("source") or str(hash(doc.page_content[:300]))) == key), None)
            ds = dense_norm.get(key, 0.0)
            bs = bm25_norm.get(key, 0.0)
            hs = 0.7 * ds + 0.3 * bs
            scored.append((d, ds, bs, hs))
        scored.sort(key=lambda x: x[3], reverse=True)
        out: List[Document] = []
        for doc, ds, bs, hs in scored:
            meta = dict(doc.metadata or {})
            meta["dense_score_norm"] = ds
            meta["bm25_score_norm"] = bs
            meta["hybrid_score"] = hs
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out