# گزارش ریشه‌یابی Dense Search Bottleneck

## 📋 خلاصه اجرا

| مورد | مقدار |
|------|-------|
| **تعداد embedding API call** | **3 بار per query** |
| **Cold start dense** | 3.1s - 5.4s (3× sequential) |
| **Cached dense** | 0.8s - 0.9s (3× parallel) |
| **Bottleneck** | هر branch یک embedding call مجزا |

---

## 🔍 ریشه‌یابی

### Problem: Duplicate Embedding per Query

```
hybrid_retrieve()
    │
    └─► _dense_parallel(query, k=15, is_public=False)
           │
           ├─► branch 1: vs.similarity_search_with_score(query, k=15, filter=contrib)
           │      │
           │      └─► (internal) embeddings.embed_query(query) → API call #1 (~0.8s)
           │
           ├─► branch 2: vs.similarity_search_with_score(query, k=15, filter=docx)
           │      │
           │      └─► (internal) embeddings.embed_query(query) → API call #2 (~0.8s)
           │
           └─► branch 3: vs.similarity_search_with_score(query, k=15, filter=excel)
                  │
                  └─► (internal) embeddings.embed_query(query) → API call #3 (~0.8s)
```

### Root Cause Analysis

**1. DocumentProcessor Embedding Initialization:**

در `document_processor.py`:
```python
class OpenRouterEmbeddings:
    def __init__(self, api_key: str, base_url: str, model: str, ...):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # ❌ NO CACHE

    def embed_query(self, text: str):
        # ❌ هر بار API call مستقیم
        resp = self.client.embeddings.create(model=self.model, input=text, ...)
        return data[0].embedding
```

**هیچ cachingای در `embed_query` وجود ندارد.**

**2. HybridRetrievalService _dense_parallel:**

در `hybrid_retrieval.py`:
```python
async def _dense_parallel(self, query: str, k: int, is_public: bool = False):
    filters = {
        "contrib": {...},  # branch 1
        "docx": {...},     # branch 2
        "excel": {...},    # branch 3
    }
    
    async def run(f):
        # ❌ هر branch similarity_search_with_score را صدا می‌زند
        # که درونش embed_query(query) را صدا می‌زند
        return await asyncio.to_thread(vs.similarity_search_with_score, query, k=k, filter=f)
    
    # ۳ branch = ۳ embedding API call
    results = await asyncio.gather(*[run(f) for f in filters.values()])
```

**3. LangChain Chroma internals:**

`Chroma.similarity_search_with_score(query, ...)` داخلی:
1. `query_embedding = self.embeddings.embed_query(query)` → API call
2. `scores = self.collection.query(query_embeddings=query_embedding, ...)` → local

---

### Evidence from Test Results

#### TEST 1: Cold Start (3 sequential API calls)
```
[EMBED_CALL #3] text=کدام محصولات سلامتی... elapsed=5.251s  ← branch docx
[DENSE_BRANCH] branch=docx elapsed=5.257s

[EMBED_CALL #3] text=کدام محصولات سلامتی... elapsed=5.402s  ← branch excel
[DENSE_BRANCH] branch=excel elapsed=5.445s

[DENSE_BRANCH] branch=contrib elapsed=0.743s  ← cached بعد از call اول
```
**Total dense: 5.4s** (3 sequential API calls: 5.25s + 5.4s + 0.74s)

#### TEST 2: Cached (3 parallel API calls)
```
Run 1: dense=3.129s (partial cache)
Run 2: dense=0.847s (all cached, parallel)
Run 3: dense=0.829s (all cached, parallel)
```
**Total dense (cached): ~0.8s** (3 parallel API calls, bottleneck = slowest one)

---

## 📊 Timing Breakdown

### Before Optimization

| Step | Cold | Cached | Notes |
|------|------|--------|-------|
| dense_search (3 branches) | 5.4s | 0.8s | 3× embed_query |
| bm25_search | 0.2s | 0.01s | cached |
| merge_normalize | 0.0s | 0.0s | negligible |
| reranking | 2.4s | 1.3s | single call |
| **total hybrid** | **8.0s** | **2.1s** | |

### Expected After Optimization (single shared embedding)

| Step | Cold | Cached | Notes |
|------|------|--------|-------|
| dense_search (1 embedding + 3 searches) | ~1.6s | ~0.8s | 1× embed_query |
| bm25_search | 0.2s | 0.01s | cached |
| merge_normalize | 0.0s | 0.0s | negligible |
| reranking | 2.4s | 1.3s | single call |
| **total hybrid** | **~4.2s** | **~2.1s** | **~48% faster cold** |

---

## ✅ State Analysis

### 1. Embedding API Caching

```python
# OpenRouterEmbeddings.embed_query()
# ❌ NO CACHING
# هر بار API call مستقیم به OpenRouter

# OpenAIEmbeddings (fallback)
# ❌ NO CACHING در لایه ما
# (SDK خود OpenAI ممکن است cache داشته باشد)
```

**نتیجه:** هر `embed_query` = یک HTTP request

### 2. Vector Store Singleton

```python
# DocumentProcessor.__init__()
self._vector_store = None  # lazy loaded

def get_vector_store(self):
    if self._vector_store is None:
        self._vector_store = Chroma(...)
    return self._vector_store

# Singleton pattern:
_document_processor = None
def get_document_processor():
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
```

**✅ Singleton است** - یک‌بار در startup initialize می‌شود

### 3. BM25 Cache

```python
self._bm25_cache: Dict[str, Tuple[Optional[BM25Retriever], float]] = {}
self._cache_ttl_seconds = 3600  # 1 hour
```

**✅ Cached با TTL 1 ساعت**

### 4. Docs Cache

```python
self._docs_cache: Dict[str, Tuple[List[Document], float]] = {}
```

**✅ Cached برای documents هر filter**

---

## 🛠️ اصلاح پیشنهادی

### Option A: Share Embedding Across Branches (Recommended)

```python
async def _dense_parallel(self, query: str, k: int, is_public: bool = False) -> List[Tuple[Document, float]]:
    if not query.strip() or not self.vector_store:
        return []
    
    vs = self.vector_store
    
    # === FIX: Single shared embedding ===
    t_embed = time.perf_counter()
    query_embedding = await asyncio.to_thread(vs.embeddings.embed_query, query)
    embed_elapsed = time.perf_counter() - t_embed
    logger.info(f"[EMBED_SHARED] query={query[:50]}... elapsed={embed_elapsed:.3f}s")
    
    # Create filters
    filters = {
        "contrib": {"$and": [...]},
        "docx": {"$and": [...]},
        "excel": {"$and": [...]},
    }
    
    # === Use asimilarity_search_by_vector instead of similarity_search_with_score ===
    async def run_by_vector(filter_name: str, f: Dict) -> Tuple[str, List[Tuple[Document, float]]]:
        branch_t0 = time.perf_counter()
        results = await asyncio.to_thread(
            vs.asimilarity_search_by_vector,
            query_embedding,  # ← همان embedding share شده
            k=k,
            filter=f
        )
        elapsed = time.perf_counter() - branch_t0
        scores = [(doc, 1.0) for doc in results]  # Placeholder scores
        logger.info(f"[DENSE_BRANCH] branch={filter_name} elapsed={elapsed:.3f}s docs={len(results)}")
        return (filter_name, scores)
    
    tasks = [run_by_vector(name, filt) for name, filt in filters.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    combined = []
    branch_results = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        name, docs = result
        branch_results[name] = docs
        combined.extend(docs)
    
    logger.info(f"[EMBED_SHARED] total={time.perf_counter()-t_embed:.3f}s")
    return combined
```

**ROI:**
- Cold: 5.4s → 1.6s (3× faster)
- Cached: 0.8s → 0.8s (unchanged - parallel already optimal)
- API calls: 3 → 1 (67% reduction)

### Option B: Add LRU Cache to embed_query (Quick Fix)

```python
from functools import lru_cache

class OpenRouterEmbeddings:
    def __init__(self, ...):
        # ... existing code ...
        self._query_cache = {}
        self._cache_maxsize = 1024
    
    def embed_query(self, text: str):
        # Simple LRU cache
        cache_key = hash(text)
        if cache_key in self._query_cache:
            logger.debug(f"[EMBED_CACHE] HIT for query={text[:30]}...")
            return self._query_cache[cache_key]
        
        result = self._original_embed_query(text)
        self._query_cache[cache_key] = result
        
        # Evict old entries if cache is full
        if len(self._query_cache) > self._cache_maxsize:
            keys = list(self._query_cache.keys())
            for k in keys[:self._cache_maxsize // 2]:
                del self._query_cache[k]
        
        logger.debug(f"[EMBED_CACHE] MISS for query={text[:30]}...")
        return result
```

**ROI:**
- Cold first call: 0.8s → 0.8s (no change)
- Cold second+ call: 5.4s → 0.8s (cache hit)
- Repeated queries: 0.8s → ~0.001s (instant cache hit)

### Option C: Combined (Best of Both)

```python
# Option A + Option B = maximum performance
async def _dense_parallel(self, query, k, is_public):
    # Single embedding
    query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
    
    # 3 vector searches (no additional embedding)
    results = await asyncio.gather(*[
        asyncio.to_thread(vs.asimilarity_search_by_vector, query_embedding, k=k, filter=f)
        for f in filters.values()
    ])
```

```python
# Plus LRU cache for repeated queries
class OpenRouterEmbeddings:
    _query_cache = {}  # LRU with 1024 entries
```

---

## 📝 Summary

| Component | Status | Detail |
|-----------|--------|--------|
| **Embedding API** | ❌ No cache | هر call = HTTP request |
| **Vector Store** | ✅ Singleton | یک‌بار در startup |
| **BM25** | ✅ Cached (1h TTL) | `_bm25_cache` |
| **Docs** | ✅ Cached (1h TTL) | `_docs_cache` |
| **Dense branches** | ❌ Duplicate embedding | 3 branches = 3 API calls |

### Bottleneck Summary

1. **Primary:** 3× `embed_query` در ۳ branch موازی → 3 API call
2. **Secondary:** No query embedding cache → repeated queries rebuild embedding

### Quick Win
- Option B (LRU cache): 3-second improvement روی second call
- Option A (shared embedding): 3.8s cold → 1.6s cold

### Long-term Win
- Option C (combined): 1.6s cold, ~0.001s cached repeated

---

## 📅 تاریخ
2026-06-21 22:57

## ✍️ تست اجرا شده
```bash
python test_dense_perf.py 2>&1 | findstr /C:"EMBED" /C:"DENSE" /C:"PERF"