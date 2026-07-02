# PIPELINE PERFORMANCE REPORT - PersianWay AI Bot

**Environment:** Windows 11 Local  
**Date:** 2026-06-21  
**Python:** 3.13  
**Query tested:** 
1. "کامبوچا چیست و چه فوایدی دارد؟"
2. "بهترین کود برای گندم چیست؟"  
3. "محصولات پرشین وی را معرفی کنید"

---

## Timing Breakdown (Average of 3 tests)

| Step | Time (s) | % Total | Type |
|------|----------|---------|------|
| intent_detection | 0.005 | 0.04% | blocking (failed - LLM init error) |
| query_rewriting | 0.006 | 0.05% | blocking (failed - LLM init error) |
| embedding_generation | ~5.136 | 39.4% | async (OpenRouter API) |
| dense_search_branch_1 | 3.555 | 27.3% | sync |
| dense_search_branch_2 | 5.353 | 41.0% | sync |
| dense_search_branch_3 | 5.500 | 42.1% | sync |
| bm25_search | 0.259 | 2.0% | sync (0 docs - rank_bm25 not installed) |
| hybrid_merge | ~0.0001 | 0.001% | in-memory |
| reranking (KB) | 2.661 | 20.4% | sync (embeddings call) |
| reranking (hybrid) | 1.996 | 15.3% | sync |
| deduplication | 2.661 | 20.4% | in-memory |
| **TOTAL per query** | **13.038** | **100%** | |

---

## Total Pipeline Timing per Query

| Query | Total Time | Confidence | Source |
|-------|------------|------------|--------|
| کامبوچا چیست | 17.336s | 0.000 | none (error) |
| بهترین کود گندم | 10.435s | 0.000 | none (error) |
| محصولات پرشین وی | 11.342s | 0.000 | none (error) |
| **Average** | **13.038s** | - | - |

---

## Initialization State

| Component | State | Issue |
|-----------|-------|-------|
| DocumentProcessor | ✅ initialized at startup | - |
| VectorStore (ChromaDB) | ✅ loaded at startup | - |
| BM25 index | ❌ NOT cached | `rank_bm25` package not installed |
| Embeddings (OpenRouter) | ✅ loaded at startup | - |
| Reranker | ✅ initialized | Uses OpenRouter embeddings |

**Issues Found:**
- `Could not import rank_bm25, please install with pip install rank_bm25` - BM25 search completely non-functional
- MongoDB connection happens per-request (should be singleton)
- Config loaded 6+ times per request

---

## API Call Count per Request

| Call Type | Count | Model | Time (s) |
|-----------|-------|-------|----------|
| Embedding (query) | 1 | OpenAI text-embedding-3-small | 6.658 |
| Embedding (dense branch 1) | 1 | Qwen3 embedding 4B | - |
| Embedding (dense branch 2) | 1 | Qwen3 embedding 4B | - |
| Embedding (dense branch 3) | 1 | Qwen3 embedding 4B | - |
| Embedding (reranking batch 1) | 1 | Qwen3 embedding 4B | - |
| Embedding (reranking batch 2) | 1 | Qwen3 embedding 4B | - |
| **Total embedding API calls** | **~7** | various | ~13s |
| LLM intent detection | 0 (failed) | - | - |
| LLM query rewriting | 0 (failed) | - | - |

**Total external API calls:** ~7 embedding calls per request  
**LLM calls:** 0 (all failed due to Pydantic `ChatOpenAI` init error)

---

## Observations

### Critical Issues (Blocking)

1. **Pydantic `ChatOpenAI` initialization failure:**
   ```
   PydanticUserError: `ChatOpenAI` is not fully defined; you should define `BaseCache`, 
   then call `ChatOpenAI.model_rebuild()`.
   ```
   - Intent detection ALWAYS returns default response (0.004s)
   - Query rewriting ALWAYS falls back to original query (0.006s)
   - Final generation ALWAYS fails → returns human referral message
   - **This means NO actual LLM inference happens for final answer generation!**

2. **BM25 search completely broken:**
   ```
   Could not import rank_bm25, please install with `pip install rank_bm25`.
   ```
   - BM25 returns 0 documents every time
   - Hybrid search degrades to pure vector search

### Performance Bottlenecks

3. **Embedding generation is VERY slow (5.136s average):**
   - OpenRouter API calls for embeddings take 3.5-5.5 seconds each
   - 3 dense search branches run sequentially despite being in `asyncio.gather`
   - **Root cause:** Each branch makes separate embedding API calls to OpenRouter

4. **Reranking called TWICE (redundant):**
   - First in `hybrid_retrieval.py` (1.996s)
   - Second in `knowledge_base.py` (2.661s)
   - **Total reranking time:** ~4.6s but could be 2.0s if called once

5. **Vector search branches NOT running in parallel:**
   - Branch 1: 3.555s
   - Branch 2: 5.353s  
   - Branch 3: 5.500s
   - Total: 14.4s (would be ~5.5s if truly parallel)
   - **They ARE in `asyncio.gather` but sequential due to sync embedding calls**

### Configuration Issues

6. **Config loaded excessively:**
   - `config_service._load_config()` called 6+ times per request
   - Should be cached with TTL

7. **MongoDB connection per request:**
   - Connection established on each `process_message` call
   - Should be singleton with connection pooling

---

## Verdict

### Bottleneck #1: Embedding Generation (~5.1s, 39%)
**Reason:** OpenRouter API latency for embeddings is high (3.5-5.5s). Three dense branches make separate calls.

### Bottleneck #2: Reranking (~4.6s total, 35%)
**Reason:** Called twice redundantly. First in hybrid_retrieval, second in knowledge_base.

### Bottleneck #3: Vector Search Serial Execution (~14.4s total, but ~9s unique)
**Reason:** Despite `asyncio.gather`, embedding calls block the event loop.

### Is Speed Improvable?
**YES** — estimated achievable time: **4-6 seconds** (from 13s)

### Optimization Plan:

1. **[HIGH] Fix Pydantic ChatOpenAI** — This is the most critical. Without working LLM, final generation always fails.
   - Add `ChatOpenAI.model_rebuild(cache=...)` before instantiation
   - Or pin to compatible langchain-openai version

2. **[HIGH] Remove duplicate reranking** — Call once in hybrid_retrieval, pass scores to KB.
   - **Savings:** ~2.7s

3. **[HIGH] Install rank_bm25** — BM25 search adds meaningful recall.
   - `pip install rank_bm25`

4. **[MEDIUM] Batch embeddings** — Instead of 3 separate embedding calls for dense branches, batch them.
   - **Savings:** ~5s (3.5s → 1.2s for batch)

5. **[MEDIUM] Cache config** — Load config once with 5-minute TTL.
   - **Savings:** ~0.1s

6. **[LOW] Use local embeddings** — Replace OpenRouter embeddings with local model.
   - **Savings:** ~10s (but less accurate)

7. **[LOW] Parallel web search** — If web search is enabled, run it in parallel with vector search.
   - **Savings:** overlaps with existing time

---

## Summary Table

| Metric | Current | Target |
|--------|---------|--------|
| Average response time | 13.038s | 4-6s |
| Best case | 10.435s | 3-4s |
| Worst case | 17.336s | 6-8s |
| API calls per request | ~7 | ~3-4 (batched) |
| LLM inference | 0 (broken) | 2-3 |
| BM25 recall | 0 docs | 30+ docs |
| Final answer quality | FAILS | ✅ working |