"""Test script to measure dense search performance and embedding call counts."""
import asyncio
import time
import logging
from unittest.mock import patch, MagicMock

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("dense_perf_test")


async def test_dense_search():
    """Test dense search timing and count embedding API calls."""
    logger.info("=" * 60)
    logger.info("DENSE SEARCH PERFORMANCE TEST")
    logger.info("=" * 60)
    
    from app.services.document_processor import get_document_processor
    from app.services.hybrid_retrieval import HybridRetrievalService
    
    # Get document processor
    doc_processor = get_document_processor()
    
    # === TEST 1: Count embedding API calls ===
    logger.info("\n--- TEST 1: Count Embedding API Calls ---")
    
    # Mock the original embed_query to count calls
    original_embed_query = None
    if hasattr(doc_processor, 'embeddings') and doc_processor.embeddings:
        original_embed_query = doc_processor.embeddings.embed_query
        call_count = 0
        call_times = []
        
        def counting_embed_query(text):
            nonlocal call_count, call_times
            call_count += 1
            t0 = time.perf_counter()
            try:
                result = original_embed_query(text)
                elapsed = time.perf_counter() - t0
                call_times.append(elapsed)
                logger.info(f"[EMBED_CALL #{call_count}] text={text[:50]}... elapsed={elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                call_times.append(elapsed)
                logger.error(f"[EMBED_CALL #{call_count}] FAILED: {e} after {elapsed:.3f}s")
                raise
        
        with patch.object(doc_processor.embeddings, 'embed_query', side_effect=counting_embed_query):
            hrs = HybridRetrievalService(doc_processor)
            
            test_query = "کدام محصولات سلامتی برای تقویت سیستم ایمنی مناسب هستند؟"
            
            # Run dense search
            t_start = time.perf_counter()
            results = await hrs.hybrid_retrieve(test_query, is_public=False)
            t_end = time.perf_counter()
            
            logger.info(f"\n--- RESULTS ---")
            logger.info(f"Total hybrid retrieval: {t_end - t_start:.3f}s")
            logger.info(f"Embedding API calls: {call_count}")
            logger.info(f"Results returned: {len(results)} docs")
            
            if call_times:
                logger.info(f"Embedding call times: {[f'{t:.3f}s' for t in call_times]}")
                logger.info(f"Total embedding time: {sum(call_times):.3f}s")
                logger.info(f"Avg embedding time: {sum(call_times)/len(call_times):.3f}s")
                logger.info(f"Max embedding time: {max(call_times):.3f}s")
    
    # === TEST 2: Single dense search timing ===
    logger.info("\n--- TEST 2: Dense Search Branch Timing ---")
    
    if hasattr(doc_processor, 'embeddings') and doc_processor.embeddings:
        original_embed_query = doc_processor.embeddings.embed_query
        branch_counts = {'contrib': 0, 'docx': 0, 'excel': 0}
        branch_times = {'contrib': [], 'docx': [], 'excel': []}
        
        def timing_embed_query(text):
            t0 = time.perf_counter()
            try:
                result = original_embed_query(text)
                elapsed = time.perf_counter() - t0
                # We can't know which branch this is from just the query text
                # The branch timing is logged by [DENSE_BRANCH]
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error(f"Embedding failed after {elapsed:.3f}s: {e}")
                raise
        
        with patch.object(doc_processor.embeddings, 'embed_query', side_effect=timing_embed_query):
            hrs = HybridRetrievalService(doc_processor)
            
            # Run hybrid retrieve 3 times
            for run in range(1, 4):
                t_start = time.perf_counter()
                results = await hrs.hybrid_retrieve(test_query, is_public=False)
                t_end = time.perf_counter()
                logger.info(f"Run {run}: hybrid={t_end - t_start:.3f}s docs={len(results)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_dense_search())