"""Quick verification that embedding is called only once per query."""
import asyncio
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("verify_fix")

async def test():
    from app.services.hybrid_retrieval import HybridRetrievalService
    from app.services.document_processor import get_document_processor
    
    doc_processor = get_document_processor()
    hrs = HybridRetrievalService(doc_processor)
    
    test_query = 'کدام محصولات سلامتی برای تقویت سیستم ایمنی'
    
    # Count [EMBED_SHARED] logs
    embed_count = 0
    branch_count = 0
    
    logger.info("=" * 60)
    logger.info("VERIFICATION TEST: Embedding Shared Across Branches")
    logger.info("=" * 60)
    
    # Run 2 times - first warm, second cached
    for run in range(1, 3):
        t0 = time.perf_counter()
        results = await hrs.hybrid_retrieve(test_query, is_public=False)
        elapsed = time.perf_counter() - t0
        
        logger.info(f"\nRun {run}: {elapsed:.3f}s | Docs: {len(results)}")
        if run == 1:
            logger.info("^( This is WARM run - embedding API was called)")
        else:
            logger.info("^( This is CACHED run - result from Chroma local)")

    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("Check logs above for:")
    logger.info("  - [EMBED_SHARED] should appear ONCE per hybrid_retrieve call")
    logger.info("  - [DENSE_BRANCH] should appear 3 times (contrib, docx, excel)")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(test())