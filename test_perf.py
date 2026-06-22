"""
Performance Testing Script for PersianWay RAG Pipeline

This script runs a test query and captures detailed timing information
from each stage of the pipeline.

Usage:
    python test_perf.py
"""

import asyncio
import time
import logging
import sys
import json

# Set up logging to capture all PERF logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('perf_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger('perf_test')


async def run_performance_test():
    """Run performance test on the chat pipeline."""
    
    logger.info("=" * 80)
    logger.info("PERFORMANCE TEST START")
    logger.info("=" * 80)
    
    # Import the chat service
    try:
        from app.services.chat_service import get_chat_service
    except Exception as e:
        logger.error(f"Failed to import chat service: {e}")
        return
    
    # Initialize service
    chat_service = get_chat_service()
    
    # Test queries (in Persian)
    test_queries = [
        "کامبوچا چیست و چه فوایدی دارد؟",  # What is kombucha and its benefits?
        "بهترین کود برای گندم چیست؟",  # Best fertilizer for wheat?
        "محصولات پرشین وی را معرفی کنید",  # Introduce PersianWay products
    ]
    
    all_results = []
    
    for idx, query in enumerate(test_queries, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST QUERY {idx}/{len(test_queries)}: {query}")
        logger.info(f"{'=' * 60}")
        
        # Start total pipeline timing
        t_total_start = time.perf_counter()
        
        try:
            # Process the message
            result = await chat_service.process_message(
                user_id=f"perf_test_{idx}",
                message=query,
                conversation_history=None,
                model=None,
                parameters={}
            )
            
            t_total_elapsed = time.perf_counter() - t_total_start
            
            # Extract timing info from result
            answer = result.get('answer', '')
            confidence = result.get('query_analysis', {}).get('confidence_score', 0)
            knowledge_source = result.get('query_analysis', {}).get('knowledge_source', 'unknown')
            
            all_results.append({
                'query': query,
                'total_time': t_total_elapsed,
                'confidence': confidence,
                'knowledge_source': knowledge_source,
                'answer_length': len(answer),
                'requires_human': result.get('query_analysis', {}).get('requires_human_referral', False)
            })
            
            logger.info(f"RESULT: total={t_total_elapsed:.3f}s confidence={confidence:.3f} source={knowledge_source}")
            
        except Exception as e:
            t_total_elapsed = time.perf_counter() - t_total_start
            logger.error(f"Error processing query: {e}")
            
            all_results.append({
                'query': query,
                'total_time': t_total_elapsed,
                'error': str(e)
            })
    
    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info("PERFORMANCE TEST SUMMARY")
    logger.info(f"{'=' * 80}")
    
    for idx, result in enumerate(all_results, 1):
        logger.info(f"\nQuery {idx}: {result['query']}")
        logger.info(f"  Total Time: {result['total_time']:.3f}s")
        if 'confidence' in result:
            logger.info(f"  Confidence: {result['confidence']}")
            logger.info(f"  Knowledge Source: {result['knowledge_source']}")
            logger.info(f"  Answer Length: {result['answer_length']} chars")
            logger.info(f"  Requires Human: {result['requires_human']}")
        if 'error' in result:
            logger.info(f"  Error: {result['error']}")
    
    # Calculate averages
    total_times = [r['total_time'] for r in all_results if 'error' not in r]
    if total_times:
        avg_time = sum(total_times) / len(total_times)
        min_time = min(total_times)
        max_time = max(total_times)
        
        logger.info(f"\n{'=' * 40}")
        logger.info(f"STATISTICS:")
        logger.info(f"  Average Time: {avg_time:.3f}s")
        logger.info(f"  Min Time: {min_time:.3f}s")
        logger.info(f"  Max Time: {max_time:.3f}s")
        logger.info(f"  Number of Tests: {len(total_times)}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("PERFORMANCE TEST COMPLETE")
    logger.info(f"{'=' * 80}")
    
    return all_results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PersianWay RAG Pipeline Performance Test")
    print("=" * 80)
    print("\nStarting test... Please wait.\n")
    
    result = asyncio.run(run_performance_test())
    
    print("\n" + "=" * 80)
    print("Test completed! Check perf_test.log for detailed timing information.")
    print("=" * 80 + "\n")