"""
Simple test to verify qa_chain variable handling.
"""

import sys
import os
import asyncio
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_qa_chain_variables():
    """Simple test to check qa_chain variables."""
    
    try:
        # Import the service
        from app.services.knowledge_base import KnowledgeBaseService
        
        logger.info("=" * 60)
        logger.info("TESTING QA CHAIN VARIABLES")
        logger.info("=" * 60)
        
        # Create service instance
        service = KnowledgeBaseService()
        
        # Initialize the service
        await service._load_config()
        
        # Get QA chain
        logger.info("Getting QA chain...")
        qa_chain = await service._get_qa_chain()
        
        logger.info(f"QA chain type: {type(qa_chain)}")
        logger.info(f"QA chain attributes: {dir(qa_chain)}")
        
        # Check if it has the expected attributes
        if hasattr(qa_chain, 'combine_documents_chain'):
            combine_chain = qa_chain.combine_documents_chain
            logger.info(f"Combine documents chain type: {type(combine_chain)}")
            
            if hasattr(combine_chain, 'llm_chain'):
                llm_chain = combine_chain.llm_chain
                logger.info(f"LLM chain type: {type(llm_chain)}")
                
                if hasattr(llm_chain, 'prompt'):
                    prompt = llm_chain.prompt
                    logger.info(f"Prompt type: {type(prompt)}")
                    logger.info(f"Prompt input variables: {prompt.input_variables}")
                    logger.info(f"Prompt template: {prompt.template}")
                    
                    # This is what we're looking for!
                    logger.info("✓ Found prompt template with input variables!")
                    logger.info(f"✓ Input variables: {prompt.input_variables}")
                    
                    # Check if the variables are correct
                    expected_vars = ["context", "question"]
                    if set(prompt.input_variables) == set(expected_vars):
                        logger.info("✓ Input variables are correct: ['context', 'question']")
                    else:
                        logger.warning(f"✗ Input variables mismatch. Expected: {expected_vars}, Got: {prompt.input_variables}")
                    
                    # Check template content
                    if "{context}" in prompt.template and "{question}" in prompt.template:
                        logger.info("✓ Template contains both {context} and {question} placeholders")
                    else:
                        logger.warning("✗ Template missing required placeholders")
        
        # Test a simple query
        logger.info("\n" + "=" * 60)
        logger.info("TESTING QUERY EXECUTION")
        logger.info("=" * 60)
        
        test_query = "تست سیستم"
        logger.info(f"Testing query: {test_query}")
        
        # Try to call qa_chain directly
        try:
            # This should work with {"question": query} based on our analysis
            result = qa_chain({"question": test_query})
            logger.info("✓ QA chain called successfully with 'question' key")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
        except Exception as e:
            logger.error(f"✗ Error calling qa_chain with 'question' key: {str(e)}")
            
            # Try with "query" key
            try:
                result = qa_chain({"query": test_query})
                logger.info("✓ QA chain called successfully with 'query' key")
                logger.info(f"Result type: {type(result)}")
                logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            except Exception as e2:
                logger.error(f"✗ Error calling qa_chain with 'query' key: {str(e2)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETED")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_qa_chain_variables())