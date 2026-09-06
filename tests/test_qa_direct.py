"""
Direct test to verify qa_chain variable handling without circular imports.
"""

import sys
import os
import asyncio
import logging
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_qa_chain_variables_direct():
    """Direct test to check qa_chain variables without circular imports."""
    
    try:
        logger.info("=" * 60)
        logger.info("TESTING QA CHAIN VARIABLES (DIRECT)")
        logger.info("=" * 60)
        
        # Mock the circular import issue
        with patch('app.services.chat_service.get_llm') as mock_get_llm:
            # Mock LLM
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            
            # Now import after mocking
            from app.services.knowledge_base import KnowledgeBaseService
            
            # Create service instance
            service = KnowledgeBaseService()
            
            # Mock other dependencies
            service.document_processor = MagicMock()
            service.excel_processor = MagicMock()
            service.config_service = MagicMock()
            
            # Mock config
            mock_rag_settings = MagicMock()
            mock_rag_settings.top_k_results = 5
            mock_rag_settings.qa_match_threshold = 0.8
            mock_rag_settings.knowledge_base_confidence_threshold = 0.7
            mock_rag_settings.human_referral_message = "متأسفانه، پاسخ مناسبی یافت نشد."
            mock_rag_settings.system_prompt = "شما یک دستیار هوشمند هستید."
            mock_rag_settings.prompt_template = """با استفاده از اطلاعات زیر، به سوال پاسخ دهید. اگر اطلاعات کافی نیست، صادقانه بگویید که نمی‌دانید.

اطلاعات مرجع:
{context}

سوال: {question}"""
            
            service.config_service.get_rag_settings = AsyncMock(return_value=mock_rag_settings)
            service.config_service._load_config = AsyncMock()
            
            # Mock vector store
            mock_vector_store = MagicMock()
            mock_retriever = MagicMock()
            mock_vector_store.as_retriever.return_value = mock_retriever
            service.document_processor.get_vector_store.return_value = mock_vector_store
            
            # Initialize the service (call config_service._load_config instead)
            await service.config_service._load_config()
            
            # Get QA chain
            logger.info("Getting QA chain...")
            qa_chain = await service._get_qa_chain()
            
            logger.info(f"QA chain type: {type(qa_chain)}")
            
            # Check the chain structure
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
                            
                        # Check what RetrievalQA expects as input
                        logger.info("\n" + "=" * 40)
                        logger.info("CHECKING RETRIEVAL QA INPUT EXPECTATIONS")
                        logger.info("=" * 40)
                        
                        # Check the input_keys of the qa_chain
                        if hasattr(qa_chain, 'input_keys'):
                            logger.info(f"QA chain input keys: {qa_chain.input_keys}")
                        
                        if hasattr(qa_chain, 'output_keys'):
                            logger.info(f"QA chain output keys: {qa_chain.output_keys}")
                        
                        # Test calling the chain
                        logger.info("\n" + "=" * 40)
                        logger.info("TESTING CHAIN CALLS")
                        logger.info("=" * 40)
                        
                        test_query = "تست سیستم"
                        
                        # Mock the retriever to return some documents
                        from langchain.schema import Document
                        mock_docs = [
                            Document(page_content="این یک متن تست است", metadata={"source": "test.pdf"}),
                            Document(page_content="اطلاعات اضافی", metadata={"source": "test2.pdf"})
                        ]
                        mock_retriever.get_relevant_documents.return_value = mock_docs
                        
                        # Mock the LLM response
                        mock_llm.return_value = "پاسخ تست"
                        
                        # Try different input formats
                        input_formats = [
                            {"question": test_query},
                            {"query": test_query}
                        ]
                        
                        for input_format in input_formats:
                            try:
                                logger.info(f"Trying input format: {input_format}")
                                result = qa_chain(input_format)
                                logger.info(f"✓ Success with input format: {input_format}")
                                logger.info(f"Result type: {type(result)}")
                                if isinstance(result, dict):
                                    logger.info(f"Result keys: {list(result.keys())}")
                                break
                            except Exception as e:
                                logger.warning(f"✗ Failed with input format {input_format}: {str(e)}")
                        
            logger.info("\n" + "=" * 60)
            logger.info("TEST COMPLETED")
            logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_qa_chain_variables_direct())