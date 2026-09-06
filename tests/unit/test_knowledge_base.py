import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock the problematic imports to avoid circular import
with patch('app.services.chat_service.get_llm'):
    with patch('app.services.document_processor.get_document_processor'):
        with patch('app.services.excel_processor.get_excel_qa_processor'):
            from app.services.knowledge_base import KnowledgeBaseService


@pytest.fixture
def knowledge_base_service():
    """Create a KnowledgeBaseService instance for testing."""
    return KnowledgeBaseService()


class MockAsyncResponse:
    """Mock response for AsyncOpenAI client."""
    def __init__(self, content):
        self.choices = [
            MagicMock(
                message=MagicMock(
                    content=content
                )
            )
        ]


@pytest.mark.asyncio
async def test_expand_query_success(knowledge_base_service):
    """Test successful query expansion with valid response."""
    # Arrange
    test_query = "How to make Persian tea?"
    mock_response = {
        "expanded_queries": [
            "What is the traditional method for brewing Persian tea?",
            "Persian tea preparation techniques and tips",
            "How to prepare authentic Iranian chai?"
        ]
    }
    
    # Mock the AsyncOpenAI client
    with patch('openai.AsyncOpenAI') as mock_openai:
        # Configure the mock
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MockAsyncResponse(
            json.dumps(mock_response)
        )
        mock_openai.return_value = mock_client
        
        # Act
        result = await knowledge_base_service.expand_query(test_query)
        
        # Assert
        assert result["original_query"] == test_query
        assert len(result["expanded_queries"]) == 3
        assert "Persian tea preparation" in result["expanded_queries"][1]
        assert "Iranian chai" in result["expanded_queries"][2]
        
        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o-mini"
        assert call_args["temperature"] == 0.3
        assert call_args["response_format"] == {"type": "json_object"}
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert test_query in call_args["messages"][1]["content"]


@pytest.mark.asyncio
async def test_expand_query_malformed_response(knowledge_base_service):
    """Test handling of malformed JSON response."""
    # Arrange
    test_query = "How to make Persian tea?"
    
    # Mock the AsyncOpenAI client to return malformed JSON
    with patch('openai.AsyncOpenAI') as mock_openai:
        # Configure the mock
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MockAsyncResponse(
            "This is not valid JSON"
        )
        mock_openai.return_value = mock_client
        
        # Act
        result = await knowledge_base_service.expand_query(test_query)
        
        # Assert
        assert result["original_query"] == test_query
        assert result["expanded_queries"] == []


@pytest.mark.asyncio
async def test_expand_query_api_error(knowledge_base_service):
    """Test handling of API errors."""
    # Arrange
    test_query = "How to make Persian tea?"
    
    # Mock the AsyncOpenAI client to raise an exception
    with patch('openai.AsyncOpenAI') as mock_openai:
        # Configure the mock
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Mock the logging to verify error is logged
        with patch('logging.error') as mock_logging:
            # Act
            result = await knowledge_base_service.expand_query(test_query)
            
            # Assert
            assert result["original_query"] == test_query
            assert result["expanded_queries"] == []
            mock_logging.assert_called_once()
            assert "API Error" in mock_logging.call_args[0][0]


@pytest.mark.asyncio
async def test_expand_query_empty_query(knowledge_base_service):
    """Test expansion with empty query."""
    # Arrange
    test_query = ""
    mock_response = {
        "expanded_queries": []
    }
    
    # Mock the AsyncOpenAI client
    with patch('openai.AsyncOpenAI') as mock_openai:
        # Configure the mock
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MockAsyncResponse(
            json.dumps(mock_response)
        )
        mock_openai.return_value = mock_client
        
        # Act
        result = await knowledge_base_service.expand_query(test_query)
        
        # Assert
        assert result["original_query"] == ""
        assert result["expanded_queries"] == []


# Unit tests for query_knowledge_base method

@pytest.mark.asyncio
async def test_query_knowledge_base_high_confidence_qa_match(knowledge_base_service):
    """Test successful QA match with high confidence."""
    # Arrange
    test_query = "How to make Persian tea?"
    
    # Mock document with QA metadata
    mock_doc = MagicMock()
    mock_doc.page_content = "Persian tea preparation instructions"
    mock_doc.metadata = {
        "source_type": "excel_qa",
        "question": "How to make Persian tea?",
        "answer": "To make Persian tea, boil water, add tea leaves, and steep for 5 minutes.",
        "source": "tea_guide.xlsx",
        "page": 1
    }
    
    # Mock expanded query result
    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": ["Persian tea preparation", "Iranian chai brewing"]
    }
    
    # Mock RAG settings
    mock_rag_settings = MagicMock()
    mock_rag_settings.top_k_results = 5
    mock_rag_settings.qa_match_threshold = 0.8
    
    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query):
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store') as mock_vector_store:
            with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                with patch.object(knowledge_base_service.config_service, '_load_config'):
                    with patch.object(knowledge_base_service, '_is_content_relevant', return_value=True):
                        with patch.object(knowledge_base_service, '_calculate_confidence_score', return_value=0.9):
                            # Configure vector store mock
                            mock_vector_store.return_value.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
                            
                            # Act
                            result = await knowledge_base_service.query_knowledge_base(test_query)
                            
                            # Assert
                            assert result["answer"] == "To make Persian tea, boil water, add tea leaves, and steep for 5 minutes."
                            assert result["confidence_score"] == 0.9
                            assert result["source_type"] == "excel_qa"
                            assert result["requires_human_support"] is False
                            assert result["query_id"] is None
                            assert len(result["sources"]) == 1
                            assert result["sources"][0]["title"] == "How to make Persian tea?"


@pytest.mark.asyncio
async def test_query_knowledge_base_pdf_fallback(knowledge_base_service):
    """Test PDF-based knowledge retrieval fallback when no high-confidence QA match."""
    # Arrange
    test_query = "What is Persian culture?"
    
    # Mock document with PDF metadata
    mock_doc = MagicMock()
    mock_doc.page_content = "Persian culture is rich and diverse with a long history..."
    mock_doc.metadata = {
        "source_type": "pdf",
        "source": "culture_guide.pdf",
        "page": 15
    }
    
    # Mock expanded query result
    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": ["Iranian culture", "Persian traditions"]
    }
    
    # Mock RAG settings
    mock_rag_settings = MagicMock()
    mock_rag_settings.top_k_results = 5
    mock_rag_settings.qa_match_threshold = 0.8
    mock_rag_settings.knowledge_base_confidence_threshold = 0.7
    
    # Mock QA chain result
    mock_qa_result = {
        "result": "Persian culture encompasses art, literature, and traditions spanning thousands of years.",
        "source_documents": [mock_doc]
    }
    
    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query):
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store') as mock_vector_store:
            with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                with patch.object(knowledge_base_service.config_service, '_load_config', new_callable=AsyncMock):
                    with patch.object(knowledge_base_service, '_get_qa_chain') as mock_get_qa_chain:
                        with patch.object(knowledge_base_service, '_calculate_confidence_score', return_value=0.75):
                            # Configure mocks
                            mock_vector_store.return_value.similarity_search_with_score.return_value = [(mock_doc, 0.3)]
                            mock_qa_chain = MagicMock()
                            mock_qa_chain.invoke.return_value = mock_qa_result
                            mock_get_qa_chain.return_value = mock_qa_chain
                            
                            # Act
                            result = await knowledge_base_service.query_knowledge_base(test_query)
                            
                            # Assert
                            assert "Persian culture encompasses art" in result["answer"]
                            assert result["confidence_score"] == 0.75
                            assert result["source_type"] == "pdf"
                            assert result["requires_human_support"] is False
                            assert len(result["sources"]) == 1


@pytest.mark.asyncio
async def test_query_knowledge_base_public_filter_applied(knowledge_base_service):
    """Ensure public queries apply metadata filtering before retrieval."""
    test_query = "What does PersianWay do?"

    mock_doc = MagicMock()
    mock_doc.page_content = "PersianWay provides public services."
    mock_doc.metadata = {
        "source_type": "pdf",
        "source": "public_overview.pdf",
        "page": 2
    }

    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": []
    }

    mock_rag_settings = MagicMock()
    mock_rag_settings.top_k_results = 5
    mock_rag_settings.qa_match_threshold = 0.8
    mock_rag_settings.knowledge_base_confidence_threshold = 0.7
    mock_rag_settings.human_referral_message = "Need human support"

    mock_qa_chain = MagicMock()
    mock_qa_chain.invoke.return_value = {
        "answer": "PersianWay is a public company.",
        "context": [mock_doc]
    }

    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query):
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store') as mock_vector_store:
            vector_store_instance = MagicMock()
            vector_store_instance.similarity_search_with_score.return_value = [(mock_doc, 0.2)]
            mock_vector_store.return_value = vector_store_instance

            with patch.object(knowledge_base_service.config_service, '_load_config', new_callable=AsyncMock):
                with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                    with patch.object(knowledge_base_service, '_get_qa_chain') as mock_get_qa_chain:
                        mock_get_qa_chain.return_value = mock_qa_chain

                        with patch.object(knowledge_base_service, '_calculate_confidence_score', return_value=0.9):
                            result = await knowledge_base_service.query_knowledge_base(
                                test_query,
                                is_public=True
                            )

    assert result["confidence_score"] == 0.9
    assert result["requires_human_support"] is False
    assert vector_store_instance.similarity_search_with_score.called
    for call in vector_store_instance.similarity_search_with_score.call_args_list:
        assert call.kwargs.get("filter") == {"is_public": True}


@pytest.mark.asyncio
async def test_query_knowledge_base_low_confidence_human_referral(knowledge_base_service):
    """Test human referral when confidence is below threshold."""
    # Arrange
    test_query = "Very specific technical question"
    
    # Mock document
    mock_doc = MagicMock()
    mock_doc.page_content = "Some general information..."
    mock_doc.metadata = {
        "source_type": "pdf",
        "source": "general.pdf",
        "page": 1
    }
    
    # Mock expanded query result
    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": ["technical question", "specific query"]
    }
    
    # Mock RAG settings
    mock_rag_settings = MagicMock()
    mock_rag_settings.top_k_results = 5
    mock_rag_settings.qa_match_threshold = 0.8
    mock_rag_settings.knowledge_base_confidence_threshold = 0.7
    
    # Mock QA chain result
    mock_qa_result = {
        "result": "I'm not sure about this specific question.",
        "source_documents": [mock_doc]
    }
    
    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query):
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store') as mock_vector_store:
            with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                with patch.object(knowledge_base_service.config_service, '_load_config', new_callable=AsyncMock):
                    with patch.object(knowledge_base_service, '_get_qa_chain') as mock_get_qa_chain:
                        with patch.object(knowledge_base_service, '_calculate_confidence_score', return_value=0.4):
                            with patch.object(knowledge_base_service, '_log_human_referral'):
                                # Configure mocks
                                mock_vector_store.return_value.similarity_search_with_score.return_value = [(mock_doc, 0.8)]
                                mock_qa_chain = MagicMock()
                                mock_qa_chain.invoke.return_value = mock_qa_result
                                mock_get_qa_chain.return_value = mock_qa_chain
                                
                                # Act
                                result = await knowledge_base_service.query_knowledge_base(test_query)
                                
                                # Assert
                                assert result["confidence_score"] == 0.4
                                assert result["requires_human_support"] is True
                                assert result["query_id"] is not None
                                assert len(result["query_id"]) > 0  # UUID should be generated


@pytest.mark.asyncio
async def test_query_knowledge_base_vector_store_unavailable(knowledge_base_service):
    """Test error handling when vector store is unavailable."""
    # Arrange
    test_query = "Any question"
    
    # Mock expanded query result
    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": ["alternative query"]
    }
    
    # Mock RAG settings
    mock_rag_settings = MagicMock()
    mock_rag_settings.human_referral_message = "متأسفانه، سیستم در حال حاضر در دسترس نیست."
    
    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query):
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store', return_value=None):
            with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                with patch.object(knowledge_base_service.config_service, '_load_config'):
                    # Act
                    result = await knowledge_base_service.query_knowledge_base(test_query)
                    
                    # Assert
                    assert "سیستم در حال حاضر در دسترس نیست" in result["answer"]
                    assert result["confidence_score"] == 0.0
                    assert result["source_type"] == "system"
                    assert result["requires_human_support"] is True
                    assert result["query_id"] is not None


@pytest.mark.asyncio
async def test_query_knowledge_base_qa_chain_unavailable(knowledge_base_service):
    """Test fallback when QA chain is unavailable."""
    # Arrange
    test_query = "Question when QA chain fails"
    
    # Mock document
    mock_doc = MagicMock()
    mock_doc.page_content = "Some content"
    mock_doc.metadata = {
        "source_type": "pdf",
        "source": "test.pdf",
        "page": 1
    }
    
    # Mock expanded query result
    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": ["alternative question"]
    }
    
    # Mock RAG settings
    mock_rag_settings = MagicMock()
    mock_rag_settings.top_k_results = 5
    mock_rag_settings.qa_match_threshold = 0.8
    mock_rag_settings.human_referral_message = "لطفاً با پشتیبانی تماس بگیرید."
    
    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query):
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store') as mock_vector_store:
            with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                with patch.object(knowledge_base_service.config_service, '_load_config'):
                    with patch.object(knowledge_base_service, '_get_qa_chain', return_value=None):
                        # Configure vector store mock (no QA docs)
                        mock_vector_store.return_value.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
                        
                        # Act
                        result = await knowledge_base_service.query_knowledge_base(test_query)
                        
                        # Assert
                        assert "پشتیبانی تماس بگیرید" in result["answer"]
                        assert result["confidence_score"] == 0.0
                        assert result["source_type"] == "system"
                        assert result["requires_human_support"] is True


@pytest.mark.asyncio
async def test_query_knowledge_base_exception_handling(knowledge_base_service):
    """Test general exception handling in query_knowledge_base."""
    # Arrange
    test_query = "Question that causes exception"
    
    # Mock RAG settings for error case
    mock_rag_settings = MagicMock()
    mock_rag_settings.human_referral_message = "خطایی رخ داده است."
    
    with patch.object(knowledge_base_service, 'expand_query', side_effect=Exception("Test exception")):
        with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
            with patch.object(knowledge_base_service.config_service, '_load_config'):
                with patch('logging.error') as mock_logging:
                    # Act
                    result = await knowledge_base_service.query_knowledge_base(test_query)
                    
                    # Assert
                    assert "خطایی رخ داده است" in result["answer"]
                    assert result["confidence_score"] == 0.0
                    assert result["source_type"] == "system"
                    assert result["requires_human_support"] is True
                    assert result["query_id"] is not None
                    mock_logging.assert_called_once()


@pytest.mark.asyncio
async def test_query_knowledge_base_query_expansion_integration(knowledge_base_service):
    """Test that query expansion is properly integrated and multiple queries are used."""
    # Arrange
    test_query = "Persian tea"
    
    # Mock documents for different queries
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Persian tea content 1"
    mock_doc1.metadata = {"source_type": "pdf", "source": "tea1.pdf", "page": 1}
    
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Persian tea content 2"
    mock_doc2.metadata = {"source_type": "pdf", "source": "tea2.pdf", "page": 2}
    
    # Mock expanded query result
    mock_expanded_query = {
        "original_query": test_query,
        "expanded_queries": ["Iranian tea", "Persian chai"]
    }
    
    # Mock RAG settings
    mock_rag_settings = MagicMock()
    mock_rag_settings.top_k_results = 3
    mock_rag_settings.qa_match_threshold = 0.8
    mock_rag_settings.knowledge_base_confidence_threshold = 0.7
    
    # Mock QA chain result
    mock_qa_result = {
        "result": "Persian tea is a traditional beverage.",
        "source_documents": [mock_doc1, mock_doc2]
    }
    
    with patch.object(knowledge_base_service, 'expand_query', return_value=mock_expanded_query) as mock_expand:
        with patch.object(knowledge_base_service.document_processor, 'get_vector_store') as mock_vector_store:
            with patch.object(knowledge_base_service.config_service, 'get_rag_settings', return_value=mock_rag_settings):
                with patch.object(knowledge_base_service.config_service, '_load_config', new_callable=AsyncMock):
                    with patch.object(knowledge_base_service, '_get_qa_chain') as mock_get_qa_chain:
                        with patch.object(knowledge_base_service, '_calculate_confidence_score', return_value=0.8):
                            # Configure mocks to return different docs for different queries
                            def mock_similarity_search(query, k):
                                if query == "Persian tea":
                                    return [(mock_doc1, 0.2)]
                                elif query == "Iranian tea":
                                    return [(mock_doc2, 0.3)]
                                elif query == "Persian chai":
                                    return [(mock_doc1, 0.4)]  # Duplicate, should be filtered
                                return []
                            
                            mock_vector_store.return_value.similarity_search_with_score.side_effect = mock_similarity_search
                            mock_qa_chain = MagicMock()
                            mock_qa_chain.invoke.return_value = mock_qa_result
                            mock_get_qa_chain.return_value = mock_qa_chain
                            
                            # Act
                            result = await knowledge_base_service.query_knowledge_base(test_query)
                            
                            # Assert
                            mock_expand.assert_called_once_with(test_query)
                            # Verify similarity_search_with_score was called for each query
                            assert mock_vector_store.return_value.similarity_search_with_score.call_count == 3
                            assert result["confidence_score"] == 0.8
                            assert result["requires_human_support"] is False