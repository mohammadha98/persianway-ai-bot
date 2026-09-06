import pytest
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with patch('app.services.chat_service.get_llm'):
    with patch('app.services.document_processor.get_document_processor') as mock_dp:
        with patch('app.services.excel_processor.get_excel_qa_processor'):
            from app.services.knowledge_base import KnowledgeBaseService


@pytest.mark.asyncio
async def test_is_public_true_filters_only_public():
    kb = KnowledgeBaseService()

    # Mock vector store
    vector_store = MagicMock()
    mock_doc_public = MagicMock()
    mock_doc_public.page_content = "public content"
    mock_doc_public.metadata = {"source": "pub.pdf", "page": 1, "is_public": True}

    mock_doc_private = MagicMock()
    mock_doc_private.page_content = "private content"
    mock_doc_private.metadata = {"source": "priv.pdf", "page": 2, "is_public": False}

    vector_store.max_marginal_relevance_search.return_value = [mock_doc_public, mock_doc_private]
    vector_store.similarity_search_with_score.return_value = [(mock_doc_public, 0.3), (mock_doc_private, 0.2)]

    mock_dp.return_value.get_vector_store.return_value = vector_store

    # Mock QA chain
    qa_chain = MagicMock()
    qa_chain.invoke.return_value = {
        "answer": "ok",
        "source_documents": [mock_doc_public]
    }
    with patch.object(kb, '_get_qa_chain', return_value=qa_chain):
        with patch.object(kb, 'expand_query_with_context', return_value={
            "original_query": "سوال تستی",
            "rewritten_query": "سوال تستی",
            "expanded_queries": [],
            "all_queries": ["سوال تستی"],
        }):
            with patch.object(kb.config_service, 'get_rag_settings') as get_rag:
                mock_rag = MagicMock()
                mock_rag.top_k_results = 5
                mock_rag.fetch_k_multiplier = 2
                mock_rag.mmr_diversity_score = 0.3
                mock_rag.original_query_weight = 1.0
                mock_rag.expanded_query_weight = 0.7
                mock_rag.similarity_threshold = 1.5
                mock_rag.reranker_alpha = 0.7
                get_rag.return_value = mock_rag
                result = await kb.query_knowledge_base("سوال تستی", is_public=True)
        assert result["sources"], "No sources returned"
        assert all(s.get("is_public") is True for s in result["sources"]), "Found non-public doc when is_public=True"


@pytest.mark.asyncio
async def test_is_public_false_allows_both():
    kb = KnowledgeBaseService()

    vector_store = MagicMock()
    mock_doc_public = MagicMock()
    mock_doc_public.page_content = "public content"
    mock_doc_public.metadata = {"source": "pub.pdf", "page": 1, "is_public": True}

    mock_doc_private = MagicMock()
    mock_doc_private.page_content = "private content"
    mock_doc_private.metadata = {"source": "priv.pdf", "page": 2, "is_public": False}

    vector_store.max_marginal_relevance_search.return_value = [mock_doc_public, mock_doc_private]
    vector_store.similarity_search_with_score.return_value = [(mock_doc_public, 0.3), (mock_doc_private, 0.2)]

    with patch('app.services.document_processor.get_document_processor') as dp:
        dp.return_value.get_vector_store.return_value = vector_store
        qa_chain = MagicMock()
        qa_chain.invoke.return_value = {
            "answer": "ok",
            "source_documents": [mock_doc_public, mock_doc_private]
        }
        with patch.object(kb, '_get_qa_chain', return_value=qa_chain):
            with patch.object(kb, 'expand_query_with_context', return_value={
                "original_query": "سوال تستی",
                "rewritten_query": "سوال تستی",
                "expanded_queries": [],
                "all_queries": ["سوال تستی"],
            }):
                with patch.object(kb.config_service, 'get_rag_settings') as get_rag:
                    mock_rag = MagicMock()
                    mock_rag.top_k_results = 5
                    mock_rag.fetch_k_multiplier = 2
                    mock_rag.mmr_diversity_score = 0.3
                    mock_rag.original_query_weight = 1.0
                    mock_rag.expanded_query_weight = 0.7
                    mock_rag.similarity_threshold = 1.5
                    mock_rag.reranker_alpha = 0.7
                    get_rag.return_value = mock_rag
                    result = await kb.query_knowledge_base("سوال تستی", is_public=False)
            assert result["sources"], "No sources returned"
            assert any(s.get("is_public") is False for s in result["sources"]), "Expected at least one non-public doc when is_public=False"
