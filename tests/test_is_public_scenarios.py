import pytest
import sys, os
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.mark.asyncio
async def test_company_info_public_filtering():
    from app.services.chat_service import get_llm
    with patch('app.services.chat_service.get_llm'):
        with patch('app.services.document_processor.get_document_processor'):
            with patch('app.services.excel_processor.get_excel_qa_processor'):
                from app.services.knowledge_base import KnowledgeBaseService
                kb = KnowledgeBaseService()
    vector_store = MagicMock()
    pub = MagicMock(); pub.page_content = "public content"; pub.metadata = {"source":"pub.pdf","page":1,"is_public":True}
    priv = MagicMock(); priv.page_content = "private content"; priv.metadata = {"source":"priv.pdf","page":2,"is_public":False}
    vector_store.max_marginal_relevance_search.return_value = [pub, priv]
    vector_store.similarity_search_with_score.return_value = [(pub, 0.3), (priv, 0.2)]
    kb.document_processor.get_vector_store = MagicMock(return_value=vector_store)
    with patch.object(kb, '_get_document_chain') as get_chain:
        chain = MagicMock(); chain.invoke.return_value = "ANSWER"; get_chain.return_value = chain
        with patch.object(kb.config_service, 'get_rag_settings') as get_rag:
            mock_rag = MagicMock()
            mock_rag.top_k_results = 5
            mock_rag.fetch_k_multiplier = 2
            mock_rag.mmr_diversity_score = 0.3
            mock_rag.original_query_weight = 1.0
            mock_rag.expanded_query_weight = 0.7
            mock_rag.similarity_threshold = 1.5
            mock_rag.reranker_alpha = 0.7
            mock_rag.knowledge_base_confidence_threshold = 0.5
            mock_rag.human_referral_message = "نیاز به پشتیبانی انسانی"
            get_rag.return_value = mock_rag
            with patch.object(kb, 'expand_query_with_context', return_value={
                "original_query": "پرشین وی چیست؟",
                "rewritten_query": "پرشین وی چیست؟",
                "expanded_queries": [],
                "all_queries": ["پرشین وی چیست؟"],
            }):
                result = await kb.query_knowledge_base("پرشین وی چیست؟", is_public=True)
        assert result.get('sources'), "No sources returned"
        assert all(s.get('is_public') is True for s in result['sources'])


@pytest.mark.asyncio
async def test_agriculture_private_allows_both():
    with patch('app.services.chat_service.get_llm'):
        with patch('app.services.document_processor.get_document_processor'):
            with patch('app.services.excel_processor.get_excel_qa_processor'):
                from app.services.knowledge_base import KnowledgeBaseService
                kb = KnowledgeBaseService()
    vector_store = MagicMock()
    pub = MagicMock(); pub.page_content = "public agri"; pub.metadata = {"source":"pub2.pdf","page":3,"is_public":True}
    priv = MagicMock(); priv.page_content = "private agri"; priv.metadata = {"source":"priv2.pdf","page":4,"is_public":False}
    vector_store.max_marginal_relevance_search.return_value = [pub, priv]
    vector_store.similarity_search_with_score.return_value = [(pub, 0.3), (priv, 0.25)]
    kb.document_processor.get_vector_store = MagicMock(return_value=vector_store)
    with patch.object(kb, '_get_document_chain') as get_chain:
        chain = MagicMock(); chain.invoke.return_value = "ANSWER"; get_chain.return_value = chain
        with patch.object(kb.config_service, 'get_rag_settings') as get_rag:
            mock_rag = MagicMock()
            mock_rag.top_k_results = 5
            mock_rag.fetch_k_multiplier = 2
            mock_rag.mmr_diversity_score = 0.3
            mock_rag.original_query_weight = 1.0
            mock_rag.expanded_query_weight = 0.7
            mock_rag.similarity_threshold = 1.5
            mock_rag.reranker_alpha = 0.7
            mock_rag.knowledge_base_confidence_threshold = 0.5
            mock_rag.human_referral_message = "نیاز به پشتیبانی انسانی"
            get_rag.return_value = mock_rag
            with patch.object(kb, 'expand_query_with_context', return_value={
                "original_query": "روش های کوددهی خاک چیست؟",
                "rewritten_query": "روش های کوددهی خاک چیست؟",
                "expanded_queries": [],
                "all_queries": ["روش های کوددهی خاک چیست؟"],
            }):
                result = await kb.query_knowledge_base("روش های کوددهی خاک چیست؟", is_public=False)
        assert result.get('sources'), "No sources returned"
        assert any(s.get('is_public') is False for s in result['sources'])
