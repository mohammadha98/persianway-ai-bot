import pytest
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class StubLLM:
    class Resp:
        def __init__(self, content):
            self.content = content

    async def ainvoke(self, messages):
        text = "".join(getattr(m, 'content', '') for m in messages)
        if "پرشین وی" in text:
            return StubLLM.Resp('{"intent":"PUBLIC","category":"company_info","confidence":0.9,"explanation":"company info"}')
        return StubLLM.Resp('{"intent":"PRIVATE","category":"agriculture","confidence":0.9,"explanation":"agriculture"}')


@pytest.mark.asyncio
async def test_company_info_is_public_true_and_filter():
    with patch('app.services.chat_service.get_llm', return_value=StubLLM()):
        from app.services.chat_service import ChatService
        from app.services.knowledge_base import KnowledgeBaseService

        kb = KnowledgeBaseService()
        vector_store = MagicMock()
        pub = MagicMock()
        pub.page_content = "public content"
        pub.metadata = {"source": "pub.pdf", "page": 1, "is_public": True}
        priv = MagicMock()
        priv.page_content = "private content"
        priv.metadata = {"source": "priv.pdf", "page": 2, "is_public": False}
        vector_store.max_marginal_relevance_search.return_value = [pub, priv]
        vector_store.similarity_search_with_score.return_value = [(pub, 0.3), (priv, 0.2)]
        kb.document_processor.get_vector_store = MagicMock(return_value=vector_store)

        # Simplify chain invoke
        with patch.object(kb, '_get_document_chain') as get_chain:
            chain = MagicMock()
            chain.invoke.return_value = "ANSWER"
            get_chain.return_value = chain

            chat = ChatService()
            intent = await chat.detect_query_intent("پرشین وی چیست؟")
            assert intent.get('is_public') is True
            result = await kb.query_knowledge_base("پرشین وی چیست؟", is_public=intent.get('is_public'))
            assert result.get('sources'), "No sources returned"
            assert all(s.get('is_public') is True for s in result['sources'])


@pytest.mark.asyncio
async def test_agriculture_is_public_false_and_filter():
    with patch('app.services.chat_service.get_llm', return_value=StubLLM()):
        from app.services.chat_service import ChatService
        from app.services.knowledge_base import KnowledgeBaseService

        kb = KnowledgeBaseService()
        vector_store = MagicMock()
        pub = MagicMock()
        pub.page_content = "public agri"
        pub.metadata = {"source": "pub2.pdf", "page": 3, "is_public": True}
        priv = MagicMock()
        priv.page_content = "private agri"
        priv.metadata = {"source": "priv2.pdf", "page": 4, "is_public": False}
        vector_store.max_marginal_relevance_search.return_value = [pub, priv]
        vector_store.similarity_search_with_score.return_value = [(pub, 0.3), (priv, 0.25)]
        kb.document_processor.get_vector_store = MagicMock(return_value=vector_store)

        with patch.object(kb, '_get_document_chain') as get_chain:
            chain = MagicMock()
            chain.invoke.return_value = "ANSWER"
            get_chain.return_value = chain

            chat = ChatService()
            intent = await chat.detect_query_intent("روش های کوددهی خاک چیست؟")
            assert intent.get('is_public') is False
            result = await kb.query_knowledge_base("روش های کوددهی خاک چیست؟", is_public=intent.get('is_public'))
            assert result.get('sources'), "No sources returned"
            assert any(s.get('is_public') is False for s in result['sources'])
