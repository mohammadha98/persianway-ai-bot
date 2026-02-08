import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
import os

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.services.chat_service import ChatService
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool

class TestWebSearchIntegration(unittest.TestCase):
    @patch('app.services.chat_service.get_llm', new_callable=AsyncMock)
    @patch('app.services.chat_service.search_persianway')
    @patch('app.services.chat_service.get_knowledge_base_service')
    @patch('app.services.chat_service.ConfigService')
    def test_web_search_integration(self, mock_config_service_cls, mock_get_kb, mock_search, mock_get_llm):
        # Setup mocks
        
        # Mock ConfigService
        mock_config_instance = mock_config_service_cls.return_value
        mock_config_instance._load_config = AsyncMock()
        mock_config_instance.get_llm_settings = AsyncMock(return_value=MagicMock(
            preferred_api_provider="openai",
            openai_api_key="test_key",
            temperature=0.7,
            max_tokens=100
        ))
        
        # Mock Config object
        mock_config_obj = MagicMock()
        mock_config_obj.updated_at = "2024-01-01"
        mock_config_obj.created_at = "2024-01-01"
        mock_config_instance.get_config = AsyncMock(return_value=mock_config_obj)
        
        # Mock RAG settings
        mock_rag_settings = MagicMock()
        mock_rag_settings.human_referral_message = "Referral"
        mock_rag_settings.knowledge_base_confidence_threshold = 0.8
        mock_rag_settings.system_prompt = "System Prompt"
        mock_config_instance.get_rag_settings = AsyncMock(return_value=mock_rag_settings)
        
        # Mock LLM for translation
        # get_llm returns an LLM object. The LLM object has an ainvoke method.
        # We use spec=ChatOpenAI so it passes Pydantic validation
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm.ainvoke = AsyncMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Translated Persian Query"
        mock_llm.ainvoke.return_value = mock_llm_response
        
        # Configure mock_get_llm to return our mock_llm
        mock_get_llm.return_value = mock_llm
        
        # Mock Search Tool
        mock_search.name = "search_persianway"
        mock_search.description = "Search tool for PersianWay"
        mock_search.return_value = "Search Result: Found product X"
        # Make it pass isinstance check for BaseTool if possible, or just skip validation if we could
        # Since we patch the object itself, we can't easily change its type to BaseTool unless we use spec=BaseTool
        # But we can't change the spec of an existing mock easily if it was created by patch without spec.
        # So we should use new_callable or autospec in patch.
        # However, for now, let's just make it look like a BaseTool instance
        mock_search.__class__ = BaseTool 
        
        # Mock KB Service
        mock_kb_service = MagicMock()
        mock_get_kb.return_value = mock_kb_service
        
        # Mock KB Query responses
        # First call: Low confidence (triggers search)
        # Second call: High confidence (after search injection)
        async def side_effect(*args, **kwargs):
            if kwargs.get('external_context'):
                return {
                    "answer": "Final Answer based on search",
                    "confidence_score": 0.9,
                    "source_type": "web_search_augmented"
                }
            else:
                return {
                    "answer": "I don't know",
                    "confidence_score": 0.5,
                    "source_type": "knowledge_base"
                }
        
        mock_kb_service.query_knowledge_base = AsyncMock(side_effect=side_effect)
        
        # Initialize ChatService
        chat_service = ChatService()
        chat_service.generalAnswer = True
        
        # Run process_message
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # We also need to mock detect_query_intent because it's called inside process_message
            # Since it's a method on chat_service, we can patch it on the instance or the class
            # But simpler is to rely on it working or mock it if it's complex.
            # It uses get_llm internally if not provided, or logic. 
            # Looking at code: it calls self.detect_query_intent
            
            # Let's mock detect_query_intent to avoid LLM calls there and simplify flow
            chat_service.detect_query_intent = AsyncMock(return_value={
                "intent": "PUBLIC",
                "is_public": True,
                "explanation": "Test explanation"
            })
            
            result = loop.run_until_complete(
                chat_service.process_message(
                    user_id="user1",
                    message="What are the latest products?",
                    conversation_history=[]
                )
            )
            
            # Verify
            print("\nTest Results:")
            print(f"Result Answer: {result['answer']}")
            print(f"Result Source: {result['query_analysis']['knowledge_source']}")
            
            # Check translation call
            # mock_get_llm was called?
            print(f"get_llm called: {mock_get_llm.called}")
            
            # mock_llm.ainvoke called?
            if mock_llm.ainvoke.call_count > 0:
                print("Translation called: OK")
                args, _ = mock_llm.ainvoke.call_args
                print(f"Translation prompt: {args[0][0].content}")
            else:
                print("Translation called: FAILED")
            
            # Check search call
            if mock_search.call_count > 0:
                print("Search tool called: OK")
                mock_search.assert_called_with("Translated Persian Query")
            else:
                print("Search tool called: FAILED")
            
            # Check KB calls
            print(f"KB called count: {mock_kb_service.query_knowledge_base.call_count}")
            
            if mock_kb_service.query_knowledge_base.call_count >= 2:
                call_args = mock_kb_service.query_knowledge_base.call_args_list
                
                # Verify first call (no external context)
                args1, kwargs1 = call_args[0]
                has_context_1 = kwargs1.get('external_context') is not None
                print(f"First KB call external_context: {kwargs1.get('external_context')}")
                
                # Verify second call (with external context)
                args2, kwargs2 = call_args[1]
                context_2 = kwargs2.get('external_context')
                print(f"Second KB call external_context: {context_2}")
                
                if context_2 == "Search Result: Found product X":
                    print("Context injection: OK")
                else:
                    print("Context injection: FAILED")
            
            self.assertEqual(result['answer'], "Final Answer based on search")
            
        finally:
            loop.close()

if __name__ == '__main__':
    unittest.main()
