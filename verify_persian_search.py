import asyncio
import os
import sys
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch
from app.services.chat_service import ChatService
from app.services.utility import search_persianway
from langchain.globals import set_debug

# Enable debug logging for LangChain to see the tool inputs
set_debug(True)

# Load environment variables
load_dotenv()

async def verify_persian_search():
    chat_service = ChatService()
    # chat_service.initialize() # No initialize method needed
    
    # Force general answer to be True so it falls back to agent/tools
    chat_service.generalAnswer = True
    
    user_id = "test_user_persian_search"
    query = "Search online for the latest products of PersianWay company"
    model = "openai/gpt-4o" 
    
    print(f"\nProcessing message: '{query}' with model '{model}'")
    print("Expected behavior: The agent should translate 'latest products of PersianWay' to Persian before calling the tool.")
    
    # Mock the knowledge base service to return low confidence
    with patch('app.services.chat_service.get_knowledge_base_service') as mock_kb_service, \
         patch('app.services.config_service.get_database_service') as mock_db_service, \
         patch('app.services.config_service.ConfigService') as MockConfigServiceLib, \
         patch('app.services.chat_service.ConfigService') as MockConfigServiceChat:
         
        # Make both patches return the same mock instance for consistency
        mock_config_instance = MagicMock()
        MockConfigServiceLib.return_value = mock_config_instance
        MockConfigServiceChat.return_value = mock_config_instance
        
        # Mock DB service
        mock_db_instance = MagicMock()
        mock_db_instance.connect.return_value = None # async return? connect is async
        
        # We need to handle async connect
        future_connect = asyncio.Future()
        future_connect.set_result(None)
        mock_db_instance.connect.return_value = future_connect
        
        mock_db_service.return_value = mock_db_instance
        
        # Mock _load_config
        future_load = asyncio.Future()
        future_load.set_result(None)
        mock_config_instance._load_config.return_value = future_load
        
        # Mock get_config
        mock_config = MagicMock()
        mock_config.updated_at = "2024-01-01"
        mock_config.created_at = "2024-01-01"
        
        future_get_config = asyncio.Future()
        future_get_config.set_result(mock_config)
        mock_config_instance.get_config.return_value = future_get_config
        
        # Mock get_rag_settings
        mock_rag_settings = MagicMock()
        mock_rag_settings.system_prompt = "You are a helpful assistant."
        mock_rag_settings.knowledge_base_confidence_threshold = 0.85
        mock_rag_settings.human_referral_message = "Please contact human support."
        
        future_rag = asyncio.Future()
        future_rag.set_result(mock_rag_settings)
        mock_config_instance.get_rag_settings.return_value = future_rag
        
        # Mock get_llm_settings
        mock_llm_settings = MagicMock()
        mock_llm_settings.preferred_api_provider = "openai"
        mock_llm_settings.openai_api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
        mock_llm_settings.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")
        mock_llm_settings.default_model = "gpt-3.5-turbo"
        mock_llm_settings.temperature = 0.7
        mock_llm_settings.max_tokens = 1000
        mock_llm_settings.openrouter_api_base = "https://openrouter.ai/api/v1"
        
        future_llm = asyncio.Future()
        future_llm.set_result(mock_llm_settings)
        mock_config_instance.get_llm_settings.return_value = future_llm
        
        # Assign the mock to chat_service instance (it was already created, so we need to patch the instance attribute or re-create it)
        # Re-creating ChatService might be safer if we patch ConfigService class
        chat_service = ChatService()
        chat_service.generalAnswer = True
 
        mock_instance = MagicMock()
        mock_instance.query_knowledge_base.return_value = {
            "confidence_score": 0.1,
            "answer": "I don't know",
            "source_type": "knowledge_base"
        }
        # Because get_knowledge_base_service is called inside the method, we need to ensure the return value has the method
        # Wait, get_knowledge_base_service() returns a service instance.
        # But get_knowledge_base_service might be async? No, usually factory.
        # query_knowledge_base IS async. So we need an AsyncMock or set return_value to a future.
        
        # Let's check get_knowledge_base_service definition if possible, but easier to just assume async method needs awaitable.
        # Actually, let's use a wrapper or just simple MagicMock with async return.
        
        future = asyncio.Future()
        future.set_result({
            "confidence_score": 0.1,
            "answer": "I don't know",
            "source_type": "knowledge_base"
        })
        mock_instance.query_knowledge_base.return_value = future
        mock_kb_service.return_value = mock_instance
        
        try:
            response = await chat_service.process_message(user_id, query, model=model)
            print("\nResponse:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_persian_search())
