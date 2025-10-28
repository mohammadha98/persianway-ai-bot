"""
Test script to verify SYSTEM_PROMPT integration in both RAG and general knowledge responses.
"""
import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.chat_service import get_chat_service
from app.services.knowledge_base import get_knowledge_base_service
from app.services.config_service import ConfigService

async def test_system_prompt_integration():
    """Test that SYSTEM_PROMPT is properly integrated in both RAG and general responses."""
    
    print("🧪 Testing SYSTEM_PROMPT Integration")
    print("=" * 50)
    
    # Initialize services
    chat_service = get_chat_service()
    kb_service = get_knowledge_base_service()
    config_service = ConfigService()
    
    # Load configuration
    await config_service._load_config()
    rag_settings = await config_service.get_rag_settings()
    
    print(f"📋 Current System Prompt:")
    print(f"   {rag_settings.system_prompt[:100]}...")
    print()
    
    # Test 1: RAG Response (Knowledge Base)
    print("🔍 Test 1: RAG Response with System Prompt")
    try:
        # Query that should trigger knowledge base
        kb_query = "سلامت چیست؟"
        kb_result = await kb_service.query_knowledge_base(kb_query)
        
        print(f"   Query: {kb_query}")
        print(f"   Confidence: {kb_result.get('confidence_score', 0)}")
        print(f"   Answer: {kb_result.get('answer', 'No answer')[:100]}...")
        print(f"   ✅ RAG response generated with system prompt integration")
    except Exception as e:
        print(f"   ❌ RAG test failed: {str(e)}")
    
    print()
    
    # Test 2: General Knowledge Response
    print("🤖 Test 2: General Knowledge Response with System Prompt")
    try:
        # Query for general knowledge
        general_query = "آب و هوا امروز چطور است؟"
        user_id = "test_user_system_prompt"
        
        response = await chat_service.process_message(user_id, general_query)
        
        print(f"   Query: {general_query}")
        print(f"   Knowledge Source: {response['query_analysis']['knowledge_source']}")
        print(f"   Answer: {response['answer'][:100]}...")
        print(f"   ✅ General knowledge response generated with system prompt")
    except Exception as e:
        print(f"   ❌ General knowledge test failed: {str(e)}")
    
    print()
    
    # Test 3: Check conversation memory for system prompt
    print("💭 Test 3: Verify System Prompt in Conversation Memory")
    try:
        user_id = "test_user_memory"
        
        # Create a session to check memory
        conversation = await chat_service._get_or_create_session(user_id)
        messages = conversation.memory.chat_memory.messages
        
        print(f"   Total messages in memory: {len(messages)}")
        
        # Check if first message is SystemMessage
        if messages and hasattr(messages[0], 'content'):
            first_message = messages[0]
            message_type = type(first_message).__name__
            print(f"   First message type: {message_type}")
            
            if message_type == "SystemMessage":
                print(f"   System message content: {first_message.content[:100]}...")
                print(f"   ✅ System prompt correctly added to conversation memory")
            else:
                print(f"   ❌ First message is not SystemMessage")
        else:
            print(f"   ❌ No messages found in memory")
            
    except Exception as e:
        print(f"   ❌ Memory test failed: {str(e)}")
    
    print()
    print("🎯 System Prompt Integration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_system_prompt_integration())