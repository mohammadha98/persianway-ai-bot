import asyncio
import logging
import sys
from datetime import datetime
import uuid

# Add project root to path
import os
sys.path.append(os.getcwd())

from app.services.chat_service import ChatService
from app.services.config_service import ConfigService
from app.services.database import get_database_service
from langchain_core.messages import SystemMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_test():
    logger.info("Starting reproduction test...")
    
    # 1. Initialize Services
    db_service = await get_database_service()
    config_service = ConfigService()
    await config_service.initialize()
    
    chat_service = ChatService()
    # ChatService creates its own config_service internally, so we need to rely on that one 
    # or we can access it via chat_service.config_service
    
    user_id = f"test_user_{uuid.uuid4()}"
    
    # 2. Create initial session
    logger.info(f"Creating session for user {user_id}")
    # This triggers _get_or_create_session -> _ensure_latest_config -> _load_config
    session = await chat_service._get_or_create_session(user_id)
    
    # Check initial system prompt
    messages = session.memory.chat_memory.messages
    system_msg = next((m for m in messages if isinstance(m, SystemMessage)), None)
    initial_prompt = system_msg.content if system_msg else "NONE"
    logger.info(f"Initial System Prompt: {initial_prompt[:50]}...")
    
    # 3. Update Config in DB
    new_prompt = f"You are a specialized AI updated at {datetime.now().isoformat()}"
    logger.info(f"Updating system prompt in DB to: {new_prompt}")
    
    # We use the external config_service to update the DB, simulating an API call or another worker
    current_config = await config_service.get_config()
    updates = {
        "rag_settings": {
            "system_prompt": new_prompt
        }
    }
    await config_service.update_config(updates)
    
    # 4. Simulate next request
    logger.info("Simulating next request (should trigger refresh)...")
    
    # In a real request, process_message calls _ensure_latest_config
    # We'll call _get_or_create_session which also calls _ensure_latest_config
    
    # Ensure the DB update has propagated (MongoDB is usually fast but let's be sure)
    await asyncio.sleep(1)
    
    session_v2 = await chat_service._get_or_create_session(user_id)
    
    # Check if session object is different (it should be if cleared)
    if session is session_v2:
        logger.error("Session object IS THE SAME! Refresh did not happen or did not clear session.")
    else:
        logger.info("Session object is different (Good).")
        
    # Check new system prompt
    messages_v2 = session_v2.memory.chat_memory.messages
    system_msg_v2 = next((m for m in messages_v2 if isinstance(m, SystemMessage)), None)
    current_prompt = system_msg_v2.content if system_msg_v2 else "NONE"
    
    logger.info(f"New System Prompt: {current_prompt}")
    
    if current_prompt == new_prompt:
        logger.info("SUCCESS: System prompt updated correctly.")
    else:
        logger.error("FAILURE: System prompt did NOT update.")
        logger.error(f"Expected: {new_prompt}")
        logger.error(f"Got: {current_prompt}")

    # Restore original config (optional, but good practice)
    # For now we leave it as is since it's a dev env
    
if __name__ == "__main__":
    asyncio.run(run_test())
