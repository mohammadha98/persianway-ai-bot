import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from app.core.config import settings
from app.services.chat_service import get_llm, ChatService


def test_chat_with_different_models():
    """Demonstrate using different model providers with the chat service."""
    # Print the current settings
    logger.info(f"\nCurrent settings:")
    logger.info(f"PREFERRED_API_PROVIDER: {settings.PREFERRED_API_PROVIDER}")
    logger.info(f"DEFAULT_MODEL: {settings.DEFAULT_MODEL}")
    
    # Initialize the chat service
    chat_service = ChatService()
    
    # Test message
    test_message = "What are the benefits of organic farming?"
    user_id = "test-user-123"
    
    # Test with different models
    test_models = [
        None,                    # Use default model
        "openai/gpt-3.5-turbo",  # OpenAI model
        "google/gemini-pro",     # Google model
        "anthropic/claude-2"     # Anthropic model
    ]
    
    for model in test_models:
        model_display = model if model else f"{settings.DEFAULT_MODEL} (default)"
        logger.info(f"\n--- Testing with model: {model_display} ---")
        
        try:
            # Get response using the specified model
            response = chat_service.get_response(user_id, test_message, model)
            
            # Print the response
            logger.info(f"Response from {model_display}:")
            logger.info(f"{response}")
            
        except Exception as e:
            logger.error(f"Error with model {model_display}: {str(e)}")


if __name__ == "__main__":
    # Make sure we're using OpenRouter to access multiple providers
    original_provider = settings.PREFERRED_API_PROVIDER
    settings.PREFERRED_API_PROVIDER = "openrouter"
    
    try:
        test_chat_with_different_models()
    finally:
        # Restore original setting
        settings.PREFERRED_API_PROVIDER = original_provider