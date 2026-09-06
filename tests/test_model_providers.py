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
from app.services.chat_service import get_llm


def test_different_model_providers():
    """Test the model provider handling with different model types."""
    # Print the current settings
    logger.info(f"\nCurrent settings:")
    logger.info(f"PREFERRED_API_PROVIDER: {settings.PREFERRED_API_PROVIDER}")
    logger.info(f"OPENAI_API_KEY: {'Set' if settings.OPENAI_API_KEY else 'Not set'}")
    logger.info(f"OPENROUTER_API_KEY: {'Set' if settings.OPENROUTER_API_KEY else 'Not set'}")
    logger.info(f"DEFAULT_MODEL: {settings.DEFAULT_MODEL}")
    
    # Make sure we're using OpenRouter
    original_provider = settings.PREFERRED_API_PROVIDER
    settings.PREFERRED_API_PROVIDER = "openrouter"
    
    # Test with different model providers
    test_models = [
        "openai/gpt-3.5-turbo",  # OpenAI model with prefix
        "gpt-4",                 # OpenAI model without prefix
        "google/gemini-pro",     # Google model
        "anthropic/claude-2",    # Anthropic model
        "meta-llama/llama-2-70b-chat"  # Meta model
    ]
    
    for model in test_models:
        logger.info(f"\nTesting with model: {model}")
        try:
            llm = get_llm(model_name=model)
            logger.info(f"Successfully initialized LLM with model '{model}'")
            logger.info(f"Final model name used: {llm.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM with model '{model}': {e}")
    
    # Restore original setting
    settings.PREFERRED_API_PROVIDER = original_provider


if __name__ == "__main__":
    test_different_model_providers()