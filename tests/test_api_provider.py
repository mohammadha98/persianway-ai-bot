import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from app.core.config import settings
from app.services.chat_service import get_llm


def test_api_provider_selection():
    """Test the API provider selection logic."""
    # Print the current settings
    print(f"\nCurrent settings:")
    print(f"PREFERRED_API_PROVIDER: {settings.PREFERRED_API_PROVIDER}")
    print(f"OPENAI_API_KEY: {'Set' if settings.OPENAI_API_KEY else 'Not set'}")
    print(f"OPENROUTER_API_KEY: {'Set' if settings.OPENROUTER_API_KEY else 'Not set'}")
    
    # Test with the current settings
    print(f"\nTesting with current settings (PREFERRED_API_PROVIDER={settings.PREFERRED_API_PROVIDER})...")
    try:
        llm = get_llm()
        print(f"Successfully initialized LLM: {llm}")
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
    
    # Test with each provider setting
    for provider in ["auto", "openai", "openrouter"]:
        print(f"\nTesting with PREFERRED_API_PROVIDER={provider}...")
        
        # Temporarily override the setting
        original_provider = settings.PREFERRED_API_PROVIDER
        settings.PREFERRED_API_PROVIDER = provider
        
        try:
            llm = get_llm()
            print(f"Successfully initialized LLM: {llm}")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
        
        # Restore the original setting
        settings.PREFERRED_API_PROVIDER = original_provider


if __name__ == "__main__":
    test_api_provider_selection()