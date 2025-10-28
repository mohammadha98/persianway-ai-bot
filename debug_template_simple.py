#!/usr/bin/env python3
"""
Simple debug script to check the actual prompt template configuration
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.config_service import ConfigService

async def debug_template_config():
    """Debug the prompt template configuration"""
    print("🔍 Debugging Prompt Template Configuration...")
    
    try:
        # Get config service
        config_service = ConfigService()
        
        # Load configuration
        await config_service._load_config()
        
        # Get RAG settings
        rag_settings = await config_service.get_rag_settings()
        
        print("✅ Configuration loaded successfully")
        print(f"📝 Prompt Template:")
        print("-" * 50)
        print(rag_settings.prompt_template)
        print("-" * 50)
        
        # Check for problematic variables
        if "{query}" in rag_settings.prompt_template:
            print("⚠️  PROBLEM FOUND: Template contains {query} instead of {question}")
            print("🔧 This is likely the cause of the ValueError!")
        elif "{question}" in rag_settings.prompt_template:
            print("✅ Template correctly uses {question}")
        else:
            print("❓ Template doesn't contain {question} or {query}")
            
        # Check for context variable
        if "{context}" in rag_settings.prompt_template:
            print("✅ Template correctly uses {context}")
        else:
            print("⚠️  Template doesn't contain {context}")
            
        print(f"\n📋 System Prompt:")
        print("-" * 30)
        print(rag_settings.system_prompt[:200] + "..." if len(rag_settings.system_prompt) > 200 else rag_settings.system_prompt)
        print("-" * 30)
        
        # Check system prompt for problematic variables
        if "{query}" in rag_settings.system_prompt:
            print("⚠️  PROBLEM FOUND: System prompt contains {query}")
        elif "{question}" in rag_settings.system_prompt:
            print("⚠️  System prompt contains {question} - this might be problematic")
            
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_template_config())