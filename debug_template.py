#!/usr/bin/env python3
"""
Debug script to check the actual prompt template being used
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.knowledge_base import get_knowledge_base_service

async def debug_template():
    """Debug the actual prompt template being used"""
    print("🔍 Debugging Prompt Template...")
    
    try:
        # Get knowledge base service
        kb_service = get_knowledge_base_service()
        
        # Get QA chain
        qa_chain = await kb_service._get_qa_chain()
        
        if qa_chain is None:
            print("❌ QA Chain is None - vector store not available")
            return
            
        print("✅ QA Chain created successfully")
        
        # Access the prompt template
        if hasattr(qa_chain, 'combine_documents_chain'):
            combine_chain = qa_chain.combine_documents_chain
            print(f"📋 Combine chain type: {type(combine_chain)}")
            
            if hasattr(combine_chain, 'llm_chain'):
                llm_chain = combine_chain.llm_chain
                print(f"🔗 LLM chain type: {type(llm_chain)}")
                
                if hasattr(llm_chain, 'prompt'):
                    prompt = llm_chain.prompt
                    print(f"📝 Prompt type: {type(prompt)}")
                    print(f"🔑 Input variables: {prompt.input_variables}")
                    print(f"📄 Template preview (first 200 chars):")
                    print("-" * 50)
                    print(prompt.template[:200] + "..." if len(prompt.template) > 200 else prompt.template)
                    print("-" * 50)
                    
                    # Check for problematic variables
                    if "{query}" in prompt.template:
                        print("⚠️  PROBLEM FOUND: Template contains {query} instead of {question}")
                    elif "{question}" in prompt.template:
                        print("✅ Template correctly uses {question}")
                    else:
                        print("❓ Template doesn't contain {question} or {query}")
                        
                    # Check input_variables
                    if "query" in prompt.input_variables:
                        print("⚠️  PROBLEM FOUND: input_variables contains 'query' instead of 'question'")
                    elif "question" in prompt.input_variables:
                        print("✅ input_variables correctly contains 'question'")
                    else:
                        print("❓ input_variables doesn't contain 'question' or 'query'")
                        
                else:
                    print("❌ LLM chain has no prompt attribute")
            else:
                print("❌ Combine chain has no llm_chain attribute")
        else:
            print("❌ QA chain has no combine_documents_chain attribute")
            
        # Test with different input formats
        print("\n🧪 Testing different input formats:")
        
        test_query = "تست"
        
        # Test 1: {"question": query}
        try:
            print("Testing: {'question': 'تست'}")
            result = qa_chain({"question": test_query})
            print("✅ Success with 'question' key")
        except Exception as e:
            print(f"❌ Failed with 'question' key: {str(e)}")
            
        # Test 2: {"query": query}  
        try:
            print("Testing: {'query': 'تست'}")
            result = qa_chain({"query": test_query})
            print("✅ Success with 'query' key")
        except Exception as e:
            print(f"❌ Failed with 'query' key: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_template())