"""
Test script for the new clarification intent detection feature.

This script demonstrates how the new intent detection system works
with three categories: PUBLIC, PRIVATE, and NEEDS_CLARIFICATION.
"""

import asyncio
from app.services.chat_service import ChatService

async def test_intent_detection():
    """Test various queries to see how they are classified."""
    
    chat_service = ChatService()
    
    # Test cases
    test_queries = [
        # Should be NEEDS_CLARIFICATION
        "چطور؟",
        "اینا چیه؟",
        "بهتر",
        "مشکل داره",
        "این کارو کنم؟",
        
        # Should be PUBLIC
        "شرکت پرشین وی چیه؟",
        "در مورد شرکت شما بگو",
        "دفتر شما کجاست؟",
        
        # Should be PRIVATE (Agriculture)
        "بهترین کود برای گندم چیست؟",
        "چطور کود بزنم؟",
        "چطور آفت رو از بین ببرم؟",
        
        # Should be PRIVATE (Health & Beauty)
        "چه ویتامینی برای پوست خوبه؟",
        "کرم ضد آفتاب خوب معرفی کن",
        "درمان سردرد چیه؟",
        
        # Should be PRIVATE (Services)
        "خدمات شما چیه؟",
        "محصولات شما چیه؟",
        "قیمت محصولات شما چقدره؟",
        
        # Should be OFF_TOPIC
        "بهترین تیم فوتبال کدومه؟",
        "چطور برنامه نویسی یاد بگیرم؟",
        "قیمت دلار امروز چقدره؟",
        "فیلم خوب پیشنهاد بده",
        "نظرت درباره انتخابات چیه؟"
    ]
    
    print("=" * 80)
    print("Testing Intent Classification System")
    print("=" * 80)
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")
        
        try:
            result = await chat_service.detect_query_intent(query)
            
            print(f"   Intent: {result['intent']}")
            print(f"   Is Public: {result['is_public']}")
            print(f"   Explanation: {result['explanation']}")
            
            if result.get('clarification_prompt'):
                print(f"   Clarification Prompt: {result['clarification_prompt']}")
            
            if result.get('off_topic_message'):
                print(f"   Off-Topic Message: {result['off_topic_message'][:100]}...")
            
            print()
            
        except Exception as e:
            print(f"   ERROR: {str(e)}")
            print()
    
    print("=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_intent_detection())

