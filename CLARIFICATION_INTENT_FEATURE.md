# Intent Detection System - PersianWay AI Chatbot

## Overview

This document describes the **multi-intent detection system** for the PersianWay AI chatbot. The system now intelligently classifies user queries into four categories to provide optimal responses and guide users effectively.

## What Changed

### Previous System (2 Intents)
- **PUBLIC**: Questions about the PersianWay company itself
- **PRIVATE**: Technical/domain expertise questions (default)

### Current System (4 Intents)
- **PUBLIC**: Questions about the PersianWay company itself
- **PRIVATE**: Technical/domain expertise questions (default for relevant topics)
- **NEEDS_CLARIFICATION**: ✨ Unclear or vague questions that need more details
- **OFF_TOPIC**: ✨ Questions clearly outside our expertise areas

## Features

### 1. Intent Classification

The system now classifies user queries into four categories:

#### PUBLIC Intent
Questions specifically about the PersianWay company:
- Company history, mission, vision, background
- Business model, organizational structure
- Office locations, contact information
- Company announcements, policies

**Examples:**
- "شرکت پرشین وی چیه؟" (What is PersianWay?)
- "دفتر شما کجاست؟" (Where is your office?)
- "شرکت شما چه کاری انجام میده؟" (What does your company do?)

#### PRIVATE Intent
Questions related to our core expertise areas (DEFAULT for relevant topics):

**Three Main Areas:**
1. **Agriculture (کشاورزی)**: Farming, crops, fertilizers, irrigation, soil, pests, etc.
2. **Health (سلامت)**: Wellness, nutrition, medical questions, health products
3. **Beauty (زیبایی)**: Skincare, cosmetics, beauty products, treatments

Also includes:
- Product recommendations, troubleshooting
- Questions about services and features
- Personal account issues
- General knowledge that might relate to these areas

**Examples:**
- **Agriculture**: "بهترین کود برای گندم؟" (Best fertilizer for wheat?)
- **Health**: "چه ویتامینی برای پوست خوبه؟" (Which vitamin is good for skin?)
- **Beauty**: "کرم ضد آفتاب خوب معرفی کن" (Recommend a good sunscreen)
- **Services**: "خدمات شما چیه؟" (What are your services?)

#### NEEDS_CLARIFICATION Intent ✨
Unclear or vague questions that need more details:
- Questions too vague or ambiguous
- Questions missing critical context
- Single words or very short phrases without clear meaning
- Questions with pronouns (این, اون, اینا) without clear references
- Incomplete questions or fragmented sentences

**Examples:**
- "چطور؟" (How?)
- "اینا چیه؟" (What are these?)
- "بهتر" (Better)
- "مشکل داره" (Has a problem)

#### OFF_TOPIC Intent ✨
Questions clearly UNRELATED to our expertise areas:
- Topics completely outside agriculture, health, beauty, and company info
- Politics, sports, entertainment, technology, real estate, finance
- Unrelated products or services we don't provide
- Questions about other companies or brands

**Important:** The system is LENIENT - only uses OFF_TOPIC if question is CLEARLY and OBVIOUSLY unrelated.

**Examples:**
- "بهترین تیم فوتبال کدومه؟" (Which is the best football team?)
- "چطور برنامه نویسی یاد بگیرم؟" (How to learn programming?)
- "قیمت دلار امروز چقدره؟" (What's the dollar price today?)
- "فیلم خوب پیشنهاد بده" (Recommend a good movie)
- "نظرت درباره انتخابات چیه؟" (What's your opinion about elections?)

### 2. Response Handling

#### A. Clarification Response

When a query is classified as `NEEDS_CLARIFICATION`, the system:

1. **Skips the knowledge base search** (saves computational resources)
2. **Responds with a helpful clarification prompt** asking for more details
3. **Provides guidance** on what information would be helpful
4. **Maintains conversation context** so the user can provide clarification in the next message

##### Default Clarification Message

```
سوال شما کمی مبهم است. لطفاً جزئیات بیشتری ارائه دهید تا بتوانم بهتر به شما کمک کنم.

مثلاً:
- به چه محصول یا موضوع خاصی اشاره دارید؟
- چه اطلاعاتی نیاز دارید؟
- مشکل یا سوال دقیق شما چیست؟
```

Translation:
> Your question is somewhat unclear. Please provide more details so I can better help you.
>
> For example:
> - Which product or specific topic are you referring to?
> - What information do you need?
> - What is your exact problem or question?

##### Custom Clarification Prompts

The LLM can also generate **custom clarification prompts** specific to the user's query, making the response more contextual and helpful.

#### B. Off-Topic Response

When a query is classified as `OFF_TOPIC`, the system:

1. **Skips the knowledge base search** (saves resources)
2. **Responds with a friendly redirect message** explaining what topics we cover
3. **Guides the user** to ask about relevant topics (agriculture, health, beauty, company)
4. **Maintains positive tone** while setting boundaries

##### Default Off-Topic Message

```
درود! 🌹

متأسفانه این سوال خارج از حوزه تخصص ماست. پرشین وی در حوزه‌های زیر آماده کمک به شماست:

🌱 **کشاورزی**: کاشت، داشت، کود، آبیاری، مبارزه با آفات
💊 **سلامت**: تغذیه، ویتامین‌ها، محصولات سلامتی
💄 **زیبایی**: مراقبت از پوست، محصولات آرایشی و بهداشتی
🏢 **اطلاعات شرکت**: درباره پرشین وی، خدمات و محصولات

چطور می‌تونم در این زمینه‌ها بهتون کمک کنم؟
```

##### Custom Off-Topic Messages

The LLM can generate **custom redirect messages** tailored to the specific off-topic question, making the response more natural.

### 3. API Changes

#### New Method: `detect_query_intent()`

```python
async def detect_query_intent(
    self,
    message: str,
    conversation_history: Optional[Any] = None,
    *,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """Detect the intent of the user's query.
    
    Returns:
        {
            "intent": "PUBLIC" | "PRIVATE" | "NEEDS_CLARIFICATION" | "OFF_TOPIC",
            "is_public": bool,  # For backward compatibility
            "explanation": str,  # Reason for classification
            "clarification_prompt": str | None,  # Custom message for unclear queries
            "off_topic_message": str | None  # Custom message for off-topic queries
        }
    """
```

#### Legacy Method (Backward Compatible)

The old `detect_public_data_intent()` method still exists as a wrapper:

```python
async def detect_public_data_intent(
    self,
    message: str,
    conversation_history: Optional[Any] = None,
    *,
    llm: Optional[Any] = None
) -> bool:
    """Legacy method - returns is_public as boolean."""
    result = await self.detect_query_intent(message, conversation_history, llm=llm)
    return result.get("is_public", False)
```

### 4. Process Flow

```
User Query
    ↓
Intent Classification
    ↓
    ├─→ [OFF_TOPIC] → Send friendly redirect message
    │                  (Skip KB search, guide to relevant topics)
    │
    ├─→ [NEEDS_CLARIFICATION] → Send clarification prompt
    │                             (Skip KB search, ask for details)
    │
    ├─→ [PUBLIC] → Query knowledge base (public documents only)
    │               Answer from company information
    │
    └─→ [PRIVATE] → Query knowledge base (all documents)
                     Answer from agriculture/health/beauty expertise
```

## Benefits

### 1. **Improved User Experience**
- Users get immediate feedback when their question is unclear OR off-topic
- Clear guidance on what topics are covered
- Friendly redirects instead of confusing or irrelevant answers
- Reduces frustration and sets proper expectations

### 2. **Resource Optimization**
- Avoids expensive vector DB searches for unclear AND off-topic queries
- Reduces LLM token usage on unanswerable questions
- Faster response time for both clarification and redirect messages
- Saves computational costs on irrelevant queries

### 3. **Better Answer Quality**
- System only attempts KB search when question is clear AND relevant
- Focuses expertise on agriculture, health, and beauty domains
- Reduces low-confidence responses
- Fewer false positive answers

### 4. **Clear Boundaries**
- Establishes scope of service (agriculture, health, beauty, company info)
- Politely declines off-topic questions
- Guides users to ask relevant questions
- Maintains professional brand image

### 5. **Conversation Flow**
- Maintains natural dialogue with users
- Conversation history helps resolve pronouns and references
- Users can iteratively refine their questions
- Positive tone even when declining off-topic questions

## Implementation Details

### File Modified
- `app/services/chat_service.py`

### Key Changes

1. **Renamed and enhanced `detect_public_data_intent()` → `detect_query_intent()`**
   - Now returns a dictionary instead of boolean
   - Supports three intent categories
   - Includes clarification prompts

2. **Updated `process_message()` flow**
   - Checks for `NEEDS_CLARIFICATION` intent first
   - Responds immediately with clarification prompt
   - Skips knowledge base query for unclear questions

3. **Enhanced LLM prompt for intent classification**
   - Added detailed NEEDS_CLARIFICATION criteria
   - Included examples for each intent category
   - Better guidance on edge cases

4. **Maintained backward compatibility**
   - Old `detect_public_data_intent()` method still works
   - Existing code continues to function

### Response Format

When `NEEDS_CLARIFICATION` is detected:

```json
{
    "query_analysis": {
        "confidence_score": 0.5,
        "knowledge_source": "clarification_request",
        "requires_human_referral": false,
        "reasoning": "Query needs clarification: Too vague - missing context"
    },
    "response_parameters": {
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 1.0
    },
    "answer": "سوال شما کمی مبهم است. لطفاً جزئیات بیشتری ارائه دهید..."
}
```

## Testing

### Test Script
Use `test_intent_clarification.py` to test the feature:

```bash
python test_intent_clarification.py
```

### Test Cases

The test script includes queries for all three intent categories:

**NEEDS_CLARIFICATION:**
- "چطور؟"
- "اینا چیه؟"
- "بهتر"
- "مشکل داره"

**PUBLIC:**
- "شرکت پرشین وی چیه؟"
- "در مورد شرکت شما بگو"

**PRIVATE:**
- "بهترین کود برای گندم چیست؟"
- "چطور کود بزنم؟"

## Configuration

### LLM Settings
- Model: `gpt-4o-mini` (for intent classification)
- Temperature: `0.1` (for consistent classification)
- Top P: `0.1` (for focused responses)

### Confidence Score
- Clarification requests get a confidence score of `0.5`
- Does not trigger human referral (`requires_human_referral: false`)

## Future Enhancements

Potential improvements for future versions:

1. **Multi-turn Clarification**
   - Track how many times clarification was requested
   - Escalate to human support after multiple unclear attempts

2. **Context-Aware Clarification**
   - Use conversation history more deeply
   - Suggest specific follow-up questions based on previous context

3. **Analytics Dashboard**
   - Track clarification request frequency
   - Identify common unclear query patterns
   - Improve system prompts based on data

4. **Smart Suggestions**
   - Provide multiple choice clarification options
   - Suggest related topics user might be asking about

5. **Language-Specific Rules**
   - Enhanced Persian language patterns
   - Better handling of Persian pronouns and references

## Conclusion

The clarification intent detection feature significantly improves the chatbot's ability to handle unclear queries, leading to better user experience, optimized resource usage, and higher quality responses. The implementation maintains backward compatibility while adding powerful new capabilities to guide users toward more productive conversations.

---

**Version:** 1.0  
**Date:** 2025-11-10  
**Author:** PersianWay AI Team

