# Human Referral System for Persian Agriculture Knowledge Base

## Overview

This document describes the human referral system implemented for the Persian Agriculture Knowledge Base. The system automatically detects when the AI model cannot provide an adequate response due to insufficient information in the knowledge base and triggers a referral to human experts.

## Features

- **Confidence Threshold Detection**: Automatically evaluates the relevance of retrieved documents to determine if an answer is adequate
- **Human Referral Flagging**: Adds a `requires_human_support` flag to responses when confidence is low
- **Query Logging**: Logs queries requiring human attention to a dedicated log file
- **Unique Query Identification**: Generates a unique ID for each query requiring human support
- **Configurable Parameters**: Threshold values and response messages are configurable via environment variables

## System Components

### Configuration Settings

The following settings in `app/core/config.py` control the human referral system:

- `KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD`: Minimum confidence score (0.0-1.0) to consider an answer adequate
- `HUMAN_REFERRAL_MESSAGE`: Custom message to display when a query is referred to human experts

### Knowledge Base Service

The `KnowledgeBaseService` class in `app/services/knowledge_base.py` has been enhanced with:

- `_calculate_confidence_score()`: Evaluates the relevance of retrieved documents
- `_log_human_referral()`: Logs queries requiring human attention
- Updated `query_knowledge_base()`: Includes human referral detection logic

### API Response Schema

The `KnowledgeBaseResponse` schema in `app/schemas/knowledge_base.py` now includes:

- `requires_human_support`: Boolean flag indicating if human intervention is needed
- `query_id`: Unique identifier for queries requiring human support

## How It Works

1. When a query is received, the system retrieves relevant documents from the knowledge base
2. A confidence score is calculated based on the relevance of the retrieved documents
3. If the confidence score is below the configured threshold, the system:
   - Sets `requires_human_support` to `true` in the response
   - Generates a unique `query_id`
   - Replaces the answer with the configured human referral message
   - Logs the query details to `human_referrals.log`
4. The API returns the response with the appropriate flags and message

## Example Response

When a query can be answered adequately:

```json
{
  "answer": "انواع کودهای شیمیایی عبارتند از: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
  "sources": [
    {
      "content": "کودهای شیمیایی به چند دسته تقسیم می‌شوند: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
      "source": "fertilization-guide-table.pdf",
      "page": 5
    }
  ],
  "requires_human_support": false,
  "query_id": null
}
```

When a query requires human support:

```json
{
  "answer": "متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد. سؤال شما برای بررسی بیشتر توسط کارشناسان ما ثبت شده است.",
  "sources": [],
  "requires_human_support": true,
  "query_id": "f8e7d6c5-b4a3-42f1-9e8d-7c6b5a4f3d2e"
}
```

## Log Format

Queries requiring human attention are logged to `human_referrals.log` with the following format:

```
2023-07-15 14:30:45,123 - INFO - HUMAN REFERRAL NEEDED
Query ID: f8e7d6c5-b4a3-42f1-9e8d-7c6b5a4f3d2e
Query: روش های کشت هیدروپونیک برای گیاهان دارویی چیست؟
Retrieved Sources: 2
Timestamp: 2023-07-15T14:30:45.123456
==================================================
```

## Customization

### Adjusting the Confidence Threshold

To adjust how sensitive the system is to referring queries to humans, modify the `KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD` in `.env` or directly in `app/core/config.py`:

```
# More strict (refers more queries to humans)
KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD=0.8

# More lenient (refers fewer queries to humans)
KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD=0.5
```

### Customizing the Referral Message

To change the message shown to users when a query is referred to humans, modify the `HUMAN_REFERRAL_MESSAGE` setting:

```
HUMAN_REFERRAL_MESSAGE="پاسخ به سوال شما نیاز به بررسی بیشتر دارد. کارشناسان ما به زودی با شما تماس خواهند گرفت."
```

## Future Improvements

- Implement a more sophisticated confidence scoring algorithm
- Add a web interface for human experts to view and respond to referred queries
- Implement a feedback loop to improve the knowledge base based on human expert responses
- Add support for conversation history in human referrals
- Implement priority levels for different types of queries requiring human attention