# Conversation Storage System

This document describes the conversation storage system implementation for the Persian Agricultural Knowledge Base AI Bot.

## Overview

The conversation storage system automatically logs all conversations between users and the AI system without modifying existing APIs. It uses MongoDB as the database and implements a middleware-based approach for transparent conversation logging.

## Features

- **Automatic Logging**: All chat conversations are automatically logged via middleware
- **No API Changes**: Existing client APIs remain unchanged
- **MongoDB Storage**: Uses MongoDB for scalable conversation storage
- **Rich Metadata**: Stores comprehensive conversation metadata including confidence scores, sources, and response parameters
- **Search & Retrieval**: Provides powerful search and retrieval capabilities
- **Privacy Protection**: IP addresses are hashed for privacy
- **TTL Support**: Automatic data cleanup using MongoDB TTL indexes
- **Performance Optimized**: Uses appropriate indexes for fast queries

## Architecture

### Components

1. **ConversationLoggerMiddleware**: Automatically intercepts and logs chat requests/responses
2. **ConversationService**: Handles conversation storage and retrieval operations
3. **DatabaseService**: Manages MongoDB connections and indexing
4. **Conversation APIs**: New endpoints for accessing conversation history
5. **Pydantic Schemas**: Type-safe data models for conversations

### Data Flow

```
Client Request → Chat API → ConversationLoggerMiddleware → Chat Service → Response
                     ↓
              ConversationService → MongoDB
```

## Database Schema

### Conversation Document Structure

```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "user_question": "string",
  "system_response": "string",
  "timestamp": "datetime",
  "is_agriculture_related": "boolean",
  "confidence_score": "float",
  "knowledge_source": "string",
  "requires_human_referral": "boolean",
  "reasoning": "string",
  "model_used": "string",
  "temperature": "float",
  "max_tokens": "integer",
  "sources_used": ["string"],
  "session_id": "string",
  "user_agent": "string",
  "ip_address": "string (hashed)",
  "response_time_ms": "float"
}
```

### Indexes

The system creates the following indexes for optimal performance:

- `user_id` - Single field index
- `timestamp` - Single field index
- `knowledge_source` - Single field index
- `requires_human_referral` - Single field index
- `confidence_score` - Single field index
- `is_agriculture_related` - Single field index
- `(user_id, timestamp)` - Compound index for user-specific queries
- `(timestamp, confidence_score)` - Compound index for time-based analytics
- Text index on `(user_question, system_response)` - For full-text search
- TTL index on `timestamp` - For automatic data cleanup

## Configuration

### Environment Variables

Add the following to your `.env` file:

```env
# MongoDB Settings
MONGODB_URL="mongodb://localhost:27017"
MONGODB_DATABASE="persian_agriculture_db"
MONGODB_CONVERSATIONS_COLLECTION="conversations"
CONVERSATION_TTL_DAYS=365
```

### Configuration Options

- `MONGODB_URL`: MongoDB connection string
- `MONGODB_DATABASE`: Database name for storing conversations
- `MONGODB_CONVERSATIONS_COLLECTION`: Collection name for conversations
- `CONVERSATION_TTL_DAYS`: Time-to-live for conversations (0 = no expiration)

## API Endpoints

### New Conversation APIs

#### 1. Get User Conversations
```http
GET /api/conversations/{user_id}
```

Retrieve all conversations for a specific user with pagination.

**Parameters:**
- `user_id` (path): User identifier
- `limit` (query): Maximum results (1-100, default: 50)
- `skip` (query): Results to skip for pagination (default: 0)

#### 2. Get Latest User Conversations
```http
GET /api/conversations/{user_id}/latest
```

Retrieve the most recent conversations for a user.

**Parameters:**
- `user_id` (path): User identifier
- `limit` (query): Maximum results (1-50, default: 10)

#### 3. Search Conversations (POST)
```http
POST /api/conversations/search
```

Search conversations using multiple criteria.

**Request Body:**
```json
{
  "user_id": "string (optional)",
  "start_date": "datetime (optional)",
  "end_date": "datetime (optional)",
  "search_text": "string (optional)",
  "knowledge_source": "string (optional)",
  "requires_human_referral": "boolean (optional)",
  "min_confidence": "float (optional)",
  "max_confidence": "float (optional)",
  "limit": "integer (default: 50)",
  "skip": "integer (default: 0)"
}
```

#### 4. Advanced Search (GET)
```http
GET /api/conversations/search/advanced
```

Same functionality as POST search but using query parameters.

#### 5. Conversation Statistics
```http
GET /api/conversations/stats/overview
```

Get conversation statistics.

**Parameters:**
- `user_id` (query, optional): Filter stats by specific user

**Response:**
```json
{
  "total_conversations": 1250,
  "agriculture_related": 1100,
  "human_referrals": 150,
  "avg_confidence": 0.85,
  "knowledge_base_responses": 800,
  "general_knowledge_responses": 300
}
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install motor pymongo
```

### 2. Configure MongoDB

Ensure MongoDB is running and accessible. Update your `.env` file with the correct connection details.

### 3. Start the Application

The conversation storage system will automatically initialize when the FastAPI application starts.

```bash
python main.py
```

### 4. Test the System

Run the test script to verify everything is working:

```bash
python test_conversation_storage.py
```

## Usage Examples

### Python Client Example

```python
import httpx
import asyncio

async def get_user_conversations(user_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/conversations/{user_id}",
            params={"limit": 20}
        )
        return response.json()

async def search_conversations(search_text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/conversations/search",
            json={
                "search_text": search_text,
                "knowledge_source": "knowledge_base",
                "min_confidence": 0.8
            }
        )
        return response.json()
```

### JavaScript/Fetch Example

```javascript
// Get user conversations
async function getUserConversations(userId) {
    const response = await fetch(`/api/conversations/${userId}?limit=20`);
    return await response.json();
}

// Search conversations
async function searchConversations(searchText) {
    const response = await fetch('/api/conversations/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            search_text: searchText,
            knowledge_source: 'knowledge_base',
            min_confidence: 0.8
        })
    });
    return await response.json();
}
```

## Security Considerations

### Data Privacy

- **IP Address Hashing**: User IP addresses are hashed using SHA-256 before storage
- **No Sensitive Data**: The system avoids storing sensitive personal information
- **Access Control**: Consider implementing authentication for conversation retrieval APIs

### Data Sanitization

- All conversation data is validated using Pydantic schemas
- Text content is stored as-is but should be sanitized by the client if displayed in web interfaces

### Recommended Security Measures

1. **Authentication**: Implement user authentication for conversation APIs
2. **Authorization**: Ensure users can only access their own conversations
3. **Rate Limiting**: Implement rate limiting on search endpoints
4. **Data Encryption**: Consider encrypting sensitive conversation content
5. **Audit Logging**: Log access to conversation data for security auditing

## Performance Considerations

### Database Performance

- **Indexes**: The system creates appropriate indexes for common query patterns
- **Connection Pooling**: Motor (MongoDB async driver) handles connection pooling automatically
- **Query Optimization**: Use pagination and filtering to limit result sets

### Monitoring

- Monitor MongoDB performance and index usage
- Track conversation storage success rates
- Monitor API response times for conversation retrieval

### Scaling

- **Horizontal Scaling**: MongoDB supports sharding for large datasets
- **Read Replicas**: Use read replicas for conversation retrieval to reduce load on primary
- **Archiving**: Implement archiving strategy for old conversations

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check MongoDB is running: `mongosh --eval "db.adminCommand('ping')"`
   - Verify connection string in `.env` file
   - Check network connectivity and firewall settings

2. **Conversations Not Being Logged**
   - Check middleware is properly registered in `main.py`
   - Verify chat endpoints are being called correctly
   - Check application logs for error messages

3. **Search Not Working**
   - Ensure text indexes are created properly
   - Check search query syntax
   - Verify MongoDB version supports text search

4. **Performance Issues**
   - Check index usage with MongoDB explain plans
   - Monitor query execution times
   - Consider adding more specific indexes for your query patterns

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing

Run the test script to verify system functionality:

```bash
python test_conversation_storage.py
```

## Future Enhancements

### Planned Features

1. **Analytics Dashboard**: Web interface for conversation analytics
2. **Export Functionality**: Export conversations to various formats (CSV, JSON)
3. **Advanced Search**: More sophisticated search with filters and sorting
4. **Conversation Tagging**: Add custom tags to conversations
5. **User Feedback**: Store user feedback on AI responses
6. **A/B Testing**: Support for A/B testing different AI models

### Integration Opportunities

1. **Machine Learning**: Use conversation data for model training and improvement
2. **Business Intelligence**: Integration with BI tools for advanced analytics
3. **Customer Support**: Integration with customer support systems
4. **Quality Assurance**: Automated quality scoring of AI responses

## Support

For issues or questions regarding the conversation storage system:

1. Check this documentation
2. Review the test script for usage examples
3. Check application logs for error messages
4. Verify MongoDB connectivity and configuration

## License

This conversation storage system is part of the Persian Agricultural Knowledge Base project and follows the same licensing terms.