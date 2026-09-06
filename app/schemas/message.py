from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceDocument(BaseModel):
    """Schema for normalized source documents used in RAG responses."""
    
    content: Optional[str] = Field(default=None, description="Document content snippet")
    source: Optional[str] = Field(default=None, description="Source file or origin identifier")
    title: Optional[str] = Field(default=None, description="Document title if available")
    page: Optional[int] = Field(default=None, description="Page number for paginated documents")
    source_type: Optional[str] = Field(default=None, description="Type of source (qa_contribution, web_search, etc.)")
    is_public: Optional[bool] = Field(default=None, description="Whether the source is public")
    meta_tags: Optional[str] = Field(default=None, description="Comma-separated meta tags")
    question: Optional[str] = Field(default=None, description="Question for QA-type documents")
    answer: Optional[str] = Field(default=None, description="Answer for QA-type documents")
    hash_id: Optional[str] = Field(default=None, description="Knowledge base entry hash ID")
    score: Optional[float] = Field(default=None, description="Retrieval relevance score")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8...",
                "source": "agriculture_guide.pdf",
                "title": "Tomato Cultivation Guide",
                "page": 15,
                "source_type": "qa_contribution",
                "is_public": True
            }
        }
    }


class MessageDocument(BaseModel):
    """Schema for individual messages within a conversation."""
    
    message_id: Optional[str] = Field(default=None, description="Unique identifier for this message (used for feedback tracking)")
    role: MessageRole = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the message was created")
    
    # Analysis data (mainly for assistant messages)
    confidence_score: Optional[float] = Field(default=None, description="Confidence score of the response")
    knowledge_source: Optional[str] = Field(default=None, description="Source of knowledge used (knowledge_base, general_knowledge, none)")
    requires_human_referral: Optional[bool] = Field(default=None, description="Whether human referral was required")
    reasoning: Optional[str] = Field(default=None, description="Reasoning behind the response decision")
    
    # Response parameters (for assistant messages)
    model_used: Optional[str] = Field(default=None, description="The AI model used for the response")
    temperature: Optional[float] = Field(default=None, description="Temperature setting used")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens setting used")
    
    # RAG-related metadata
    sources_used: Optional[List[str]] = Field(default=None, description="List of sources used in RAG response")
    normalized_sources: Optional[List[SourceDocument]] = Field(default=None, description="Normalized source documents used in RAG response")
    prompt_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Coach-only snapshot of the exact prompt/context used to generate this message",
    )
    
    # Performance metadata
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    
    # Feedback relationship
    feedback_ids: Optional[List[str]] = Field(default=None, description="List of feedback IDs associated with this message")
    has_feedback: Optional[bool] = Field(default=False, description="Quick flag indicating if any feedback exists")
    
    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "example": {
                "role": "user",
                "content": "What is the best pH for tomato cultivation?",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class MessageResponse(BaseModel):
    """Schema for message API responses."""
    
    message_id: Optional[str] = Field(default=None, description="Unique identifier for this message (for feedback tracking)")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(..., description="Message timestamp")
    confidence_score: Optional[float] = Field(default=None, description="Response confidence score (for assistant messages)")
    knowledge_source: Optional[str] = Field(default=None, description="Knowledge source used (for assistant messages)")
    requires_human_referral: Optional[bool] = Field(default=None, description="Whether human referral was required (for assistant messages)")
    sources_used: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of normalized sources used in RAG response (for assistant messages)")
    normalized_sources: Optional[List[SourceDocument]] = Field(default=None, description="Normalized source documents used in RAG response")
    
    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "example": {
                "role": "assistant",
                "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8.",
                "timestamp": "2024-01-15T10:30:01Z",
                "confidence_score": 0.95,
                "knowledge_source": "knowledge_base",
                "requires_human_referral": False,
                "normalized_sources": [
                    {
                        "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8...",
                        "source": "agriculture_guide.pdf",
                        "title": "Tomato Cultivation Guide",
                        "page": 15,
                        "source_type": "qa_contribution",
                        "is_public": True
                    }
                ]
            }
        }
    }


class MessageDetailResponse(BaseModel):
    """Schema for detailed message response including all metadata and sources."""

    message_id: Optional[str] = Field(default=None, description="Unique identifier for this message")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")

    # Analysis data
    confidence_score: Optional[float] = Field(default=None, description="Confidence score of the response")
    knowledge_source: Optional[str] = Field(default=None, description="Source of knowledge used")
    requires_human_referral: Optional[bool] = Field(default=None, description="Whether human referral was required")
    reasoning: Optional[str] = Field(default=None, description="Reasoning behind the response decision")

    # Response parameters
    model_used: Optional[str] = Field(default=None, description="The AI model used")
    temperature: Optional[float] = Field(default=None, description="Temperature setting used")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens setting used")

    # RAG metadata
    sources_used: Optional[List[str]] = Field(default=None, description="List of source identifiers")
    normalized_sources: Optional[List[SourceDocument]] = Field(default=None, description="Normalized source documents used in RAG response")

    # Performance metadata
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")

    # Feedback info
    feedback_ids: Optional[List[str]] = Field(default=None, description="List of feedback IDs")
    has_feedback: Optional[bool] = Field(default=False, description="Whether any feedback exists")

    # Conversation context
    conversation_id: Optional[str] = Field(default=None, description="ID of the parent conversation")
    user_id: Optional[str] = Field(default=None, description="User ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")

    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "example": {
                "message_id": "msg_abc123",
                "role": "assistant",
                "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8.",
                "confidence_score": 0.95,
                "knowledge_source": "knowledge_base",
                "requires_human_referral": False,
                "reasoning": "High confidence answer found in knowledge base.",
                "model_used": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 150,
                "normalized_sources": [
                    {
                        "content": "Tomatoes prefer slightly acidic soil...",
                        "source": "agriculture_guide.pdf",
                        "title": "Tomato Cultivation Guide",
                        "page": 15,
                        "source_type": "qa_contribution",
                        "is_public": True
                    }
                ],
                "conversation_id": "conv_xyz789",
                "user_id": "user123",
                "session_id": "session_abc"
            }
        }
    }
