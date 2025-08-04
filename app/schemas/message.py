from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageDocument(BaseModel):
    """Schema for individual messages within a conversation."""
    
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
    
    # Performance metadata
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the best pH for tomato cultivation?",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class MessageResponse(BaseModel):
    """Schema for message API responses."""
    
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(..., description="Message timestamp")
    confidence_score: Optional[float] = Field(default=None, description="Response confidence score (for assistant messages)")
    knowledge_source: Optional[str] = Field(default=None, description="Knowledge source used (for assistant messages)")
    requires_human_referral: Optional[bool] = Field(default=None, description="Whether human referral was required (for assistant messages)")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "role": "assistant",
                "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8.",
                "timestamp": "2024-01-15T10:30:01Z",
                "confidence_score": 0.95,
                "knowledge_source": "knowledge_base",
                "requires_human_referral": False
            }
        }