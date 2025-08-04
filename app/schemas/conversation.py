from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from .message import MessageDocument, MessageResponse


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")
        return field_schema


class ConversationDocument(BaseModel):
    """Schema for storing conversations in MongoDB with embedded messages."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="Unique identifier for the user")
    title: Optional[str] = Field(default=None, description="Optional title for the conversation")
    messages: List[MessageDocument] = Field(default_factory=list, description="List of messages in the conversation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the conversation was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the conversation was last updated")
    
    # Conversation-level metadata
    session_id: Optional[str] = Field(default=None, description="Session identifier if applicable")
    user_agent: Optional[str] = Field(default=None, description="User agent information")
    ip_address: Optional[str] = Field(default=None, description="User IP address (hashed for privacy)")
    
    # Conversation statistics (computed from messages)
    total_messages: int = Field(default=0, description="Total number of messages in the conversation")
    is_active: bool = Field(default=True, description="Whether the conversation is still active")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "title": "Tomato Cultivation Questions",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the best pH for tomato cultivation?",
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8.",
                        "timestamp": "2024-01-15T10:30:01Z",
                        "confidence_score": 0.95,
                        "knowledge_source": "knowledge_base",
                        "requires_human_referral": False,
                        "reasoning": "High confidence answer found in knowledge base",
                        "model_used": "gpt-3.5-turbo",
                        "temperature": 0.1,
                        "max_tokens": 500,
                        "sources_used": ["agriculture_handbook.pdf", "soil_science_guide.pdf"]
                    }
                ],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:01Z",
                "total_messages": 2,
                "is_active": True
            }
        }


class ConversationSearchRequest(BaseModel):
    """Schema for conversation search requests."""
    
    user_id: Optional[str] = Field(default=None, description="Filter by user ID")
    start_date: Optional[datetime] = Field(default=None, description="Start date for date range filter")
    end_date: Optional[datetime] = Field(default=None, description="End date for date range filter")
    search_text: Optional[str] = Field(default=None, description="Text to search in questions and responses")
    knowledge_source: Optional[str] = Field(default=None, description="Filter by knowledge source")
    requires_human_referral: Optional[bool] = Field(default=None, description="Filter by human referral status")
    min_confidence: Optional[float] = Field(default=None, description="Minimum confidence score")
    max_confidence: Optional[float] = Field(default=None, description="Maximum confidence score")
    limit: int = Field(default=50, description="Maximum number of results to return")
    skip: int = Field(default=0, description="Number of results to skip for pagination")
      
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "search_text": "tomato",
                "knowledge_source": "knowledge_base",
                "min_confidence": 0.8,
                "limit": 20,
                "skip": 0
            }
        }


class ConversationResponse(BaseModel):
    """Schema for conversation API responses."""
    
    id: str = Field(..., description="Conversation ID")
    user_id: str = Field(..., description="User ID")
    title: Optional[str] = Field(default=None, description="Conversation title")
    messages: List[MessageResponse] = Field(..., description="List of messages in the conversation")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    updated_at: datetime = Field(..., description="Conversation last update timestamp")
    total_messages: int = Field(..., description="Total number of messages")
    is_active: bool = Field(..., description="Whether the conversation is active")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "user_id": "user123",
                "title": "Tomato Cultivation Questions",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the best pH for tomato cultivation?",
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8.",
                        "timestamp": "2024-01-15T10:30:01Z",
                        "confidence_score": 0.95,
                        "knowledge_source": "knowledge_base",
                        "requires_human_referral": False
                    }
                ],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:01Z",
                "total_messages": 2,
                "is_active": True
            }
        }


class ConversationListResponse(BaseModel):
    """Schema for paginated conversation list responses."""
    
    conversations: List[ConversationResponse] = Field(..., description="List of conversations")
    total_count: int = Field(..., description="Total number of conversations matching the criteria")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversations": [
                    {
                        "id": "507f1f77bcf86cd799439011",
                        "user_id": "user123",
                        "title": "Tomato Cultivation Questions",
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the best pH for tomato cultivation?",
                                "timestamp": "2024-01-15T10:30:00Z"
                            },
                            {
                                "role": "assistant",
                                "content": "The optimal pH for tomato cultivation is between 6.0 and 6.8.",
                                "timestamp": "2024-01-15T10:30:01Z",
                                "confidence_score": 0.95,
                                "knowledge_source": "knowledge_base",
                                "requires_human_referral": False
                            }
                        ],
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:30:01Z",
                        "total_messages": 2,
                        "is_active": True
                    }
                ],
                "total_count": 1,
                "page": 1,
                "page_size": 50,
                "has_next": False
            }
        }