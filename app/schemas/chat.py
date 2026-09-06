from pydantic import BaseModel, Field
from typing import List, Optional, Any

from app.schemas.message import SourceDocument


class ChatMessage(BaseModel):
    """Schema for a single chat message."""
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")


class RequestParameters(BaseModel):
    """Optional parameters to override the model's default settings."""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="The temperature setting for the model.")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="The top_p setting for the model.")
    max_tokens: Optional[int] = Field(None, gt=0, description="The maximum number of tokens for the response.")

class ChatRequest(BaseModel):
    """Schema for chat request data.
    
    This defines the expected input format for the chat API.
    """
    user_id: str = Field(..., description="Unique identifier for the user session")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")
    user_email: Optional[str] = Field(None, description="Email address of the user")
    message: str = Field(..., description="The message from the user")
    simplified_response: Optional[bool] = Field(False, description="If true, returns a simplified response with only title, answer, and requires_human_referral")
   
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user123",
                "session_id": "session_abc123",
                "user_email": "user@example.com",
                "message": "Hello, how can you help me today?",
                "simplified_response": False
            }
        }
    }


class QueryAnalysis(BaseModel):
    """Schema for the analysis of the user's query."""
    confidence_score: float = Field(..., description="Confidence score of the answer between 0.0 and 1.0")
    knowledge_source: str = Field(..., description="Source of the knowledge used for the response (knowledge_base, general_knowledge, none)")
    requires_human_referral: bool = Field(..., description="Whether the query requires human referral")
    reasoning: str = Field(..., description="Brief explanation of the decision-making process")


class ResponseParameters(BaseModel):
    """Schema for the parameters used to generate the response."""
    model: str = Field(..., description="The model used to generate the response")
    temperature: float = Field(..., description="The temperature setting used for the model")
    max_tokens: int = Field(..., description="The maximum number of tokens for the response")
    top_p: Optional[float] = Field(None, description="The top_p setting used for the model")


class SimplifiedChatResponse(BaseModel):
    """Schema for simplified chat response data.
    
    This defines a simplified output format with only essential fields.
    """
    answer: str = Field(..., description="The actual response to the user's query")
    title: Optional[str] = Field(None, description="Generated title for the conversation")
    requires_human_referral: bool = Field(..., description="Whether the query requires human referral")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Based on our knowledge base, the optimal pH for tomato cultivation is between 6.0 and 6.8.",
                "title": "Tomato pH Requirements",
                "requires_human_referral": False
            }
        }
    }


class ChatResponse(BaseModel):
    """Schema for chat response data.
    
    This defines the expected output format from the chat API according to the new advanced response system.
    """
    query_analysis: QueryAnalysis = Field(..., description="Analysis of the user's query")
    response_parameters: ResponseParameters = Field(..., description="Parameters used for generating the response")
    answer: str = Field(..., description="The actual response to the user's query")
    title: Optional[str] = Field(None, description="Generated title for the conversation")
    conversation_history: Optional[List[ChatMessage]] = Field(None, description="The conversation history, if applicable")
    message_id: Optional[str] = Field(None, description="Unique identifier for this AI message (for feedback tracking)")
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation (for navigation)")
    feedback: Optional[dict[str, Any]] = Field(None, description="Existing feedback for this message if any")
    normalized_sources: Optional[List[SourceDocument]] = Field(None, description="Normalized source documents used in the RAG response")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query_analysis": {
                    "confidence_score": 0.95,
                    "knowledge_source": "knowledge_base",
                    "requires_human_referral": False,
                    "reasoning": "Query is agriculture-related and high confidence answer found in knowledge base."
                },
                "response_parameters": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 150
                },
                "answer": "Based on our knowledge base, the optimal pH for tomato cultivation is between 6.0 and 6.8.",
                "title": "Tomato pH Requirements",
                "message_id": "msg_abc123",
                "conversation_id": "conv_xyz789",
                "feedback": None,
                "conversation_history": [
                    {"role": "user", "content": "What is the best pH for tomatoes?"},
                    {"role": "assistant", "content": "Based on our knowledge base, the optimal pH for tomato cultivation is between 6.0 and 6.8."}
                ]
            }
        }
    }


# ==================== AI Feedback System ====================

class FeedbackType:
    """Enum-like class for feedback types"""
    APPROVE = "approve"
    REPORT = "report"
    EDIT = "edit"
    ADD_TO_KB = "add_to_kb"


class FeedbackCategory:
    """Enum-like class for feedback issue categories"""
    INCORRECT_INFO = "incorrect_info"
    INCOMPLETE = "incomplete"
    IRRELEVANT = "irrelevant"
    HARMFUL = "harmful"
    OUTDATED = "outdated"
    GRAMMAR = "grammar"
    OTHER = "other"


class FeedbackRequest(BaseModel):
    """Schema for submitting feedback on an AI message."""
    message_id: str = Field(..., description="Unique identifier of the AI message")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation this message belongs to")
    feedback: str = Field(..., description="Type of feedback: 'approve', 'report', 'edit', or 'add_to_kb'")
    comment: Optional[str] = Field("", description="Optional comment explaining the feedback (required for 'report' and 'edit')")
    category: Optional[str] = Field(None, description="Category of issue: incorrect_info, incomplete, irrelevant, etc.")
    
    # Context (to support navigation)
    user_email: Optional[str] = Field(None, description="Email of the user giving feedback")
    user_id: Optional[str] = Field(None, description="ID of the trainer giving feedback")
    
    # Original message context (for traceability)
    message_content: Optional[str] = Field(None, description="The AI's message content (cached for reference)")
    question: Optional[str] = Field(None, description="The user's question that led to this answer")
    
    # For edit feedback
    edited_answer: Optional[str] = Field(None, description="The corrected/edited answer (for 'edit' feedback)")
    
    timestamp: Optional[str] = Field(None, description="When the feedback was submitted")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message_id": "msg_abc123",
                "conversation_id": "conv_xyz789",
                "feedback": "report",
                "comment": "The pH range is incorrect, it should be 6.0-6.5",
                "category": "incorrect_info",
                "user_email": "trainer@example.com",
                "user_id": "trainer_001",
                "message_content": "Based on our knowledge base, the optimal pH for tomato cultivation is between 6.0 and 6.8.",
                "question": "What is the best pH for tomatoes?"
            }
        }
    }


class FeedbackResponse(BaseModel):
    """Schema for feedback submission response."""
    success: bool = Field(..., description="Whether the feedback was saved successfully")
    feedback_id: str = Field(..., description="Unique ID of the saved feedback")
    message_id: str = Field(..., description="The message ID the feedback is for")
    conversation_id: Optional[str] = Field(None, description="The conversation ID")
    feedback_type: str = Field(..., description="The type of feedback")
    created_at: str = Field(..., description="When the feedback was created")
    message: str = Field("بازخورد شما با موفقیت ثبت شد", description="Human-readable message")


class FeedbackDetail(BaseModel):
    """Schema for full feedback details (when retrieving)."""
    feedback_id: str = Field(..., description="Unique ID of the feedback")
    message_id: str = Field(..., description="The message ID")
    conversation_id: Optional[str] = Field(None, description="The conversation ID")
    feedback: str = Field(..., description="Type of feedback")
    comment: Optional[str] = Field(None, description="Comment text")
    category: Optional[str] = Field(None, description="Issue category")
    user_email: Optional[str] = Field(None, description="Trainer email")
    user_id: Optional[str] = Field(None, description="Trainer ID")
    message_content: Optional[str] = Field(None, description="Original AI message content")
    question: Optional[str] = Field(None, description="Original user question")
    edited_answer: Optional[str] = Field(None, description="Edited answer (if applicable)")
    created_at: str = Field(..., description="When the feedback was created")
    updated_at: Optional[str] = Field(None, description="When it was last updated")
    status: str = Field("active", description="Status: active, resolved, dismissed")


class FeedbackListResponse(BaseModel):
    """Schema for paginated feedback list response."""
    items: List[FeedbackDetail] = Field(..., description="List of feedbacks for the current page")
    total: int = Field(..., description="Total number of feedbacks")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")


class FeedbackStats(BaseModel):
    """Schema for feedback statistics."""
    total_feedbacks: int = Field(..., description="Total number of feedbacks")
    approved_count: int = Field(..., description="Number of approved messages")
    reported_count: int = Field(..., description="Number of reported issues")
    edit_count: int = Field(..., description="Number of edits")
    add_to_kb_count: int = Field(..., description="Number of added to KB")
    by_category: dict[str, int] = Field(default_factory=dict, description="Counts by category")
    approval_rate: float = Field(..., description="Approval rate as a percentage")