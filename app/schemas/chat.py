from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    """Schema for a single chat message."""
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")


class ChatRequest(BaseModel):
    """Schema for chat request data.
    
    This defines the expected input format for the chat API.
    """
    class RequestParameters(BaseModel):
        """Optional parameters to override the model's default settings."""
        temperature: Optional[float] = Field(None, description="The temperature setting for the model.")
        top_p: Optional[float] = Field(None, description="The top_p setting for the model.")
        max_tokens: Optional[int] = Field(None, description="The maximum number of tokens for the response.")

    user_id: str = Field(..., description="Unique identifier for the user session")
    message: str = Field(..., description="The message from the user")
    model: Optional[str] = Field(None, description="The model to use for the response, overriding the default.")
    parameters: Optional[RequestParameters] = Field(None, description="Parameters to override model settings.")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "message": "Hello, how can you help me today?",
                "model": "gpt-4-turbo",
                "parameters": {
                    "temperature": 0.5
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


class ChatResponse(BaseModel):
    """Schema for chat response data.
    
    This defines the expected output format from the chat API according to the new advanced response system.
    """
    query_analysis: QueryAnalysis = Field(..., description="Analysis of the user's query")
    response_parameters: ResponseParameters = Field(..., description="Parameters used for generating the response")
    answer: str = Field(..., description="The actual response to the user's query")
    conversation_history: Optional[List[ChatMessage]] = Field(None, description="The conversation history, if applicable")

    class Config:
        schema_extra = {
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
                "conversation_history": [
                    {"role": "user", "content": "What is the best pH for tomatoes?"},
                    {"role": "assistant", "content": "Based on our knowledge base, the optimal pH for tomato cultivation is between 6.0 and 6.8."}
                ]
            }
        }