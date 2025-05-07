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
    user_id: str = Field(..., description="Unique identifier for the user session")
    message: str = Field(..., description="The message from the user")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "message": "Hello, how can you help me today?"
            }
        }


class ChatResponse(BaseModel):
    """Schema for chat response data.
    
    This defines the expected output format from the chat API.
    """
    response: str = Field(..., description="The AI model's response")
    conversation_history: Optional[List[ChatMessage]] = Field(None, description="The conversation history")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "I'm an AI assistant. I can help you with information, answer questions, or assist with various tasks. What would you like to know?",
                "conversation_history": [
                    {"role": "user", "content": "Hello, how can you help me today?"},
                    {"role": "assistant", "content": "I'm an AI assistant. I can help you with information, answer questions, or assist with various tasks. What would you like to know?"}
                ]
            }
        }