from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service

# Create router for chat endpoints
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def create_chat(request: ChatRequest, chat_service=Depends(get_chat_service)):
    """Process a chat message and get a response from the AI.
    
    This endpoint accepts a user message and returns an AI-generated response.
    It maintains conversation history for each user session.
    """
    try:
        # Process the message
        response = await chat_service.process_message(
            user_id=request.user_id,
            message=request.message
        )
        
        # Get conversation history
        conversation_history = chat_service.get_conversation_history(request.user_id)
        
        # Return the response
        return ChatResponse(
            response=response,
            conversation_history=conversation_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")