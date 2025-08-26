from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import time

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service
from app.services.conversation_service import get_conversation_service

# Create router for chat endpoints
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def create_chat(
    request: ChatRequest, 
    chat_service=Depends(get_chat_service),
    conversation_service=Depends(get_conversation_service)
):
    """Process a chat message and get a response from the AI.
    
    This endpoint accepts a user message and returns an AI-generated response.
    It maintains conversation history for each user session.
    """
    try:
        # Record start time for response time calculation
        start_time = time.time()
        
        # Process the message
        result = await chat_service.process_message(
            user_id=request.user_id,
            message=request.message,
        )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        

               # Get or generate conversation title
        title: Optional[str] = None
        if request.session_id:
            # Check if a conversation with this session ID already exists
   

            existing_conversation = await conversation_service.get_conversations_by_session_id(request.session_id)
            if existing_conversation and len(existing_conversation) > 0 and existing_conversation[0].title:
                title = existing_conversation[0].title
        
        if not title:
            # If no title exists, generate one
            title = await chat_service.generate_conversation_title(request.message)

        
        # # Store the conversation in the database
        await conversation_service.store_conversation(
            session_id=request.session_id,
            user_email=request.user_email,
            user_id=request.user_id,
            user_question=request.message,
            system_response=result["answer"],
            query_analysis=result["query_analysis"],
            response_parameters=result["response_parameters"],
            response_time_ms=response_time_ms
        )
        
        # Get conversation history
        conversation_history = chat_service.get_conversation_history(request.user_id)
        
      
        # Return the response
        return ChatResponse(
            query_analysis=result["query_analysis"],
            response_parameters=result["response_parameters"],
            answer=result["answer"],
            title=title,
            conversation_history=conversation_history
         )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")