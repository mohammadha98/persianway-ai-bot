from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Union, Optional
import time

from app.schemas.chat import ChatRequest, ChatResponse, SimplifiedChatResponse
from app.services.chat_service import get_chat_service
from app.services.conversation_service import get_conversation_service
from app.services.spell_corrector import get_spell_corrector

# Create router for chat endpoints
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=Union[ChatResponse, SimplifiedChatResponse])
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
        
        # Get or generate conversation title
        title: Optional[str] = ""
        existing_conversation = None
        
        if request.session_id:
            # Check if a conversation with this session ID already exists
            existing_conversation = await conversation_service.get_conversations_by_session_id(request.session_id)
            if existing_conversation and len(existing_conversation) > 0 and existing_conversation[0].title:
                title = existing_conversation[0].title
        
        if not title:
            # If no title exists, generate one
            title = await chat_service.generate_conversation_title(request.message)

        # Process the message
        result = await chat_service.process_message(
            user_id=request.user_id,
            message=request.message,
            conversation_history=existing_conversation
        )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Store the conversation in the database
        await conversation_service.store_conversation(
            session_id=request.session_id,
            user_email=request.user_email,
            user_id=request.user_id,
            user_question=request.message,
            system_response=result["answer"],
            query_analysis=result["query_analysis"],
            response_parameters=result["response_parameters"],
            response_time_ms=response_time_ms,
        )
     
        # Get conversation history
        conversation_history = chat_service.get_conversation_history(request.user_id)
        
      
        # Return the response based on simplified_response parameter
        if request.simplified_response:
            return SimplifiedChatResponse(
                answer=result["answer"],
                title=title,
                requires_human_referral=result["query_analysis"]["requires_human_referral"]
            )
        else:
            return ChatResponse(
                query_analysis=result["query_analysis"],
                response_parameters=result["response_parameters"],
                answer=result["answer"],
                title=title,
                conversation_history=conversation_history
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post("/corrected", response_model=Union[ChatResponse, SimplifiedChatResponse])
async def create_chat_with_correction(
    request: ChatRequest, 
    chat_service=Depends(get_chat_service),
    conversation_service=Depends(get_conversation_service)
):
    """Process a chat message with spell correction and get a response from the AI.
    
    This endpoint accepts a user message, applies spell correction, and returns an AI-generated response.
    It maintains conversation history for each user session and provides information about spell corrections.
    """
    try:
        # Record start time for response time calculation
        start_time = time.time()
        
        # Apply spell correction to the input message
        original_message = request.message
        corrected_message = original_message
        spell_corrections = []
        
        try:
            spell_corrector = get_spell_corrector()
            corrected_message = spell_corrector.correct_text(original_message)
            
            # Track corrections made
            if corrected_message != original_message:
                spell_corrections.append({
                    "original": original_message,
                    "corrected": corrected_message,
                    "type": "spell_correction"
                })
        except Exception as spell_error:
            # Continue with original message if spell correction fails
            pass
        
        # Update request with corrected message
        request.message = corrected_message
        
        # Process the corrected message
        result = await chat_service.process_message(
            user_id=request.user_id,
            message=corrected_message,
        )
        
        # Add spell correction information to query analysis
        if spell_corrections:
            result["query_analysis"]["spell_corrections"] = spell_corrections
            result["query_analysis"]["original_message"] = original_message
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Store the conversation in the database (with corrected message)
        await conversation_service.store_conversation(
            session_id=request.session_id,
            user_email=request.user_email,
            user_id=request.user_id,
            user_question=corrected_message,  # Store corrected version
            system_response=result["answer"],
            query_analysis=result["query_analysis"],
            response_parameters=result["response_parameters"],
            response_time_ms=response_time_ms,
        )
        
        # Get or generate conversation title
        title: Optional[str] = ""
        if request.session_id:
            # Check if a conversation with this session ID already exists
            existing_conversation = await conversation_service.get_conversations_by_session_id(request.session_id)
            if existing_conversation and len(existing_conversation) > 0 and existing_conversation[0].title:
                title = existing_conversation[0].title
        
        if not title:
            # If no title exists, generate one using corrected message
            title = await chat_service.generate_conversation_title(corrected_message)
        
        # Get conversation history
        conversation_history = chat_service.get_conversation_history(request.user_id)
        
        # Return the response based on simplified_response parameter
        if request.simplified_response:
            return SimplifiedChatResponse(
                answer=result["answer"],
                title=title,
                requires_human_referral=result["query_analysis"]["requires_human_referral"]
            )
        else:
            return ChatResponse(
                query_analysis=result["query_analysis"],
                response_parameters=result["response_parameters"],
                answer=result["answer"],
                title=title,
                conversation_history=conversation_history
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing with correction error: {str(e)}")