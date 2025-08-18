from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.schemas.conversation import (
    ConversationSearchRequest,
    ConversationListResponse,
    ConversationResponse
)
from app.services.conversation_service import get_conversation_service

# Create router for conversation endpoints
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/{user_id}", response_model=ConversationListResponse)
async def get_user_conversations(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of conversations to return"),
    skip: int = Query(default=0, ge=0, description="Number of conversations to skip for pagination"),
    conversation_service=Depends(get_conversation_service)
):
    """Get all conversations for a specific user.
    
    This endpoint retrieves all conversations for a given user with pagination support.
    Conversations are returned in reverse chronological order (newest first).
    """
    try:
        result = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=limit,
            skip=skip
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversations: {str(e)}")


@router.get("/{user_id}/latest", response_model=List[ConversationResponse])
async def get_latest_user_conversations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of latest conversations to return"),
    conversation_service=Depends(get_conversation_service)
):
    """Get the latest conversations for a specific user.
    
    This endpoint retrieves the most recent conversations for a given user.
    Useful for displaying recent chat history or continuing conversations.
    """
    try:
        conversations = await conversation_service.get_latest_user_conversations(
            user_id=user_id,
            limit=limit
        )
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve latest conversations: {str(e)}")


@router.post("/search", response_model=ConversationListResponse)
async def search_conversations(
    search_request: ConversationSearchRequest,
    conversation_service=Depends(get_conversation_service)
):
    """Search conversations based on various criteria.
    
    This endpoint allows searching conversations using multiple filters:
    - User ID
    - Date range
    - Text search in questions and responses
    - Knowledge source (knowledge_base, general_knowledge, none)
    - Human referral status
    - Confidence score range
    
    Results are paginated and sorted by timestamp (newest first).
    """
    try:
        result = await conversation_service.search_conversations(search_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search conversations: {str(e)}")


@router.get("/stats/overview")
async def get_conversation_stats(
    user_id: Optional[str] = Query(default=None, description="Optional user ID to filter stats by specific user"),
    conversation_service=Depends(get_conversation_service)
) -> Dict[str, Any]:
    """Get conversation statistics.
    
    This endpoint provides various statistics about conversations:
    - Total number of conversations
    - Number of agriculture-related queries
    - Number of human referrals
    - Average confidence score
    - Distribution by knowledge source
    
    If user_id is provided, statistics are filtered for that specific user.
    """
    try:
        stats = await conversation_service.get_conversation_stats(user_id=user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation stats: {str(e)}")


@router.get("/email/{user_email}", response_model=ConversationListResponse)
async def get_conversations_by_email(
    user_email: str,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of conversations to return"),
    skip: int = Query(default=0, ge=0, description="Number of conversations to skip for pagination"),
    conversation_service=Depends(get_conversation_service)
):
    """Get all conversations for a specific user email.
    
    This endpoint retrieves all conversations for a given user email with pagination support.
    Conversations are returned in reverse chronological order (newest first).
    """
    try:
        result = await conversation_service.get_conversations_by_email(
            user_email=user_email,
            limit=limit,
            skip=skip
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversations by email: {str(e)}")


@router.get("/session/{session_id}", response_model=List[ConversationResponse])
async def get_conversations_by_session_id(
    session_id: str,
    conversation_service=Depends(get_conversation_service)
):
    """Get conversations for a specific session ID.
    
    This endpoint retrieves all conversations associated with a given session ID.
    Typically, there should be only one conversation per session, but this returns
    a list to handle edge cases where multiple conversations might share a session ID.
    """
    try:
        conversations = await conversation_service.get_conversations_by_session_id(
            session_id=session_id
        )
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversations by session ID: {str(e)}")


@router.get("/search/advanced", response_model=ConversationListResponse)
async def advanced_conversation_search(
    user_id: Optional[str] = Query(default=None, description="Filter by user ID"),
    start_date: Optional[datetime] = Query(default=None, description="Start date for date range filter (ISO format)"),
    end_date: Optional[datetime] = Query(default=None, description="End date for date range filter (ISO format)"),
    search_text: Optional[str] = Query(default=None, description="Text to search in questions and responses"),
    knowledge_source: Optional[str] = Query(default=None, description="Filter by knowledge source"),
    requires_human_referral: Optional[bool] = Query(default=None, description="Filter by human referral status"),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum confidence score"),
    max_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Maximum confidence score"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of results to return"),
    skip: int = Query(default=0, ge=0, description="Number of results to skip for pagination"),
    conversation_service=Depends(get_conversation_service)
):
    """Advanced conversation search with query parameters.
    
    This endpoint provides the same functionality as the POST /search endpoint
    but uses query parameters instead of a request body for easier integration
    with some client applications.
    """
    try:
        search_request = ConversationSearchRequest(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            search_text=search_text,
            knowledge_source=knowledge_source,
            requires_human_referral=requires_human_referral,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            limit=limit,
            skip=skip
        )
        
        result = await conversation_service.search_conversations(search_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search conversations: {str(e)}")