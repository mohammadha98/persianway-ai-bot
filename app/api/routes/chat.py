from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Union, Optional, List
import time
import json
import logging

from app.schemas.chat import (
    ChatRequest, ChatResponse, SimplifiedChatResponse,
    FeedbackRequest, FeedbackResponse, FeedbackDetail,
    FeedbackListResponse, FeedbackStats
)
from app.services.chat_service import get_chat_service
from app.services.conversation_service import get_conversation_service
from app.services.feedback_service import get_feedback_service
from app.services.spell_corrector import get_spell_corrector
from app.api.routes.users import get_admin_user

logger = logging.getLogger(__name__)

# Create router for chat endpoints
router = APIRouter(prefix="/chat", tags=["chat"])


# ==================== Helper Functions ====================

def _format_feedback(fb: Dict[str, Any]) -> FeedbackDetail:
    """Format a feedback document from DB into FeedbackDetail schema."""
    if not fb:
        return None
    return FeedbackDetail(
        feedback_id=fb.get("feedback_id"),
        message_id=fb.get("message_id"),
        conversation_id=fb.get("conversation_id"),
        feedback=fb.get("feedback"),
        comment=fb.get("comment"),
        category=fb.get("category"),
        user_email=fb.get("user_email"),
        user_id=fb.get("user_id"),
        message_content=fb.get("message_content"),
        question=fb.get("question"),
        edited_answer=fb.get("edited_answer"),
        created_at=fb.get("created_at"),
        updated_at=fb.get("updated_at"),
        status=fb.get("status", "active")
    )


# ==================== Feedback Endpoints ====================

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback_request: FeedbackRequest):
    """
    Submit feedback (approve, report, edit, or add_to_kb) for an AI message.
    
    The feedback is stored in MongoDB with a relationship to both the message_id
    and conversation_id, allowing easy navigation between feedbacks and their
    source conversations.
    """
    try:
        feedback_service = await get_feedback_service()
        
        # Validate feedback type
        valid_types = ["approve", "report", "edit", "add_to_kb"]
        if feedback_request.feedback not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback type. Must be one of: {', '.join(valid_types)}"
            )
        
        # For 'report' and 'edit', require a comment
        if feedback_request.feedback in ["report", "edit"] and not feedback_request.comment:
            raise HTTPException(
                status_code=400,
                detail=f"Comment is required for '{feedback_request.feedback}' feedback"
            )
        
        # Save the feedback
        saved = await feedback_service.submit_feedback(
            message_id=feedback_request.message_id,
            feedback=feedback_request.feedback,
            comment=feedback_request.comment or "",
            category=feedback_request.category,
            user_email=feedback_request.user_email,
            user_id=feedback_request.user_id,
            message_content=feedback_request.message_content,
            question=feedback_request.question,
            conversation_id=feedback_request.conversation_id,
            edited_answer=feedback_request.edited_answer
        )
        
        return FeedbackResponse(
            success=True,
            feedback_id=saved["feedback_id"],
            message_id=saved["message_id"],
            conversation_id=saved.get("conversation_id"),
            feedback_type=saved["feedback"],
            created_at=saved["created_at"],
            message=f"بازخورد '{saved['feedback']}' با موفقیت ثبت شد"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/feedback/{feedback_id}", response_model=FeedbackDetail)
async def get_feedback(feedback_id: str):
    """Get a specific feedback by its ID."""
    try:
        feedback_service = await get_feedback_service()
        feedback = await feedback_service.get_feedback_by_id(feedback_id)
        
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        return _format_feedback(feedback)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")


@router.get("/feedback/message/{message_id}", response_model=List[FeedbackDetail])
async def get_feedback_by_message(
    message_id: str,
    user_email: Optional[str] = Query(None, description="Filter by user email")
):
    """Get all feedback for a specific message."""
    try:
        feedback_service = await get_feedback_service()
        feedbacks = await feedback_service.get_feedback_by_message(
            message_id=message_id,
            user_email=user_email
        )
        return [_format_feedback(fb) for fb in feedbacks if fb]
    except Exception as e:
        logger.error(f"Error getting feedback by message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")


@router.get("/feedback/conversation/{conversation_id}", response_model=List[FeedbackDetail])
async def get_feedback_by_conversation(conversation_id: str):
    """Get all feedback for a specific conversation."""
    try:
        feedback_service = await get_feedback_service()
        feedbacks = await feedback_service.get_feedback_by_conversation(conversation_id)
        return [_format_feedback(fb) for fb in feedbacks if fb]
    except Exception as e:
        logger.error(f"Error getting feedback by conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")


@router.get("/feedback", response_model=FeedbackListResponse)
async def list_feedbacks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    feedback_type: Optional[str] = Query(None, description="Filter by feedback type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    user_email: Optional[str] = Query(None, description="Filter by trainer email"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get paginated list of feedbacks with optional filters."""
    try:
        feedback_service = await get_feedback_service()
        items, total = await feedback_service.list_feedbacks(
            page=page,
            page_size=page_size,
            feedback_type=feedback_type,
            category=category,
            user_email=user_email,
            status=status
        )
        
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        
        return FeedbackListResponse(
            items=[_format_feedback(fb) for fb in items if fb],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        logger.error(f"Error listing feedbacks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list feedbacks: {str(e)}")


@router.patch("/feedback/{feedback_id}/status")
async def update_feedback_status(feedback_id: str, status: str):
    """Update the status of a feedback (active, resolved, dismissed)."""
    try:
        valid_statuses = ["active", "resolved", "dismissed"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        
        feedback_service = await get_feedback_service()
        success = await feedback_service.update_feedback_status(feedback_id, status)
        
        if not success:
            raise HTTPException(status_code=404, detail="Feedback not found or update failed")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "status": status,
            "message": f"وضعیت بازخورد به '{status}' تغییر یافت"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating feedback status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """Soft delete a feedback (sets status to 'dismissed')."""
    try:
        feedback_service = await get_feedback_service()
        success = await feedback_service.delete_feedback(feedback_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "بازخورد با موفقیت حذف شد"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete feedback: {str(e)}")


@router.get("/feedback/stats/summary", response_model=FeedbackStats)
async def get_feedback_stats(
    user_email: Optional[str] = Query(None, description="Filter stats by trainer email")
):
    """Get feedback statistics and analytics."""
    try:
        feedback_service = await get_feedback_service()
        stats = await feedback_service.get_feedback_stats(user_email=user_email)
        return FeedbackStats(**stats)
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


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
        
        # Store the conversation in the database (with normalized sources for traceability)
        store_result = await conversation_service.store_conversation(
            session_id=request.session_id,
            user_email=request.user_email,
            user_id=request.user_id,
            user_question=request.message,
            system_response=result["answer"],
            query_analysis=result["query_analysis"],
            response_parameters=result["response_parameters"],
            sources_used=result.get("normalized_sources", []),
            prompt_snapshot=result.get("prompt_snapshot"),
            response_time_ms=response_time_ms,
        )
        
        # Extract IDs (support both old str return and new dict return).
        # store_result is None when persistence failed: the answer is still
        # returned, but message_id/conversation_id are set to None so the
        # frontend does not enable the context button for unrecoverable messages.
        if isinstance(store_result, dict):
            conversation_id = store_result.get("conversation_id")
            assistant_message_id = store_result.get("assistant_message_id")
        elif store_result:
            # Backward compatibility
            conversation_id = store_result
            assistant_message_id = None
        else:
            logger.warning("Conversation persistence failed; returning response without message_id")
            conversation_id = None
            assistant_message_id = None
     
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
                conversation_history=conversation_history,
                message_id=assistant_message_id,
                conversation_id=conversation_id
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.get("/context/{message_id}")
async def get_message_context(
    message_id: str,
    current_user=Depends(get_admin_user),
    conversation_service=Depends(get_conversation_service),
):
    """Return the persisted prompt snapshot to administrators only."""
    message = await conversation_service.get_message_by_id(message_id)
    if not message or message.get("role") != "assistant":
        raise HTTPException(status_code=404, detail="Message not found")
    snapshot = message.get("prompt_snapshot")
    if not snapshot:
        raise HTTPException(status_code=404, detail="no context snapshot for this message")
    return {"success": True, "data": snapshot}


@router.post("/stream")
@router.get("/stream")
async def stream_chat(
    user_id: str = Query(...),
    message: str = Query(...),
    session_id: str = Query(""),
    user_email: str = Query(""),
    chat_service=Depends(get_chat_service),
    conversation_service=Depends(get_conversation_service)
):
    """Stream a chat message response using Server-Sent Events (SSE).
    
    This endpoint processes a message and streams the response token-by-token
    as it's generated, enabling real-time UI updates on the client side.
    
    The response is formatted as Server-Sent Events (SSE) with JSON data.
    
    Supports both GET (for EventSource) and POST methods.
    """
    try:
        # Record start time for response time calculation
        start_time = time.time()
        
        # Get or generate conversation title
        title: Optional[str] = ""
        existing_conversation = None
        
        if session_id:
            existing_conversation = await conversation_service.get_conversations_by_session_id(session_id)
            if existing_conversation and len(existing_conversation) > 0 and existing_conversation[0].title:
                title = existing_conversation[0].title
        
        if not title:
            title = await chat_service.generate_conversation_title(message)

        # Generator function to stream the response
        async def stream_generator():
            try:
                # Stream the message processing
                full_answer = ""
                query_analysis = None
                final_query_analysis = None
                response_parameters = None
                normalized_sources = []
                prompt_snapshot = None
                stream_error = None
                
                async for chunk in chat_service.process_message_stream(
                    user_id=user_id,
                    message=message,
                    conversation_history=existing_conversation
                ):
                    event_type = chunk.get("type")
                    
                    # Handle metadata chunks (sent once, early)
                    if event_type == "metadata":
                        query_analysis = chunk.get("query_analysis")
                        normalized_sources = chunk.get("normalized_sources", [])
                        response_parameters = chunk.get("response_parameters") or response_parameters
                        
                        # Send metadata to client
                        yield f"data: {json.dumps({'type': 'metadata', 'data': chunk}, ensure_ascii=False)}\n\n"
                    
                    # Handle content chunks (incremental tokens)
                    elif event_type == "chunk":
                        content = chunk.get("content", "")
                        full_answer += content
                        
                        # Send content chunk to client
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}\n\n"
                    
                    # Handle service done event: authoritative final answer + final analysis.
                    # Not forwarded as-is; the route emits its own done event below with
                    # conversation metadata (title, IDs, history).
                    elif event_type == "done":
                        final_query_analysis = chunk.get("query_analysis") or query_analysis
                        prompt_snapshot = chunk.get("prompt_snapshot")
                        if chunk.get("answer"):
                            full_answer = chunk.get("answer")
                    
                    # Handle structured service error: forward once and stop streaming.
                    elif event_type == "error":
                        stream_error = chunk
                        break
                
                if stream_error is not None:
                    error_data = {
                        "type": "error",
                        "message": stream_error.get("message") or "خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید.",
                        "code": stream_error.get("code", "stream_internal_error"),
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
                if final_query_analysis is not None:
                    query_analysis = final_query_analysis
                
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Store the conversation in the database
                store_result = await conversation_service.store_conversation(
                    session_id=session_id,
                    user_email=user_email,
                    user_id=user_id,
                    user_question=message,
                    system_response=full_answer,
                    query_analysis=query_analysis or {},
                    response_parameters=response_parameters or {},
                    sources_used=normalized_sources,
                    response_time_ms=response_time_ms,
                    prompt_snapshot=prompt_snapshot,
                )
                
                # Extract IDs. store_result is None when persistence failed:
                # the answer still streams back, but message_id is None so the
                # frontend won't show the context button for unpersisted messages.
                if isinstance(store_result, dict):
                    conversation_id = store_result.get("conversation_id")
                    assistant_message_id = store_result.get("assistant_message_id")
                elif store_result:
                    conversation_id = store_result
                    assistant_message_id = None
                else:
                    logger.warning("Conversation persistence failed; done event without message_id")
                    conversation_id = None
                    assistant_message_id = None
                
                # Get conversation history
                conversation_history = chat_service.get_conversation_history(user_id)
                
                # Send completion event with metadata
                completion_data = {
                    "type": "done",
                    "answer": full_answer,
                    "query_analysis": query_analysis or {},
                    "normalized_sources": normalized_sources,
                    "title": title,
                    "conversation_id": conversation_id,
                    "message_id": assistant_message_id,
                    "conversation_history": [
                        {"role": msg.role, "content": msg.content} 
                        for msg in (conversation_history or [])
                    ]
                }
                yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                # Structured error event: user-safe message + internal code, never raw details.
                logger.error(f"Error in stream_generator: {str(e)}")
                error_data = {
                    "type": "error",
                    "message": "خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید.",
                    "code": "stream_route_error",
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )
    except Exception as e:
        logger.error(f"Stream chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stream chat error: {str(e)}")
