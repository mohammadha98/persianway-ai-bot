from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Union, Optional, List
import asyncio
import time
import json
import logging

from app.core.config import settings
from app.schemas.chat import (
    ChatRequest, ChatResponse, SimplifiedChatResponse,
    FeedbackRequest, FeedbackResponse, FeedbackDetail,
    FeedbackListResponse, FeedbackStats
)
from app.services.chat_service import get_chat_service
from app.services.conversation_service import get_conversation_service
from app.services.feedback_service import get_feedback_service
from app.services.spell_corrector import get_spell_corrector
from app.api.routes.users import get_admin_user, require_permission
from app.schemas.user import PermissionType, UserResponse

logger = logging.getLogger(__name__)

# ==================== SSE keep-alive tuning ====================
# Production symptom these guard against: `GET /api/chat/stream` answered 200
# but the body never arrived -- nginx logged "upstream prematurely closed
# connection while reading upstream" and the browser reported
# `net::ERR_HTTP2_PROTOCOL_ERROR` (surfaced by the frontend as the structured
# `stream_read_error` event). The route used to send nothing at all until intent
# detection, retrieval and the first LLM round trip had finished, so a proxy
# idle timeout (nginx `proxy_read_timeout` defaults to 60s) reset the HTTP/2
# stream of a request that was still perfectly healthy.
#
# The stream now opens with a comment frame and emits one every
# `SSE_HEARTBEAT_INTERVAL_SECONDS` while it waits for the next chunk. Comment
# frames carry no `data:` line, so the frontend parser
# (chat.service.ts -> processFrame) ignores them, while nginx/HTTP2 see real
# bytes on the wire and keep the connection open.
SSE_HEARTBEAT_INTERVAL_SECONDS = 10.0

# `generate_conversation_title` is an LLM round trip executed *before* the
# StreamingResponse is returned (i.e. before the client receives a single byte).
# Bounded so an unresponsive provider can never delay the response headers.
TITLE_GENERATION_TIMEOUT_SECONDS = 15.0

# Emitted while waiting for the next chunk of the answer. A comment frame, i.e.
# it carries no `data:` line, so the frontend parser (chat.service.ts ->
# processFrame) ignores it while nginx/HTTP2 see real bytes on the wire.
SSE_HEARTBEAT_FRAME = ": ping\n\n"

# Emitted as the first frame of every stream: same keep-alive comment as the
# periodic heartbeat, written before any slow work runs, so the status line and
# the headers are flushed immediately (a real byte on the wire), `fetch()`
# resolves right away and every proxy starts forwarding data.
SSE_STREAM_OPEN_FRAME = SSE_HEARTBEAT_FRAME

# Explicit end-of-stream marker, written after the JSON `done` / `error` event so
# a client never has to infer the end of the stream from a closed connection
# (that inference is what turns a proxy hiccup into a truncated answer). It is
# recognised and skipped by the frontend parser, see chat.service.ts.
SSE_DONE_FRAME = "data: [DONE]\n\n"


def _sse_data_frame(payload) -> str:
    """Serialise one SSE event: `data: <json>\\n\\n`, UTF-8, never ASCII-escaped.

    Every payload the route sends goes through here so the wire format stays
    identical for metadata, tokens, errors and the final event.
    """
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False)
    return f"data: {payload}\n\n"


class _SSEHeartbeat:
    """Sentinel yielded by `_with_sse_heartbeats` while `source` is silent.

    A dedicated type (instead of a string) so the caller has to decide explicitly
    how to serialise it, and so a heartbeat can never be mistaken for one of the
    `dict` events coming out of `ChatService.process_message_stream`.
    """


SSE_HEARTBEAT_SENTINEL = _SSEHeartbeat()


async def _with_sse_heartbeats(source):
    """Yield every item of `source`, plus heartbeats while it is silent.

    `source` (the async generator from `ChatService.process_message_stream`) is
    drained by a producer task so this generator stays responsive: whenever no
    item has arrived for `SSE_HEARTBEAT_INTERVAL_SECONDS` it yields
    `SSE_HEARTBEAT_SENTINEL`, which lets the route write a byte to the socket and
    thereby stops nginx (or any other proxy in front of it) from treating a slow
    generation as an idle connection and resetting the HTTP/2 stream mid-answer.

    Exceptions raised by `source` are re-raised on the consuming side, so the
    route's existing error handling keeps working unchanged.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _STREAM_END = object()

    async def _producer():
        try:
            async for item in source:
                queue.put_nowait(item)
        except asyncio.CancelledError:
            raise  # the consumer went away: don't mask the cancellation
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            queue.put_nowait(exc)
        finally:
            queue.put_nowait(_STREAM_END)

    producer = asyncio.create_task(_producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), SSE_HEARTBEAT_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                yield SSE_HEARTBEAT_SENTINEL
                continue

            if item is _STREAM_END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # Client disconnected, an error was raised or the answer finished:
        # never leave the producer running past the response it feeds.
        if not producer.done():
            producer.cancel()


async def _await_with_sse_heartbeats(coro):
    """Await `coro`, yielding heartbeat sentinels while it is still pending.

    Yields `SSE_HEARTBEAT_SENTINEL` every `SSE_HEARTBEAT_INTERVAL_SECONDS`, then
    the awaited result (re-raising its exception on the consuming side), then
    stops.

    Needed because `_with_sse_heartbeats` only covers the answer generator: the
    route's post-processing (persisting the conversation, which itself may run a
    title LLM round trip) happens *after* the last token and used to write nothing
    at all, so the client waited for the `done` event in silence -- exactly the
    silence that makes a proxy reset a finished-but-unterminated stream.
    """
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait(
                {task}, timeout=SSE_HEARTBEAT_INTERVAL_SECONDS
            )
            if done:
                yield task.result()
                return
            yield SSE_HEARTBEAT_SENTINEL
    finally:
        # Client disconnected or an error was raised: never leave the coroutine
        # running past the response it feeds.
        if not task.done():
            task.cancel()


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
async def submit_feedback(
    feedback_request: FeedbackRequest,
    current_user: UserResponse = Depends(require_permission(PermissionType.ANALYSIS))
):
    """
    Submit feedback (approve, report, edit, or add_to_kb) for an AI message.

    Requires the 'Analysis' permission (admins always allowed). Users without it
    can only chat and cannot send AI-trainer feedback.

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

    Wire format of the response (``text/event-stream``, never buffered):

    * ``: ping`` -- comment frame written before any slow work, and then every
      ``SSE_HEARTBEAT_INTERVAL_SECONDS`` while the answer is still being
      produced, so no proxy mistakes a slow generation for an idle connection.
    * ``data: {...}`` -- one JSON event per frame (``status``, ``metadata``,
      ``chunk``, ``done`` or ``error``), serialised through ``_sse_data_frame``.
    * ``data: [DONE]`` -- explicit end-of-stream marker written last.

    Transport mode is chosen by ``settings.CHAT_STREAMING_ENABLED`` and changes
    exactly one thing: whether the ``chunk`` frames are written as the model
    produces them (``True``) or the tokens are accumulated server-side and handed
    over once inside the ``done`` frame (``False``, single-frame SSE). Everything
    else -- the open frame, the heartbeats, ``status`` / ``metadata`` / ``error``
    frames, the persistence tail and the end-of-stream marker -- is shared by both
    modes, so the two can never drift apart and neither one loses the keep-alive
    bytes that stop a proxy from resetting a slow request.

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
            # Bounded: this LLM round trip happens before the first byte of the
            # response, so an unresponsive provider would otherwise keep the
            # client (and every proxy in between) waiting with no headers at
            # all. `generate_conversation_title` already falls back on errors.
            try:
                title = await asyncio.wait_for(
                    chat_service.generate_conversation_title(message),
                    timeout=TITLE_GENERATION_TIMEOUT_SECONDS,
                )
            except Exception as title_error:
                logger.warning(f"Conversation title not generated: {title_error}")
                title = "New Conversation"

        # Generator function to stream the response
        async def stream_generator():
            try:
                # Flush the status line + headers (a real byte on the wire)
                # before the first slow step, so no proxy can consider this
                # connection idle while retrieval + the first LLM round trip
                # run, and the client starts rendering immediately.
                yield SSE_STREAM_OPEN_FRAME

                # Stream the message processing
                full_answer = ""
                query_analysis = None
                final_query_analysis = None
                response_parameters = None
                normalized_sources = []
                prompt_snapshot = None
                stream_error = None

                # Transport mode (see the route docstring). Only the answer tokens
                # are affected: `status`, `metadata` and `error` frames are forwarded
                # in both modes, the heartbeats keep flowing while the tokens are
                # buffered, and the `done` frame below already carries the complete
                # answer, so single-frame mode needs no second code path.
                stream_tokens = settings.CHAT_STREAMING_ENABLED
                
                _stream = _with_sse_heartbeats(chat_service.process_message_stream(
                    user_id=user_id,
                    message=message,
                    conversation_history=existing_conversation
                ))
                async for chunk in _stream:
                    # Keep-alive tick: the service is still working, so write a
                    # comment frame (no `data:` line, ignored by the frontend
                    # parser) and keep waiting.
                    if chunk is SSE_HEARTBEAT_SENTINEL:
                        yield SSE_HEARTBEAT_FRAME
                        continue

                    event_type = chunk.get("type")

                    # Handle progress notifications: forwarded verbatim so the UI
                    # can say what the pipeline is doing ("searching…", "writing
                    # the answer…") instead of showing a silent spinner while
                    # retrieval + the first LLM round trip run.
                    if event_type == "status":
                        yield _sse_data_frame({
                            "type": "status",
                            "stage": chunk.get("stage"),
                            "message": chunk.get("message"),
                        })

                    # Handle metadata chunks (sent once, early)
                    elif event_type == "metadata":
                        query_analysis = chunk.get("query_analysis")
                        normalized_sources = chunk.get("normalized_sources", [])
                        response_parameters = chunk.get("response_parameters") or response_parameters
                        
                        # Send metadata to client
                        yield _sse_data_frame({'type': 'metadata', 'data': chunk})
                    
                    # Handle content chunks (incremental tokens)
                    elif event_type == "chunk":
                        content = chunk.get("content", "")
                        full_answer += content
                        if not stream_tokens:
                            # Single-frame mode: the token is accumulated above and
                            # written by the `done` frame instead of here.
                            continue
                        
                        # Send content chunk to client
                        yield _sse_data_frame({'type': 'chunk', 'content': content})
                    
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
                
                # Deterministic cleanup for the `break` above: closing the wrapper
                # cancels the producer, so the service generator stops immediately
                # instead of running until garbage collection.
                await _stream.aclose()

                if stream_error is not None:
                    error_data = {
                        "type": "error",
                        "message": stream_error.get("message") or "خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید.",
                        "code": stream_error.get("code", "stream_internal_error"),
                    }
                    yield _sse_data_frame(error_data)
                    yield SSE_DONE_FRAME
                    return
                
                if final_query_analysis is not None:
                    query_analysis = final_query_analysis
                
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Store the conversation in the database. Heartbeat-wrapped: the
                # write (plus the title LLM call inside `store_conversation`) happens
                # after the last token, and without a byte on the wire here the
                # client waits for `done` in silence.
                store_result = None
                async for item in _await_with_sse_heartbeats(
                    conversation_service.store_conversation(
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
                ):
                    if item is SSE_HEARTBEAT_SENTINEL:
                        yield SSE_HEARTBEAT_FRAME
                    else:
                        store_result = item
                
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
                yield _sse_data_frame(completion_data)
                # Explicit end-of-stream marker: the client never has to infer the
                # end of the answer from the connection being closed.
                yield SSE_DONE_FRAME

            except asyncio.CancelledError:
                # The client went away (or the server is shutting down) while the
                # answer was still being produced. There is nobody left to receive
                # an SSE error frame, so this is a normal end of the request, not a
                # failure. It must never be swallowed: re-raising lets the
                # generator unwind and cancels the service stream instead of
                # leaving a half-open producer behind.
                logger.info(
                    "Chat stream cancelled (client disconnected) for user_id=%s", user_id
                )
                raise

            except Exception as e:
                # Structured error event: user-safe message + internal code, never raw details.
                logger.error(f"Error in stream_generator: {str(e)}")
                error_data = {
                    "type": "error",
                    "message": "خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید.",
                    "code": "stream_route_error",
                }
                yield _sse_data_frame(error_data)
                yield SSE_DONE_FRAME

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Hop-by-hop, and therefore ignored under HTTP/2 (where nginx
                # manages the connection itself), but explicit for HTTP/1.1
                # clients and proxies that only look at this header.
                "Connection": "keep-alive",
                # Tells nginx not to buffer the stream: without it the heartbeats
                # sit in proxy buffers and the client still sees silence.
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )
    except Exception as e:
        logger.error(f"Stream chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stream chat error: {str(e)}")
