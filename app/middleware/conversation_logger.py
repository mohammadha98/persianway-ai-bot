import json
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from app.services.conversation_service import get_conversation_service


class ConversationLoggerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_chat_endpoints: bool = True):
        super().__init__(app)
        self.log_chat_endpoints = log_chat_endpoints

    async def dispatch(self, request: Request, call_next):
        # Simple debug logging
        print(f"[MIDDLEWARE] {request.method} {request.url.path}")
        
        # Check if this is a request we should log
        if self._should_log_request(request):
            print(f"[MIDDLEWARE] Logging chat request")
            # Get request data
            user_id, message = await self._extract_request_data(request)
            
            if user_id and message:
                # Store conversation
                await self._store_conversation(user_id, message)
        
        # Continue with the request
        response = await call_next(request)
        return response

    def _should_log_request(self, request: Request) -> bool:
        """Determine if this request should be logged"""
        if not self.log_chat_endpoints:
            return False
        
        path = request.url.path
        method = request.method
        
        # Check for chat endpoints - simplified logic
        is_chat_endpoint = "/chat" in path
        is_post = method == "POST"
        
        return is_chat_endpoint and is_post

    async def _extract_request_data(self, request: Request) -> tuple[Optional[str], Optional[str]]:
        """Extract user_id and message from request"""
        try:
            # Read request body
            body = await request.body()
            if not body:
                return None, None
            
            # Parse JSON
            data = json.loads(body.decode('utf-8'))
            
            user_id = data.get('user_id')
            message = data.get('message')
            
            return user_id, message
            
        except Exception as e:
            print(f"[MIDDLEWARE] Error extracting request data: {e}")
            return None, None

    async def _store_conversation(self, user_id: str, message: str):
        """Store conversation in database"""
        try:
            print(f"[MIDDLEWARE] Storing conversation for user {user_id}")
            
            # Get conversation service
            conversation_service = await get_conversation_service()
            
            # Store the conversation
            conversation_id = await conversation_service.store_conversation(
                user_id=user_id,
                user_question=message,
                system_response="",  # Will be updated later if needed
                query_analysis={},
                response_parameters={},
                user_email=None  # Default to None when email is not available
            )
            
            print(f"[MIDDLEWARE] Conversation stored successfully with ID: {conversation_id}")
            
        except Exception as e:
            print(f"[MIDDLEWARE] Error storing conversation: {e}")