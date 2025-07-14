from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from pymongo import DESCENDING
from motor.motor_asyncio import AsyncIOMotorCollection

from app.schemas.conversation import (
    ConversationDocument,
    ConversationSearchRequest,
    ConversationResponse,
    ConversationListResponse
)
from app.schemas.message import MessageDocument, MessageRole
from app.services.database import get_database_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversation storage and retrieval."""
    
    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the conversations collection."""
        if self._collection is None:
            db_service = await get_database_service()
            self._collection = db_service.get_conversations_collection()
        return self._collection
    
    async def store_conversation(
        self,
        user_id: str,
        user_question: str,
        system_response: str,
        query_analysis: Dict[str, Any],
        response_parameters: Dict[str, Any],
        sources_used: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        response_time_ms: Optional[float] = None
    ) -> str:
        """Store a conversation in the database using the new embedded message structure.
        
        Args:
            user_id: Unique identifier for the user
            user_question: The user's question/message
            system_response: The AI system's response
            query_analysis: Analysis data from the chat service
            response_parameters: Parameters used for the response
            sources_used: List of sources used in RAG response
            session_id: Session identifier if applicable
            user_agent: User agent information
            ip_address: User IP address (should be hashed for privacy)
            response_time_ms: Response time in milliseconds
            
        Returns:
            The ID of the stored conversation
        """
        try:
            collection = await self._get_collection()
            
            # Create user message
            user_message = MessageDocument(
                role=MessageRole.USER,
                content=user_question,
                timestamp=datetime.utcnow()
            )
            
            # Create assistant message with analysis data
            assistant_message = MessageDocument(
                role=MessageRole.ASSISTANT,
                content=system_response,
                timestamp=datetime.utcnow(),
                confidence_score=query_analysis.get("confidence_score"),
                knowledge_source=query_analysis.get("knowledge_source"),
                requires_human_referral=query_analysis.get("requires_human_referral"),
                reasoning=query_analysis.get("reasoning"),
                model_used=response_parameters.get("model"),
                temperature=response_parameters.get("temperature"),
                max_tokens=response_parameters.get("max_tokens"),
                sources_used=sources_used,
                response_time_ms=response_time_ms
            )
            
            # Create conversation document with embedded messages
            conversation = ConversationDocument(
                user_id=user_id,
                messages=[user_message, assistant_message],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                session_id=session_id,
                user_agent=user_agent,
                ip_address=ip_address,
                total_messages=2,
                is_active=True
            )
            
            # Insert into database
            result = await collection.insert_one(conversation.dict(by_alias=True, exclude={"id"}))
            
            logger.info(f"Stored conversation for user {user_id} with ID {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {str(e)}")
            # Don't raise the exception to avoid breaking the chat flow
            return ""
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> ConversationListResponse:
        """Get all conversations for a specific user.
        
        Args:
            user_id: The user ID to filter by
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip for pagination
            
        Returns:
            A list of conversations for the user
        """
        try:
            collection = await self._get_collection()
            
            # Build query
            query = {"user_id": user_id}
            
            # Get total count
            total_count = await collection.count_documents(query)
            
            # Get conversations with pagination, sorted by updated_at (newest first)
            cursor = collection.find(query).sort("updated_at", DESCENDING).skip(skip).limit(limit)
            conversations_data = await cursor.to_list(length=limit)
            
            # Convert to response format
            conversations = []
            for conv in conversations_data:
                # Convert messages to MessageResponse format
                message_responses = []
                for msg in conv.get("messages", []):
                    message_responses.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"],
                        "confidence_score": msg.get("confidence_score"),
                        "knowledge_source": msg.get("knowledge_source"),
                        "requires_human_referral": msg.get("requires_human_referral")
                    })
                
                conversations.append(ConversationResponse(
                    id=str(conv["_id"]),
                    user_id=conv["user_id"],
                    title=conv.get("title"),
                    messages=message_responses,
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    total_messages=conv.get("total_messages", len(conv.get("messages", []))),
                    is_active=conv.get("is_active", True)
                ))
            
            # Calculate pagination info
            page = (skip // limit) + 1
            has_next = (skip + limit) < total_count
            
            return ConversationListResponse(
                conversations=conversations,
                total_count=total_count,
                page=page,
                page_size=limit,
                has_next=has_next
            )
            
        except Exception as e:
            logger.error(f"Failed to get user conversations: {str(e)}")
            return ConversationListResponse(
                conversations=[],
                total_count=0,
                page=1,
                page_size=limit,
                has_next=False
            )
    
    async def get_latest_user_conversations(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[ConversationResponse]:
        """Get the latest conversations for a specific user.
        
        Args:
            user_id: The user ID to filter by
            limit: Maximum number of conversations to return
            
        Returns:
            A list of the latest conversations for the user
        """
        try:
            collection = await self._get_collection()
            
            # Get latest conversations, sorted by updated_at (newest first)
            cursor = collection.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(limit)
            conversations_data = await cursor.to_list(length=limit)
            
            # Convert to response format
            conversations = []
            for conv in conversations_data:
                # Convert messages to MessageResponse format
                message_responses = []
                for msg in conv.get("messages", []):
                    message_responses.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"],
                        "confidence_score": msg.get("confidence_score"),
                        "knowledge_source": msg.get("knowledge_source"),
                        "requires_human_referral": msg.get("requires_human_referral")
                    })
                
                conversations.append(ConversationResponse(
                    id=str(conv["_id"]),
                    user_id=conv["user_id"],
                    title=conv.get("title"),
                    messages=message_responses,
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    total_messages=conv.get("total_messages", len(conv.get("messages", []))),
                    is_active=conv.get("is_active", True)
                ))
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get latest user conversations: {str(e)}")
            return []
    
    async def search_conversations(
        self,
        search_request: ConversationSearchRequest
    ) -> ConversationListResponse:
        """Search conversations based on various criteria.
        
        Args:
            search_request: The search criteria
            
        Returns:
            A list of conversations matching the criteria
        """
        try:
            collection = await self._get_collection()
            
            # Build query
            query = {}
            
            if search_request.user_id:
                query["user_id"] = search_request.user_id
            
            if search_request.start_date or search_request.end_date:
                timestamp_query = {}
                if search_request.start_date:
                    timestamp_query["$gte"] = search_request.start_date
                if search_request.end_date:
                    timestamp_query["$lte"] = search_request.end_date
                query["updated_at"] = timestamp_query
            
            if search_request.search_text:
                # Search in title and message content
                query["$or"] = [
                    {"title": {"$regex": search_request.search_text, "$options": "i"}},
                    {"messages.content": {"$regex": search_request.search_text, "$options": "i"}}
                ]
            
            if search_request.knowledge_source:
                query["messages.knowledge_source"] = search_request.knowledge_source
            
            if search_request.requires_human_referral is not None:
                query["messages.requires_human_referral"] = search_request.requires_human_referral
            
            if search_request.min_confidence is not None or search_request.max_confidence is not None:
                confidence_query = {}
                if search_request.min_confidence is not None:
                    confidence_query["$gte"] = search_request.min_confidence
                if search_request.max_confidence is not None:
                    confidence_query["$lte"] = search_request.max_confidence
                query["messages.confidence_score"] = confidence_query
            
            # Get total count
            total_count = await collection.count_documents(query)
            
            # Get conversations with pagination, sorted by updated_at (newest first)
            cursor = collection.find(query).sort("updated_at", DESCENDING).skip(search_request.skip).limit(search_request.limit)
            conversations_data = await cursor.to_list(length=search_request.limit)
            
            # Convert to response format
            conversations = []
            for conv in conversations_data:
                # Convert messages to MessageResponse format
                message_responses = []
                for msg in conv.get("messages", []):
                    message_responses.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"],
                        "confidence_score": msg.get("confidence_score"),
                        "knowledge_source": msg.get("knowledge_source"),
                        "requires_human_referral": msg.get("requires_human_referral")
                    })
                
                conversations.append(ConversationResponse(
                    id=str(conv["_id"]),
                    user_id=conv["user_id"],
                    title=conv.get("title"),
                    messages=message_responses,
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    total_messages=conv.get("total_messages", len(conv.get("messages", []))),
                    is_active=conv.get("is_active", True)
                ))
            
            # Calculate pagination info
            page = (search_request.skip // search_request.limit) + 1
            has_next = (search_request.skip + search_request.limit) < total_count
            
            return ConversationListResponse(
                conversations=conversations,
                total_count=total_count,
                page=page,
                page_size=search_request.limit,
                has_next=has_next
            )
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {str(e)}")
            return ConversationListResponse(
                conversations=[],
                total_count=0,
                page=1,
                page_size=search_request.limit,
                has_next=False
            )
    
    async def get_conversation_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics.
        
        Args:
            user_id: Optional user ID to filter stats by specific user
            
        Returns:
            Dictionary containing various statistics
        """
        try:
            collection = await self._get_collection()
            
            # Build base query
            base_query = {}
            if user_id:
                base_query["user_id"] = user_id
            
            # Aggregate statistics
            pipeline = [
                {"$match": base_query},
                {"$unwind": "$messages"},
                {"$match": {"messages.role": "assistant"}},
                {
                    "$group": {
                        "_id": None,
                        "total_conversations": {"$addToSet": "$_id"},
                        "human_referrals": {
                            "$sum": {"$cond": [{"$eq": ["$messages.requires_human_referral", True]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$messages.confidence_score"},
                        "knowledge_base_responses": {
                            "$sum": {"$cond": [{"$eq": ["$messages.knowledge_source", "knowledge_base"]}, 1, 0]}
                        },
                        "general_knowledge_responses": {
                            "$sum": {"$cond": [{"$eq": ["$messages.knowledge_source", "general_knowledge"]}, 1, 0]}
                        }
                    }
                },
                {
                    "$project": {
                        "total_conversations": {"$size": "$total_conversations"},
                        "human_referrals": 1,
                        "avg_confidence": 1,
                        "knowledge_base_responses": 1,
                        "general_knowledge_responses": 1
                    }
                }
            ]
            
            result = await collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                stats.pop("_id", None)  # Remove the _id field
                return stats
            else:
                return {
                    "total_conversations": 0,
                    "human_referrals": 0,
                    "avg_confidence": 0,
                    "knowledge_base_responses": 0,
                    "general_knowledge_responses": 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {str(e)}")
            return {}


# Singleton instance
_conversation_service: Optional[ConversationService] = None


async def get_conversation_service() -> ConversationService:
    """Get the conversation service instance.
    
    Returns:
        A singleton instance of the ConversationService
    """
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service