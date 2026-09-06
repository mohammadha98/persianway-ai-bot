"""
AI Feedback Service

Handles storage and retrieval of feedback (approve, report, edit, add_to_kb)
on AI-generated messages. Provides relationship between feedback, messages,
and conversations in MongoDB.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from app.services.database import DatabaseService, get_database_service

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for managing AI message feedback in MongoDB."""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.collection_name = "ai_feedback"
    
    def get_collection(self):
        """Get the feedback collection."""
        return self.db_service.get_feedback_collection()
    
    async def create_indexes(self):
        """Create indexes for the feedback collection."""
        try:
            collection = self.get_collection()
            await collection.create_index("feedback_id", unique=True)
            await collection.create_index("message_id")
            await collection.create_index("conversation_id")
            await collection.create_index("feedback")
            await collection.create_index("created_at")
            await collection.create_index([("message_id", 1), ("user_email", 1)])
            logger.info("Feedback collection indexes created")
        except Exception as e:
            logger.error(f"Failed to create feedback indexes: {str(e)}")
    
    async def submit_feedback(
        self,
        message_id: str,
        feedback: str,
        comment: str = "",
        category: Optional[str] = None,
        user_email: Optional[str] = None,
        user_id: Optional[str] = None,
        message_content: Optional[str] = None,
        question: Optional[str] = None,
        conversation_id: Optional[str] = None,
        edited_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit feedback for an AI message.
        
        Returns the saved feedback document with feedback_id.
        """
        try:
            collection = self.get_collection()
            
            feedback_id = f"fb_{uuid.uuid4().hex[:16]}"
            now = datetime.utcnow().isoformat()
            
            feedback_doc = {
                "feedback_id": feedback_id,
                "message_id": message_id,
                "conversation_id": conversation_id,
                "feedback": feedback,
                "comment": comment,
                "category": category,
                "user_email": user_email,
                "user_id": user_id,
                "message_content": message_content,
                "question": question,
                "edited_answer": edited_answer,
                "created_at": now,
                "updated_at": None,
                "status": "active"
            }
            
            await collection.insert_one(feedback_doc)
            
            # Also update the conversation to reference this feedback (if applicable)
            if conversation_id:
                await self._link_feedback_to_conversation(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    feedback_id=feedback_id,
                    feedback_type=feedback
                )
            
            logger.info(
                f"Feedback {feedback_id} ({feedback}) saved for message {message_id}"
            )
            return feedback_doc
        except Exception as e:
            logger.error(f"Failed to submit feedback: {str(e)}")
            raise RuntimeError(f"Failed to submit feedback: {str(e)}")
    
    async def _link_feedback_to_conversation(
        self,
        conversation_id: str,
        message_id: str,
        feedback_id: str,
        feedback_type: str
    ):
        """
        Link the feedback to the conversation document so we can navigate
        from feedback -> conversation in MongoDB.
        """
        try:
            conversations = self.db_service.get_conversations_collection()
            
            # Update conversation: add to feedback list
            await conversations.update_one(
                {"conversation_id": conversation_id},
                {
                    "$push": {
                        "feedbacks": {
                            "feedback_id": feedback_id,
                            "message_id": message_id,
                            "feedback_type": feedback_type,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    },
                    "$set": {
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
        except Exception as e:
            logger.warning(f"Failed to link feedback to conversation: {str(e)}")
    
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """Get a single feedback by its feedback_id."""
        try:
            collection = self.get_collection()
            feedback = await collection.find_one({"feedback_id": feedback_id})
            return feedback
        except Exception as e:
            logger.error(f"Failed to get feedback: {str(e)}")
            return None
    
    async def get_feedback_by_message(
        self,
        message_id: str,
        user_email: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific message.
        If user_email is provided, returns only that user's feedback.
        """
        try:
            collection = self.get_collection()
            query = {"message_id": message_id}
            if user_email:
                query["user_email"] = user_email
            
            cursor = collection.find(query).sort("created_at", -1)
            feedbacks = await cursor.to_list(length=100)
            return feedbacks
        except Exception as e:
            logger.error(f"Failed to get feedback by message: {str(e)}")
            return []
    
    async def get_feedback_by_conversation(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get all feedback for a specific conversation."""
        try:
            collection = self.get_collection()
            cursor = collection.find(
                {"conversation_id": conversation_id}
            ).sort("created_at", -1)
            feedbacks = await cursor.to_list(length=500)
            return feedbacks
        except Exception as e:
            logger.error(f"Failed to get feedback by conversation: {str(e)}")
            return []
    
    async def list_feedbacks(
        self,
        page: int = 1,
        page_size: int = 20,
        feedback_type: Optional[str] = None,
        category: Optional[str] = None,
        user_email: Optional[str] = None,
        status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated list of feedbacks with optional filters.
        Returns (feedbacks, total_count).
        """
        try:
            collection = self.get_collection()
            
            # Build query
            query = {}
            if feedback_type:
                query["feedback"] = feedback_type
            if category:
                query["category"] = category
            if user_email:
                query["user_email"] = user_email
            if status:
                query["status"] = status
            
            # Get total
            total = await collection.count_documents(query)
            
            # Get paginated
            skip = (page - 1) * page_size
            cursor = (
                collection
                .find(query)
                .sort("created_at", -1)
                .skip(skip)
                .limit(page_size)
            )
            items = await cursor.to_list(length=page_size)
            
            return items, total
        except Exception as e:
            logger.error(f"Failed to list feedbacks: {str(e)}")
            return [], 0
    
    async def update_feedback_status(
        self,
        feedback_id: str,
        status: str
    ) -> bool:
        """Update the status of a feedback (active, resolved, dismissed)."""
        try:
            collection = self.get_collection()
            result = await collection.update_one(
                {"feedback_id": feedback_id},
                {
                    "$set": {
                        "status": status,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update feedback status: {str(e)}")
            return False
    
    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback (soft delete by marking as dismissed)."""
        try:
            collection = self.get_collection()
            result = await collection.update_one(
                {"feedback_id": feedback_id},
                {
                    "$set": {
                        "status": "dismissed",
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to delete feedback: {str(e)}")
            return False
    
    async def get_feedback_stats(
        self,
        user_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get feedback statistics."""
        try:
            collection = self.get_collection()
            
            base_query = {}
            if user_email:
                base_query["user_email"] = user_email
            
            # Count by type
            pipeline = [
                {"$match": {**base_query, "status": "active"}},
                {"$group": {"_id": "$feedback", "count": {"$sum": 1}}}
            ]
            type_counts_cursor = collection.aggregate(pipeline)
            type_counts = {item["_id"]: item["count"] async for item in type_counts_cursor}
            
            # Count by category
            cat_pipeline = [
                {"$match": {**base_query, "status": "active", "category": {"$ne": None}}},
                {"$group": {"_id": "$category", "count": {"$sum": 1}}}
            ]
            cat_cursor = collection.aggregate(cat_pipeline)
            cat_counts = {item["_id"]: item["count"] async for item in cat_cursor}
            
            # Totals
            total = await collection.count_documents({**base_query, "status": "active"})
            approved = type_counts.get("approve", 0)
            reported = type_counts.get("report", 0)
            edits = type_counts.get("edit", 0)
            add_to_kb = type_counts.get("add_to_kb", 0)
            
            # Approval rate
            approval_rate = 0.0
            if total > 0:
                approval_rate = round((approved / total) * 100, 2)
            
            return {
                "total_feedbacks": total,
                "approved_count": approved,
                "reported_count": reported,
                "edit_count": edits,
                "add_to_kb_count": add_to_kb,
                "by_category": cat_counts,
                "approval_rate": approval_rate
            }
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {str(e)}")
            return {
                "total_feedbacks": 0,
                "approved_count": 0,
                "reported_count": 0,
                "edit_count": 0,
                "add_to_kb_count": 0,
                "by_category": {},
                "approval_rate": 0.0
            }


# Global instance holder
_feedback_service_instance: Optional[FeedbackService] = None


async def get_feedback_service() -> FeedbackService:
    """Get or create the feedback service singleton."""
    global _feedback_service_instance
    if _feedback_service_instance is None:
        db_service = await get_database_service()
        _feedback_service_instance = FeedbackService(db_service)
        # Create indexes (idempotent)
        try:
            await _feedback_service_instance.create_indexes()
        except Exception as e:
            logger.warning(f"Could not create feedback indexes: {str(e)}")
    return _feedback_service_instance