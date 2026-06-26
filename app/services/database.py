from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing MongoDB connections and operations."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.database = self.client[settings.MONGODB_DATABASE]
            
            # Test the connection
            await self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB at {settings.MONGODB_URL}")
            
            # Create indexes for better performance
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client is not None:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for all collections."""
        try:
            # Create indexes for conversations
            conversations_collection = self.database[settings.MONGODB_CONVERSATIONS_COLLECTION]
            
            await conversations_collection.create_index("user_id")
            await conversations_collection.create_index("messages.knowledge_source")
            await conversations_collection.create_index("messages.requires_human_referral")
            await conversations_collection.create_index("messages.confidence_score")
            await conversations_collection.create_index("messages.is_agriculture_related")
            await conversations_collection.create_index("is_active")
            await conversations_collection.create_index("total_messages")
            
            if settings.CONVERSATION_TTL_DAYS > 0:
                try:
                    await conversations_collection.create_index(
                        "updated_at",
                        expireAfterSeconds=settings.CONVERSATION_TTL_DAYS * 24 * 60 * 60
                    )
                except Exception as ttl_error:
                    if "IndexOptionsConflict" in str(ttl_error):
                        logger.warning("TTL index conflict detected. Dropping existing updated_at index and recreating with TTL.")
                        await conversations_collection.drop_index("updated_at_1")
                        await conversations_collection.create_index(
                            "updated_at",
                            expireAfterSeconds=settings.CONVERSATION_TTL_DAYS * 24 * 60 * 60
                        )
                    else:
                        raise ttl_error
            else:
                await conversations_collection.create_index("updated_at")
            
            await conversations_collection.create_index("created_at")
            await conversations_collection.create_index([("user_id", 1), ("updated_at", -1)])
            await conversations_collection.create_index([("updated_at", -1), ("messages.confidence_score", -1)])
            await conversations_collection.create_index([("user_id", 1), ("is_active", 1), ("updated_at", -1)])
            await conversations_collection.create_index([("title", "text"), ("messages.content", "text")])
            
            # Create indexes for tasks
            tasks_collection = self.get_tasks_collection()
            await tasks_collection.create_index("task_id", unique=True)
            await tasks_collection.create_index("status")
            await tasks_collection.create_index("created_at")
            await tasks_collection.create_index("knowledge_hash_id")
            
            logger.info("Successfully created all MongoDB indexes")
            
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {str(e)}")
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if self.database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.database
    
    def get_conversations_collection(self):
        """Get the conversations collection."""
        database = self.get_database()
        return database[settings.MONGODB_CONVERSATIONS_COLLECTION]
    
    def get_config_collection(self):
        """Get the config collection."""
        database = self.get_database()
        return database["config"]
        
    def get_knowledgebase_collection(self):
        """Get the knowledgebase collection."""
        database = self.get_database()
        return database["knowledgebase"]
        
    def get_tasks_collection(self):
        """Get the tasks collection."""
        database = self.get_database()
        return database["tasks"]
    
    async def insert_task(self, task: Dict[str, Any]) -> str:
        """Insert a new task into the tasks collection."""
        try:
            collection = self.get_tasks_collection()
            result = await collection.insert_one(task)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert task: {str(e)}")
            raise RuntimeError(f"Failed to insert task: {str(e)}")
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by task_id."""
        try:
            collection = self.get_tasks_collection()
            task = await collection.find_one({"task_id": task_id})
            return task
        except Exception as e:
            logger.error(f"Failed to get task: {str(e)}")
            raise RuntimeError(f"Failed to get task: {str(e)}")
    
    async def update_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a task by task_id."""
        try:
            collection = self.get_tasks_collection()
            update_data["updated_at"] = datetime.now().isoformat()
            result = await collection.update_one(
                {"task_id": task_id},
                {"$set": update_data}
            )
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Failed to update task: {str(e)}")
            raise RuntimeError(f"Failed to update task: {str(e)}")
        
    async def insert_knowledge_document(self, document: dict) -> str:
        """
        Insert a document into the knowledgebase collection.
        
        Args:
            document: A dictionary containing the knowledge document data with the following fields:
                - hash_id: A unique identifier for the document
                - title: The title of the knowledge entry
                - content: The main content of the knowledge entry
                - meta_tags: An array of tags for categorization
                - author_name: The name of the author who created the entry
                - additional_references: An array of related references or sources
                - submission_timestamp: The date and time when the entry was submitted
                - entry_type: The type of knowledge entry
                
        Returns:
            The inserted document ID
            
        Raises:
            ValueError: If required fields are missing
            RuntimeError: If database operation fails
        """
        # Validate required fields
        required_fields = ["hash_id", "title", "content", "meta_tags", 
                          "author_name", "submission_timestamp", "entry_type"]
        
        missing_fields = [field for field in required_fields if field not in document]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Ensure arrays are properly formatted
        if "meta_tags" in document and not isinstance(document["meta_tags"], list):
            document["meta_tags"] = [document["meta_tags"]]
            
        if "additional_references" in document and not isinstance(document["additional_references"], list):
            document["additional_references"] = [document["additional_references"]]
        
        try:
            # Get the knowledgebase collection
            collection = self.get_knowledgebase_collection()
            
            # Insert the document
            result = await collection.insert_one(document)
            
            # Return the inserted document ID
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert knowledge document: {str(e)}")
            raise RuntimeError(f"Failed to insert knowledge document: {str(e)}")
            
    async def get_knowledge_documents(self, limit: int = 100, skip: int = 0) -> list[dict]:
        """
        Retrieve knowledge documents from the knowledgebase collection.
        
        Args:
            limit: Maximum number of documents to return
            skip: Number of documents to skip for pagination
            
        Returns:
            List of knowledge documents
        """
        try:
            collection = self.get_knowledgebase_collection()
            cursor = collection.find().sort("submission_timestamp", -1).skip(skip).limit(limit)
            documents = await cursor.to_list(length=limit)
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge documents: {str(e)}")
            raise RuntimeError(f"Failed to retrieve knowledge documents: {str(e)}")
            
    async def get_knowledge_documents_paginated(self, page: int = 1, page_size: int = 10) -> tuple[list[dict], int]:
        """
        Retrieve paginated knowledge documents and total count.
        
        Args:
            page: Page number (starting from 1)
            page_size: Number of documents per page
            
        Returns:
            Tuple of (list of documents, total count)
        """
        try:
            collection = self.get_knowledgebase_collection()
            skip = (page - 1) * page_size
            
            # Get total count first
            total_count = await collection.count_documents({})
            
            # Get paginated documents
            cursor = collection.find().sort("submission_timestamp", -1).skip(skip).limit(page_size)
            documents = await cursor.to_list(length=page_size)
            
            return documents, total_count
        except Exception as e:
            logger.error(f"Failed to retrieve paginated knowledge documents: {str(e)}")
            raise RuntimeError(f"Failed to retrieve paginated knowledge documents: {str(e)}")
            
    async def update_knowledge_document_sync_status(self, hash_id: str, synced: bool = False) -> bool:
        """
        Update the synced field for a knowledge document by hash_id.
        If the synced field doesn't exist, it will be added with default value True for all documents,
        then updated to the specified value for the target document.
        
        Args:
            hash_id: The unique identifier of the document to update
            synced: The value to set for the synced field (default: False)
            
        Returns:
            True if the document was found and updated, False otherwise
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            collection = self.get_knowledgebase_collection()
        
            # Now update the specific document's synced status
            result = await collection.update_one(
                {"hash_id": hash_id},
                {"$set": {"synced": synced}}
            )
            
            if result.matched_count > 0:
                logger.info(f"Updated synced status to {synced} for document with hash_id: {hash_id}")
                return True
            else:
                logger.warning(f"No document found with hash_id: {hash_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update synced status for document {hash_id}: {str(e)}")
            raise RuntimeError(f"Failed to update synced status for document {hash_id}: {str(e)}")


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service() -> DatabaseService:
    """Get the database service instance."""
    global _database_service
    if _database_service is None:
        _database_service = DatabaseService()
        await _database_service.connect()
    return _database_service


async def close_database_connection():
    """Close the database connection."""
    global _database_service
    if _database_service is not None:
        await _database_service.disconnect()
        _database_service = None