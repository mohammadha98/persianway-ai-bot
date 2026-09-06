from typing import Dict, Any, Optional
import uuid
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskService:
    """Service for managing background processing tasks."""
    
    def __init__(self, db_service):
        self.db_service = db_service
    
    async def create_task(
        self,
        task_type: str,
        knowledge_hash_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new background task."""
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": TaskStatus.PENDING,
            "knowledge_hash_id": knowledge_hash_id,
            "progress": 0,
            "metadata": metadata or {},
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None
        }
        await self.db_service.insert_task(task)
        logger.info(f"Created task {task_id} of type {task_type}")
        return task_id
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update task status and progress."""
        update_data = {"status": status}
        if progress is not None:
            update_data["progress"] = progress
        if error is not None:
            update_data["error"] = error
        if metadata is not None:
            update_data["metadata"] = metadata
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            update_data["completed_at"] = datetime.now().isoformat()
        
        success = await self.db_service.update_task(task_id, update_data)
        if success:
            logger.info(f"Updated task {task_id} to status {status}")
        return success
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task details by task_id."""
        return await self.db_service.get_task(task_id)


# Singleton instance will be set when needed
_task_service_instance: Optional[TaskService] = None


async def get_task_service() -> TaskService:
    """Get the task service instance."""
    from app.services.database import get_database_service
    global _task_service_instance
    if _task_service_instance is None:
        db_service = await get_database_service()
        _task_service_instance = TaskService(db_service)
    return _task_service_instance
