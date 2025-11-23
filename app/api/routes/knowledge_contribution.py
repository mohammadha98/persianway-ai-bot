from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.services.database import get_database_service
from app.services.knowledge_base import get_knowledge_base_service
from app.services.document_processor import get_document_processor
from app.schemas.knowledge_base import SyncResponse

# Create router for knowledge contribution endpoints
router = APIRouter(prefix="/knowledge-contribution", tags=["knowledge-contribution"])


@router.post("/sync-mongodb-to-vectordb", response_model=SyncResponse)
async def sync_mongodb_to_vectordb():
    """
    Sync user_contribution documents from MongoDB to vector database.
    
    This endpoint triggers the synchronization of user contribution documents
    from MongoDB to the vector database. Only documents with entry_type: "user_contribution"
    are processed.
    
    Returns:
        SyncResponse with detailed sync statistics and operation status
    """
    try:
        # Get the document processor and knowledge base service
        doc_processor = get_document_processor()
        kb_service = get_knowledge_base_service()
        
        # Execute the sync operation
        result = await doc_processor.sync_mongodb_to_vectordb(knowledge_base_service=kb_service)
        
        logging.info(f"MongoDB to VectorDB sync completed: {result}")
        
        # Convert the result dictionary to SyncResponse
        return SyncResponse(
            success=result.get('error_count', 0) == 0,
            processed_count=result.get('processed_count', 0),
            skipped_count=result.get('skipped_count', 0),
            error_count=result.get('error_count', 0),
            total_documents=result.get('total_documents', 0),
            start_time=result.get('start_time'),
            end_time=result.get('end_time'),
            total_time_seconds=result.get('total_time_seconds'),
            message=result.get('message', 'Sync completed successfully')
        )
        
    except Exception as e:
        logging.error(f"Failed to sync MongoDB to VectorDB: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"MongoDB to VectorDB sync error: {str(e)}"
        )


@router.post("/sync-mongodb-to-vectordb-background", response_model=Dict[str, Any])
async def sync_mongodb_to_vectordb_background(background_tasks: BackgroundTasks):
    """
    Sync user_contribution documents from MongoDB to vector database in background.
    
    This endpoint triggers the synchronization of user contribution documents
    from MongoDB to the vector database as a background task.
    
    Returns:
        Dictionary with sync status message
    """
    try:
        # Get the document processor and knowledge base service
        doc_processor = get_document_processor()
        kb_service = get_knowledge_base_service()
        
        # Add the sync operation to background tasks
        background_tasks.add_task(
            doc_processor.sync_mongodb_to_vectordb,
            knowledge_base_service=kb_service
        )
        
        return {
            "message": "MongoDB to VectorDB sync started in background",
            "status": "processing"
        }
        
    except Exception as e:
        logging.error(f"Failed to start MongoDB to VectorDB sync: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"MongoDB to VectorDB sync start error: {str(e)}"
        )