from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form, Body, UploadFile, File
from typing import Dict, Any, List, Optional
from enum import Enum
import os
import uuid
from datetime import datetime
from pydantic import BaseModel
from app.services.database import get_database_service
from app.core.config import settings

from app.schemas.knowledge_base import (
    KnowledgeBaseQuery, 
    KnowledgeBaseResponse, 
    ProcessDocsResponse,
    KnowledgeContributionResponse,
    KnowledgeRemovalResponse,
    KnowledgeContributionItem,
    KnowledgeItemDb,
    PaginatedKnowledgeListResponse
)
from app.services.knowledge_base import get_knowledge_base_service
from app.services.document_processor import get_document_processor
from app.services.task_service import get_task_service, TaskStatus

# Create router for knowledge base endpoints
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None


@router.get("/knowledge-list", response_model=PaginatedKnowledgeListResponse)
async def get_knowledge_list(
    page: int = 1,
    page_size: int = 10,
    db_service=Depends(get_database_service)
):
    """Retrieve paginated list of knowledge contributions.
    
    This endpoint returns a paginated list of all knowledge contributions,
    including questions, answers, and metadata.
    """
    try:
        # Validate page and page size
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 10
        if page_size > 100:
            page_size = 100
            
        # Query the database for paginated knowledge contributions
        documents, total_count = await db_service.get_knowledge_documents_paginated(
            page=page,
            page_size=page_size
        )
        
        # Convert documents to KnowledgeContributionItem format
        knowledge_list = []
        for doc in documents:
            # Handle different document structures based on entry_type
            if doc.get("entry_type") == "user_contribution":
                knowledge_list.append(KnowledgeItemDb(
                    hash_id=doc["hash_id"],
                    title=doc["title"],
                    content=doc["content"],
                    author_name=doc["author_name"],
                    meta_tags=doc["meta_tags"],
                    entry_type=doc["entry_type"],
                    submission_timestamp=doc["submission_timestamp"],
                    synced=doc.get("synced", True),
                    file_type=doc.get("file_type"),
                    file_name=doc.get("file_name"),
                    task_id=doc.get("task_id")
                ))
            # Add handling for other entry types if needed
        
        # Calculate total pages
        total_pages = (total_count + page_size - 1) // page_size
        
        return PaginatedKnowledgeListResponse(
            items=knowledge_list,
            total=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge list retrieval error: {str(e)}")

@router.post("/query", response_model=KnowledgeBaseResponse)
async def query_knowledge_base(query: KnowledgeBaseQuery, kb_service=Depends(get_knowledge_base_service)):
    """Query the knowledge base with a question.
    
    This endpoint accepts a question and returns an answer based on the
    agricultural documents in the knowledge base. If the system cannot find
    relevant information, it will flag the query for human expert review.
    """
    try:
        # Query the knowledge base
        result = await kb_service.query_knowledge_base(query.question, is_public=query.is_public)
    
        # Return the response with human referral information if needed
        return KnowledgeBaseResponse(
            answer=result["answer"],
            sources=result["sources"],
            requires_human_support=result["requires_human_support"],
            query_id=result["query_id"],
            source_type=result.get("source_type", "pdf"),
            confidence_score=result["confidence_score"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge base query error: {str(e)}")


@router.post("/process", response_model=ProcessDocsResponse)
async def process_documents(background_tasks: BackgroundTasks):
    """Process all PDF documents in the docs directory.
    
    This endpoint triggers the processing of all PDF documents in the docs directory
    and adds them to the vector database. This operation runs in the background.
    """
    try:
        # Get the document processor
        doc_processor = get_document_processor()
        
        # Process documents in the background
        background_tasks.add_task(doc_processor.process_all_pdfs)
        
        return ProcessDocsResponse(
            message="Document processing started in the background",
            status="processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")


@router.post("/process-excel", response_model=ProcessDocsResponse)
async def process_excel_files(background_tasks: BackgroundTasks):
    """Process all Excel QA files in the configured directory.
    
    This endpoint triggers the processing of all Excel QA files in the docs directory
    and adds them to the vector database. This operation runs in the background.
    """
    try:
        # Get the knowledge base service
        kb_service = get_knowledge_base_service()
        
        # Process Excel files in the background
        background_tasks.add_task(kb_service.process_excel_files)
        
        return ProcessDocsResponse(
            message="Excel QA processing started in the background",
            status="processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel processing error: {str(e)}")


@router.get("/status")
async def get_processing_status():
    """Get the status of the document processing.
    
    This endpoint returns information about the vector database,
    including the number of documents processed.
    """
    try:
        # Get the document processor
        doc_processor = get_document_processor()
        
        # Get the vector store
        vector_store = doc_processor.get_vector_store()
        
        # Check if vector store is None (embeddings not available)
        if vector_store is None:
            return {
                "status": "unavailable",
                "message": "Vector store is not available. Check if OpenAI API key is configured correctly.",
                "document_count": 0,
                "pdf_document_count": 0,
                "excel_qa_count": 0
            }
        
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        
        # Count documents by source type
        pdf_count = 0
        excel_qa_count = 0
        
        # This is a simple approach - in a production system you might want to use more efficient queries
        for doc in collection.get()['metadatas']:
            if doc.get('source_type') == 'excel_qa':
                excel_qa_count += 1
            else:
                pdf_count += 1
        
        return {
            "status": "ready" if count > 0 else "empty",
            "document_count": count,
            "pdf_document_count": pdf_count,
            "excel_qa_count": excel_qa_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check error: {str(e)}")



@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, task_service=Depends(get_task_service)):
    """Get the status of a background processing task."""
    try:
        task = await task_service.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskStatusResponse(
            task_id=task["task_id"],
            status=TaskStatus(task["status"]),
            progress=task.get("progress", 0),
            error=task.get("error"),
            metadata=task.get("metadata"),
            created_at=task["created_at"],
            updated_at=task["updated_at"],
            completed_at=task.get("completed_at")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.post("/contribute", response_model=KnowledgeContributionResponse)
async def contribute_knowledge(
    background_tasks: BackgroundTasks,
    kb_service=Depends(get_knowledge_base_service),
    title: str = Form(..., description="Title of the knowledge entry."),
    content: str = Form(..., description="Main body/content of the knowledge in Persian."),
    source:Optional[str] = Form(None, description="The origin or reference for the knowledge (optional)."),
    meta_tags: str = Form(..., description="Comma-separated keywords for categorization (e.g., soil,fertilizer,wheat)."),
    author_name: Optional[str] = Form(None, description="Name of the contributor (optional)."),
    additional_references: Optional[str] = Form(None, description="URLs or citation text for further reading (optional)."),
    is_public: bool = Form(False, description="Mark the contribution as public information."),
    file: Optional[UploadFile] = File(None, description="Optional PDF, Word (docx), or Excel file to be processed and added to the knowledge base.")
):
    """Allows users to contribute new agricultural knowledge entries.

    The endpoint accepts form-data for various fields describing the knowledge.
    It validates the input, processes it, and stores it in the vector knowledge base.
    
    Supported file formats:
    - PDF (.pdf): Text and tables extracted
    - Word (.docx): Text and tables extracted with Persian support
    - Excel (.xlsx, .xls): Question-Answer pairs extracted
    """
    try:
        # Basic input sanitation (FastAPI handles basic type validation)
        cleaned_title = title.strip()
        cleaned_content = content.strip()
        cleaned_source = source.strip() if source else None
        
        if not cleaned_title or not cleaned_content:
            return KnowledgeContributionResponse(success=False, message="Title, content, and source cannot be empty.")

        parsed_meta_tags = [tag.strip() for tag in meta_tags.split(',') if tag.strip()]
        if not parsed_meta_tags:
            return KnowledgeContributionResponse(success=False, message="Meta tags cannot be empty and must be comma-separated.")

        # Process uploaded file if provided
        uploaded_file_path = None
        if file:
            # Validate file type
            file_ext = file.filename.lower().split('.')[-1]
            if file_ext not in ['pdf', 'docx', 'xlsx', 'xls']:
                return KnowledgeContributionResponse(
                    success=False, 
                    message="Unsupported file format. Only PDF, Word (docx), and Excel (xlsx, xls) files are supported."
                )
            
            # Save the uploaded file to the docs directory
            import shutil
            
            # Resolve storage root from SETTINGS or project root
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            _storage_root = settings.STORAGE_ROOT.strip() if settings.STORAGE_ROOT else _project_root
            docs_dir = os.path.join(_storage_root, "docs")
            os.makedirs(docs_dir, exist_ok=True)
            
            # Generate a unique filename to avoid conflicts
            import uuid
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(docs_dir, unique_filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_file_path = file_path

        # Call the service to add the contribution
        contribution_details = await kb_service.add_knowledge_contribution(
            title=cleaned_title,
            content=cleaned_content,
            source=cleaned_source,
            meta_tags=parsed_meta_tags,
            author_name=author_name.strip() if author_name else None,
            additional_references=additional_references.strip() if additional_references else None,
            uploaded_file_path=uploaded_file_path,
            is_public=is_public
        )
        
        # Add the background task to process the contribution
        task_id = contribution_details.get("task_id")
        if task_id:
            background_tasks.add_task(
                kb_service.process_knowledge_contribution_background,
                task_id,
                contribution_details["id"],
                {
                    "title": cleaned_title,
                    "content": cleaned_content,
                    "source": cleaned_source,
                    "author_name": author_name.strip() if author_name else None,
                    "additional_references": additional_references.strip() if additional_references else None,
                    "uploaded_file_path": uploaded_file_path,
                    "is_public": is_public,
                    "meta_tags": parsed_meta_tags
                }
            )
        
        return KnowledgeContributionResponse(
            success=True,
            contribution=KnowledgeContributionItem(**contribution_details)
        )

    except ValueError as ve:
        # Handle specific validation errors from service or here
        return KnowledgeContributionResponse(success=False, message=str(ve))
    except Exception as e:
        # Log the exception for debugging
        # logger.error(f"Error during knowledge contribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to contribute knowledge: {str(e)}")


@router.delete("/remove/{hash_id}", response_model=KnowledgeRemovalResponse)
async def remove_knowledge_contribution(
    hash_id: str,
    kb_service=Depends(get_knowledge_base_service)
):
    """Remove a knowledge contribution by its hash_id.
    
    This endpoint removes a knowledge contribution from both the vector store
    and the database. It also updates the sync status before removal.
    
    Args:
        hash_id: The unique hash identifier of the knowledge contribution to remove
        
    Returns:
        KnowledgeRemovalResponse: Response indicating success/failure with details
    """
    try:
        # Validate hash_id format (basic validation)
        if not hash_id or not hash_id.strip():
            return KnowledgeRemovalResponse(
                success=False,
                message="Hash ID cannot be empty",
                hash_id=hash_id,
                removed_count=0
            )
        
        # Call the service to remove the knowledge contribution
        result = await kb_service.remove_knowledge_contribution(hash_id.strip())
        
        if result.get("success", False):
            removed_count = result.get("documents_removed_count", 0)
            return KnowledgeRemovalResponse(
                success=True,
                message=f"Knowledge contribution removed successfully. {removed_count} items removed from vector store. Document marked as unsynced in database.",
                hash_id=hash_id,
                removed_count=removed_count
            )
        else:
            error_msg = result.get("error", "Knowledge contribution not found or could not be removed")
            return KnowledgeRemovalResponse(
                success=False,
                message=error_msg,
                hash_id=hash_id,
                removed_count=0
            )
            
    except ValueError as ve:
        # Handle specific validation errors from service
        return KnowledgeRemovalResponse(
            success=False,
            message=str(ve),
            hash_id=hash_id,
            removed_count=0
        )
    except Exception as e:
        # Log the exception for debugging
        # logger.error(f"Error during knowledge removal: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to remove knowledge contribution: {str(e)}"
        )