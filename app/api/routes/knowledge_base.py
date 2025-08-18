from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form, Body, UploadFile, File
from typing import Dict, Any, List, Optional
from enum import Enum
import uuid
from datetime import datetime

from app.schemas.knowledge_base import (
    KnowledgeBaseQuery, 
    KnowledgeBaseResponse, 
    ProcessDocsResponse,
    KnowledgeContributionResponse,
    KnowledgeContributionItem
)
from app.services.knowledge_base import get_knowledge_base_service
from app.services.document_processor import get_document_processor

# Create router for knowledge base endpoints
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/query", response_model=KnowledgeBaseResponse)
async def query_knowledge_base(query: KnowledgeBaseQuery, kb_service=Depends(get_knowledge_base_service)):
    """Query the knowledge base with a question.
    
    This endpoint accepts a question and returns an answer based on the
    agricultural documents in the knowledge base. If the system cannot find
    relevant information, it will flag the query for human expert review.
    """
    try:
        # Query the knowledge base
        result = await kb_service.query_knowledge_base(query.question)
    
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



@router.post("/contribute", response_model=KnowledgeContributionResponse)
async def contribute_knowledge(
    kb_service=Depends(get_knowledge_base_service),
    title: str = Form(..., description="Title of the knowledge entry."),
    content: str = Form(..., description="Main body/content of the knowledge in Persian."),
    source: str = Form(..., description="The origin or reference for the knowledge."),
    meta_tags: str = Form(..., description="Comma-separated keywords for categorization (e.g., soil,fertilizer,wheat)."),
    author_name: Optional[str] = Form(None, description="Name of the contributor (optional)."),
    additional_references: Optional[str] = Form(None, description="URLs or citation text for further reading (optional)."),
    file: Optional[UploadFile] = File(None, description="Optional PDF or Excel file to be processed and added to the knowledge base.")
):
    """Allows users to contribute new agricultural knowledge entries.

    The endpoint accepts form-data for various fields describing the knowledge.
    It validates the input, processes it, and stores it in the vector knowledge base.
    """
    try:
        # Basic input sanitation (FastAPI handles basic type validation)
        # More advanced sanitation (e.g., HTML stripping) could be added here if content allows HTML
        cleaned_title = title.strip()
        cleaned_content = content.strip()
        cleaned_source = source.strip()
        
        if not cleaned_title or not cleaned_content or not cleaned_source:
            return KnowledgeContributionResponse(success=False, message="Title, content, and source cannot be empty.")

        parsed_meta_tags = [tag.strip() for tag in meta_tags.split(',') if tag.strip()]
        if not parsed_meta_tags:
            return KnowledgeContributionResponse(success=False, message="Meta tags cannot be empty and must be comma-separated.")

        # Process uploaded file if provided
        uploaded_file_path = None
        if file:
            # Validate file type
            file_ext = file.filename.lower().split('.')[-1]
            if file_ext not in ['pdf', 'xlsx', 'xls']:
                return KnowledgeContributionResponse(
                    success=False, 
                    message="Unsupported file format. Only PDF and Excel (xlsx, xls) files are supported."
                )
            
            # Save the uploaded file to the docs directory
            import os
            from app.core.config import settings
            import shutil
            
            # Create docs directory if it doesn't exist
            docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "docs")
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
            uploaded_file_path=uploaded_file_path
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