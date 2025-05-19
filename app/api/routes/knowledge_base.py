from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List

from app.schemas.knowledge_base import KnowledgeBaseQuery, KnowledgeBaseResponse, ProcessDocsResponse
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
            query_id=result["query_id"]
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