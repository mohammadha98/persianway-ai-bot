from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import os
import uuid
import shutil
import time
from pathlib import Path
from fastapi.responses import JSONResponse

from app.schemas.upload import FileUploadResponse, PDFProcessingResponse
from app.services.document_processor import DocumentProcessor

# Create router for upload endpoints
router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/", response_model=FileUploadResponse, status_code=200)
async def upload_file(file: UploadFile = File(...), description: Optional[str] = Form(None)):
    """Upload a file to the server.
    
    This endpoint accepts a file upload and stores it in the docs directory.
    It returns information about the uploaded file.
    
    Args:
        file: The file to upload (PDF or Excel)
        description: Optional description of the file
        
    Returns:
        FileUploadResponse with status, message, and filename
    """
    try:
        # Validate file type
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['pdf', 'xlsx', 'xls']:
            return JSONResponse(
                status_code=400,
                content=FileUploadResponse(
                    status="error",
                    message="Unsupported file format. Only PDF and Excel (xlsx, xls) files are supported."
                ).dict()
            )
        
        # Create docs directory if it doesn't exist
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Generate a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(docs_dir, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return FileUploadResponse(
            status="success",
            message="File uploaded successfully",
            filename=unique_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")


@router.post("/pdf/process", response_model=PDFProcessingResponse, status_code=200)
async def process_pdf_to_vectors(
    file: UploadFile = File(...),
    create_vectors: bool = Form(True),
    description: Optional[str] = Form(None)
):
    """Upload a PDF file and convert it to vector data for Persian text processing.
    
    This endpoint accepts a PDF file upload, processes it using Persian-aware text processing,
    extracts tables, converts to markdown, and creates vector embeddings for search.
    
    Args:
        file: The PDF file to upload and process
        create_vectors: Whether to create vector embeddings (default: True)
        description: Optional description of the file
        
    Returns:
        PDFProcessingResponse with processing statistics and results
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content=PDFProcessingResponse(
                    status="error",
                    message="Only PDF files are supported for vector processing.",
                    filename=file.filename or "unknown"
                ).dict()
            )
        
        # Create temporary directory for processing
        temp_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) / "temp_processing"
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        temp_file_path = temp_dir / file.filename
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Check if embeddings are available
        if create_vectors and not processor.embeddings_available:
            return JSONResponse(
                status_code=503,
                content=PDFProcessingResponse(
                    status="error",
                    message="Vector embeddings service is not available. Please check OpenAI API configuration.",
                    filename=file.filename
                ).dict()
            )
        
        # Create output directory for processed files
        output_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) / "processed_pdfs"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Process the PDF using Persian-aware processing
            markdown_content = processor.load_persian_pdf_as_markdown(
                str(temp_file_path),
                str(output_dir)
            )
            
            # Extract tables from PDF
            tables = processor.extract_tables_from_pdf(str(temp_file_path))
            
            # Get page count
            import pymupdf
            doc = pymupdf.open(str(temp_file_path))
            total_pages = len(doc)
            doc.close()
            
            # Create vector embeddings if requested
            total_chunks = 0
            vector_store_updated = False
            
            if create_vectors:
                # Split text using Persian-aware splitter
                chunks = processor.persian_text_splitter.split_text(markdown_content)
                total_chunks = len(chunks)
                
                if chunks:
                    # Create documents for vector store
                    from langchain.schema import Document
                    import hashlib
                    from datetime import datetime
                    
                    # Generate file checksum
                    file_hash = hashlib.md5(temp_file_path.read_bytes()).hexdigest()
                    
                    documents = []
                    for i, chunk in enumerate(chunks):
                        # Check if chunk contains table
                        has_table = 'جدول' in chunk or '|' in chunk
                        
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                'source': str(temp_file_path),
                                'filename': file.filename,
                                'chunk_index': i,
                                'has_table': has_table,
                                'file_checksum': file_hash,
                                'processed_at': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                    
                    # Add to vector store
                    vector_store = processor.get_vector_store()
                    vector_store.add_documents(documents)
                    vector_store.persist()
                    vector_store_updated = True
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Clean up temporary file
            temp_file_path.unlink()
            
            return PDFProcessingResponse(
                status="success",
                message="PDF processed and converted to vector data successfully",
                filename=file.filename,
                total_pages=total_pages,
                total_tables=len(tables),
                total_chunks=total_chunks,
                processing_time=processing_time,
                vector_store_updated=vector_store_updated
            )
            
        except Exception as processing_error:
            # Clean up temporary file on error
            if temp_file_path.exists():
                temp_file_path.unlink()
            
            return JSONResponse(
                status_code=422,
                content=PDFProcessingResponse(
                    status="error",
                    message=f"PDF processing failed: {str(processing_error)}",
                    filename=file.filename,
                    processing_time=time.time() - start_time
                ).dict()
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=PDFProcessingResponse(
                status="error",
                message=f"Server error during PDF processing: {str(e)}",
                filename=file.filename or "unknown",
                processing_time=time.time() - start_time
            ).dict()
        )