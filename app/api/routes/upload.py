from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import os
import uuid
import shutil
from fastapi.responses import JSONResponse

from app.schemas.upload import FileUploadResponse

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