from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class FileUploadResponse(BaseModel):
    """Schema for file upload response.
    
    This defines the expected output format for the file upload API.
    """
    status: str = Field(..., description="Status of the upload operation (success or error)")
    message: str = Field(..., description="Message describing the result of the operation")
    filename: Optional[str] = Field(None, description="The unique filename of the uploaded file")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "message": "File uploaded successfully",
                "filename": "550e8400-e29b-41d4-a716-446655440000_example.pdf"
            }
        }
    }


class PDFProcessingResponse(BaseModel):
    """Schema for PDF processing and vector conversion response."""
    status: str = Field(..., description="Status of the processing operation (success or error)")
    message: str = Field(..., description="Message describing the result of the operation")
    filename: str = Field(..., description="The filename of the processed PDF")
    processing_stats: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")
    total_pages: Optional[int] = Field(None, description="Total number of pages processed")
    total_tables: Optional[int] = Field(None, description="Total number of tables extracted")
    total_chunks: Optional[int] = Field(None, description="Total number of text chunks created")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    vector_store_updated: Optional[bool] = Field(None, description="Whether vector store was updated")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "message": "PDF processed and converted to vector data successfully",
                "filename": "example.pdf",
                "total_pages": 10,
                "total_tables": 3,
                "total_chunks": 25,
                "processing_time": 15.5,
                "vector_store_updated": True
            }
        }
    }