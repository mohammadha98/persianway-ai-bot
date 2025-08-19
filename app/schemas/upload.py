from pydantic import BaseModel, Field
from typing import Optional


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