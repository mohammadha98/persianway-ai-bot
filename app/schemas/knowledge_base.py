from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class KnowledgeBaseQuery(BaseModel):
    """Schema for knowledge base query request.
    
    This defines the expected input format for the knowledge base query API.
    """
    question: str = Field(..., description="The question to ask the knowledge base")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "انواع کودهای شیمیایی کدامند؟"
            }
        }


class DocumentSource(BaseModel):
    """Schema for a document source."""
    content: str = Field(..., description="The content of the document")
    source: str = Field(..., description="The source of the document")
    page: int = Field(..., description="The page number of the document")


class KnowledgeBaseResponse(BaseModel):
    """Schema for knowledge base response data.
    
    This defines the expected output format from the knowledge base query API.
    """
    answer: str = Field(..., description="The answer to the question")
    sources: List[DocumentSource] = Field(..., description="The sources used to generate the answer")
    requires_human_support: bool = Field(False, description="Indicates if the query requires human expert attention")
    query_id: Optional[str] = Field(None, description="Unique identifier for the query when human support is required")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "انواع کودهای شیمیایی عبارتند از: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
                "sources": [
                    {
                        "content": "کودهای شیمیایی به چند دسته تقسیم می‌شوند: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
                        "source": "fertilization-guide-table.pdf",
                        "page": 5
                    }
                ],
                "requires_human_support": False,
                "query_id": None
            }
        }


class ProcessDocsResponse(BaseModel):
    """Schema for document processing response."""
    message: str = Field(..., description="A message about the document processing")
    status: str = Field(..., description="The status of the document processing")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Document processing started in the background",
                "status": "processing"
            }
        }