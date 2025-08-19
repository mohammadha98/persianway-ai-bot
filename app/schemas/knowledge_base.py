from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class KnowledgeBaseQuery(BaseModel):
    """Schema for knowledge base query request.
    
    This defines the expected input format for the knowledge base query API.
    """
    question: str = Field(..., description="The question to ask the knowledge base")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "انواع کودهای شیمیایی کدامند؟"
            }
        }
    }


class KnowledgeContributionItem(BaseModel):
    """Schema for a single knowledge contribution item, used in the response."""
    id: str = Field(..., description="Unique identifier for the contribution")
    title: str = Field(..., description="Title of the entry")
    submitted_at: str = Field(..., description="Timestamp of submission")
    meta_tags: List[str] = Field(..., description="Keywords for categorization")
    source: str = Field(..., description="The origin or reference for the knowledge")
    author_name: Optional[str] = Field(None, description="Name of the contributor")
    additional_references: Optional[str] = Field(None, description="URLs or citation text for further reading")
    file_processed: Optional[bool] = Field(None, description="Indicates if a file was processed")
    file_type: Optional[str] = Field(None, description="Type of file processed (pdf or excel)")
    file_name: Optional[str] = Field(None, description="Name of the processed file")
    qa_count: Optional[int] = Field(None, description="Number of QA pairs processed from Excel file")


class KnowledgeContributionResponse(BaseModel):
    """Schema for the knowledge contribution API response."""
    success: bool = Field(..., description="Indicates if the contribution was successful")
    contribution: Optional[KnowledgeContributionItem] = Field(None, description="Details of the submitted contribution")
    message: Optional[str] = Field(None, description="Error message in case of failure")

    model_config = {
        "json_schema_extra": {
            "example_success": {
                "success": True,
                "contribution": {
                    "id": "4dfaaf98-4036-4e8a-9235-18f915d21a24",
                    "title": "Properties of Loamy Soil",
                    "submitted_at": "2024-06-01T14:10:22+03:30",
                    "meta_tags": ["loam", "soil", "agriculture"],
                    "source": "Expert observation",
                    "author_name": "Dr. KhakShenas",
                    "additional_references": "https://example.com/loamy-soil-guide",
                    "file_processed": True,
                    "file_type": "pdf",
                    "file_name": "soil_analysis.pdf",
                    "qa_count": None
                }
            },
            "example_failure": {
                "success": False,
                "message": "Invalid content_type provided."
            }
        }
    }


class DocumentSource(BaseModel):
    """Schema for a document source."""
    content: str = Field(..., description="The content of the document")
    source: str = Field(..., description="The source of the document")
    page: int = Field(..., description="The page number of the document")
    source_type: str = Field("pdf", description="The type of source (pdf or excel_qa)")
    title: Optional[str] = Field(None, description="The title of the QA pair (for excel_qa sources)")


class KnowledgeBaseResponse(BaseModel):
    """Schema for knowledge base response data.
    
    This defines the expected output format from the knowledge base query API.
    """
    answer: str = Field(..., description="The answer to the question")
    sources: List[DocumentSource] = Field(..., description="The sources used to generate the answer")
    requires_human_support: bool = Field(False, description="Indicates if the query requires human expert attention")
    query_id: Optional[str] = Field(None, description="Unique identifier for the query when human support is required")
    source_type: str = Field("pdf", description="The primary source type for this answer (pdf or excel_qa)")
    confidence_score: float = Field(0.0, description="Confidence score of the answer between 0 and 1")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "انواع کودهای شیمیایی عبارتند از: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
                "sources": [
                    {
                        "content": "کودهای شیمیایی به چند دسته تقسیم می‌شوند: کودهای نیتروژنی، کودهای فسفاته، کودهای پتاسیمی، و کودهای میکرو.",
                        "source": "fertilization-guide-table.pdf",
                        "page": 5
                    }
                ],
                "confidence_score": 0.85,
                "requires_human_support": False,
                "query_id": None
            }
        }
    }


class KnowledgeContributionItem(BaseModel):
    """Schema for a single knowledge contribution item, used in the response."""
    id: str = Field(..., description="Unique identifier for the contribution")
    title: str = Field(..., description="Title of the entry")
    submitted_at: str = Field(..., description="Timestamp of submission")
    meta_tags: List[str] = Field(..., description="Keywords for categorization")
    source: str = Field(..., description="The origin or reference for the knowledge")
    author_name: Optional[str] = Field(None, description="Name of the contributor")
    additional_references: Optional[str] = Field(None, description="URLs or citation text for further reading")


class KnowledgeContributionResponse(BaseModel):
    """Schema for the knowledge contribution API response."""
    success: bool = Field(..., description="Indicates if the contribution was successful")
    contribution: Optional[KnowledgeContributionItem] = Field(None, description="Details of the submitted contribution")
    message: Optional[str] = Field(None, description="Error message in case of failure")

    model_config = {
        "json_schema_extra": {
            "example_success": {
                "success": True,
                "contribution": {
                    "id": "4dfaaf98-4036-4e8a-9235-18f915d21a24",
                    "title": "Properties of Loamy Soil",
                    "submitted_at": "2024-06-01T14:10:22+03:30",
                    "meta_tags": ["loam", "soil", "agriculture"],
                    "source": "Expert observation",
                    "author_name": "Dr. KhakShenas",
                    "additional_references": "https://example.com/loamy-soil-guide"
                }
            },
            "example_failure": {
                "success": False,
                "message": "Invalid content_type provided."
            }
        }
    }


class ProcessDocsResponse(BaseModel):
    """Schema for document processing response."""
    message: str = Field(..., description="A message about the document processing")
    status: str = Field(..., description="The status of the document processing")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Document processing started in the background",
                "status": "processing"
            }
        }
    }


class KnowledgeContributionItem(BaseModel):
    """Schema for a single knowledge contribution item, used in the response."""
    id: str = Field(..., description="Unique identifier for the contribution")
    title: str = Field(..., description="Title of the entry")
    submitted_at: str = Field(..., description="Timestamp of submission")
    meta_tags: List[str] = Field(..., description="Keywords for categorization")
    source: str = Field(..., description="The origin or reference for the knowledge")
    author_name: Optional[str] = Field(None, description="Name of the contributor")
    additional_references: Optional[str] = Field(None, description="URLs or citation text for further reading")


class KnowledgeContributionResponse(BaseModel):
    """Schema for the knowledge contribution API response."""
    success: bool = Field(..., description="Indicates if the contribution was successful")
    contribution: Optional[KnowledgeContributionItem] = Field(None, description="Details of the submitted contribution")
    message: Optional[str] = Field(None, description="Error message in case of failure")

    model_config = {
        "json_schema_extra": {
            "example_success": {
                "success": True,
                "contribution": {
                    "id": "4dfaaf98-4036-4e8a-9235-18f915d21a24",
                    "title": "Properties of Loamy Soil",
                    "submitted_at": "2024-06-01T14:10:22+03:30",
                    "meta_tags": ["loam", "soil", "agriculture"],
                    "source": "Expert observation",
                    "author_name": "Dr. KhakShenas",
                    "additional_references": "https://example.com/loamy-soil-guide"
                }
            },
            "example_failure": {
                "success": False,
                "message": "Invalid content_type provided."
            }
        }
    }