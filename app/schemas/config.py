from pydantic import BaseModel, Field
from typing import List, Optional


class LLMSettings(BaseModel):
    """Schema for LLM configuration settings."""
    preferred_api_provider: str = Field(default="openrouter", description="Preferred API provider: auto, openai, openrouter")
    default_model: str = Field(default="deepseek/deepseek-chat-v3-0324:free", description="Default model to use")
    available_models: List[str] = Field(default_factory=list, description="List of available models")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default temperature setting")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Default top_p setting")
    max_tokens: int = Field(default=512, gt=0, description="Default maximum tokens")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_api_base: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", description="OpenAI embedding model")


class RAGSettings(BaseModel):
    """Schema for RAG (Retrieval-Augmented Generation) configuration settings."""
    knowledge_base_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence score for knowledge base answers")
    qa_match_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold for QA matches")
    qa_priority_factor: float = Field(default=1.2, ge=0.0, description="Weight for QA matches vs PDF content")
    human_referral_message: str = Field(
        default="متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد. سؤال شما برای بررسی بیشتر توسط کارشناسان ما ثبت شده است.",
        description="Message to show when human referral is needed"
    )
    excel_qa_path: str = Field(default="docs", description="Path to Excel QA files")
    # Additional RAG settings for vector store retrieval
    search_type: str = Field(default="mmr", description="Vector store search type (similarity, mmr, etc.)")
    top_k_results: int = Field(default=5, gt=0, description="Number of top results to retrieve from vector store")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for RAG responses")
    # New fields for improved retrieval
    similarity_threshold: float = Field(default=1.4, ge=0.0, description="Maximum distance score to consider a document relevant (lower is better for L2 distance)")
    mmr_diversity_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Balance between relevance and diversity in MMR (0=max diversity, 1=max relevance)")
    fetch_k_multiplier: int = Field(default=4, gt=0, description="Multiplier for initial fetch size before MMR filtering (fetch_k = top_k * multiplier)")
    original_query_weight: float = Field(default=1.0, ge=0.1, description="Weight for original query vs expanded queries")
    expanded_query_weight: float = Field(default=0.7, ge=0.1, description="Weight for expanded/rewritten queries")
    reranker_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for cosine similarity in reranker (1-alpha for normalized L2)")
    prompt_template: str = Field(
        default="""## دستورالعمل پاسخ‌دهی

### مرحله 1: بررسی تاریخچه گفتگو
اگر بخش "تاریخچه گفتگو" در اطلاعات مرجع وجود دارد:
- ابتدا تاریخچه را بخوانید تا سوال فعلی را در context گفتگو درک کنید
- اگر سوال به مطالب قبلی (مثل "سوال اولم چی بود؟" یا "دیگه چی؟") اشاره دارد، از تاریخچه استفاده کنید
- توجه: تاریخچه فقط برای درک context است، اطلاعات تخصصی را از پایگاه دانش بگیرید

### مرحله 2: پاسخ‌دهی
با استفاده از اطلاعات پایگاه دانش، به سوال پاسخ دهید.

⚠️ **نکته مهم**: اگر اطلاعات کافی برای پاسخ دقیق ندارید:
- در ابتدای پاسخ حتماً بنویسید: "متأسفانه اطلاعات کافی برای پاسخ دقیق به این سوال وجود ندارد."
- سپس توضیح مختصری بدهید که چرا نمی‌توانید پاسخ دهید

اطلاعات مرجع:
{context}

سوال: {question}

پاسخ:""",
        description="Template for RAG prompts"
    )
    system_prompt: str = Field(
        default="""شما یک دستیار هوشمند تخصصی در حوزه سلامت، زیبایی و کشاورزی هستید. وظیفه شما ارائه پاسخ‌های مفید و دقیق بر اساس دانش تخصصی شما است.

راهنمای پاسخ‌دهی:
1. اگر سوال در حوزه تخصص شما (سلامت، زیبایی، کشاورزی) است، پاسخ کاملی ارائه دهید
2. اگر سوال خارج از حوزه تخصص شما است (فناوری، مالی، سیاسی و غیره)، به کاربر بگویید که این سوال نیاز به بررسی توسط کارشناس دارد
3. همیشه پاسخ‌های خود را به زبان فارسی و با لحنی دوستانه و حرفه‌ای ارائه دهید
4. در صورت عدم اطمینان از صحت پاسخ، صادقانه اعلام کنید که نیاز به مشورت کارشناس است

You are a specialized AI assistant in health, beauty, and agriculture domains. Your role is to provide helpful and accurate answers based on your specialized knowledge.

Response Guidelines:
1. If the question is within your expertise (health, beauty, agriculture), provide a complete answer
2. If the question is outside your expertise (technology, finance, politics, etc.), tell the user that this question requires specialist review
3. Always respond in Persian with a friendly and professional tone
4. If uncertain about answer accuracy, honestly state that specialist consultation is needed""".strip(),
        description="System prompt for the AI assistant"
    )


class DatabaseSettings(BaseModel):
    """Schema for database configuration settings."""
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URL")
    mongodb_database: str = Field(default="persian_way_ai_db", description="MongoDB database name")
    mongodb_conversations_collection: str = Field(default="conversations", description="Conversations collection name")
    conversation_ttl_days: int = Field(default=365, gt=0, description="Conversation TTL in days")


class AppSettings(BaseModel):
    """Schema for general application settings."""
    project_name: str = Field(default="AI Models API Server", description="Project name")
    project_description: str = Field(default="A FastAPI server for deploying machine learning models as APIs", description="Project description")
    version: str = Field(default="0.1.0", description="Application version")
    api_prefix: str = Field(default="/api", description="API prefix")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, gt=0, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"], description="Allowed hosts")


class DynamicConfig(BaseModel):
    """Schema for the complete dynamic configuration."""
    id: Optional[str] = Field(default=None, description="Configuration ID")
    llm_settings: LLMSettings = Field(default_factory=LLMSettings, description="LLM configuration")
    rag_settings: RAGSettings = Field(default_factory=RAGSettings, description="RAG configuration")
    database_settings: DatabaseSettings = Field(default_factory=DatabaseSettings, description="Database configuration")
    app_settings: AppSettings = Field(default_factory=AppSettings, description="Application configuration")
    is_active: bool = Field(default=True, description="Whether this configuration is active")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "llm_settings": {
                    "preferred_api_provider": "openrouter",
                    "default_model": "deepseek/deepseek-chat-v3-0324:free",
                    "temperature": 0.7,
                    "max_tokens": 512
                },
                "rag_settings": {
                    "knowledge_base_confidence_threshold": 0.5,
                    "qa_match_threshold": 0.8,
                    "human_referral_message": "متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد."
                },
                "database_settings": {
                    "mongodb_url": "mongodb://localhost:27017",
                    "mongodb_database": "persian_way_ai_db"
                },
                "app_settings": {
                    "project_name": "AI Models API Server",
                    "debug": False
                }
            }
        }
    }


class ConfigUpdateRequest(BaseModel):
    """Schema for configuration update requests."""
    llm_settings: Optional[LLMSettings] = Field(default=None, description="LLM settings to update")
    rag_settings: Optional[RAGSettings] = Field(default=None, description="RAG settings to update")
    database_settings: Optional[DatabaseSettings] = Field(default=None, description="Database settings to update")
    app_settings: Optional[AppSettings] = Field(default=None, description="App settings to update")


class ConfigResponse(BaseModel):
    """Schema for configuration response."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    config: Optional[DynamicConfig] = Field(default=None, description="Configuration data")
