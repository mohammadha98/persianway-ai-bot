import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    # Project info
    PROJECT_NAME: str = "AI Models API Server"
    PROJECT_DESCRIPTION: str = "A FastAPI server for deploying machine learning models as APIs"
    VERSION: str = "0.1.0"
    
    # API settings
    API_PREFIX: str = "/api"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model settings
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4.1"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 500
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Human referral settings
    KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence score to consider an answer adequate
    HUMAN_REFERRAL_MESSAGE: str = "متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد. سؤال شما برای بررسی بیشتر توسط کارشناسان ما ثبت شده است."
    
    # Excel QA settings
    EXCEL_QA_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
    QA_MATCH_THRESHOLD: float = 0.8  # Confidence threshold for considering a QA match sufficient
    QA_PRIORITY_FACTOR: float = 1.2  # Weight to assign to QA matches versus PDF content matches
    
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "persian_way_ai_db"
    MONGODB_CONVERSATIONS_COLLECTION: str = "conversations"
    CONVERSATION_TTL_DAYS: int = 365  # Time-to-live for conversations in days
    
    # Knowledge Base settings
    SYSTEM_PROMPT: str = """شما یک دستیار هوشمند تخصصی در حوزه سلامت، زیبایی و کشاورزی هستید. وظیفه شما ارائه پاسخ‌های مفید و دقیق بر اساس دانش تخصصی شما است.

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
4. If uncertain about answer accuracy, honestly state that specialist consultation is needed
    """.strip()
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()