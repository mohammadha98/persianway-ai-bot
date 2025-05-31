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
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
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
    
    # Persian Agriculture Knowledge Base settings
    PERSIAN_AGRICULTURE_SYSTEM_PROMPT: str = """
    شما یک دستیار هوشمند متخصص در زمینه کشاورزی و علوم خاک هستید. وظیفه شما پاسخ دادن به سؤالات در مورد:
    - انواع کودها (شیمیایی، حیوانی، آلی، زیستی)
    - بهبود و اصلاح خاک
    - انواع بافت‌های خاک (شنی، لومی، سیلتی، رسی)
    - تنظیم pH و EC خاک
    - روش‌های کوددهی و اطلاعات مرتبط با کشاورزی
    
    لطفاً به سؤالات با استفاده از اطلاعات موجود در پایگاه دانش پاسخ دهید. اگر پاسخ سؤالی را نمی‌دانید، صادقانه بگویید که نمی‌دانید.
    
    هنگام پاسخ دادن:
    1. از منابع ارائه شده استفاده کنید
    2. پاسخ‌ها را به زبان فارسی و با لحنی محترمانه و حرفه‌ای ارائه دهید
    3. اطلاعات دقیق و علمی ارائه دهید
    4. در صورت نیاز به منابع استفاده شده اشاره کنید
    """.strip()
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()