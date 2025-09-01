import os
from typing import Optional, List
from pydantic_settings import BaseSettings

# Determine the project root directory and construct the absolute path to the .env file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")

class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # LLM Configuration
    PREFERRED_API_PROVIDER: str = "openrouter"
    DEFAULT_MODEL: str = "deepseek/deepseek-chat-v3-0324:free"
    AVAILABLE_MODELS: List[str] = []
    TEMPERATURE: float = 0.7
    TOP_P: float = 1.0
    MAX_TOKENS: int = 512
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # RAG Configuration
    KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD: float = 0.5
    QA_MATCH_THRESHOLD: float = 0.8
    QA_PRIORITY_FACTOR: float = 1.2
    HUMAN_REFERRAL_MESSAGE: str = "متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد. سؤال شما برای بررسی بیشتر توسط کارشناسان ما ثبت شده است."
    EXCEL_QA_PATH: str = "docs"
    SYSTEM_PROMPT: str = "شما یک دستیار هوشمند تخصصی در حوزه سلامت، زیبایی و کشاورزی هستید."
    
    # Database Configuration
    database_url: str = "sqlite:///./persian_way_ai.db"
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "persian_way_ai"
    MONGODB_CONVERSATIONS_COLLECTION: str = "conversations"
    CONVERSATION_TTL_DAYS: int = 30
    
    # Application Configuration
    app_name: str = "Persian Way AI Bot"
    PROJECT_NAME: str = "Persian Way AI Bot"
    PROJECT_DESCRIPTION: str = "Persian Way AI Consultation Bot"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Security
    secret_key: str = "your-secret-key-here"
    
    # CORS Configuration
    allowed_origins: list = ["*"]
    ALLOWED_HOSTS: list = ["*"]
    
    model_config = {
        "env_file": ENV_FILE_PATH,
        "case_sensitive": False,
        "extra": "allow"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Create global settings instance
settings = Settings()