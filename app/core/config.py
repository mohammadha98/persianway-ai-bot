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
    
    # Model settings
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 500
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()