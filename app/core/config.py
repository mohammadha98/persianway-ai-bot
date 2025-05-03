import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    # Project info
    PROJECT_NAME: str = "ML Model API Server"
    PROJECT_DESCRIPTION: str = "A FastAPI server for deploying machine learning models as APIs"
    VERSION: str = "0.1.0"
    
    # API settings
    API_PREFIX: str = "/api"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Model settings
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()