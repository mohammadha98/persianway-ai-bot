import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Database Configuration
    database_url: str = "sqlite:///./persian_way_ai.db"
    
    # Application Configuration
    app_name: str = "Persian Way AI Bot"
    debug: bool = False
    
    # Security
    secret_key: str = "your-secret-key-here"
    
    # CORS Configuration
    allowed_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load API keys from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Create global settings instance
settings = Settings()