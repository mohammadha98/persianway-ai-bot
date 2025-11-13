from typing import Optional, Dict, Any
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection

from app.schemas.config import DynamicConfig, LLMSettings, RAGSettings, DatabaseSettings, AppSettings
from app.utils.validators import validate_rag_settings
from app.core.config import settings as static_settings
from app.services.database import get_database_service

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for managing dynamic configuration with database storage and fallback to static settings."""
    
    def __init__(self):
        self._cached_config: Optional[DynamicConfig] = None
        self._config_collection: Optional[AsyncIOMotorCollection] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the configuration service."""
        db_service = await get_database_service()
        database = db_service.get_database()
        self._config_collection = database["config"]
        
        # Create index for faster queries
        await self._config_collection.create_index("is_active")
        await self._config_collection.create_index("updated_at")
        
        # Load initial configuration
        await self._load_config()
        self._initialized = True
        logger.info("Configuration service initialized successfully")
    
    async def _load_config(self):
        """Load configuration from database or create default if not exists."""
        if self._config_collection is None:
            logger.error("Configuration collection is None, reinitializing")
            db_service = await get_database_service()
            database = db_service.get_database()
            self._config_collection = database["config"]
        
        # Try to find active configuration
        config_doc = await self._config_collection.find_one({"is_active": True})
        
        if config_doc:
            # Remove MongoDB _id field
            config_doc.pop('_id', None)
            self._cached_config = DynamicConfig(**config_doc)
            logger.info("Loaded configuration from database")
        else:
            # Create default configuration from static settings
            default_config = self._create_fallback_config()
            await self._save_config(default_config)
            self._cached_config = default_config
            logger.info("Created default configuration in database")
    
    def _create_fallback_config(self) -> DynamicConfig:
        """Create configuration from static settings as fallback."""
        return DynamicConfig(
            llm_settings=LLMSettings(
                preferred_api_provider=static_settings.PREFERRED_API_PROVIDER,
                default_model=static_settings.DEFAULT_MODEL,
                available_models=static_settings.AVAILABLE_MODELS,
                temperature=static_settings.TEMPERATURE,
                top_p=static_settings.TOP_P,
                max_tokens=static_settings.MAX_TOKENS,
                openai_api_key=static_settings.OPENAI_API_KEY,
                openrouter_api_key=static_settings.OPENROUTER_API_KEY,
                openrouter_api_base=static_settings.OPENROUTER_API_BASE,
                openai_embedding_model=static_settings.OPENAI_EMBEDDING_MODEL
            ),
            rag_settings=RAGSettings(
                knowledge_base_confidence_threshold=static_settings.KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD,
                qa_match_threshold=static_settings.QA_MATCH_THRESHOLD,
                qa_priority_factor=static_settings.QA_PRIORITY_FACTOR,
                human_referral_message=static_settings.HUMAN_REFERRAL_MESSAGE,
                excel_qa_path=static_settings.EXCEL_QA_PATH,
                system_prompt=static_settings.SYSTEM_PROMPT
            ),
            database_settings=DatabaseSettings(
                mongodb_url=static_settings.MONGODB_URL,
                mongodb_database=static_settings.MONGODB_DATABASE,
                mongodb_conversations_collection=static_settings.MONGODB_CONVERSATIONS_COLLECTION,
                conversation_ttl_days=static_settings.CONVERSATION_TTL_DAYS
            ),
            app_settings=AppSettings(
                project_name=static_settings.PROJECT_NAME,
                project_description=static_settings.PROJECT_DESCRIPTION,
                version=static_settings.VERSION,
                api_prefix=static_settings.API_PREFIX,
                host=static_settings.HOST,
                port=static_settings.PORT,
                debug=static_settings.DEBUG,
                allowed_hosts=static_settings.ALLOWED_HOSTS
            ),
            is_active=True,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
    
    async def get_config(self) -> DynamicConfig:
        """Get the current configuration."""
        if not self._initialized:
            await self.initialize()
        
        if self._cached_config is None:
            await self._load_config()
        
        return self._cached_config
    
    async def update_config(self, updates: Dict[str, Any]) -> DynamicConfig:
        """Update configuration with new values."""
        if not self._initialized:
            await self.initialize()
        
        current_config = await self.get_config()
        
        # Create updated configuration
        config_dict = current_config.dict()
        
        # Apply updates
        for section, values in updates.items():
            if section in config_dict and isinstance(values, dict):
                config_dict[section].update(values)
        
        # Update timestamp
        config_dict['updated_at'] = datetime.utcnow().isoformat()
        
        # Create new config object
        updated_config = DynamicConfig(**config_dict)
        
        # Save to database
        await self._save_config(updated_config)
        
        # Update cache
        self._cached_config = updated_config
        
        logger.info("Configuration updated successfully")
        return updated_config
    
    async def _save_config(self, config: DynamicConfig):
        """Save configuration to database."""
        if self._config_collection is None:
            logger.warning("Cannot save config: database not available")
            return
        
        config_dict = config.dict()
        config_dict['is_active'] = True
        
        # Update existing active config or insert if none exists
        result = await self._config_collection.replace_one(
            {"is_active": True},
            config_dict,
            upsert=True
        )
        
        logger.info("Configuration saved to database")
    
    async def reset_to_defaults(self) -> DynamicConfig:
        """Reset configuration to default values from static settings."""
        default_config = self._create_fallback_config()
        await self._save_config(default_config)
        self._cached_config = default_config
        logger.info("Configuration reset to defaults")
        return default_config
    
    async def get_llm_settings(self) -> LLMSettings:
        """Get LLM settings with fallback."""
        config = await self.get_config()
        return config.llm_settings
    
    async def get_rag_settings(self) -> RAGSettings:
        config = await self.get_config()
        validated = validate_rag_settings(config.rag_settings.dict())
        return RAGSettings(**validated)
    
    async def get_database_settings(self) -> DatabaseSettings:
        """Get database settings with fallback."""
        config = await self.get_config()
        return config.database_settings
    
    async def get_app_settings(self) -> AppSettings:
        """Get application settings with fallback."""
        config = await self.get_config()
        return config.app_settings


# Global configuration service instance
_config_service: Optional[ConfigService] = None


async def get_config_service() -> ConfigService:
    """Get the configuration service instance."""
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
        await _config_service.initialize()
    elif not _config_service._initialized:
        await _config_service.initialize()
    return _config_service


async def get_dynamic_llm_settings() -> LLMSettings:
    """Get dynamic LLM settings."""
    config_service = await get_config_service()
    return await config_service.get_llm_settings()


async def get_dynamic_rag_settings() -> RAGSettings:
    """Get dynamic RAG settings."""
    config_service = await get_config_service()
    return await config_service.get_rag_settings()


async def get_dynamic_database_settings() -> DatabaseSettings:
    """Get dynamic database settings."""
    config_service = await get_config_service()
    return await config_service.get_database_settings()


async def get_dynamic_app_settings() -> AppSettings:
    """Get dynamic application settings."""
    config_service = await get_config_service()
    return await config_service.get_app_settings()
