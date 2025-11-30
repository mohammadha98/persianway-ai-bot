from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.schemas.config import DynamicConfig, ConfigUpdateRequest, ConfigResponse
from app.services.config_service import get_config_service, ConfigService
from app.services.knowledge_base import get_knowledge_base_service, KnowledgeBaseService
from app.services.chat_service import get_chat_service, ChatService

# Create router for configuration endpoints
router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("/", response_model=ConfigResponse)
async def get_configuration(config_service: ConfigService = Depends(get_config_service)):
    """Get the current dynamic configuration.
    
    Returns the current configuration settings including LLM, RAG, database, and app settings.
    If database is not available, returns the fallback static configuration.
    """
    try:
        config = await config_service.get_config()
        return ConfigResponse(
            success=True,
            message="Configuration retrieved successfully",
            config=config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve configuration: {str(e)}")


@router.put("/", response_model=ConfigResponse)
async def update_configuration(
    request: ConfigUpdateRequest,
    config_service: ConfigService = Depends(get_config_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    chat_service: ChatService = Depends(get_chat_service)
):
    """Update the dynamic configuration.
    
    Updates the configuration with the provided values. Only the specified sections
    will be updated, leaving other sections unchanged.
    """
    try:
        # Convert request to dictionary, excluding None values
        updates = {}
        
        if request.llm_settings is not None:
            updates["llm_settings"] = request.llm_settings.dict(exclude_none=True)
        
        if request.rag_settings is not None:
            updates["rag_settings"] = request.rag_settings.dict(exclude_none=True)
        
        if request.database_settings is not None:
            updates["database_settings"] = request.database_settings.dict(exclude_none=True)
        
        if request.app_settings is not None:
            updates["app_settings"] = request.app_settings.dict(exclude_none=True)
        
        if not updates:
            return ConfigResponse(
                success=False,
                message="No valid updates provided",
                config=None
            )
        
        updated_config = await config_service.update_config(updates)
        await kb_service.refresh()
        await chat_service.refresh()
        return ConfigResponse(
            success=True,
            message="Configuration updated successfully",
            config=updated_config
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.post("/reset", response_model=ConfigResponse)
async def reset_configuration(config_service: ConfigService = Depends(get_config_service)):
    """Reset configuration to default values.
    
    Resets all configuration settings to their default values from the static configuration.
    """
    try:
        default_config = await config_service.reset_to_defaults()
        
        return ConfigResponse(
            success=True,
            message="Configuration reset to defaults successfully",
            config=default_config
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset configuration: {str(e)}")


@router.get("/llm", response_model=Dict[str, Any])
async def get_llm_settings(config_service: ConfigService = Depends(get_config_service)):
    """Get current LLM settings."""
    try:
        llm_settings = await config_service.get_llm_settings()
        return {
            "success": True,
            "message": "LLM settings retrieved successfully",
            "settings": llm_settings.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve LLM settings: {str(e)}")


@router.get("/rag", response_model=Dict[str, Any])
async def get_rag_settings(config_service: ConfigService = Depends(get_config_service)):
    """Get current RAG settings."""
    try:
        rag_settings = await config_service.get_rag_settings()

        return {
            "success": True,
            "message": "RAG settings retrieved successfully",
            "settings": rag_settings.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve RAG settings: {str(e)}")


@router.get("/database", response_model=Dict[str, Any])
async def get_database_settings(config_service: ConfigService = Depends(get_config_service)):
    """Get current database settings."""
    try:
        db_settings = await config_service.get_database_settings()
        return {
            "success": True,
            "message": "Database settings retrieved successfully",
            "settings": db_settings.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve database settings: {str(e)}")


@router.get("/app", response_model=Dict[str, Any])
async def get_app_settings(config_service: ConfigService = Depends(get_config_service)):
    """Get current application settings."""
    try:
        app_settings = await config_service.get_app_settings()
        return {
            "success": True,
            "message": "Application settings retrieved successfully",
            "settings": app_settings.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve application settings: {str(e)}")


@router.put("/llm", response_model=Dict[str, Any])
async def update_llm_settings(
    settings: Dict[str, Any],
    config_service: ConfigService = Depends(get_config_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Update LLM settings only."""
    try:
        updates = {"llm_settings": settings}
        updated_config = await config_service.update_config(updates)
        await kb_service.refresh()
        await chat_service.refresh()
        return {
            "success": True,
            "message": "LLM settings updated successfully",
            "settings": updated_config.llm_settings.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update LLM settings: {str(e)}")


@router.put("/rag", response_model=Dict[str, Any])
async def update_rag_settings(
    settings: Dict[str, Any],
    config_service: ConfigService = Depends(get_config_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Update RAG settings only."""
    try:
        updates = {"rag_settings": settings}
        updated_config = await config_service.update_config(updates)
        await kb_service.refresh()
        await chat_service.refresh()
        return {
            "success": True,
            "message": "RAG settings updated successfully",
            "settings": updated_config.rag_settings.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update RAG settings: {str(e)}")
