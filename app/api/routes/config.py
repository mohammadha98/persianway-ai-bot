from typing import Any, Dict, List
import time

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import ValidationError

from app.core.config import settings as static_settings
from app.schemas.config import (
    ConfigResponse,
    ConfigUpdateRequest,
    TavilySearchSettings,
)
from app.services.chat_service import ChatService, get_chat_service
from app.services.config_service import ConfigService, get_config_service
from app.services.knowledge_base import KnowledgeBaseService, get_knowledge_base_service
from app.services.utility import search_persianway

# Create router for configuration endpoints
router = APIRouter(prefix="/config", tags=["configuration"])


def _set_no_cache_headers(response: Response) -> None:
    """Prevent browsers/proxies/CDNs from serving stale config responses."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"


# Tavily only accepts these two depths; the panel offers exactly these options.
TAVILY_SEARCH_DEPTHS = ("basic", "advanced")


def _normalize_domain_list(value: Any, field_name: str) -> List[str]:
    """Coerce a domain list (list or comma separated string) into clean hostnames.

    The panel sends whatever the operator typed, so `https://www.persianway.ir/x`,
    `PersianWay.IR` and `persianway.ir,persianway.ir` all have to end up as the
    single hostname Tavily can be restricted to.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, (list, tuple)):
        candidates = [str(item) for item in value]
    else:
        raise HTTPException(
            status_code=400, detail=f"{field_name} must be a list of domains"
        )

    normalized: List[str] = []
    for candidate in candidates:
        domain = candidate.strip().lower()
        if not domain:
            continue
        domain = domain.removeprefix("http://").removeprefix("https://")
        domain = domain.split("/")[0].split("?")[0].strip()
        if domain.startswith("*."):
            domain = domain[2:]
        if domain and domain not in normalized:
            normalized.append(domain)
    return normalized


def _validate_tavily_updates(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the known Tavily fields, normalise the domains and validate the ranges.

    Only the fields present in the request are returned, so the panel can save a
    single toggle without posting the whole document, and a misspelled field can
    never reach the stored configuration.
    """
    allowed = set(TavilySearchSettings.model_fields.keys())
    updates = {key: value for key, value in payload.items() if key in allowed}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid Tavily settings provided")

    for field_name in ("include_domains", "exclude_domains"):
        if field_name in updates:
            updates[field_name] = _normalize_domain_list(updates[field_name], field_name)

    if "search_depth" in updates:
        depth = str(updates["search_depth"]).strip().lower()
        if depth not in TAVILY_SEARCH_DEPTHS:
            raise HTTPException(
                status_code=400,
                detail=f"search_depth must be one of {list(TAVILY_SEARCH_DEPTHS)}",
            )
        updates["search_depth"] = depth

    # Reuse the schema for the numeric bounds (`max_results`, `snippet_length`).
    try:
        TavilySearchSettings(**updates)
    except ValidationError as e:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'settings'}: {err['msg']}"
            for err in e.errors()
        )
        raise HTTPException(status_code=400, detail=f"Invalid Tavily settings - {detail}")

    return updates


@router.get("/", response_model=ConfigResponse)
async def get_configuration(
    config_service: ConfigService = Depends(get_config_service),
):
    """Get the current dynamic configuration.

    Returns the current configuration settings including LLM, RAG, database, and app settings.
    If database is not available, returns the fallback static configuration.
    """
    try:
        config = await config_service.get_config()
        return ConfigResponse(
            success=True, message="Configuration retrieved successfully", config=config
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve configuration: {str(e)}"
        )


@router.put("/", response_model=ConfigResponse)
async def update_configuration(
    request: ConfigUpdateRequest,
    config_service: ConfigService = Depends(get_config_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    chat_service: ChatService = Depends(get_chat_service),
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
            updates["database_settings"] = request.database_settings.dict(
                exclude_none=True
            )

        if request.app_settings is not None:
            updates["app_settings"] = request.app_settings.dict(exclude_none=True)

        if request.tavily_settings is not None:
            updates["tavily_settings"] = request.tavily_settings.dict(exclude_none=True)

        if not updates:
            return ConfigResponse(
                success=False, message="No valid updates provided", config=None
            )

        updated_config = await config_service.update_config(updates)
        await kb_service.refresh()
        await chat_service.refresh()
        return ConfigResponse(
            success=True,
            message="Configuration updated successfully",
            config=updated_config,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {str(e)}"
        )


@router.post("/reset", response_model=ConfigResponse)
async def reset_configuration(
    config_service: ConfigService = Depends(get_config_service),
):
    """Reset configuration to default values.

    Resets all configuration settings to their default values from the static configuration.
    """
    try:
        default_config = await config_service.reset_to_defaults()

        return ConfigResponse(
            success=True,
            message="Configuration reset to defaults successfully",
            config=default_config,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset configuration: {str(e)}"
        )


@router.get("/llm", response_model=Dict[str, Any])
async def get_llm_settings(
    response: Response, config_service: ConfigService = Depends(get_config_service)
):
    """Get current LLM settings."""
    try:
        _set_no_cache_headers(response)
        # Force a DB refresh to avoid stale in-process cache on hosted/multi-instance deployments.
        await config_service.clear_cache()
        llm_settings = await config_service.get_llm_settings()
        return {
            "success": True,
            "message": "LLM settings retrieved successfully",
            "settings": llm_settings.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve LLM settings: {str(e)}"
        )


@router.get("/rag", response_model=Dict[str, Any])
async def get_rag_settings(
    response: Response, config_service: ConfigService = Depends(get_config_service)
):
    """Get current RAG settings."""
    try:
        _set_no_cache_headers(response)
        # Force a DB refresh to avoid stale in-process cache on hosted/multi-instance deployments.
        await config_service.clear_cache()
        rag_settings = await config_service.get_rag_settings()

        return {
            "success": True,
            "message": "RAG settings retrieved successfully",
            "settings": rag_settings.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve RAG settings: {str(e)}"
        )


@router.get("/database", response_model=Dict[str, Any])
async def get_database_settings(
    config_service: ConfigService = Depends(get_config_service),
):
    """Get current database settings."""
    try:
        db_settings = await config_service.get_database_settings()
        return {
            "success": True,
            "message": "Database settings retrieved successfully",
            "settings": db_settings.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve database settings: {str(e)}"
        )


@router.get("/app", response_model=Dict[str, Any])
async def get_app_settings(config_service: ConfigService = Depends(get_config_service)):
    """Get current application settings."""
    try:
        app_settings = await config_service.get_app_settings()
        return {
            "success": True,
            "message": "Application settings retrieved successfully",
            "settings": app_settings.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve application settings: {str(e)}"
        )


@router.get("/tavily", response_model=Dict[str, Any])
async def get_tavily_settings(
    response: Response, config_service: ConfigService = Depends(get_config_service)
):
    """Get current Tavily settings."""
    try:
        _set_no_cache_headers(response)
        # Force a DB refresh to avoid stale in-process cache on hosted/multi-instance deployments.
        await config_service.clear_cache()
        config = await config_service.get_config()
        tavily_settings = config.tavily_settings
        return {
            "success": True,
            "message": "Tavily settings retrieved successfully",
            "settings": tavily_settings.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve Tavily settings: {str(e)}"
        )


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
            "settings": updated_config.llm_settings.dict(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update LLM settings: {str(e)}"
        )


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
            "settings": updated_config.rag_settings.dict(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update RAG settings: {str(e)}"
        )


@router.put("/tavily", response_model=Dict[str, Any])
async def update_tavily_settings(
    settings: Dict[str, Any],
    config_service: ConfigService = Depends(get_config_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Update Tavily web search settings only.

    Accepts a partial document: only the submitted keys are changed, so the panel
    can flip `is_enabled` on its own. Values are validated and the domain lists are
    normalised before they reach the stored configuration.
    """
    try:
        updates = {"tavily_settings": _validate_tavily_updates(settings)}
        updated_config = await config_service.update_config(updates)
        await kb_service.refresh()
        await chat_service.refresh()
        return {
            "success": True,
            "message": "Tavily settings updated successfully",
            "settings": updated_config.tavily_settings.dict(),
        }
    except HTTPException:
        # Validation failures keep their 400 status/message.
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update Tavily settings: {str(e)}"
        )


@router.post("/tavily/test", response_model=Dict[str, Any])
async def test_tavily_settings(
    payload: Dict[str, Any],
    config_service: ConfigService = Depends(get_config_service),
):
    """Run one real search so the panel can prove the current settings work.

    Goes through the very same path as a live question (`search_persianway`), which
    means the saved `is_enabled` switch, API key, depth, result count, domain filters
    and snippet length are all exercised exactly as production uses them.
    """
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="A non-empty search query is required")

    try:
        await config_service.clear_cache()
        config = await config_service.get_config()
        tavily_settings = config.tavily_settings

        if not tavily_settings.is_enabled:
            return {
                "success": False,
                "enabled": False,
                "query": query,
                "message": "جستجوی وب غیرفعال است؛ برای تست ابتدا آن را فعال کنید.",
                "result": "",
                "include_domains": tavily_settings.include_domains,
            }

        started_at = time.perf_counter()
        result = await search_persianway.ainvoke({"query": query})
        elapsed = time.perf_counter() - started_at

        succeeded = bool(result) and not any(
            marker in result for marker in ("Error searching web", "Error executing search")
        )
        return {
            "success": succeeded,
            "enabled": True,
            "query": query,
            "message": (
                f"تست جستجو در {elapsed:.1f} ثانیه انجام شد"
                if succeeded
                else result or "نتیجه‌ای از سرویس جستجو دریافت نشد."
            ),
            "result": result or "",
            "elapsed_seconds": round(elapsed, 2),
            "include_domains": tavily_settings.include_domains,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to test Tavily settings: {str(e)}"
        )


@router.post("/tavily/reset", response_model=Dict[str, Any])
async def reset_tavily_settings(
    config_service: ConfigService = Depends(get_config_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Restore only the Tavily section to the static (`.env`) defaults.

    `POST /config/reset` would throw away the LLM and RAG configuration too, which is
    a very heavy price for undoing a search setting.
    """
    try:
        defaults = TavilySearchSettings(tavily_api_key=static_settings.TAVILY_API_KEY)
        updated_config = await config_service.update_config(
            {"tavily_settings": defaults.dict()}
        )
        await kb_service.refresh()
        await chat_service.refresh()
        return {
            "success": True,
            "message": "Tavily settings reset to defaults",
            "settings": updated_config.tavily_settings.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset Tavily settings: {str(e)}"
        )
