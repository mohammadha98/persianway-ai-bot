from typing import Dict, Any, Optional, List
from tavily import TavilyClient
from loguru import logger
from app.services.config_service import get_config_service
from app.core.config import settings
import asyncio
from langchain.tools import tool
from pydantic import BaseModel, Field

class TavilySearchService:
    def __init__(self):
        self.client: Optional[TavilyClient] = None
        self._current_api_key: Optional[str] = None

    def _ensure_client(self, api_key: Optional[str]) -> None:
        if not api_key:
            self.client = None
            self._current_api_key = None
            return
        if self.client is None or api_key != self._current_api_key:
            try:
                self.client = TavilyClient(api_key=api_key)
                self._current_api_key = api_key
                logger.info("Tavily client initialized or rotated with new API key")
            except Exception as e:
                self.client = None
                self._current_api_key = None
                logger.error(f"Failed to initialize Tavily client: {e}")

    async def search(self, query: str) -> Dict[str, Any]:
        config_service = await get_config_service()
        config = await config_service.get_config()
        tavily_cfg = config.tavily_settings
        
        # Use dynamic API key if available, otherwise fallback to settings.TAVILY_API_KEY
        api_key = tavily_cfg.tavily_api_key or settings.TAVILY_API_KEY
        
        # If enabled in config is False, we respect that UNLESS it's the default/unconfigured state 
        # and we have an env key. But strictly following the requirement: "if config values are empty... pass from .env"
        # The is_enabled flag defaults to True in the model, so we check if key is present.
        
        if not api_key:
            logger.warning("Tavily search disabled: No API key in config or .env")
            return {"error": "Search service is disabled", "results": []}
            
        # Ensure client is initialized with the correct key
        self._ensure_client(api_key)
        
        if not self.client:
            return {"error": "Search service is not available", "results": []}
            
        try:
            logger.info(f"Searching Tavily for: {query}")
            # Tavily client is synchronous, so we run it in a thread pool to avoid blocking the event loop
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth=tavily_cfg.search_depth,
                include_answer=tavily_cfg.include_answer,
                max_results=tavily_cfg.max_results,
                include_domains=tavily_cfg.include_domains if tavily_cfg.include_domains else None,
                exclude_domains=tavily_cfg.exclude_domains if tavily_cfg.exclude_domains else None,
            )
            return response
        except Exception as e:
            logger.error(f"Error performing Tavily search: {e}")
            return {"error": str(e), "results": []}

tavily_service = TavilySearchService()

async def search_web(query: str) -> str:
    result = await tavily_service.search(query)
    if "error" in result and result["error"]:
        return f"Error searching web: {result['error']}"
    config_service = await get_config_service()
    config = await config_service.get_config()
    snippet_length = config.tavily_settings.snippet_length
    output: List[str] = []
    if result.get("answer"):
        output.append(f"Answer: {result['answer']}\n")
    if result.get("results"):
        output.append("Sources:")
        for res in result["results"]:
            title = res.get("title", "No title")
            url = res.get("url", "#")
            content = res.get("content", "")
            output.append(f"- [{title}]({url}): {content[:snippet_length]}...")
    return "\n".join(output)

class SearchInput(BaseModel):
    query: str = Field(description="The search query string. MUST be in Persian language.")

@tool(args_schema=SearchInput)
async def search_persianway(query: str) -> str:
    """
    Domain-aware web search tool using Tavily with dynamic configuration.
    
    Uses the current TavilySearchSettings from the database to:
    - Restrict queries to configured include_domains (e.g., site:persianway.ir)
    - Apply dynamic search_depth, max_results, include/exclude domains
    - Format output snippets using configured snippet_length
    
    Input must be a Persian-language search query string.
    Returns a formatted string containing answer and sources or an error message.
    """
    try:
        config_service = await get_config_service()
        config = await config_service.get_config()
        include_domains = config.tavily_settings.include_domains
        restricted_query = query
        if include_domains:
            domains_expr = " OR ".join([f"site:{d}" for d in include_domains])
            restricted_query = f"{domains_expr} {query}"
        logger.debug(f"search_persianway called with query: '{restricted_query}'")
        return await search_web(restricted_query)
    except Exception as e:
        logger.error(f"Error in search_persianway: {e}")
        return f"Error executing search: {str(e)}"


# Since the tool is async, we should explicitly allow it to be used asynchronously.
# However, LangChain sometimes defaults to sync execution if not handled carefully.
# The error "StructuredTool does not support sync invocation" means something tried to call .run() instead of .arun()
# or invoke() instead of ainvoke() on this async-only tool.
# To fix this, we can provide a synchronous wrapper that raises a helpful error or runs in a new loop (risky),
# OR ensure all callers use ainvoke. 
# But the safest way to avoid the error when a sync path might be triggered is to provide a dummy sync implementation
# that explains it's async-only, or just let it fail if we are sure we only use ainvoke.
# Given the error, it seems `StructuredTool` was instantiated with only an async coroutine. 
# We can fix this by ensuring the tool is defined properly.
# The @tool decorator on an async function creates a tool that supports async execution.
# If something calls it synchronously, it fails.

