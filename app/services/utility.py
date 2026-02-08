from typing import Dict, Any, Optional
from tavily import TavilyClient
from loguru import logger
from app.core.config import settings

class TavilySearchService:
    """
    Service for performing web searches using the Tavily API.
    Designed to be used as a tool for LLMs.
    """
    
    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.client = None
        if self.api_key:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily search service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
        else:
            logger.warning("TAVILY_API_KEY is not set. Search functionality will be disabled.")

    def search(
        self, 
        query: str, 
        search_depth: str = "advanced", 
        include_answer: bool = True,
        max_results: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query (str): The search query.
            search_depth (str): "basic" or "advanced". Defaults to "advanced".
            include_answer (bool): Whether to include an AI-generated answer. Defaults to True.
            max_results (int): Maximum number of results to return. Defaults to 5.
            **kwargs: Additional arguments to pass to the Tavily client.
            
        Returns:
            Dict[str, Any]: The search results.
        """
        if not self.client:
            logger.warning("Tavily client is not initialized.")
            return {"error": "Search service is not available", "results": []}
            
        try:
            logger.info(f"Searching Tavily for: {query}")
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                include_answer=include_answer,
                max_results=max_results,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error performing Tavily search: {e}")
            return {"error": str(e), "results": []}

# Global instance
tavily_service = TavilySearchService()

def search_web(query: str, include_answer: bool = True) -> str:
    """
    Simple wrapper function for Tavily search to be used directly by LLMs or other services.
    
    Args:
        query (str): The search query.
        include_answer (bool): Whether to include a direct answer.
        
    Returns:
        str: A formatted string containing the answer and/or top results.
    """
    result = tavily_service.search(query, include_answer=include_answer)
    
    if "error" in result and result["error"]:
        return f"Error searching web: {result['error']}"
    
    output = []
    
    # Add the AI generated answer if available
    if result.get("answer"):
        output.append(f"Answer: {result['answer']}\n")
    
    # Add sources
    if result.get("results"):
        output.append("Sources:")
        for res in result["results"]:
            title = res.get("title", "No title")
            url = res.get("url", "#")
            content = res.get("content", "")
            output.append(f"- [{title}]({url}): {content[:200]}...")
            
    return "\n".join(output)

from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query string. MUST be in Persian language.")

@tool(args_schema=SearchInput)
def search_persianway(query: str) -> str:
    """
    Performs a domain-specific search on persianway.ir.
    Use this tool when you need to find information about PersianWay, its products, or services
    that might not be in your immediate knowledge base.
    IMPORTANT: The input query MUST be in Persian language. If the user asks in English or another language,
    you must translate the search terms to Persian before using this tool.
    Input should be a search query string in Persian.
    """
    logger.debug(f"search_persianway called with query: '{query}'")
    # Enforce domain restriction
    restricted_query = f"site:persianway.ir {query}"
    return search_web(restricted_query, include_answer=True)
