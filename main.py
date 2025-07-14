from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.config import settings
from app.middleware.conversation_logger import ConversationLoggerMiddleware
from app.services.database import get_database_service, close_database_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup: Initialize database connection
    try:
        await get_database_service()
        print("Database connection established")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        # Continue without database - conversations won't be logged but app will work
    
    yield
    
    # Shutdown: Close database connection
    await close_database_connection()
    print("Database connection closed")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        docs_url=None,  # We'll customize the docs URL
        redoc_url=None,  # We'll customize the redoc URL
        lifespan=lifespan
    )

    # Set up CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add conversation logging middleware
    application.add_middleware(
        ConversationLoggerMiddleware,
        log_chat_endpoints=True
    )

    # Include API router
    application.include_router(api_router)

    # Custom Swagger UI
    @application.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=application.openapi_url,
            title=f"{settings.PROJECT_NAME} - Swagger UI",
            oauth2_redirect_url=application.swagger_ui_oauth2_redirect_url,
        )

    # Health check endpoint
    @application.get("/health", tags=["health"])
    async def health_check():
        return {"status": "healthy", "version": settings.VERSION}

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)