from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import os

from app.api.routes import router as api_router, ui_router
from app.core.config import settings
from app.core.templates import configure_templates
from app.middleware.conversation_logger import ConversationLoggerMiddleware
from app.services.database import get_database_service, close_database_connection
from app.services.config_service import get_config_service, get_dynamic_app_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup: Initialize database, config, and middleware
    try:
        await get_database_service()
        print("Database connection established")

        # Initialize config service and load settings
        await get_config_service()
        app_settings = await get_dynamic_app_settings()

        # Update app instance with dynamic settings
        app.title = app_settings.project_name
        app.description = app_settings.project_description
        app.version = app_settings.version

        print("Configuration loaded from database settings")

    except Exception as e:
        print(f"Failed to connect to database or load config: {e}")
        # If database fails, the app will run with static settings
        print("Running with static configuration")

    yield
    
    # Shutdown: Close database connection
    await close_database_connection()
    print("Database connection closed")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    application = FastAPI(
        title=settings.PROJECT_NAME,  # Initial static title
        description=settings.PROJECT_DESCRIPTION,  # Initial static description
        version=settings.VERSION,  # Initial static version
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan
    )

    # Set up CORS middleware with static settings initially
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

    # Configure Jinja2 templates
    configure_templates(application)

    # Include API router
    application.include_router(api_router)
    
    # Include UI router
    application.include_router(ui_router)

    # Custom Swagger UI
    @application.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=application.openapi_url,
            title=f"{application.title} - Swagger UI",
            oauth2_redirect_url=application.swagger_ui_oauth2_redirect_url,
        )

    # Health check endpoint
    @application.get("/health", tags=["health"])
    async def health_check():
        return {"status": "healthy", "version": application.version}

    # Mount static files for Angular frontend
    angular_dist_path = os.path.join(os.path.dirname(__file__), "frontend", "ai-panel", "dist", "ai-panel", "browser")
    if os.path.exists(angular_dist_path):
        # Serve Angular app for UI routes and fallback
        @application.get("/ui/{full_path:path}", include_in_schema=False)
        async def serve_angular_ui(full_path: str):
            # Serve index.html for all UI routes (Angular Router will handle routing)
            index_file = os.path.join(angular_dist_path, "index.html")
            if os.path.exists(index_file):
                return FileResponse(index_file)
            return {"detail": "Angular frontend not built"}
        
        # Serve Angular app for root path
        @application.get("/", include_in_schema=False)
        async def serve_angular_root():
            index_file = os.path.join(angular_dist_path, "index.html")
            if os.path.exists(index_file):
                return FileResponse(index_file)
            return {"detail": "Angular frontend not built"}
        
        # Serve static files (JS, CSS, etc.)
        @application.get("/{file_path:path}", include_in_schema=False)
        async def serve_angular_static(file_path: str):
            # Don't serve frontend for API routes, docs, health check, or ui routes
            if file_path.startswith(("api/", "docs", "health", "ui/")):
                return {"detail": "Not found"}
            
            # Check if it's a static file
            static_file = os.path.join(angular_dist_path, file_path)
            if os.path.exists(static_file) and os.path.isfile(static_file):
                return FileResponse(static_file)
            
            # For any other route, serve the Angular app (SPA routing)
            index_file = os.path.join(angular_dist_path, "index.html")
            if os.path.exists(index_file):
                return FileResponse(index_file)
            return {"detail": "Angular frontend not built"}

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)