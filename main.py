import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)

# The Angular bundle is a build artifact, not source: `.gitignore` excludes
# `dist/`, so a deployment that only pulls from git starts without it. In that
# case `/` used to answer FastAPI's bare 404 (which is indistinguishable from a
# routing bug), so the missing-bundle case now returns an explicit 503.
ANGULAR_BUNDLE_MISSING_MESSAGE = {
    "detail": (
        "Angular frontend bundle not found. Build it on the host with "
        "`cd frontend/ai-panel && npm ci && npm run build` (or copy an existing "
        "`dist/ai-panel/browser` directory into place) and restart the server, "
        "because main.py registers the SPA routes at import time."
    )
}

# Must run before any dependency that still references aliases removed in
# NumPy 2.0 (ChromaDB < 0.5.0 used np.float_ at import time, which crashed
# every gunicorn worker with "Worker failed to boot" and caused nginx 502s).
from app.core import numpy_compat  # noqa: F401  (imported for its side effects)

from app.api.routes import router as api_router
from app.api.routes import ui_router
from app.core.config import settings
from app.core.templates import configure_templates
from app.services.config_service import get_config_service, get_dynamic_app_settings
from app.services.database import close_database_connection, get_database_service


async def _warm_up_retrieval() -> None:
    """Build the retrieval stack (Chroma, hybrid service, BM25 indexes) off-loop.

    Runs as a background task from the app lifespan; see `lifespan` for why the
    work must not stay on the first request's path.
    """
    # Imported locally: `main` is imported by tooling that must not require the
    # full retrieval stack (langchain/chroma) to be importable.
    from app.services.knowledge_base import get_knowledge_base_service

    try:
        await get_knowledge_base_service().warm_up()
    except asyncio.CancelledError:
        logger.info("[WARMUP] Retrieval warm-up cancelled (shutdown)")
        raise
    except Exception as e:  # pragma: no cover - defensive, warm_up logs its own
        logger.error("[WARMUP] Retrieval warm-up failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    retrieval_warmup: Optional[asyncio.Task] = None
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

    # Build the retrieval stack (Chroma client + collection, hybrid service, BM25
    # branch indexes) in the background, off the event loop. Doing this lazily in
    # the first streaming request is what made a cold worker look like a hung
    # stream in production: the blocking Chroma/BM25 work froze the loop, gunicorn
    # stopped receiving heartbeats and killed the worker (`WORKER TIMEOUT ...
    # SIGABRT`) while the client was still waiting for its first byte.
    # Fire-and-forget on purpose: the worker must report itself booted
    # immediately, and scheduling the work as a background task is what keeps boot
    # latency unchanged.
    try:
        retrieval_warmup = asyncio.create_task(
            _warm_up_retrieval(), name="retrieval-warmup"
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"Retrieval warm-up could not be scheduled: {e}")

    yield

    # Shutdown: stop the warm-up if it is still running, then close the database.
    if retrieval_warmup is not None and not retrieval_warmup.done():
        retrieval_warmup.cancel()
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
        lifespan=lifespan,
    )

    # Set up CORS middleware with static settings initially
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
    angular_dist_path = os.path.join(
        os.path.dirname(__file__), "frontend", "ai-panel", "dist", "ai-panel", "browser"
    )
    if os.path.exists(angular_dist_path):
        # Serve Angular app for UI routes and fallback
        @application.get("/ui/{full_path:path}", include_in_schema=False)
        async def serve_angular_ui(full_path: str):
            # If /ui points to a real static asset (e.g. /ui/main-*.js), serve the file.
            static_file = os.path.join(angular_dist_path, full_path)
            if (
                full_path
                and os.path.exists(static_file)
                and os.path.isfile(static_file)
            ):
                return FileResponse(static_file)

            # Otherwise serve index.html for Angular SPA routes.
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
                raise HTTPException(status_code=404, detail="Not found")

            # Check if it's a static file
            static_file = os.path.join(angular_dist_path, file_path)
            if os.path.exists(static_file) and os.path.isfile(static_file):
                return FileResponse(static_file)

            # If request looks like a missing asset file, return 404 (don't return index.html)
            # to avoid module MIME mismatch errors in browser.
            _, ext = os.path.splitext(file_path)
            if ext:
                raise HTTPException(status_code=404, detail="Static asset not found")

            # For Angular client-side routes, serve SPA index.
            index_file = os.path.join(angular_dist_path, "index.html")
            if os.path.exists(index_file):
                return FileResponse(index_file)
            return {"detail": "Angular frontend not built"}
    else:
        # No bundle on disk: the SPA routes above are not registered at all, so
        # without the handlers below FastAPI would answer a bare 404 on `/` and
        # the failure would look like a routing bug instead of a missing build.
        logger.warning(
            "[Frontend] Angular bundle not found at '%s'; the SPA will not be "
            "served. Run `cd frontend/ai-panel && npm ci && npm run build` on "
            "this host (or copy a build there) and restart the server.",
            angular_dist_path,
        )

        @application.get("/", include_in_schema=False)
        async def angular_bundle_missing_root():
            return JSONResponse(
                status_code=503, content=ANGULAR_BUNDLE_MISSING_MESSAGE
            )

        @application.get("/{file_path:path}", include_in_schema=False)
        async def angular_bundle_missing_fallback(file_path: str):
            # Only the SPA routes are unavailable; API, docs, health and the
            # mounted /static files must keep behaving exactly as before.
            if file_path.startswith(("api/", "docs", "health", "ui/", "static/")):
                raise HTTPException(status_code=404, detail="Not found")
            return JSONResponse(
                status_code=503, content=ANGULAR_BUNDLE_MISSING_MESSAGE
            )

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=300
    )
