from fastapi import APIRouter

from app.api.routes.predictions import router as predictions_router
from app.api.routes.chat import router as chat_router
from app.api.routes.knowledge_base import router as knowledge_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.config import router as config_router
from app.api.routes.upload import router as upload_router
from app.api.routes.users import router as users_router
from app.api.routes.ui import router as ui_routes

# Main API router
router = APIRouter(prefix="/api")

# Include all route modules
router.include_router(predictions_router)
router.include_router(chat_router)
router.include_router(knowledge_router)
router.include_router(conversations_router)
router.include_router(config_router)
router.include_router(upload_router)
router.include_router(users_router)

# UI router (separate from API router)
ui_router = APIRouter()
ui_router.include_router(ui_routes)