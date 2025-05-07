from fastapi import APIRouter

from app.api.routes.predictions import router as predictions_router
from app.api.routes.chat import router as chat_router

# Main API router
router = APIRouter(prefix="/api")

# Include all route modules
router.include_router(predictions_router)
router.include_router(chat_router)