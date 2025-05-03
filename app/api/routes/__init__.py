from fastapi import APIRouter

from app.api.routes.predictions import router as predictions_router

# Main API router
router = APIRouter(prefix="/api")

# Include all route modules
router.include_router(predictions_router)