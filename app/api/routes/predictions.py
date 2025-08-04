from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_service import get_model_service

# Create router for prediction endpoints
router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
async def create_prediction(request: PredictionRequest, model_service=Depends(get_model_service)):
    """Make a prediction using the ML model.
    
    This endpoint accepts feature values and returns model predictions.
    """
    try:
        # Get the model from the service
        model = model_service.get_model("example_model")
        
        # Make prediction
        prediction_input = request.dict()
        result = model.predict(prediction_input)
        
        # Post-process the prediction
        response = model.postprocess(result)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/models", response_model=Dict[str, str])
async def list_available_models(model_service=Depends(get_model_service)):
    """List all available ML models.
    
    Returns a dictionary of model IDs and their descriptions.
    """
    return model_service.list_models()