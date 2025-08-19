from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union


class PredictionRequest(BaseModel):
    """Schema for prediction request data.
    
    This defines the expected input format for the prediction API.
    The fields should match the features expected by your model.
    """
    feature1: float = Field(..., description="First feature value")
    feature2: float = Field(..., description="Second feature value")
    feature3: float = Field(..., description="Third feature value")
    feature4: float = Field(..., description="Fourth feature value")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "feature1": 0.5,
                "feature2": 0.75,
                "feature3": 0.2,
                "feature4": 0.9
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response data.
    
    This defines the expected output format from the prediction API.
    """
    prediction: Union[int, float, str] = Field(..., description="Raw prediction value")
    prediction_label: Optional[str] = Field(None, description="Human-readable prediction label")
    confidence: Optional[float] = Field(None, description="Confidence score for the prediction")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "prediction_label": "Positive",
                "confidence": 0.85
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests.
    
    This allows sending multiple prediction requests at once.
    """
    items: List[PredictionRequest] = Field(..., description="List of prediction requests")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses.
    
    This returns predictions for multiple inputs.
    """
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    count: int = Field(..., description="Number of predictions made")