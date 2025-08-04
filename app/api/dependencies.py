from typing import Callable, Type
from fastapi import Depends

from app.models.base import BaseModel
from app.services.model_service import get_model_instance


def get_prediction_model(model_name: str) -> Callable[[], BaseModel]:
    """Dependency for getting a specific ML model instance
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        A callable that returns a model instance
    """
    def _get_model() -> BaseModel:
        return get_model_instance(model_name)
    
    return _get_model