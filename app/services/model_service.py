from typing import Dict, Any, Optional
from functools import lru_cache

from app.models.base import BaseModel
from app.models.example_model import ExampleModel


class ModelService:
    """Service for managing machine learning models.
    
    This service is responsible for loading, caching, and providing
    access to all ML models in the application.
    """
    
    def __init__(self):
        """Initialize the model service."""
        self._models: Dict[str, BaseModel] = {}
        self._model_descriptions: Dict[str, str] = {}
        
        # Register available models
        self._register_models()
    
    def _register_models(self) -> None:
        """Register all available models.
        
        When adding a new model to the application, register it here.
        """
        # Register example model
        self.register_model(
            "example_model",
            ExampleModel(),
            "Example scikit-learn random forest classifier"
        )
        
        # Register additional models here
        # self.register_model("another_model", AnotherModel(), "Description")
    
    def register_model(self, model_id: str, model: BaseModel, description: str) -> None:
        """Register a new model with the service.
        
        Args:
            model_id: Unique identifier for the model
            model: Instance of a model implementing BaseModel
            description: Human-readable description of the model
        """
        self._models[model_id] = model
        self._model_descriptions[model_id] = description
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get a model by its ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The requested model or None if not found
            
        Raises:
            KeyError: If the model ID is not found
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found")
        
        return self._models[model_id]
    
    def list_models(self) -> Dict[str, str]:
        """List all available models.
        
        Returns:
            Dictionary mapping model IDs to their descriptions
        """
        return self._model_descriptions


@lru_cache()
def get_model_service() -> ModelService:
    """Factory function for ModelService (dependency injection).
    
    Returns:
        Singleton instance of ModelService
    """
    return ModelService()