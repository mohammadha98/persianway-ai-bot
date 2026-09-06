from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class BaseModel(ABC):
    """Base class for all machine learning models.
    
    All models deployed in the API should inherit from this class
    and implement its abstract methods.
    """
    
    @abstractmethod
    def load(self) -> None:
        """Load the model from disk or initialize it."""
        pass
    
    @abstractmethod
    def predict(self, data: Union[List, Dict[str, Any]]) -> Any:
        """Make predictions using the model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model predictions
        """
        pass
    
    def preprocess(self, data: Union[List, Dict[str, Any]]) -> Any:
        """Preprocess input data before prediction.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for model prediction
        """
        return data
    
    def postprocess(self, prediction: Any) -> Dict[str, Any]:
        """Postprocess model predictions.
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Processed prediction in API-friendly format
        """
        return {"prediction": prediction}