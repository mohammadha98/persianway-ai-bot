import os
import joblib
import numpy as np
from typing import Dict, List, Any, Union
from sklearn.ensemble import RandomForestClassifier

from app.models.base import BaseModel
from app.core.config import settings


class ExampleModel(BaseModel):
    """Example ML model implementation using scikit-learn.
    
    This class demonstrates how to implement the BaseModel interface
    for a simple scikit-learn model.
    """
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.model_path = os.path.join(settings.MODEL_PATH, "example_model.joblib")
        self.feature_names = ["feature1", "feature2", "feature3", "feature4"]
        self.load()
    
    def load(self) -> None:
        """Load the model from disk or create a new one if it doesn't exist."""
        try:
            self.model = joblib.load(self.model_path)
        except (FileNotFoundError, OSError):
            # If model doesn't exist, create a simple example model
            print("Model file not found. Creating a new example model.")
            self._create_example_model()
    
    def _create_example_model(self) -> None:
        """Create a simple example model for demonstration purposes."""
        # Create a simple random forest classifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train with some dummy data
        X = np.random.rand(100, len(self.feature_names))
        y = np.random.randint(0, 2, 100)  # Binary classification
        
        self.model.fit(X, y)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
    
    def preprocess(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess the input data.
        
        Args:
            data: Input data with features
            
        Returns:
            Numpy array ready for model prediction
        """
        # Extract features in the correct order
        features = [data.get(feature, 0.0) for feature in self.feature_names]
        return np.array([features])
    
    def predict(self, data: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Make predictions using the model.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Model predictions
        """
        if isinstance(data, dict):
            data = self.preprocess(data)
        
        return self.model.predict(data)
    
    def postprocess(self, prediction: np.ndarray) -> Dict[str, Any]:
        """Format the prediction results.
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Formatted prediction result
        """
        # Convert numpy types to Python native types for JSON serialization
        prediction_value = prediction[0].item() if prediction.size > 0 else None
        
        return {
            "prediction": prediction_value,
            "prediction_label": "Positive" if prediction_value == 1 else "Negative",
            "confidence": 0.85  # Dummy confidence value for demonstration
        }