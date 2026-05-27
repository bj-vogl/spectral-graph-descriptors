from abc import ABC, abstractmethod
from typing import Any

class PredictionStrategy(ABC):
    """
    Interface for model training and inference.
    """
    @abstractmethod
    def fit(self, X: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, model: Any, X: Any) -> Any:
        pass