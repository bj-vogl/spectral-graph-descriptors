from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from spectral_graph_metric_pavement_cells.core.interfaces.prediction_strategy import PredictionStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)


class RandomForestPrediction(PredictionStrategy):
    """
    A strategy for training a Random Forest.
    """

    def __init__(self,
                 n_estimators: int = 200,
                 random_state: int = 42,
                 **kwargs):
        """
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.
        """
        self.params = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            **kwargs
        }
        self.model = None

    def _preprocess(self, X: Any) -> np.ndarray:
        """Helper to convert to numpy."""
        X_arr = np.array(X)
        return X_arr

    def fit(self, X: Any, y: Any) -> Any:
        X_processed = self._preprocess(X)

        logger.info(f"Training RandomForest on full features. Input shape: {X_processed.shape}")

        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_processed, y)
        return self.model

    def predict(self, model: Any, X: Any) -> Any:
        estimator = model
        if estimator is None:
            raise ValueError("No model available for prediction.")

        X_processed = self._preprocess(X)
        return estimator.predict(X_processed)