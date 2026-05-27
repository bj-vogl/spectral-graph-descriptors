from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from spectral_graph_metric_pavement_cells.core.interfaces.prediction_strategy import PredictionStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.scalers import GlobalIQRScaler, GlobalStandardScaler, MaxnormScaler

logger = get_logger(__name__)


class LogisticRegressionPrediction(PredictionStrategy):
    """
    A strategy for training a Logistic Regression Pipeline (Scaler + Classifier) on the full dataset.

    Includes a self-check after fitting to log training accuracy.
    """

    def __init__(self,
                 scaler_name: str,
                 max_iter: int = 4000,
                 **kwargs):
        self.scaler_name = scaler_name

        self.params = {
            "max_iter": max_iter,
            "random_state": 42,
            "solver": "lbfgs",
            "tol": 1e-3,
            **kwargs
        }
        self.model = None

    def _get_scaler(self):
        if self.scaler_name is None or self.scaler_name == "None":
            return None

        if self.scaler_name == "StandardScaler":
            return StandardScaler()
        elif self.scaler_name == "GlobalIQRScaler":
            return GlobalIQRScaler((25, 75))
        elif self.scaler_name == "GlobalStandardScaler":
            return GlobalStandardScaler()
        elif self.scaler_name == "MaxnormScaler":
            return MaxnormScaler()
        else:
            logger.warning("Unknown scaler_name '%s'", self.scaler_name)
            return None

    def fit(self, X: Any, y: Any) -> Any:
        X_arr = np.array(X)
        scaler = self._get_scaler()

        # Adjust solver if no scaler
        if scaler is None:
            self.params["solver"] = "newton-cg"

        steps = []
        if scaler:
            steps.append(("sc", scaler))
        steps.append(("clf", LogisticRegression(**self.params)))

        self.model = Pipeline(steps)
        self.model.fit(X_arr, y)

        try:
            y_pred_check = self.model.predict(X_arr)
            train_acc = accuracy_score(y, y_pred_check)
            logger.info(f"Model fitted successfully. Self-Check (Training Accuracy): {train_acc:.2%}")
        except Exception as e:
            logger.warning(f"Self-Check during training failed: {e}")

        return self.model

    def predict(self, model: Any, X: Any) -> Any:
        estimator = model
        if estimator is None:
            raise ValueError("No model available for prediction.")

        X_arr = np.array(X)

        return estimator.predict(X_arr)