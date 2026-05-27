from typing import Any, Dict
from spectral_graph_metric_pavement_cells.core.interfaces.prediction_strategy import PredictionStrategy
from spectral_graph_metric_pavement_cells.core.runtime.context import PredictionContext
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self, strategy: PredictionStrategy):
        self.context = PredictionContext(strategy)

    def run_training(self, model: Any, X: Any, y: Any) -> Any:
        """Train a model through the strategy."""
        return self.context.fit(model, X, y)

    def run_inference(self, model: Any, data: Any) -> Any:
        """Predict using a trained model."""
        return self.context.predict(model, data)