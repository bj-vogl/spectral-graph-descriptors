import os
from typing import Any
from spectral_graph_metric_pavement_cells.core.interfaces.persistence_strategy import PersistenceStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.prediction_strategy import PredictionStrategy
from spectral_graph_metric_pavement_cells.core.runtime.context import PersistenceContext
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

class PersistencePipeline:
    def __init__(self, strategy: PersistenceStrategy, save_dir: str):
        self.context = PersistenceContext(strategy)
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def train_and_save(self, training_strategy: PredictionStrategy, X: Any, y: Any, filename: str) -> Any:
        """
        Performs the final training and immediately persists the result.

        Args:
            training_strategy: The strategy defining the model and how to fit it.
            X: Training features.
            y: Training labels.
            filename: The name of the file to save (e.g. 'model.joblib').

        Returns:
            The trained model object
        """
        logger.info(f"PersistencePipeline: Starting final training using {type(training_strategy).__name__}...")

        # 1. Train (Fit on whole dataset)
        model = training_strategy.fit(X, y)

        # 2. Save
        full_path = os.path.join(self.save_dir, filename)
        self.context.save(model, full_path)

        return model

    def load_model(self, filename_or_path: str) -> Any:
        """
        Loads the model.
        Accepts either a full path or a filename inside save_dir.
        """
        if os.path.isabs(filename_or_path):
            full_path = filename_or_path
        else:
            full_path = os.path.join(self.save_dir, filename_or_path)
        return self.context.load(full_path)