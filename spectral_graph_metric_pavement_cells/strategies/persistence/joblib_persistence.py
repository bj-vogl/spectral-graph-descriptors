import joblib
import os
from typing import Any

from spectral_graph_metric_pavement_cells.core.interfaces.persistence_strategy import PersistenceStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

class JoblibPersistence(PersistenceStrategy):
    def save(self, model: Any, filepath: str) -> None:
        try:
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            joblib.dump(model, filepath)
            logger.info(f"Model successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise

    def load(self, filepath: str) -> Any:
        if not os.path.exists(filepath):
            error_msg = f"No model found at {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            model = joblib.load(filepath)
            logger.info(f"Model successfully loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise