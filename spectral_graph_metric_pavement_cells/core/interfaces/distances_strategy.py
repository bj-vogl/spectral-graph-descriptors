from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class DistanceStrategy(ABC):

    @abstractmethod
    def compute_distance(self, features: dict[str, Any], feature_key: str, save_path: str | None = None) -> np.ndarray:
        pass