from abc import ABC, abstractmethod
from typing import Any

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class ReaderStrategy(ABC):

    @abstractmethod
    def read(self, path: str) -> dict[Any, Any]:
        pass