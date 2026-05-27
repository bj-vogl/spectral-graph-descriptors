# core/analysis_strategy.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, results: Dict[str, DistanceResult]) -> Any:
        pass