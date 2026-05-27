# core/context.py
from typing import Any

from spectral_graph_metric_pavement_cells.core.interfaces.filter_strategy import FilterStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.graph_strategy import GraphStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.persistence_strategy import PersistenceStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.prediction_strategy import PredictionStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.readers_strategy import ReaderStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.distances_strategy import DistanceStrategy
from spectral_graph_metric_pavement_cells.core.interfaces.analysis_strategy import AnalysisStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class ReaderContext:
    def __init__(self, strategy: ReaderStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: ReaderStrategy) -> None:
        self._strategy = strategy

    def execute(self, *args, **kwargs) -> Any:
        return self._strategy.read(*args, **kwargs)

class GraphContext:
    def __init__(self, strategy: GraphStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: GraphStrategy) -> None:
        self._strategy = strategy

    def execute(self, *args, **kwargs) -> Any:
        return self._strategy.compute_graph(*args, **kwargs)

class DistanceContext:
    def __init__(self, strategy: DistanceStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: DistanceStrategy) -> None:
        self._strategy = strategy

    def execute(self, *args, **kwargs) -> Any:
        return self._strategy.compute_distance(*args, **kwargs)

class FilterContext:
    def __init__(self, strategy: FilterStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: FilterStrategy) -> None:
        self._strategy = strategy

    def execute(self, graphs, cell_info):
        return self._strategy.filter(graphs, cell_info)

class ComparisonContext:
    def __init__(self, strategy: AnalysisStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: AnalysisStrategy) -> None:
        self._strategy = strategy

    def execute(self, *args, **kwargs) -> Any:
        return self._strategy.analyze(*args, **kwargs)

class PersistenceContext:
    def __init__(self, strategy: PersistenceStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: PersistenceStrategy) -> None:
        self._strategy = strategy

    def save(self, *args, **kwargs) -> None:
        return self._strategy.save(*args, **kwargs)

    def load(self, *args, **kwargs) -> Any:
        return self._strategy.load(*args, **kwargs)


class PredictionContext:
    def __init__(self, strategy: PredictionStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: PredictionStrategy) -> None:
        self._strategy = strategy

    def fit(self, *args, **kwargs) -> Any:
        return self._strategy.fit(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Any:
        return self._strategy.predict(*args, **kwargs)