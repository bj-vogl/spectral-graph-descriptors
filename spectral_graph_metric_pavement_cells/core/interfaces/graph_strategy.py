from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict

import networkx as nx

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class GraphStrategy(ABC):

    @abstractmethod
    def compute_graph(self, features) -> Tuple[Dict[str, nx.Graph], Dict[str, Dict[str, Any]]]:
        pass