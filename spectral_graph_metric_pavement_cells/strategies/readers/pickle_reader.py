import os
import pickle
from typing import Any
import networkx as nx
from spectral_graph_metric_pavement_cells.core.interfaces.readers_strategy import ReaderStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class PickleReader(ReaderStrategy):
    def read(self, path: str) -> dict[Any, Any]:
        graph_data = None

        if not os.path.exists(path):
            logger.warning(f"File {path} not found!")
            return {}

        with open(path, "rb") as f:
            graph_data = pickle.load(f)

            # Check if all objects are graphs
        for k, g in graph_data.items():
            if not isinstance(g, nx.Graph):
                logger.warning(f"Object {k} is no NetworkX Graph!")

        return graph_data