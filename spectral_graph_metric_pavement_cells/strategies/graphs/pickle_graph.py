# spectral_graph_metric_pavement_cells/strategies/graphs/pickle_graph.py

from typing import Any, Dict, Tuple, Optional
import networkx as nx
from spectral_graph_metric_pavement_cells.core.interfaces.graph_strategy import GraphStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.visualization.cells import plot_cell_graphs

logger = get_logger(__name__)

class PickleGraph(GraphStrategy):
    """
    Graph strategy for loading and visualizing precomputed graphs (e.g., from gpickle).
    No resampling or contour handling is done here.
    """
    def __init__(self, plot: bool = False):
        self.plot = plot

    def compute_graph(self, graphs: Dict[Any, nx.Graph], save_path: Optional[str] = None) -> Tuple[Dict[str, nx.Graph], Dict[str, Dict[str, Any]]]:
        """
        Accepts a dict of {name: Graph} and returns the same dict along with info.
        """
        for G in graphs.values():
            pos = nx.get_node_attributes(G, 'pos')
            if not pos:
                pos = nx.spring_layout(G, seed=42, k=0.5)
                nx.set_node_attributes(G, pos, 'pos')

        cell_info: Dict[str, Dict[str, Any]] = {}
        for fname, G in graphs.items():
            cell_info[str(fname)] = {
                "mode": "loaded",
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "resolution_um_per_pixel": float('nan'),
                "pixel_distance_used": float('nan'),
                "optimal_pixel_distance": None,
            }

        if self.plot:
            plot_cell_graphs({str(k): v for k, v in graphs.items()}, cell_info, save_path=save_path)

        return {str(k): v for k, v in graphs.items()}, cell_info