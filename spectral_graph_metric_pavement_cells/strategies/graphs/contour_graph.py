# spectral_graph_metric_pavement_cells/strategies/graphs/contour_graph.py
from typing import Any, Dict, Tuple, Optional

import networkx as nx
import numpy as np
from shapely import Polygon
from shapely.geometry.linestring import LineString

from spectral_graph_metric_pavement_cells.core.features.weight_funcs import set_cell_norm_scale
from spectral_graph_metric_pavement_cells.core.interfaces.graph_strategy import GraphStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.common import compute_convex_hull_scale
from spectral_graph_metric_pavement_cells.utils.geometry import ensure_valid_polygon
from spectral_graph_metric_pavement_cells.utils.visibility import visibility_edge_ok
from spectral_graph_metric_pavement_cells.utils.sampling import resample_contour
from spectral_graph_metric_pavement_cells.visualization.cells import plot_cell_graphs

logger = get_logger(__name__)

_DEFAULT_TOL_FACTOR = 0.085
_MIN_TOLERANCE = 1e-6

def _compute_single_graph(
        contour: np.ndarray,
        pixel_distance: Optional[float] = None,
        node_count: Optional[int] = None,
        normalize_by: Optional[str] = None)  -> Tuple[nx.Graph, np.ndarray]:

    """Compute a single visibility-like contour graph from a contour."""
    contour_resampled = resample_contour(contour, pixel_distance=pixel_distance, node_count=node_count)
    polygon = ensure_valid_polygon(Polygon(contour_resampled))

    line = LineString(polygon.exterior.coords)
    perimeter = float(line.length)

    # Dynamic tolerance based on average point spacing
    avg_distance = 0.0
    n_points = len(contour_resampled)
    if n_points > 1:
        # Calculate average point spacing
        distances = []
        for i in range(n_points):
            j = (i + 1) % n_points
            dist = np.linalg.norm(contour_resampled[i] - contour_resampled[j])
            distances.append(dist)
        avg_distance = float(np.mean(distances))
        # Tolerance is a fraction of the average point spacing
        tolerance = avg_distance * _DEFAULT_TOL_FACTOR
    else:
        tolerance = _MIN_TOLERANCE



    polygon_buffered = polygon.buffer(tolerance)

    G = nx.Graph()
    # Add nodes with their coordinates as attributes
    for i, pt in enumerate(contour_resampled):
        G.add_node(i, pos=tuple(pt), idx=i)

    max_dist = 0.0

    # Check pairwise visibility between nodes and find max distance
    for i in range(n_points):
        for j in range(i+1, n_points):
            ok = visibility_edge_ok(
                polygon_buffered=polygon_buffered,
                point_a=contour_resampled[i],
                point_b=contour_resampled[j],
            )
            if ok:
                G.add_edge(i, j, weight=None)

            dist = np.linalg.norm(contour_resampled[i] - contour_resampled[j])
            if dist > max_dist:
                max_dist = dist
    #logger.info(f"Max distance between visible nodes: {max_dist:.2f} um")

    if normalize_by == "max_dist":
        set_cell_norm_scale(max_dist if max_dist  > 0.0 else None)
    elif normalize_by == "convex_hull":
        scale = compute_convex_hull_scale(contour_resampled)
        set_cell_norm_scale(scale if scale > 0.0 else None)
    else:
        set_cell_norm_scale(None)

    return G, contour_resampled

def _extract_resolution(contour_data: Dict[str, Any]) -> Optional[float]:
    res = contour_data.get("resolution")
    if res is None:
        return None
    try:
        return float(res[0])
    except Exception:
        return None

class ContourGraph(GraphStrategy):
    """
    Graph strategy that produces contour-based graphs.

    Sampling can be controlled via:
     - node_count (exact number of nodes)
     - manual_pixel_distance (fixed pixel spacing)
     - node_coverage (fallback, yields pixel spacing = 1 / (node_coverage * resolution))

    Precedence (automatic decision):
      node_count > manual_pixel_distance > node_coverage
    """
    def __init__(self,
                 manual_pixel_distance: Optional[float] = None,
                 node_count: Optional[int] = None,
                 node_coverage: float = 0.65,
                 plot: bool = False,
                 normalize_by: Optional[str] = None) -> None:
        self.plot = plot
        self.manual_pixel_distance = manual_pixel_distance
        self.node_count = node_count
        self.node_coverage = node_coverage  # nodes per µm

        if normalize_by not in (None, "convex_hull", "max_dist"):
            raise ValueError("normalize_by must be None, 'convex_hull' or 'max_dist'")
        self.normalize_by = normalize_by

    def compute_graph(self, contours_dict: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> Tuple[
            Dict[str, nx.Graph], Dict[str, Dict[str, Any]],  Dict[str, np.ndarray]]:
        """
        Compute graphs for all contours in contours_dict.
        The sampling strategy is chosen automatically from the provided parameters.
        """
        graphs: Dict[str, nx.Graph] = {}
        cell_info: Dict[str, Dict[str, Any]] = {}
        resampled_contours: Dict[str, np.ndarray] = {}

        # Collect resolutions
        resolutions = []
        for fname, contour_data in contours_dict.items():
            if contour_data.get("contour") is not None:
                r = _extract_resolution(contour_data)
                if r is not None:
                    resolutions.append(r)

        if not resolutions:
            logger.warning("Warning: No valid contours found!")
            return {}, {}, {}

        unique_resolutions = set(resolutions)
        if len(unique_resolutions) > 1:
            raise ValueError(f"Different resolutions found: {unique_resolutions}. "
                             "All contours must have the same resolution.")

        resolution = resolutions[0]

        # Compute optimal pixel distance from node_coverage (nodes per µm)
        optimal_pixel_distance = 1.0 / (self.node_coverage * resolution)

        # Decide which sampling mode to use
        if self.node_count is not None:
            if self.manual_pixel_distance is not None:
                logger.warning("Warning: both node_count and manual_pixel_distance set — node_count will take precedence.")
            mode = "node_count"
            pixel_distance_used = None
        elif self.manual_pixel_distance is not None:
            mode = "manual"
            pixel_distance_used = self.manual_pixel_distance
        else:
            mode = "auto"
            pixel_distance_used = optimal_pixel_distance

        for fname, contour_data in contours_dict.items():
            contour = contour_data.get('contour')
            if contour is None:
                logger.warning(f"Warning: No contour for {fname} found!")
                continue

            # Create graph for this contour using the chosen sampling scheme
            G, single_contour_resampled = _compute_single_graph(
                contour,
                pixel_distance=pixel_distance_used,
                node_count=self.node_count,
                normalize_by=self.normalize_by,
            )
            graphs[fname] = G
            resampled_contours[fname] = single_contour_resampled
            # Store metadata
            cell_info[fname] = {
                "resolution_um_per_pixel": resolution,
                "optimal_pixel_distance": optimal_pixel_distance,
                "pixel_distance_used": pixel_distance_used,
                "node_count": self.node_count,
                "mode": mode,
                "avg_resolution_used": resolution,
                "sampled_contour": single_contour_resampled
            }

        if self.plot:
            plot_cell_graphs(graphs, cell_info, save_path=save_path)

        return graphs, cell_info, resampled_contours