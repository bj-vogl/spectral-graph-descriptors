# spectral_graph_metric_pavement_cells/utils/visibility.py
import numpy as np
from shapely.geometry import LineString
from shapely import Polygon

def visibility_edge_ok(
    polygon_buffered: Polygon,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> bool:
    """
    Parameters
    ----------
    polygon_buffered : shapely.geometry.Polygon
        Slightly buffered polygon to mitigate numerical boundary issues.
    point_a, point_b : np.ndarray
        Endpoints (x, y) of the candidate edge.

    Returns
    -------
    bool
        True if line lies inside the buffered polygon.
    """
    # Build the candidate segment between the two points
    line = LineString([point_a, point_b])

    return polygon_buffered.covers(line)