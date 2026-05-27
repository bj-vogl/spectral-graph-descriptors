# spectral_graph_metric_pavement_cells/utils/sampling.py
from __future__ import annotations

from typing import Optional
import numpy as np
from shapely import LineString, Polygon
from spectral_graph_metric_pavement_cells.utils.geometry import ensure_valid_polygon

def resample_contour(
    contour: np.ndarray,
    pixel_distance: Optional[float] = None,
    node_count: Optional[int] = None,
) -> np.ndarray:
    """
    Resample contour either by pixel distance or by fixed node count.
    """
    poly = ensure_valid_polygon(Polygon(contour))
    line = LineString(poly.exterior.coords)
    perimeter = line.length

    if node_count is not None:
        if node_count < 0:
            raise ValueError("node_count must be a positive integer.")
        # distribute node_count points evenly along the perimeter
        distances = np.linspace(0, perimeter, node_count, endpoint=False)
        sampled = [line.interpolate(d) for d in distances]
    else:
        if pixel_distance is None:
            raise ValueError("Either pixel_distance or node_count must be provided.")
        n_points = int(np.ceil(perimeter / pixel_distance))
        if n_points < 3:
            n_points = 3
        distances = np.linspace(0, perimeter, n_points, endpoint=False)
        sampled = [line.interpolate(d) for d in distances]

    return np.array([[p.x, p.y] for p in sampled])
