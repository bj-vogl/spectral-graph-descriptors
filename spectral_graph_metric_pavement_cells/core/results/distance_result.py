# core/distance_result.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class DistanceResult:
    """
    Stores results from a pipeline.
    """
    def __init__(
        self,
        distance_matrix: Optional[np.ndarray],
        features: Dict[str, Any],
        analysis_feature_key: str,
        manual_pixel_distance: float | None,
        node_coverage: float,
        plot_contour: bool,
        plot_distance: bool,
        contours: dict[str, np.ndarray] | None = None,
        cell_info: dict[str, dict[str, Any]] | None = None,
        um_per_pixel: tuple[float, float] = (1.0, 1.0),
    ):
        self.distance_matrix = distance_matrix
        self.analysis_feature_key = analysis_feature_key
        self.features = features
        self.manual_pixel_distance = manual_pixel_distance
        self.node_coverage = node_coverage
        self.plot_contour = plot_contour
        self.plot_distance = plot_distance
        self.contours = contours
        self.cell_info = cell_info
        self.um_per_pixel = um_per_pixel  # (µm/Pixel x, µm/Pixel y)