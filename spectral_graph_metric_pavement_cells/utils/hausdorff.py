# /utils/hausdorff.py
from typing import Dict, Optional
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)


def compute_hausdorff_distance(distance_result: DistanceResult) -> Dict[str, Optional[float]]:
    """
    Compute Hausdorff distances between the original contour and the resampled contour.

    Args:
        distance_result: DistanceResult object containing cell_info with resampled contours.

    Returns:
        Dictionary mapping each cell name to its Hausdorff distance.
        If contour information is missing, value will be None.
    """
    hausdorff_dict = {}

    for fname in distance_result.cell_info:
        cell_entry = distance_result.cell_info[fname]

        # Check if sampled contour is available
        if "sampled_contour" not in cell_entry or cell_entry["sampled_contour"] is None:
            hausdorff_dict[fname] = None
            continue

        # Check if original contour is available
        if fname not in distance_result.contours or "contour" not in distance_result.contours[fname]:
            hausdorff_dict[fname] = None
            continue

        sampled_contour = cell_entry["sampled_contour"]
        original_contour = distance_result.contours[fname]["contour"]

        sampled_contour = np.array(sampled_contour)
        original_contour = np.array(original_contour)

        if sampled_contour.size == 0 or original_contour.size == 0:
            hausdorff_dict[fname] = None
            continue

        # Compute Hausdorff distance
        d1 = directed_hausdorff(sampled_contour, original_contour)[0]
        d2 = directed_hausdorff(original_contour, sampled_contour)[0]
        hausdorff_dict[fname] = max(d1, d2)

    return hausdorff_dict