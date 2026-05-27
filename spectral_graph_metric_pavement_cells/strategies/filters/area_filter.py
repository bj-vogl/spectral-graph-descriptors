# GraphMetricOnPavementCells/strategies/filters/area_filter.py
from typing import Dict, Tuple, Any, Optional

import cv2
import numpy as np

from spectral_graph_metric_pavement_cells.core.interfaces.filter_strategy import FilterStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

class AreaFilter(FilterStrategy):
    """
    Filter graphs by cell area (in square micrometers).

    Returns:
      (kept_graphs, removed_graphs) -- both are Dict[str, nx.Graph]
    """

    def __init__(self, min_area_um2: float = 1400.0):
        self.min_area_um2 = float(min_area_um2)

    def filter(
            self,
            data: Dict[str, Any],
            meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if not data:
            return {}, {}

        kept = {}
        dropped = {}

        for name, entry in data.items():
            if not isinstance(entry, dict) or 'contour' not in entry:
                logger.warning(f"AreaFilter: Skipping {name}, expected contour dictionary entry.")
                continue

            try:
                contour_px = entry['original_coords_px']
                resolution = entry['resolution']

                c_int = contour_px.astype(np.int32)
                x, y, w, h = cv2.boundingRect(c_int)

                # Add 2px padding to ensure boundary fits
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                shifted_contour = c_int - [x - 1, y - 1]

                cv2.fillPoly(mask, [shifted_contour], 1)

                # 3. Calculate Area: Count Pixels * Pixel Area
                num_pixels = np.count_nonzero(mask)
                pixel_area_um2 = resolution[0] * resolution[1]
                area_um2 = num_pixels * pixel_area_um2
                if area_um2 >= self.min_area_um2:
                    kept[name] = entry
                    #logger.info(f"{name}: area={area_um2:.2f} um2 -> KEPT")
                else:
                    dropped[name] = entry
                    #logger.info(f"{name}: area={area_um2:.2f} um2 -> DROPPED")

            except Exception as e:
                logger.warning(f"Could not calculate area for {name}: {e}")
                dropped[name] = entry

        logger.info("dropped:\n%s", "\n".join(sorted(map(str, dropped.keys()))))
        logger.info(f"kept={len(kept)} dropped={len(dropped)}")
        return kept, dropped
