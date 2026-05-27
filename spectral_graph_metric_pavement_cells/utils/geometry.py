# spectral_graph_metric_pavement_cells/utils/geometry.py
from shapely import Polygon
from shapely.validation import explain_validity

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

def ensure_valid_polygon(poly: Polygon) -> Polygon:
    """Return polygon if valid, else rebuild from exterior only."""
    if poly.is_valid:
        return poly
    logger.warning(f"Invalid polygon detected: {explain_validity(poly)}")
    return Polygon(poly.exterior)