import math

import numpy as np
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

# --- GLOBAL per-cell normalization scale (set from ContourGraph) ---
_CELL_NORM_SCALE: float | None = None
_CELL_RBF_SIGMA: float | None = None

def set_cell_norm_scale(scale: float | None) -> None:
    """
    Set a per-cell normalization scale used by the *_norm_weight functions.
    """
    global _CELL_NORM_SCALE
    if scale is None or not np.isfinite(scale) or scale <= 0.0:
        _CELL_NORM_SCALE = None
    else:
        _CELL_NORM_SCALE = float(scale)

def set_cell_rbf_sigma(sigma: float | None) -> None:
    """
    Set a per-cell RBF sigma used by distance_rbf when called with sigma=None.
    """
    global _CELL_RBF_SIGMA
    if sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
        _CELL_RBF_SIGMA = None
        logger.warning("Setting sigma to None will result in an error.")
    else:
        _CELL_RBF_SIGMA = float(sigma)

def _dist(p1, p2) -> float:
    """Euclidean distance."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

def inv_distance_weight(p1, p2) -> float:
    """Weight = 1 / distance"""
    eps: float = 1e-12
    d = _dist(p1, p2)
    return 1.0 / (d + eps)

def inv_distance_bounded(p1, p2) -> float:
    """Bounded inverse: returns 1/(1 + d) so weight is in (0,1]."""
    d = _dist(p1, p2)
    return 1.0 / (1.0 + d)

def distance_rbf(p1, p2) -> float:
    d = _dist(p1, p2)
    sigma = _CELL_RBF_SIGMA
    return math.exp(-(d / sigma)**2)

def distance_weight(p1, p2) -> float:
    """Weight = distance"""
    return _dist(p1, p2)

def distance_norm_weight(p1, p2) -> float:
    """
    Normalized distance: (d / s)
    where s is set globally via set_cell_norm_scale(...) by ContourGraph.
    """
    d = _dist(p1, p2)
    if _CELL_NORM_SCALE is not None and _CELL_NORM_SCALE > 0.0:
        s = _CELL_NORM_SCALE
    else:
        raise ValueError(f"Normalization scale has not been set for distance_norm_.")
    dn = d / s
    return dn
