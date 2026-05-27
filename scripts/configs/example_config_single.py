from pathlib import Path

from spectral_graph_metric_pavement_cells.strategies.distances.euclidean_distance import EuclideanDistance
from spectral_graph_metric_pavement_cells.strategies.distances.ks_distance import KSDistance
from spectral_graph_metric_pavement_cells.strategies.graphs.pickle_graph import PickleGraph
from spectral_graph_metric_pavement_cells.strategies.readers.jpeg_reader import JpegReader
from spectral_graph_metric_pavement_cells.strategies.readers.pickle_reader import PickleReader
from spectral_graph_metric_pavement_cells.strategies.readers.roi_reader import RoiReader
from spectral_graph_metric_pavement_cells.strategies.graphs.contour_graph import ContourGraph
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = DATA_DIR / "results"
config_name = Path(__file__).stem

# ---------------------
# Global configuration
# ---------------------
READER_DEFAULT_RES = 0.2                # only for jpeg-input

GRAPH_SAMPLING = {
    "mode": "node_count",               # "node_count" or "manual_pixel_distance"
    "value": 25,                        # values from 30 to 140
}

NORMALIZE_BY = "max_dist"               # or "perimeter" or None
WEIGHT_FUNC_NAME = "distance_norm"      # or "distance" or "inv_distance_bounded" or "distance_rbf"
ANALYSIS_FEATURE_KEY = "eigvals_Lw"     # or "eigvals_L" or "eigvals_Qw" or "eigvals_Q" or "eigvals_Qsym" or "eigvals_Lsym" or "eigvals_Qw_sym" or "eigvals_Lw_sym"
PLOT_CONTOUR = True
COMPUTE_HAUSDORFF = False


def _set_contour_graph() -> ContourGraph:
    mode = GRAPH_SAMPLING["mode"]
    val = GRAPH_SAMPLING["value"]

    if mode == "node_count":
        return ContourGraph(
            node_count=int(val),
            plot=True,                  # toggle plotting of visibility graphs
            normalize_by=NORMALIZE_BY)
    elif mode == "manual_pixel_distance":
        return ContourGraph(
            manual_pixel_distance=float(val),
            plot=True,                  # toggle plotting of visibility graphs
            normalize_by=NORMALIZE_BY)
    else:
        raise ValueError("GRAPH_SAMPLING['mode'] must be 'node_count' or 'manual_pixel_distance'.")

# ----------------------------
# Computation pipeline
# ----------------------------
computation_pipeline = [
    {
        "name": "Col-0",
        "reader": RoiReader(),
        "graph": _set_contour_graph(),
        "distance": None,                   # No distance matrix since it's only one file
        "data_path": str(INPUT_DIR / "mutants/Col-0/rois/3-allRois/Roi_22.roi"),
        "save": True,                       # If True, the plots will be saved to disk but not shown, if False, the plots will be shown but not saved
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    },
    {
        "name": "ktn1",
        "reader": RoiReader(),
        "graph": _set_contour_graph(),
        "distance": None,                   # No distance matrix since it's only one file
        "data_path": str(INPUT_DIR / "mutants/ktn1-5/rois/4-allRois/Roi_1.roi"),
        "save": True,
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    },
{
        "name": "OX16",
        "reader": RoiReader(),
        "graph": _set_contour_graph(),
        "distance": None,                   # No distance matrix since it's only one file
        "data_path": str(INPUT_DIR / "mutants/OX16_11_1_4/rois/5-allRois/Roi_7.roi"),
        "save": True,
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    },
    {
        "name": "Convex",
        "reader": JpegReader(default_resolution=READER_DEFAULT_RES),
        "graph": _set_contour_graph(),
        "distance": EuclideanDistance(),    # or None or KSDistance()
        "data_path": str(INPUT_DIR / "convex_shapes"),
        "save": True,
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    },
    {
        "name": "Pickle",
        "reader": PickleReader(),
        "graph": PickleGraph(plot=PLOT_CONTOUR),
        "distance": KSDistance(),           # or None or EuclideanDistance()
        "data_path": str(INPUT_DIR / "GraVis_SourceData/FigS8/FigS8_visibilityGraphsPavementCells.gpickle"),
        "save": True,
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    },
]
