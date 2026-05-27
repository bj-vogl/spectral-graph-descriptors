from pathlib import Path
from typing import Optional, List

from spectral_graph_metric_pavement_cells.core.pipelines.analysis_pipeline import AnalysisPipeline
from spectral_graph_metric_pavement_cells.strategies.analysis.logistic_regression_analysis import \
    LogisticRegressionAnalysis
from spectral_graph_metric_pavement_cells.strategies.analysis.random_forest_analysis import RandomForestAnalysis
from spectral_graph_metric_pavement_cells.strategies.distances.euclidean_distance import EuclideanDistance
from spectral_graph_metric_pavement_cells.strategies.distances.ks_distance import KSDistance
from spectral_graph_metric_pavement_cells.strategies.readers.roi_reader import RoiReader
from spectral_graph_metric_pavement_cells.strategies.graphs.contour_graph import ContourGraph
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.common import load_and_sample_rois

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = DATA_DIR / "results"
config_name = Path(__file__).stem

# ------------------------
# Data selection
# ------------------------
SELECTED_GROUPS: Optional[List[str]] = [
    "Zeitserie/3_day_Coty",
    "Zeitserie/5_day_Coty",
    "Zeitserie/7_day_Coty"
]
# or
# SELECTED_GROUPS: Optional[List[str]] = [
#     "mutants/Col-0",
#     "mutants/OX16_11_1_4",
#     "mutants/ktn1-5",
# ]

_sampled, _NAME_MAP = load_and_sample_rois(SELECTED_GROUPS, INPUT_DIR)

# ----------------------------
# Computation pipeline configurations
# ----------------------------
GRAPH_SAMPLING = {
    "mode": "node_count",          # "node_count" or "manual_pixel_distance"
    "value": 50,                   # values from 30 to 140
}

NORMALIZE_BY = "max_dist"               # or "perimeter" or None
WEIGHT_FUNC_NAME = "distance_norm"      # or "distance" or "inv_distance_bounded" or "distance_rbf"
ANALYSIS_FEATURE_KEY = "eigvals_Lw"     # or "eigvals_L" or "eigvals_Qw" or "eigvals_Q" or "eigvals_Qsym" or "eigvals_Lsym" or "eigvals_Qw_sym" or "eigvals_Lw_sym"
PLOT_CONTOUR = True
COMPUTE_HAUSDORFF = False
DISTANCE = KSDistance()                         # or EuclideanDistance() or KSDistance()

computation_pipeline = []

for group_rel_path, roi_files in _sampled.items():
    group_name = Path(group_rel_path).name

    mode = GRAPH_SAMPLING["mode"]
    val = GRAPH_SAMPLING["value"]

    if mode == "node_count":
        graph = ContourGraph(
            node_count=int(val),
            plot=True,                          # toggle plotting of visibility graphs
            normalize_by=NORMALIZE_BY,
        )
    elif mode == "manual_pixel_distance":
        graph = ContourGraph(
            manual_pixel_distance=float(val),
            plot=False,                         # toggle plotting of visibility graphs
            normalize_by=NORMALIZE_BY,
        )
    else:
        raise ValueError("GRAPH_SAMPLING['mode'] must be 'node_count' or 'manual_pixel_distance'.")

    group_map = {f: _NAME_MAP[f] for f in roi_files}

    entry = {
        "name": group_name,
        "reader": RoiReader(include_files=roi_files, name_map=group_map),
        "graph": graph,
        "distance": DISTANCE,
        "data_path": str(INPUT_DIR / group_rel_path),
        "save": True,                           # If True, the plots will be saved to disk but not shown; if False, the plots will be shown but not saved
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    }
    computation_pipeline.append(entry)

# ----------------------------
# Analysis pipeline configuration
# ----------------------------
analysis_feature_keys = {cfg["name"]: cfg["analysis_feature_key"] for cfg in computation_pipeline}

analysis_pipeline = [
    AnalysisPipeline(
        analysis_strategy=LogisticRegressionAnalysis(
            analysis_feature_keys=analysis_feature_keys,
            scaler_name="StandardScaler",   # or None or "GlobalStandardScaler" or "GlobalIQRScaler" or "MaxnormScaler"
            n_folds=5,
        ),
        analysis_feature_keys=analysis_feature_keys,
        config_name=config_name,
    ),
    AnalysisPipeline(
        analysis_strategy=RandomForestAnalysis(
            analysis_feature_keys=analysis_feature_keys,
            n_folds=5,
        ),
        analysis_feature_keys=analysis_feature_keys,
        config_name=config_name,
    ),
]