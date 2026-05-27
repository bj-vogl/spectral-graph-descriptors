from pathlib import Path
from typing import Optional, List

from spectral_graph_metric_pavement_cells.core.pipelines.persistence_pipeline import PersistencePipeline
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.strategies.persistence.joblib_persistence import JoblibPersistence
from spectral_graph_metric_pavement_cells.strategies.prediction.logistic_regression_prediction import \
    LogisticRegressionPrediction
from spectral_graph_metric_pavement_cells.strategies.readers.roi_reader import RoiReader
from spectral_graph_metric_pavement_cells.strategies.graphs.contour_graph import ContourGraph
from spectral_graph_metric_pavement_cells.strategies.distances.euclidean_distance import EuclideanDistance
from spectral_graph_metric_pavement_cells.utils.common import load_and_sample_rois

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

config_name = Path(__file__).stem

# ------------------------
# Data selection
# ------------------------
TRAINING_CLASS_NAMES = ["3_day_Coty", "5_day_Coty", "7_day_Coty"]
SELECTED_GROUPS: Optional[List[str]] = [
    "Zeitserie/3_day_Coty",
    "Zeitserie/5_day_Coty",
    "Zeitserie/7_day_Coty"
]

_sampled, _NAME_MAP = load_and_sample_rois(SELECTED_GROUPS, INPUT_DIR)

# ----------------------------
# Computation pipeline configurations
# ----------------------------
GRAPH_SAMPLING = {
    "mode": "node_count",
    "value": 40,
}
NORMALIZE_BY = None
WEIGHT_FUNC_NAME = "distance"
ANALYSIS_FEATURE_KEY = "eigvals_Qw"
PLOT_CONTOUR = False
COMPUTE_HAUSDORFF = False

computation_pipeline = []
for day_rel, roi_files in _sampled.items():
    group_name = Path(day_rel).name
    mode = GRAPH_SAMPLING["mode"]
    val = GRAPH_SAMPLING["value"]

    if mode == "node_count":
        graph = ContourGraph(
            node_count=int(val),
            plot=False,
            normalize_by=NORMALIZE_BY,
        )
    elif mode == "manual_pixel_distance":
        graph = ContourGraph(
            manual_pixel_distance=float(val),
            plot=False,
            normalize_by=NORMALIZE_BY,
        )
    else:
        raise ValueError("GRAPH_SAMPLING['mode'] must be 'node_count' or 'manual_pixel_distance'.")

    group_map = {str(Path(p).resolve()): _NAME_MAP[str(Path(p).resolve())] for p in roi_files}

    entry = {
        "name": group_name,
        "reader": RoiReader(include_files=roi_files, name_map=group_map),
        "graph": graph,
        "distance": EuclideanDistance(),
        "data_path": str(INPUT_DIR / day_rel),
        "save": True,
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    }
    computation_pipeline.append(entry)

prediction_strategy = LogisticRegressionPrediction(
    scaler_name="StandardScaler",
)
persistence_pipeline = PersistencePipeline(
    strategy=JoblibPersistence(),
    save_dir=str(MODELS_DIR)
)