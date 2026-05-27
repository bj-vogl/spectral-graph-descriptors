from pathlib import Path
from typing import Optional, List

from spectral_graph_metric_pavement_cells.core.pipelines.persistence_pipeline import PersistencePipeline
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.strategies.persistence.joblib_persistence import JoblibPersistence
from spectral_graph_metric_pavement_cells.strategies.prediction.logistic_regression_prediction import \
    LogisticRegressionPrediction
from spectral_graph_metric_pavement_cells.strategies.readers.roi_reader import RoiReader
from spectral_graph_metric_pavement_cells.strategies.graphs.contour_graph import ContourGraph
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
TRAINING_CLASS_NAMES = ["Col-0", "OX16_11", "ktn1-5"]
SELECTED_GROUPS: Optional[List[str]] = [
    "mutants/Col-0",
    "mutants/ktn1-5",
    "mutants/OX16_11_1_4",
]

_sampled, _NAME_MAP = load_and_sample_rois(SELECTED_GROUPS, INPUT_DIR)

# ----------------------------
# Computation pipeline configurations
# ----------------------------
GRAPH_SAMPLING = {
    "mode": "node_count",
    "value": 30,
}
NORMALIZE_BY = 'max_dist'
WEIGHT_FUNC_NAME = "distance_norm"
ANALYSIS_FEATURE_KEY = "eigvals_Qw"
PLOT_CONTOUR = False
COMPUTE_HAUSDORFF = False

computation_pipeline = []

for group_rel_path, roi_files in _sampled.items():
    group_name = Path(group_rel_path).name

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

    group_map = {f: _NAME_MAP[f] for f in roi_files}

    entry = {
        "name": group_name,
        "reader": RoiReader(include_files=roi_files, name_map=group_map),
        "graph": graph,
        "distance": None,
        "data_path": str(INPUT_DIR / group_rel_path),
        "save": True,
        "analysis_feature_key": ANALYSIS_FEATURE_KEY,
        "weight_func_name": WEIGHT_FUNC_NAME,
        "plot_contour": PLOT_CONTOUR,
        "compute_hausdorff": COMPUTE_HAUSDORFF,
    }
    computation_pipeline.append(entry)

prediction_strategy = LogisticRegressionPrediction(
    scaler_name= 'None',
)
persistence_pipeline = PersistencePipeline(
    strategy=JoblibPersistence(),
    save_dir=str(MODELS_DIR)
)