# spectral_graph_metric_pavement_cells/configs/grid_search_config.py
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from spectral_graph_metric_pavement_cells.core.pipelines.analysis_pipeline import AnalysisPipeline
from spectral_graph_metric_pavement_cells.strategies.analysis.logistic_regression_analysis import \
    LogisticRegressionAnalysis
from spectral_graph_metric_pavement_cells.strategies.analysis.random_forest_analysis import RandomForestAnalysis
from spectral_graph_metric_pavement_cells.strategies.readers.roi_reader import RoiReader
from spectral_graph_metric_pavement_cells.strategies.graphs.contour_graph import ContourGraph
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger


logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = DATA_DIR / "results"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("__file__ resolved:", Path(__file__).resolve())

config_name = Path(__file__).stem
timestamp = datetime.now().strftime("%d%m_%H-%M-%S")

DATASETS = {
    "mutants": [
        "mutants/Col-0",
        "mutants/OX16_11_1_4",
        "mutants/ktn1-5",
    ],
    "Zeitserie": [
        "Zeitserie/3_day_Coty",
        "Zeitserie/5_day_Coty",
        "Zeitserie/7_day_Coty",
    ],
}
N_FOLDS_PER_DATASET = {
    "mutants": 10,
    "Zeitserie": 5,
}

ANALYSIS_FEATURE_KEYS = ["eigvals_Q", "eigvals_Qw", "eigvals_Qsym", "eigvals_Qw_sym", "eigvals_L", "eigvals_Lsym", "eigvals_Lw", "eigvals_Lw_sym"]
WEIGHT_FUNCS = ["distance_norm", "distance", "inv_distance_bounded", "distance_rbf"]
NODE_COUNTS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
SCALERS = [None, "StandardScaler", "GlobalIQRScaler", "GlobalStandardScaler", "MaxnormScaler"] # only for LogisticRegression
NORMALIZE_BY_OPTIONS = ["max_dist", "convex_hull"] # only for 'distance_norm'

def make_display_name(abs_path: str) -> str:
    """
    Creates deterministic display names.
    """
    p = Path(abs_path)
    try:
        rel_p = p.relative_to(INPUT_DIR)
        clean_name = str(rel_p).replace("\\", "_").replace("/", "_")
        return clean_name
    except ValueError:
        return f"{p.parent.name}_{p.name}"

def build_experiment(
    dataset_name: str,
    analysis_feature_key: str,
    weight_func: str,
    node_count: int,
    scaler_name: str = None,
    normalize_by: Optional[str] = None,
):
    """
    Creates the sampling + computation_pipeline + analysis_pipeline
    for exactly one grid combination.
    """
    if weight_func == "distance_norm" and not normalize_by:
        raise RuntimeError("distance_norm weight requires normalize_by to be set (one of NORMALIZE_BY_OPTIONS)")

    if dataset_name == "Zeitserie":
        n_folds = N_FOLDS_PER_DATASET.get("Zeitserie", 5)
    elif dataset_name == "mutants":
        n_folds = N_FOLDS_PER_DATASET.get("mutants", 10)

    groups_available = DATASETS[dataset_name]

    # -------------------------
    # 1. Loading & Sampling (Deterministic Shuffle)
    # -------------------------
    _sampled = {}
    _NAME_MAP = {}

    rng = random.Random(42)

    logger.info(f"--- Building Experiment for {dataset_name} | Node: {node_count} | {weight_func} ---")

    for group_rel in groups_available:
        group_base_path = INPUT_DIR / group_rel
        search_path = group_base_path / "rois"

        if not search_path.exists():
            logger.warning(f"Path not found: {search_path}")
            continue

        found_paths = sorted(search_path.rglob("*-allRois/*.roi"))

        if not found_paths:
            logger.warning(f"No .roi files found in {search_path}")
            continue

        rng.shuffle(found_paths)

        files_abs = []
        for p in found_paths:
            abs_p = str(p.resolve())
            files_abs.append(abs_p)
            _NAME_MAP[abs_p] = make_display_name(abs_p)

        _sampled[str(group_rel)] = files_abs
        logger.debug(f"Loaded {len(files_abs)} files for group '{group_rel}'")

    # -------------------------
    # Computation pipelines
    # -------------------------
    computation_pipeline = []

    for group_rel_path, roi_files in _sampled.items():
        group_name = Path(group_rel_path).name

        group_map = {f: _NAME_MAP[f] for f in roi_files}

        graph = ContourGraph(
            node_count=node_count,
            plot=False,
            normalize_by=normalize_by,
        )

        entry = {
            "name": group_name,
            "reader": RoiReader(include_files=roi_files, name_map=group_map),
            "graph": graph,
            "distance": None,
            "data_path": str(INPUT_DIR / group_rel_path),
            "save": False,
            "analysis_feature_key": analysis_feature_key,
            "weight_func_name": weight_func,
            "plot_contour": False,
            "compute_hausdorff": False,
        }
        computation_pipeline.append(entry)

    # ---------------------------------------
    # Analysis Pipeline →  LogReg + RandomForest
    # ---------------------------------------
    analysis_feature_keys: Dict[str, str] = {
        cfg["name"]: cfg["analysis_feature_key"]
        for cfg in computation_pipeline
    }
    cfg_base_name = f"{config_name}_{dataset_name}_{analysis_feature_key}_{node_count}_{scaler_name}_{weight_func}"
    if normalize_by:
        cfg_base_name = f"{cfg_base_name}_normby_{normalize_by}"

    analysis_pipeline = [
        AnalysisPipeline(
            analysis_strategy=LogisticRegressionAnalysis(
                analysis_feature_keys=analysis_feature_keys,
                scaler_name=scaler_name,
                n_folds=n_folds,
            ),
            analysis_feature_keys=analysis_feature_keys,
            config_name=f"{cfg_base_name}_logreg",
        ),
        AnalysisPipeline(
            analysis_strategy=RandomForestAnalysis(
                analysis_feature_keys=analysis_feature_keys,
                n_folds=n_folds,
            ),
            analysis_feature_keys=analysis_feature_keys,
            config_name=f"{cfg_base_name}_rf",
        ),
    ]

    return computation_pipeline, analysis_pipeline
