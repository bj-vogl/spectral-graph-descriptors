import importlib
import importlib.util
import random
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

def load_config_from_path(path_str: str):
    """Load a Python config module from a file path."""
    p = Path(path_str).resolve()
    if not p.exists():
        logger.error("Config file not found: %s", p)
        raise FileNotFoundError("Config file not found.")
    if p.suffix != ".py":
        raise ValueError(f"--config must be a .py file path, got: {p.name}")
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, p.stem

def smart_merge_for_training(
        results: Dict[str, DistanceResult],
        feature_key_map: Dict[str, str],
        class_names: List[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Manually merges data based on provided class names logic.
    Searches for class_name inside the group_name.
    """
    X_list = []
    y_list = []
    fnames_list = []

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    counts = {name: 0 for name in class_names}

    for group_name, res in results.items():
        if group_name not in feature_key_map:
            continue
        key = feature_key_map[group_name]

        found_label = None
        for cls in class_names:
            if cls in group_name:
                found_label = cls
                break

        if found_label is None:
            logger.warning(f"Skipping group '{group_name}' because it matches none of {class_names}")
            continue

        label_idx = class_to_idx[found_label]

        for fname, feat_dict in res.features.items():
            vec = resolve_feature_vector(feat_dict, key, debug_info=f"{group_name}|{Path(fname).name}")
            if vec is not None:
                X_list.append(vec)
                y_list.append(label_idx)
                fnames_list.append(f"{group_name} {fname}")
                counts[found_label] += 1

    if not X_list:
        raise ValueError("No valid features found for training.")

    logger.info(f"Smart Merge Counts: {counts}")
    return np.array(X_list), np.array(y_list), np.array(fnames_list), class_names


def load_and_sample_rois(
        selected_groups: List[str],
        input_dir: Path,
        seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Scans the input directory for .roi files within the specified groups,
    shuffles them deterministically, and creates a display name mapping.
    """
    sampled = {}
    name_map = {}
    rng = random.Random(seed)

    for group_rel in selected_groups:
        group_base_path = input_dir / group_rel
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
            name_map[abs_p] = make_display_name(abs_p, input_dir)

        sampled[str(group_rel)] = files_abs
        logger.info(f"Loaded {len(files_abs)} files for group '{group_rel}'")

    return sampled, name_map

def make_display_name(file_path: Union[str, Path], input_dir: Path) -> str:
    """
    Creates a deterministic display name for a given file path relative to the input directory.

    Args:
        file_path: The absolute path to the file (string or Path object).
        input_dir: The reference input directory (Path object).

    Returns:
        A cleaned string representation of the relative path.
    """
    p = Path(file_path)
    try:
        rel_p = p.relative_to(input_dir)
        clean_name = str(rel_p).replace("\\", "_").replace("/", "_")
        return clean_name
    except ValueError:
        # Fallback if the file is not within input_dir
        return f"{p.parent.name}_{p.name}"

def resolve_feature_vector(feature_data: Any, target_key: str, debug_info: str = "") -> Optional[np.ndarray]:
    """
    Extracts a numpy array.
    debug_info: string used for logging if extraction fails.
    """
    raw_val = None

    if isinstance(feature_data, dict):
        if target_key and target_key in feature_data:
            raw_val = feature_data[target_key]
        else:
            available_keys = list(feature_data.keys())
            # logger.warning(f"[{debug_info}] Key '{target_key}' not found in dict. Available: {available_keys}")
            return None
    else:
        raw_val = feature_data

    try:
        vec = np.asarray(raw_val, dtype=float).flatten()
        if vec.size > 0:
            return vec
        else:
            logger.warning(f"[{debug_info}] Vector is empty/size 0.")
            return None
    except Exception as e:
        logger.warning(f"[{debug_info}] Failed to convert to numpy: {e}")
        return None

def extract_inference_data(result: DistanceResult) -> Tuple[List[str], Any]:
    """Converts a single result into X (features) and IDs for inference."""
    ids = []
    vectors = []
    target_key = result.analysis_feature_key

    if result.features:
        for item_id, feature_data in result.features.items():
            vec = resolve_feature_vector(feature_data, target_key)
            if vec is not None:
                ids.append(item_id)
                vectors.append(vec)

    return ids, np.array(vectors)

def compute_sigma_median_pairwise_exact(points: np.ndarray) -> float:
    """
    Compute sigma as the median of all pairwise Euclidean distances among `points`.
    points: (n,2) array-like
    Returns a positive float.
    """
    pts = np.asarray(points, dtype=float)
    try:
        dists = pdist(pts, metric="euclidean")
        med = float(np.median(dists)) / 3.0
        return med if (np.isfinite(med) and med > 0.0) else float(1e-6)
    except Exception:
        return float(35.0)

def compute_convex_hull_scale(cell_points: np.ndarray) -> float:
    if len(cell_points) < 3:
        return 1.0
    hull = ConvexHull(cell_points)
    area = hull.volume
    if area <= 0 or not np.isfinite(area):
        logger.warning(f"Invalid convex hull area: {area}; defaulting to 1.0")
        return 1.0
    return np.sqrt(area)

def merge_across_pipelines(results: Dict[str, DistanceResult], analysis_feature_keys: Dict[str, str]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Merges features from several pipelines
    """
    cell_groups = list(results.keys())

    def _group_prefix(name):
        parts = name.split("_")
        if len(parts) < 2: return name
        return "_".join(parts[:2])

    class_names = sorted({_group_prefix(g) for g in cell_groups})
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    rows, y_int, fnames = [], [], []
    counts = {cls: 0 for cls in class_names}

    for group in cell_groups:
        # 1. Check Key
        if group not in analysis_feature_keys:
            logger.error(f"CRITICAL: No analysis key configured for group '{group}'!")
            continue

        key = analysis_feature_keys[group]
        cls = _group_prefix(group)
        idx = class_to_idx[cls]

        # 2. Check Features Existence
        res = results[group]
        if not res.features:
            logger.warning(f"DEBUG MERGE: Group '{group}' has empty .features dictionary!")
            continue

        # 3. Iterate Items
        success_count = 0
        fail_count = 0

        for fname, feat_dict in res.features.items():
            vec = resolve_feature_vector(feat_dict, key, debug_info=f"{group}|{Path(fname).name}")

            if vec is not None:
                rows.append(vec)
                y_int.append(idx)
                fnames.append(f"{group} {fname}")
                counts[cls] += 1
                success_count += 1
            else:
                fail_count += 1
                # Log first 3 failures only
                if fail_count <= 3:
                    logger.warning(f"  > Failed to resolve vector for {Path(fname).name}. Data type: {type(feat_dict)}")


    if not rows:
        raise ValueError("No features found across all pipelines!")

    logger.info(f"Final counts per class: {counts}")

    return np.vstack(rows), np.asarray(y_int, dtype=int), np.asarray(fnames), class_names

def file_is_empty(path: Path) -> bool:
    """Check whether a path is missing or empty. Returns True if missing or size==0."""
    try:
        return (not path.exists()) or path.stat().st_size == 0
    except Exception:
        return True