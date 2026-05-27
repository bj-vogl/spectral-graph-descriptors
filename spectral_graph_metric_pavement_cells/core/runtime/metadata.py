# core/metadata.py
import inspect
import json
import os
from pathlib import Path
from typing import Any, Optional
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)


def serialize_obj(obj: Any, project_root: Optional[Path] = None) -> Any:
    """
    Convert objects to JSON-serializable representations.
    - For classes/instances: store class name as string.
    - For basic types: keep as-is.
    - If obj is a string that looks like a path, relativize it.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        if "/" in obj or obj.startswith("~"):
            return _relativize_path(obj, project_root)
        return obj
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, str) and ("/" in k or k.startswith("~")):
                new_key = _relativize_path(k, project_root)
            else:
                new_key = k
            new_dict[new_key] = serialize_obj(v, project_root)
        return new_dict
    if isinstance(obj, (list, tuple, set)):
        return [serialize_obj(v, project_root) for v in obj]
    cls = obj if inspect.isclass(obj) else obj.__class__
    module = cls.__module__
    name = cls.__name__
    return f"{module}.{name}"


def clean_reader_params(params: dict, project_root: Optional[Path] = None) -> dict:
    """
    Recursively remove 'mask' and 'resolution' from images.
    Also serialize all params using serialize_obj with project_root.
    """
    if not isinstance(params, dict):
        return params
    clean_params = {}
    for k, v in params.items():
        if k == "images" and isinstance(v, dict):
            new_images = {}
            for img_name, img_data in v.items():
                if isinstance(img_data, dict):
                    filtered_data = {
                        kk: serialize_obj(vv, project_root)
                        for kk, vv in img_data.items()
                        if kk not in ("mask", "resolution")
                    }
                    new_images[img_name] = filtered_data
                else:
                    new_images[img_name] = serialize_obj(img_data, project_root)
            clean_params[k] = new_images
        else:
            clean_params[k] = serialize_obj(v, project_root)
    return clean_params


def _derive_project_root_from_save(save_path: str) -> Optional[Path]:
    """
    save_path usually equals <PROJECT_ROOT>/data/results/<config_name>.
    -> project_root = parents[2].
    If this fails, return None.
    """
    try:
        sp = Path(save_path).resolve()
        return sp.parents[2]
    except Exception:
        return None


def _relativize_path(value: Any, project_root: Optional[Path]) -> Any:
    """
    Make a filesystem path safe for metadata:
    - If absolute and under project_root: store as project-root-relative POSIX path.
    - If absolute but outside project_root: store only the basename.
    - If already relative: keep as-is.
    Non-string inputs are returned unchanged.
    """
    if not isinstance(value, str):
        return value

    p = Path(os.path.expanduser(value))
    if not p.is_absolute():
        return value

    if project_root is None:
        return p.name

    try:
        rel = p.resolve().relative_to(project_root)
        return rel.as_posix()
    except Exception:
        return p.name


def _relativize_data_path(value: Any, project_root: Optional[Path]) -> Any:
    """Wrapper kept for clarity/semantics; delegates to _relativize_path."""
    return _relativize_path(value, project_root)


def save_pipeline_metadata(
    computation_pipeline: list,
    analysis_pipeline: Optional[list] = None,
    persistence_pipeline: Optional[list] = None,
    prediction_pipeline: Optional[list] = None,
    save_path: Optional[str] = None,
    config_name: Optional[str] = None,
):
    if save_path is None or config_name is None:
        logger.warning("save_path or config_name not provided. Metadata not saved.")
        return

    # Derive project root before writing anything
    project_root = _derive_project_root_from_save(save_path)
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{config_name}_metadata.json")

    # Serialize computation_pipeline
    serialized_configs = {}
    for cfg in computation_pipeline:
        cfg_name = cfg.get("name", "unnamed")
        serialized_cfg = {}
        for k, v in cfg.items():
            if k == "data_path":
                # Prevent absolute path leakage for dataset location
                serialized_cfg[k] = _relativize_data_path(v, project_root)
                continue

            if k in ("reader", "graph", "distance"):
                params = v.__dict__ if hasattr(v, "__dict__") else {}
                if k == "reader":
                    params = clean_reader_params(params, project_root)
                else:
                    params = {
                        key: serialize_obj(val, project_root)
                        for key, val in params.items()
                        if not key.startswith('_')
                    }
                serialized_cfg[k] = {
                    "class": serialize_obj(v, project_root),
                    "params": params,
                }
            else:
                serialized_cfg[k] = serialize_obj(v, project_root)
        serialized_configs[cfg_name] = serialized_cfg

    # Serialize analysis_pipelines
    serialized_analysis = []
    for cp in (analysis_pipeline or []):
        cp_dict = {
            "analysis_feature_keys": getattr(cp, "analysis_feature_keys", None),
            "plot": getattr(cp, "plot", None),
            "analysis_strategy": serialize_obj(getattr(cp, "analysis_strategy", None), project_root),
        }
        comp_results = getattr(cp, "results", {}) or {}
        res_dict = {}
        for name, res in comp_results.items():
            if res is None:
                continue
            res_dict[name] = {
                "analysis_feature_key": getattr(res, "analysis_feature_key", None),
                "manual_pixel_distance": getattr(res, "manual_pixel_distance", None),
                "node_coverage": getattr(res, "node_coverage", None),
                "plot_contour": getattr(res, "plot_contour", None),
                "plot_distance": getattr(res, "plot_distance", None),
            }
        cp_dict["results"] = res_dict
        serialized_analysis.append(cp_dict)

    # Serialize persistence_pipeline
    serialized_persistence = None
    if persistence_pipeline:
        # Handle case where it might be a single instance or a list
        pp_list = persistence_pipeline if isinstance(persistence_pipeline, list) else [persistence_pipeline]
        serialized_persistence = []
        for pp in pp_list:
            # Attempt to extract strategy from the inner context
            strategy_info = None
            if hasattr(pp, "context") and hasattr(pp.context, "_strategy"):
                strategy_info = serialize_obj(pp.context._strategy, project_root)

            pp_dict = {
                "class": serialize_obj(pp, project_root),
                "save_dir": _relativize_path(getattr(pp, "save_dir", None), project_root),
                "strategy": strategy_info
            }
            serialized_persistence.append(pp_dict)

    # Serialize prediction_pipeline
    serialized_prediction = None
    if prediction_pipeline:
        # Handle case where it might be a single instance or a list
        pred_list = prediction_pipeline if isinstance(prediction_pipeline, list) else [prediction_pipeline]
        serialized_prediction = []
        for pp in pred_list:
            # Attempt to extract strategy from the inner context
            strategy_info = None
            if hasattr(pp, "context") and hasattr(pp.context, "_strategy"):
                strategy_info = serialize_obj(pp.context._strategy, project_root)

            pp_dict = {
                "class": serialize_obj(pp, project_root),
                "strategy": strategy_info
            }
            serialized_prediction.append(pp_dict)

    data = {
        "computation_pipeline": serialized_configs,
        "analysis_pipeline": serialized_analysis,
        "persistence_pipeline": serialized_persistence,
        "prediction_pipeline": serialized_prediction,
        "save_path": _relativize_path(save_path, project_root),
        "config_name": config_name,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_logreg_cv_results(
    *,
    save_path: Optional[str],
    config_name: Optional[str],
    results: dict,
    filename_suffix: str = "logreg_cv_results",
) -> None:
    """
    Save Logistic Regression CV results as a JSON file next to the other metadata.
    - Uses the same path anonymization as save_pipeline_metadata.
    - 'results' should be JSON-serializable (use serialize_obj for classes if needed).
    """
    if save_path is None or config_name is None:
        logger.warning("save_path or config_name not provided. LogReg results not saved.")
        return

    project_root = _derive_project_root_from_save(save_path)
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{config_name}_{filename_suffix}.json")

    def _relativize_obj(obj: Any):
        if isinstance(obj, dict):
            return {k: _relativize_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_relativize_obj(v) for v in obj]
        if isinstance(obj, str) and ("/" in obj or obj.startswith("~")):
            return _relativize_path(obj, project_root)
        return serialize_obj(obj, project_root)

    payload = _relativize_obj(results)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[INFO] LogReg CV results saved: {out_file}")
