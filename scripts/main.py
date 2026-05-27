import sys
import importlib
import importlib.util
import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from spectral_graph_metric_pavement_cells.visualization.confusion import plot_confusion_heatmap
from spectral_graph_metric_pavement_cells.utils.common import extract_inference_data, smart_merge_for_training, load_config_from_path
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.utils.hausdorff import compute_hausdorff_distance
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger, setup_logging
from spectral_graph_metric_pavement_cells.core.pipelines.pipeline_runner import run_pipeline
from spectral_graph_metric_pavement_cells.core.runtime.metadata import save_pipeline_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = get_logger(__name__)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to a config .py file", default=None)
    args, _ = parser.parse_known_args()
    t0 = perf_counter()

    try:
        # ---------------------------------------------------------
        # Setup & Config Loading
        # ---------------------------------------------------------
        if args.config is None:
            config_name = "example_config_batch"
            config_module = importlib.import_module(f"scripts.configs.{config_name}")
        else:
            config_module, config_name = load_config_from_path(args.config)

        base_out = (PROJECT_ROOT / "data" / "results" / config_name)
        base_out.mkdir(parents=True, exist_ok=True)

        # Retrieve Pipelines from Config
        comp_configs = getattr(config_module, "computation_pipeline", [])
        analysis_pipelines = getattr(config_module, "analysis_pipeline", [])
        persist_pipe = getattr(config_module, "persistence_pipeline", None)
        predict_pipe = getattr(config_module, "prediction_pipeline", None)

        # Inference settings
        prediction_strategy = getattr(config_module, "prediction_strategy", None)
        inference_model_path = getattr(config_module, "inference_model_path", None)
        training_class_names = getattr(config_module, "TRAINING_CLASS_NAMES", None)

        computation_results: dict[str, DistanceResult] = {}
        analysis_results: dict[str, Any] = {}
        active_model: Optional[Any] = None

        logger.info(f"=== STARTING PIPELINE | CONFIG: {config_name} ===")

        # ---------------------------------------------------------
        # Computation Pipeline (Mandatory for generating features)
        # ---------------------------------------------------------
        logger.info("--- Computation Pipeline ---")
        if not comp_configs:
            logger.warning("No computation configs found.")
        else:
            for cfg in comp_configs:
                logger.info(f"Processing: {cfg['name']}")
                save_path = str(base_out / cfg["name"]) if cfg.get("save", False) else None

                result = run_pipeline(
                    reader=cfg["reader"],
                    graph=cfg["graph"],
                    distance=cfg["distance"],
                    data_path=cfg["data_path"],
                    save_path=save_path,
                    analysis_feature_key=cfg["analysis_feature_key"],
                    weight_func=cfg["weight_func_name"],
                    plot_contour=cfg.get("plot_contour", False),
                    filter_strategy=cfg.get("filter_strategy", None),
                )
                computation_results[cfg["name"]] = result

                # Optional: Hausdorff check
                if cfg.get("compute_hausdorff", False):
                    h_dict = compute_hausdorff_distance(result)
                    for k, v in h_dict.items():
                        logger.info(f"Hausdorff {k}: {v:.3f}" if v else f"Hausdorff {k}: None")

        # ---------------------------------------------------------
        # Analysis Pipeline (Optional)
        # ---------------------------------------------------------
        if analysis_pipelines:
            logger.info("--- Analysis Pipeline ---")
            for idx, cp in enumerate(analysis_pipelines):
                cp.set_results(computation_results)

                if hasattr(cp.analysis_strategy, "set_save_info"):
                    cp.analysis_strategy.set_save_info(str(base_out), config_name)

                res = cp.run_analysis()

                # Store result
                strat_name = type(cp.analysis_strategy).__name__
                key = f"{idx:02d}_{strat_name}"
                analysis_results[key] = res
        else:
            logger.info("--- Analysis Pipeline (Skipped) ---")

        # ---------------------------------------------------------
        # Persistence Pipeline (Optional)
        # ---------------------------------------------------------
        if persist_pipe:
            logger.info("--- Persistence Pipeline ---")

            # CASE A: SAVE (Training)
            if prediction_strategy:
                logger.info(f"Running final training using {type(prediction_strategy).__name__}")

                if computation_results and training_class_names:
                    feature_key_map = {cfg["name"]: cfg["analysis_feature_key"] for cfg in comp_configs}

                    try:
                        X, y, fnames, detected_classes = smart_merge_for_training(
                            computation_results,
                            feature_key_map,
                            training_class_names
                        )

                        logger.info(f"Data merged for training. Samples: {len(X)}.")

                        fname = f"{config_name}_final_model.joblib"
                        active_model = persist_pipe.train_and_save(prediction_strategy, X, y, fname)
                        logger.info(f"Model trained and saved to {fname}")

                    except Exception as e:
                        logger.error(f"Failed to prepare training data: {e}")
                        active_model = None
                else:
                    logger.warning("No computation results or TRAINING_CLASS_NAMES available for training.")

            # CASE B: LOAD
            elif inference_model_path:
                logger.info(f"Loading inference model: {inference_model_path}")
                try:
                    active_model = persist_pipe.load_model(inference_model_path)
                    logger.info("Model loaded.")
                except Exception as e:
                    logger.error(f"Load failed: {e}")
            else:
                logger.warning("No prediction_strategy or inference_model_path defined.")
        else: logger.info("--- Persistence Pipeline (Skipped) ---")

        # ---------------------------------------------------------
        # Prediction Pipeline (Optional)
        # ---------------------------------------------------------
        if predict_pipe and active_model:
            logger.info("--- Prediction Pipeline ---")
            y_true_all = []
            y_pred_all = []

            for name, res in computation_results.items():
                ids, X_inf = extract_inference_data(res)
                if not ids:
                    logger.warning(f"No features for {name}.")
                    continue

                logger.info(f"Predicting {len(ids)} items in {name}")
                preds = predict_pipe.run_inference(active_model, X_inf)

                for i, p in zip(ids, preds):
                    status = ""
                    pred_str = str(p)

                    # If we have the mapping from training
                    if training_class_names and isinstance(p, (int, np.integer)):
                        if 0 <= p < len(training_class_names):
                            pred_str = training_class_names[p]

                    ground_truth = name
                    matched_gt = ground_truth
                    if training_class_names:
                        for tcn in training_class_names:
                            if tcn in ground_truth:
                                matched_gt = tcn
                                break

                    y_true_all.append(matched_gt)
                    y_pred_all.append(pred_str)

                    status = " [CORRECT]" if matched_gt == pred_str else f" [WRONG, exp {matched_gt}]"
                    logger.info(f"  {Path(i).name} -> {pred_str} ({p}){status}")

            if y_true_all:
                try:
                    acc = accuracy_score(y_true_all, y_pred_all)
                    total_samples = len(y_true_all)
                    correct_samples = int(acc * total_samples)
                    logger.info(f"Overall Prediction Accuracy: {acc:.2%} ({correct_samples}/{total_samples})")
                except Exception as e:
                    logger.error(f"Failed to calculate accuracy: {e}")

                logger.info("Generating Prediction Confusion Matrix...")
                try:
                    fig, ax = plot_confusion_heatmap(
                        y_true=y_true_all,
                        y_pred_clusters=y_pred_all,
                        normalize="true",
                        title=f"Inference Results: {config_name}",
                        x_label="Vorhergesagte Klasse"
                    )

                    save_path = base_out / f"{config_name}_inference_confusion.png"
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)
                    logger.info(f"Confusion matrix saved to: {save_path}")
                except Exception as e:
                    logger.error(f"Failed to plot confusion matrix: {e}")


        elif predict_pipe:
            logger.warning("Prediction active but no model available.")
        else:
            logger.info("--- Prediction Pipeline (Skipped) ---")

        # ---------------------------------------------------------
        # Metadata Saving
        # ---------------------------------------------------------
        save_pipeline_metadata(
            computation_pipeline=comp_configs,
            analysis_pipeline=analysis_pipelines,
            persistence_pipeline=persist_pipe,
            prediction_pipeline=predict_pipe,
            save_path=str(base_out),
            config_name=config_name,
        )

        logger.info("Pipeline completed. Total runtime: %.3f s", perf_counter() - t0)
        return 0
    except Exception as exc:
        logger.exception("Fatal error during execution: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
