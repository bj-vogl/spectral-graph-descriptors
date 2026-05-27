# spectral_graph_metric_pavement_cells/strategies/analysis/random_forest_analysis.py
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from spectral_graph_metric_pavement_cells.core.interfaces.analysis_strategy import AnalysisStrategy
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.common import merge_across_pipelines
from spectral_graph_metric_pavement_cells.utils.summarization import prepare_output_file, append_run_summary
from spectral_graph_metric_pavement_cells.visualization.confusion import plot_confusion_heatmap

logger = get_logger(__name__)

class RandomForestAnalysis(AnalysisStrategy):
    """
    Random Forest across ALL computation pipelines.
    Parameters:
      analysis_feature_keys: mapping pipeline_name -> feature_key
      n_folds: number of folds for cross-validation
    """

    def __init__(self, analysis_feature_keys: Dict[str, str], n_folds: Optional[int] = 10):
        self.analysis_feature_keys = analysis_feature_keys
        self.n_folds = n_folds
        self._save_info: tuple[Optional[str], Optional[str]] = (None, None)

    def set_save_info(self, save_path: Optional[str], config_name: Optional[str]) -> None:
        self._save_info = (save_path, config_name)

    def analyze(self, results: Dict[str, DistanceResult]):
        save_path, cfg = self._save_info
        summary_master = None
        run_id = None
        ts = None

        if save_path and cfg:
            try:
                summary_master, run_id, ts = prepare_output_file(save_path, cfg)
            except Exception:
                summary_master = None
                run_id = ts = None

        try:
            # Merge features across pipelines
            X, y, fnames_arr, class_names = merge_across_pipelines(results, self.analysis_feature_keys)
        except Exception as e:
            logger.error("RandomForest failed to merge datasets: %s", e)
            return {"error": str(e)}

        n_folds = self.n_folds
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)

        acc_scores: List[float] = []
        prec_scores: List[float] = []
        rec_scores: List[float] = []
        f1_scores: List[float] = []

        all_true_idx: List[np.ndarray] = []
        all_pred_idx: List[np.ndarray] = []

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            acc = rf.score(X_test, y_test)
            p_macro, r_macro, _, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)

            acc_scores.append(acc)
            prec_scores.append(p_macro)
            rec_scores.append(r_macro)
            f1_scores.append(f1_macro)

            all_true_idx.append(y_test)
            all_pred_idx.append(y_pred)

        y_true_all = np.concatenate(all_true_idx)
        y_pred_all = np.concatenate(all_pred_idx)
        y_true_txt = [class_names[i] for i in y_true_all]
        y_pred_txt = [class_names[i] for i in y_pred_all]

        fig, _ = plot_confusion_heatmap(
            y_true=y_true_txt,
            y_pred_clusters=y_pred_txt,
            normalize="true",
            title=f"RandomForest Confusion (Overall CV)",
            x_label="Predicted Class",
        )
        if save_path and cfg and ts:
            outdir = Path(save_path)
            cv_png = outdir / f"{cfg}_randomforest_confusion_{ts}.png"
            fig.savefig(cv_png, dpi=300, bbox_inches="tight")

        #plt.show()
        plt.close(fig)

        acc_arr = np.asarray(acc_scores, dtype=float)
        prec_arr = np.asarray(prec_scores, dtype=float)
        rec_arr = np.asarray(rec_scores, dtype=float)
        f1_arr = np.asarray(f1_scores, dtype=float)
        logger.info(
            "samples=%d, dim=%d, classes=%d, folds=%d\n"
            "Accuracy: Mean %.2f%% ± %.2f%%\n"
            "Precision (macro): Mean %.2f%% ± %.2f%%\n"
            "Recall (macro): Mean %.2f%% ± %.2f%%\n"
            "F1 (macro): Mean %.2f%% ± %.2f%%",
            X.shape[0], X.shape[1], len(np.unique(y)), n_folds,
            acc_arr.mean() * 100, acc_arr.std(ddof=1) * 100,
            prec_arr.mean() * 100, prec_arr.std(ddof=1) * 100,
            rec_arr.mean() * 100, rec_arr.std(ddof=1) * 100,
            f1_arr.mean() * 100, f1_arr.std(ddof=1) * 100,
        )

        summary_row: Dict[str, Any] = {
            "run_id": run_id,
            "config": cfg,
            "timestamp_utc": ts,
            "n_samples": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
            "cv_folds": n_folds,
            "acc_fold_mean": float(acc_arr.mean()) if acc_arr.size > 0 else float(np.nan),
            "acc_fold_std": float(acc_arr.std(ddof=1)) if acc_arr.size > 1 else float(0.0),
            "prec_macro_mean": float(prec_arr.mean()) if prec_arr.size > 0 else float(np.nan),
            "prec_macro_std": float(prec_arr.std(ddof=1)) if prec_arr.size > 1 else float(0.0),
            "rec_macro_mean": float(rec_arr.mean()) if rec_arr.size > 0 else float(np.nan),
            "rec_macro_std": float(rec_arr.std(ddof=1)) if rec_arr.size > 1 else float(0.0),
            "f1_macro_mean": float(f1_arr.mean()) if f1_arr.size > 0 else float(np.nan),
            "f1_macro_std": float(f1_arr.std(ddof=1)) if f1_arr.size > 1 else float(0.0),
        }
        for k in range(n_folds):
            if k < len(acc_arr):
                summary_row[f"acc_fold_{k}"] = float(acc_arr[k])
                summary_row[f"prec_fold_{k}"] = float(prec_arr[k])
                summary_row[f"rec_fold_{k}"] = float(rec_arr[k])
                summary_row[f"f1_fold_{k}"] = float(f1_arr[k])


        # --- Feature Importance Extraction ---
        # Refit on whole dataset to get global feature importance
        rf.fit(X, y)
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            for i, val in enumerate(importances):
                summary_row[f"feat_imp_{i}"] = float(val)

        if summary_master is not None:
            append_run_summary(summary_master, summary_row)

        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
            "cv_folds": n_folds,
            "acc_mean": float(acc_arr.mean()) if acc_arr.size > 0 else float(np.nan),
            "acc_std": float(acc_arr.std(ddof=1)) if acc_arr.size > 1 else float(0.0),
            "prec_macro_mean": float(prec_arr.mean()) if prec_arr.size > 0 else float(np.nan),
            "prec_macro_std": float(prec_arr.std(ddof=1)) if prec_arr.size > 1 else float(0.0),
            "rec_macro_mean": float(rec_arr.mean()) if rec_arr.size > 0 else float(np.nan),
            "rec_macro_std": float(rec_arr.std(ddof=1)) if rec_arr.size > 1 else float(0.0),
            "f1_macro_mean": float(f1_arr.mean()) if f1_arr.size > 0 else float(np.nan),
            "f1_macro_std": float(f1_arr.std(ddof=1)) if f1_arr.size > 1 else float(0.0),
        }
