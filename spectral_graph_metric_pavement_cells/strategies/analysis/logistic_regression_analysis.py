# spectral_graph_metric_pavement_cells/strategies/analysis/logistic_regression_analysis.py
from pathlib import Path
from typing import Dict, Optional, Any, List, cast

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from spectral_graph_metric_pavement_cells.core.interfaces.analysis_strategy import AnalysisStrategy
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.common import merge_across_pipelines
from spectral_graph_metric_pavement_cells.utils.scalers import GlobalIQRScaler, GlobalStandardScaler, MaxnormScaler
from spectral_graph_metric_pavement_cells.utils.summarization import append_run_summary, prepare_output_file
from spectral_graph_metric_pavement_cells.visualization.confusion import plot_confusion_heatmap

logger = get_logger(__name__)

class LogisticRegressionAnalysis(AnalysisStrategy):
    """
    Logistic Regression across ALL computation pipelines.
    Parameters:
      analysis_feature_keys: mapping pipeline_name -> feature_key
      scaler_name: name of scaler to use (e.g., "StandardScaler")
      n_folds: number of folds for cross-validation
    """

    def __init__(self, analysis_feature_keys: Dict[str, str], scaler_name: Optional[str], n_folds: Optional[int] = 10):
        self.analysis_feature_keys = analysis_feature_keys
        self.scaler_name = scaler_name
        self.n_folds = n_folds
        self._save_info: tuple[Optional[str], Optional[str]] = (None, None)

    def set_save_info(self, save_path: Optional[str], config_name: Optional[str]) -> None:
        self._save_info = (save_path, config_name)

    def _get_scaler(self):
        """Instantiate scaler based on scaler_name"""
        if self.scaler_name is None or self.scaler_name == "None":
            return None

        name = self.scaler_name
        if name == "StandardScaler": return StandardScaler()
        elif name == "GlobalIQRScaler": return GlobalIQRScaler((25, 75))
        elif name == "GlobalStandardScaler": return GlobalStandardScaler()
        elif name == "MaxnormScaler": return MaxnormScaler()
        else:
            logger.warning("Unknown scaler_name '%s'", name)
            return None

    def analyze(self, results: Dict[str, DistanceResult]) -> Dict[str, Any]:
        save_path, cfg = self._save_info
        run_id = None
        summary_master = None
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
            logger.error("LogisticRegressionAnalysis failed to merge datasets: %s", e)
            return {"error": str(e)}

        scaler = self._get_scaler()

        solver_type = "lbfgs"
        if scaler is None:
            solver_type = "newton-cg"

        pipe = Pipeline([
            ("sc", scaler), # skipped if scaler_name is None
            ("clf", LogisticRegression(max_iter=4000, solver=solver_type, tol=1e-3, random_state=42)),
        ])

        n_folds = self.n_folds
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        acc_scores: List[float] = []
        prec_scores: List[float] = []
        rec_scores: List[float] = []
        f1_scores: List[float] = []

        all_true_idx: List[np.ndarray] = []
        all_pred_idx: List[np.ndarray] = []

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = pipe.score(X_test, y_test)
            p_macro, r_macro, _, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

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
            title=f"Logistische Regression",
            x_label="Vorhergesagte Klasse",
        )

        if save_path and cfg and ts:
            outdir = Path(save_path)
            cv_png = outdir / f"{cfg}_logreg_confusion_{ts}.png"
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

        # Build summary row (compact)
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
            "scaler_name": self.scaler_name,
        }

        for k in range(n_folds):
            if k < len(acc_arr):
                summary_row[f"acc_fold_{k}"] = float(acc_arr[k])
                summary_row[f"prec_fold_{k}"] = float(prec_arr[k])
                summary_row[f"rec_fold_{k}"] = float(rec_arr[k])
                summary_row[f"f1_fold_{k}"] = float(f1_arr[k])

        # --- Feature Importance extraction ---
        # Refit on the whole dataset to extract coefficients
        pipe.fit(X, y)
        clf = cast(LogisticRegression, pipe.named_steps["clf"])

        if hasattr(clf, "coef_"):
            # coef_ shape: (n_classes, n_features) for multiclass
            # Take mean absolute value across classes to get a general importance per feature
            importances = np.mean(np.abs(clf.coef_), axis=0)

            for i, val in enumerate(importances):
                summary_row[f"feat_imp_{i}"] = float(val)

        if summary_master is not None:
            append_run_summary(summary_master, summary_row)

        return {
            "n_samples": int(X.shape[0]),
            "dim": int(X.shape[1]),
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
            "scaler_name": self.scaler_name,
        }
