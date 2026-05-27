# spectral_graph_metric_pavement_cells/utils/importances.py
from typing import List, Dict, Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline


def compute_permutation_importance_cv(
    pipe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    feature_names: list,
    n_repeats: int = 200,
    scoring: str = "neg_log_loss",
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute permutation importance across CV folds and aggregate results.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        A fitted-able pipeline (will be cloned/fitted per fold inside).
    X, y : np.ndarray
        Full dataset (2D X, 1D y).
    cv : sklearn BaseCrossValidator
        CV splitter (e.g., StratifiedKFold(...)).
    feature_names : list[str]
        Names for the D features (should match the columns of X).
    n_repeats : int
        Number of permutation repeats per fold (default 200).
    scoring : str
        Scoring passed to sklearn.inspection.permutation_importance.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Parallel jobs for sklearn.permutation_importance.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          "Feature", "Folds_Used", "Mean_LogLoss_Increase", "Std_LogLoss_Increase"
        sorted by Mean_LogLoss_Increase descending
    """
    # number of features
    D = X.shape[1]
    used_feature_names = list(feature_names)

    # prepare accumulators (per feature index)
    per_feature_means = [[] for _ in range(D)]
    per_feature_stds = [[] for _ in range(D)]

    # iterate over CV folds
    for fold_id, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # drop columns that are all-NaN in train, same as original logic
        keep_cols = ~np.all(np.isnan(X_tr), axis=0)
        if not np.any(keep_cols):
            continue

        X_tr_fold = X_tr[:, keep_cols]
        X_te_fold = X_te[:, keep_cols]

        # fit a fresh clone of the pipeline on the training fold
        from sklearn.base import clone

        pipe_fold = clone(pipe)
        pipe_fold.fit(X_tr_fold, y_tr)

        perm = permutation_importance(
            pipe_fold,
            X_te_fold,
            y_te,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        kept_idx = np.where(keep_cols)[0]
        imp_mean_kept = perm.importances_mean
        imp_std_kept = perm.importances_std

        # store fold results into global per-feature accumulators
        for local_k, j in enumerate(kept_idx):
            per_feature_means[j].append(float(imp_mean_kept[local_k]))
            per_feature_stds[j].append(float(imp_std_kept[local_k]))

    # Aggregate means/stds across folds (NaN if never used)
    counts = np.array([len(v) for v in per_feature_means], dtype=int)
    imp_mean = np.array([(np.mean(v) if len(v) > 0 else np.nan) for v in per_feature_means], dtype=float)
    imp_std = np.array([(np.mean(v) if len(v) > 0 else np.nan) for v in per_feature_stds], dtype=float)

    pi_cv_df = (
        pd.DataFrame({
            "Feature": used_feature_names,
            "Folds_Used": counts,
            "Mean_LogLoss_Increase": imp_mean,
            "Std_LogLoss_Increase": imp_std,
        })
        .sort_values(by="Mean_LogLoss_Increase", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    return pi_cv_df

def build_feature_importance_rows(
    clf: object,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    cfg: str | None,
    run_id: str | None,
) -> List[Dict[str, Any]]:
    """
    Build `rows_fi` list

      - reads coef_ as a 2D array
      - reads classifier classes_ into `classes_arr`
      - builds idx_to_label mapping where possible (int(class) -> class_names[int(class)])
      - for each class row, determines the class label like the original: try int(cls_key)
        then look up in idx_to_label, otherwise fallback to str(cls_key)

    Returns
    -------
    List[Dict[str, Any]]
        Each dict has keys: "run_id", "config", "class", "feature", "coef", "odds_ratio".
    """
    rows_fi: List[Dict[str, Any]] = []

    # coefs: shape (n_classes, n_features) for multi-class; ensure 2D
    coefs = np.atleast_2d(getattr(clf, "coef_", np.empty((0, 0))))
    classes_arr = getattr(clf, "classes_", np.arange(coefs.shape[0]))

    # Build idx_to_label mapping
    idx_to_label: Dict[int, str] = {}
    try:
        # Attempt to convert each class label to int and map to class_names[int]
        for i in np.asarray(classes_arr):
            try:
                i_int = int(i)
                if 0 <= i_int < len(class_names):
                    idx_to_label[i_int] = class_names[i_int]
                else:
                    # out-of-range ints are ignored
                    idx_to_label[i_int] = str(i)
            except Exception:
                continue
    except Exception:
        # if classes_arr isn't iterable or conversion fails, leave mapping empty
        idx_to_label = {}

    # for each class (row in coefs) -> for each feature
    for c_idx, class_coefs in enumerate(coefs):
        cls_key = classes_arr[c_idx] if c_idx < len(classes_arr) else c_idx
        try:
            cls_int = int(cls_key)
            cls_label = idx_to_label.get(cls_int, str(cls_key))
        except Exception:
            cls_label = str(cls_key)

        for f_idx, weight in enumerate(class_coefs):
            feat_name = feature_names[f_idx] if f_idx < len(feature_names) else f"feat_{f_idx}"
            rows_fi.append({
                "run_id": run_id,
                "config": cfg,
                "class": cls_label,
                "feature": feat_name,
                "coef": float(weight),
                "odds_ratio": float(np.exp(float(weight))),
            })

    return rows_fi

