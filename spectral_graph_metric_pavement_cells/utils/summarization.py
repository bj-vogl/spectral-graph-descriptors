# spectral_graph_metric_pavement_cells/utils/summarization.py
from datetime import datetime
import math
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Sequence
import csv

from spectral_graph_metric_pavement_cells.utils.common import file_is_empty

def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        path.write_text("", encoding="utf-8")

def prepare_output_file(save_path: str, cfg: str):
    """
    Only prepare summaries_master.csv
    """
    base_outdir = Path(save_path)
    base_outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%d%m_%H-%M-%S")
    run_id = f"{cfg}_{ts}"
    summary_master = base_outdir / "summaries_master.csv"
    ensure_file_exists(summary_master)
    return summary_master, run_id, ts

def append_run_summary(summary_master: Path, summary_row: Dict[str, Any]) -> None:
    """
    Append a single summary_row (dict) to the CSV at summary_master.
    If the file does not exist, it is created. If empty, a header is written
    (based on the keys of summary_row), then the row is appended.

    Parameters
    ----------
    summary_master : pathlib.Path
        Path to the master summary CSV.
    summary_row : dict
        Row dictionary whose keys determine the CSV header order.
    """
    # ensure parent dir exists
    summary_master.parent.mkdir(parents=True, exist_ok=True)

    file_empty = file_is_empty(summary_master)
    with open(summary_master, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if file_empty:
            writer.writeheader()
        writer.writerow(summary_row)


def append_predictions_master(
    pred_master: Path,
    run_id: Optional[str],
    fold_id: int,
    fn_test: Sequence[str],
    y_test: Sequence[int],
    y_pred: Sequence[int],
    probs: Sequence[Sequence[float]],
    class_names: Sequence[str],
    clf: Optional[Any] = None,
    accuracy: Optional[float] = None,
) -> None:
    """
    Append per-sample prediction rows to `pred_master`.
    - create file if missing
    - write header if empty; header contains prob_{cls} columns for each class in class_names
    - for each sample write [run_id, fold, fname, true_label, pred_label] + probs for classes
    - append a fold-summary row with the accuracy value
    """

    pred_master.parent.mkdir(parents=True, exist_ok=True)

    # ensure header present if file empty
    if file_is_empty(pred_master):
        prob_cols = [f"prob_{cls}" for cls in class_names]
        header = ["run_id", "fold", "fname", "true_label", "pred_label"] + prob_cols
        with open(pred_master, "w", newline="", encoding="utf-8") as f:
            _w = csv.writer(f)
            _w.writerow(header)

    # write rows
    with open(pred_master, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        clf_classes = getattr(clf, "classes_", None) if clf is not None else None

        for fn, yt_i, yp_i, p_row in zip(fn_test, y_test, y_pred, probs):
            yt = class_names[int(yt_i)]
            yp = class_names[int(yp_i)]
            # reorder p_row into class_names order if classifier classes don't match
            if clf_classes is not None and list(clf_classes) != list(class_names):
                try:
                    mapping = {int(c): idx for idx, c in enumerate(clf_classes)}
                    reorder_idx = [mapping.get(i, None) for i in range(len(class_names))]
                    p_ordered = [p_row[i] if i is not None and i < len(p_row) else float("nan") for i in reorder_idx]
                except Exception:
                    p_ordered = list(p_row)
            else:
                p_ordered = list(p_row)

            row_probs = []
            for x in p_ordered:
                try:
                    fx = float(x)
                    if math.isnan(fx):
                        row_probs.append("nan")
                    else:
                        row_probs.append(f"{fx:.6f}")
                except Exception:
                    row_probs.append("nan")

            row = [run_id, fold_id, fn, yt, yp] + row_probs
            w.writerow(row)

        _w = csv.writer(f)
        if accuracy is not None:
            _w.writerow(["fold_summary", fold_id, "", "Accuracy", f"{float(accuracy):.6f}"])
        else:
            _w.writerow(["fold_summary", fold_id, "", "Accuracy", ""])
        _w.writerow([])


def append_feature_importances_master(fi_master: Path, rows_fi: Iterable[Dict[str, Any]]) -> None:
    """
    Append feature importance rows (rows_fi) to fi_master CSV.
    rows_fi is an iterable of dicts with keys:
      "run_id", "config", "class", "feature", "coef", "odds_ratio"
    Behaviour: create file if missing, write header if empty, append rows.
    """
    fi_master.parent.mkdir(parents=True, exist_ok=True)

    fi_fieldnames = ["run_id", "config", "class", "feature", "coef", "odds_ratio"]
    file_empty = file_is_empty(fi_master)
    with open(fi_master, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fi_fieldnames)
        if file_empty:
            w.writeheader()
        for row in rows_fi:
            safe_row = {
                "run_id": row.get("run_id"),
                "config": row.get("config"),
                "class": row.get("class"),
                "feature": row.get("feature"),
                "coef": float(row.get("coef")) if row.get("coef") is not None else "",
                "odds_ratio": float(row.get("odds_ratio")) if row.get("odds_ratio") is not None else "",
            }
            w.writerow(safe_row)
