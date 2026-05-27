# spectral_graph_metric_pavement_cells/visualization/confusion.py
from typing import Sequence, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAME_MAP = {
    "3_day": "3 DAG",
    "5_day": "5 DAG",
    "7_day": "7 DAG",
    "Col-0": "Col-0",
    "ktn1-5": "ktn1-5",
    "OX16_11": "IQD16",
    "OX16_11_1_4": "IQD16"
}

def plot_confusion_heatmap(
    y_true: Sequence[str],
    y_pred_clusters: Sequence[int] | Sequence[str],
    normalize: Literal["true", "pred", "all", None] = "true",
    figsize: Tuple[int, int] = (5, 4.5),
    title: str = "Zeitserie: Verwechslungsmatrix",
    x_label: str | None = None,
):
    """
    Draw a confusion heatmap between ground-truth classes (rows) and predicted cluster IDs (columns).
    No file output; the caller decides whether to show or save.

    normalize:
      - "true": row-wise normalization (each GT class sums to 1)
      - "pred": column-wise normalization (each cluster sums to 1)
      - "all" : matrix divided by total
      - None  : raw counts
    """

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Computer Modern Sans serif']

    y_true = [str(x) for x in y_true]
    y_pred = [str(int(x)) if isinstance(x, (int, np.integer)) else str(x) for x in y_pred_clusters]

    def _rename_class(name):
        return CLASS_NAME_MAP.get(name, name)

    true_classes = sorted(np.unique(y_true).tolist())
    def _cluster_key(s: str):
        return (0, int(s)) if s.isdigit() else (1, s)
    pred_clusters = sorted(np.unique(y_pred).tolist(), key=_cluster_key)

    ti = {c: i for i, c in enumerate(true_classes)}
    pj = {c: j for j, c in enumerate(pred_clusters)}

    M = np.zeros((len(true_classes), len(pred_clusters)), dtype=float)
    for t, p in zip(y_true, y_pred):
        M[ti[t], pj[p]] += 1.0

    is_norm = normalize in {"true", "pred", "all"}
    if normalize == "true":
        row_sums = M.sum(axis=1, keepdims=True)
        M = np.divide(M, np.maximum(row_sums, 1e-12))
        fmt = ".2f"
    elif normalize == "pred":
        col_sums = M.sum(axis=0, keepdims=True)
        M = np.divide(M, np.maximum(col_sums, 1e-12))
        fmt = ".2f"
    elif normalize == "all":
        total = M.sum()
        M = M / (total + 1e-12)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    if is_norm:
        M = np.clip(M, 0.0, 1.0)
        im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        thresh = 0.5
    else:
        im = ax.imshow(M, aspect="auto")
        vmax = M.max() if M.size else 0.0
        thresh = (vmax / 2.0) if vmax else 0.0

    ax.grid(False)
    ax.set_yticks(np.arange(len(true_classes)))
    ax.set_yticklabels([_rename_class(c) for c in true_classes], fontsize=10)
    ax.set_xticks(np.arange(len(pred_clusters)))
    ax.set_xticklabels([_rename_class(c) for c in pred_clusters], rotation=45, ha="right", fontsize=10)

    ax.set_ylabel("Tatsächliche Klasse", fontsize=12)
    if x_label is None:
        all_pred_are_digits = all(s.isdigit() for s in pred_clusters)
        x_label_auto = "Vorhergesagte Cluster-ID" if all_pred_are_digits else "Vorhergesagte Klasse"
        ax.set_xlabel(x_label_auto, fontsize=11)
    else:
        ax.set_xlabel(x_label, fontsize=12)

    cmap = im.get_cmap()
    norm = im.norm

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            bg_rgba = cmap(norm(val))
            luminance = 0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
            text_color = "black" if luminance > 0.5 else "white"

            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                fontsize=10,
                color=text_color
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    if is_norm:
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.subplots_adjust(top=0.82, bottom=0.25, left=0.15, right=0.95)
    return fig, ax

