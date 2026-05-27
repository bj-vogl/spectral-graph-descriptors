# spectral_graph_metric_pavement_cells/visualization/cells.py
import os
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, cm

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)


def plot_cell_graphs(graphs: Dict[str, nx.Graph], cell_info: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
    """Plot graphs and save to save_path/cell_graphs/ or show them."""
    n_graphs = len(graphs)
    if n_graphs == 0:
        print("No graphs available for plotting.")
        return

    graph_out_dir = None
    if save_path:
        graph_out_dir = os.path.join(save_path, "cell_graphs")
        os.makedirs(graph_out_dir, exist_ok=True)

    for fname, G in graphs.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        try:
            pos = nx.get_node_attributes(G, 'pos')
            if not pos:
                pos = nx.spring_layout(G, seed=42, k=0.5)
                logger.info("Spring Layout used for plotting")

            nx.draw(
                G, pos,
                node_size=20, node_color='red', edge_color='blue',
                alpha=0.5, width=0.2, ax=ax
            )
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()

            info = cell_info.get(fname, {})
            mode = info.get("mode", "unknown")

            if mode == "node_count":
                subtitle = (
                    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
                    #f"Mode: node_count = {info.get('node_count', 'n/a')}"
                )
            else:
                subtitle = (
                    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
                )

            ax.set_title(f"{subtitle}")
            ax.axis('off')
            ax.margins(y=0.05)
            fig.subplots_adjust(top=0.88)

            if save_path:
                safe_name = str(fname).replace(os.sep, "_").replace("/", "_")
                file_path = os.path.join(graph_out_dir, f"{safe_name}_graph.png")
                plt.savefig(file_path, dpi=300)
            else:
                plt.show()

        finally:
            plt.close(fig)
    if save_path:
        logger.info(f"Visibilty graphs saved to: {graph_out_dir}")


def _ensure_closed_xy(sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure that the x and y arrays are closed by appending the first point."""
    if sx.size == 0 or sy.size == 0:
        return sx, sy
    need_close = (not np.isclose(sx[0], sx[-1])) or (not np.isclose(sy[0], sy[-1]))
    if need_close:
        sx = np.concatenate([sx, sx[:1]])
        sy = np.concatenate([sy, sy[:1]])
    return sx, sy

def plot_contours_grid(contours_dict: Dict[Any, Dict[str, Any]], scale=1.0, max_cols=5, padding=10, save_path: Optional[str] = None):
    """
    Plot all cell contours in a grid.
    Works with int keys or string keys like 'cell_01'.
    If save_path is provided, it is treated as a directory to save 'contours_grid.png'.
    If no save_path is provided, the plot is shown.
    """

    fig = plt.figure(figsize=(14, 10))
    colors = cm.tab20(np.linspace(0, 1, len(contours_dict)))

    # Sort contours by number if possible
    def sort_key(item):
        key = item[0]
        if isinstance(key, int):
            return key
        try:
            return int(str(key).split('_')[-1])
        except (ValueError, IndexError):
            return 0

    sorted_items = sorted(contours_dict.items(), key=sort_key)

    # Plot title
    if sorted_items:
        first_key, first_meta = sorted_items[0]
        name = None
        if isinstance(first_meta, dict):
            name = first_meta.get("collection_name")
        if not name and isinstance(first_key, str) and (os.sep in first_key or "/" in first_key):
            _p = os.path.normpath(first_key)
            _parent = os.path.basename(os.path.dirname(_p))
            if _parent:
                name = _parent

        plot_title = f"Cell Contours - {name}" if name else f"Cell Contours - {str(first_key)}"
    else:
        plot_title = "Cell Contours"

    # Compute grid layout
    n_cells = len(sorted_items)
    n_cols = min(max_cols, n_cells)

    # Compute slot widths and heights
    slot_widths = []
    slot_heights = []
    scaled_contours = []

    for fname, data in sorted_items:
        contour = data['contour']
        umx, umy = data['resolution']
        if contour is not None:
            scaled_x = contour[:, 0] * umx * scale
            scaled_y = contour[:, 1] * umy * scale
            scaled_contours.append((fname, scaled_x, scaled_y))
            slot_widths.append(scaled_x.max() - scaled_x.min())
            slot_heights.append(scaled_y.max() - scaled_y.min())
        else:
            scaled_contours.append((fname, None, None))
            slot_widths.append(0)
            slot_heights.append(0)

    max_slot_width = max(slot_widths) + padding
    max_slot_height = max(slot_heights) + padding

    # Plot contours in grid
    for i, (fname, sx, sy) in enumerate(scaled_contours):
        if sx is None:
            continue

        row = i // n_cols
        col = i % n_cols

        x_offset = col * max_slot_width - sx.min()
        y_offset = -row * max_slot_height - sy.min()  # negative y for top-down stacking

        color = colors[i % len(colors)]

        sx_closed, sy_closed = _ensure_closed_xy(sx, sy)
        plt.plot(sx_closed + x_offset, sy_closed + y_offset, color=color)

        # Annotate with last part of filename or just the int key
        if isinstance(fname, int):
            short_label = str(fname)
        else:
            parts = str(fname).split('_')
            short_label = parts[-1] if parts else str(fname)

        centroid_x = np.mean(sx) + x_offset
        centroid_y = np.mean(sy) + y_offset
        plt.text(centroid_x, centroid_y, short_label, color='black', fontsize=8,
                 ha='center', va='center', zorder=10)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(plot_title)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_file_path = os.path.join(save_path, "contours_grid.png")
        plt.savefig(full_file_path, dpi=300)
        logger.info(f"Contour grid saved to: {full_file_path}")
    else:
        plt.show()

    plt.close(fig)
