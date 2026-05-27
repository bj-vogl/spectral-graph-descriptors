from typing import List, Dict, Any

import cv2
import networkx as nx
import numpy as np

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)


def extract_contour(binary_mask):
    contours: List[np.ndarray]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=len).squeeze()
    if contour.ndim == 1:
        contour = contour.reshape(-1, 2)
    return contour


def build_contours_from_images(images: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract contours and return dict"""
    contours_dict = {}

    for fname, data in images.items():
        mask = data['mask']
        resolution = data['resolution']  # (um_per_pixel_x, um_per_pixel_y)
        if 'raw_contour' in data:
            contour = data['raw_contour']
        else:
            contour = extract_contour(mask)
        item = {
            'contour': contour,
            'resolution': resolution,
            'mask': mask,
        }
        if isinstance(data, dict) and 'collection_name' in data:
            item['collection_name'] = data['collection_name']

        contours_dict[fname] = item

    return contours_dict

def build_contours_from_graphs(graphs: Dict[str, nx.Graph]):
    contours_dict = {}

    for fname, G in graphs.items():
        # Try to get node positions
        pos = nx.get_node_attributes(G, 'pos')
        if pos:
            coords = np.array([pos[n] for n in G.nodes()])
        else:
            raise ValueError(f"Graph {fname} has no node positions!")

        meta: Dict[str, Any] = {
            "contour": coords,           # Nx2 array
            "resolution": (1.0, 1.0),    # Dummy
        }

        cn = None
        if hasattr(G, "graph"):
            val = G.graph.get("collection_name")
            if isinstance(val, str) and val.strip():
                cn = val.strip()
        if cn:
            meta["collection_name"] = cn

        # Store in contours_dict compatible format
        contours_dict[fname] = meta

    return contours_dict

def build_contours_from_lsm(images: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build contours_dict converting pixel coordinates from ROIs/masks to µm using the LSM's Pixel width/height.

    Expects each images[key] to be a dict that contains:
      - 'lsm_path' : path to the corresponding .lsm
      - 'mask'      : binary mask
      - 'resolution' : (x_um_per_px, y_um_per_px)

    Returns dict with same keys; each value contains at least:
      - 'contour' : Nx2 numpy array in µm
      - 'resolution' : (x_um_per_px, y_um_per_px)
      - 'original_coords_px' : Nx2 numpy array (pixel coords)
      - 'lsm_path' : str
      - 'units' : 'um'
    """
    contours_dict: Dict[str, Dict[str, Any]] = {}

    for key, entry in images.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {key} must be a dict with 'lsm_path' and ROI data.")

        # extract pixel size from LSM (µm/px)
        px = entry["resolution"]
        if px is None:
            raise ValueError(f"Entry '{key}' missing 'resolution'.")
        umx, umy = float(px[0]), float(px[1])

        # obtain pixel coordinates
        contour_px = entry["raw_contour"]
        if contour_px is None:
            raise ValueError(f"Could not extract contour for entry {key}")

        # convert to µm
        contour_um = contour_px * np.array([umx, umy])[None, :] # elementwise multiplication for each row

        item: Dict[str, Any] = {
            "contour": contour_um,
            "resolution": (umx, umy),
            "original_coords_px": contour_px,
            "units": "um",
        }
        if isinstance(entry, dict) and "collection_name" in entry:
            item["collection_name"] = entry["collection_name"]

        contours_dict[key] = item

    return contours_dict