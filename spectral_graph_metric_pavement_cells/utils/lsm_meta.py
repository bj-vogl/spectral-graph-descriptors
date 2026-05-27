# spectral_graph_metric_pavement_cells/utils/lsm_meta.py
import re
from pathlib import Path
import warnings
from typing import Dict, Any, Optional, Tuple

import tifffile

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

def extract_pixel_size_from_lsm(lsm_path: str) -> Optional[Tuple[float, float]]:
    """
    Try to extract Pixel width/height (in µm) from an LSM/TIFF file.
    Looks at page.tags['ImageDescription'] and imagej metadata.
    Returns (x_um_per_px, y_um_per_px) or None if not found.
    """
    p = Path(lsm_path)
    if not p.exists():
        return None
    tif = tifffile.TiffFile(str(p))
    try:
        if hasattr(tif, "lsm_metadata") and tif.lsm_metadata:
            md = tif.lsm_metadata
            x = float(md['VoxelSizeX'])
            y = float(md['VoxelSizeY'])
            return (x, y)
    finally:
        tif.close()
    return None


def _build_lsm_candidates_from_stem(stem: str) -> list[str]:
    """
    Given a directory stem like "18-allRois" or "6_2-allRois" produce candidate lsm filenames:
      - "18.lsm"
      - "6 (2).lsm"
    """
    m = re.match(r'^(\d+)(?:_(\d+))?', stem)
    mm = re.search(r'(\d+)', stem)
    candidates: list[str] = []
    if m:
        base = m.group(1)
        sub = m.group(2)
        if sub:
            candidates.append(f"{base}_{sub}.lsm")
        elif mm:
                candidates.append(f"{mm.group(1)}.lsm")

    return candidates

def find_lsm_for_roi_dir(data_path: str) -> Optional[str]:
    p = Path(data_path)
    par = p.parent
    grandpar = par.parent
    grandgrandpar = grandpar.parent
    stem = par.name
    candidates = _build_lsm_candidates_from_stem(stem)

    for cand in candidates:
        cand_path = grandgrandpar / cand
        if cand_path.exists():
            return str(cand_path)

    return None


def attach_lsm_path_to_images(images: Dict[str, Any], data_path: str, display_key: str) -> str|None:
    """
    Shallow-copy images dict and attach 'lsm_path' to each entry if not present.
    Uses find_lsm_for_roi_dir(data_path) to find a matching .lsm in the parent folder.
    """
    lsm_candidate = find_lsm_for_roi_dir(data_path)
    if lsm_candidate is None:
        warnings.warn(f"No LSM found for roi directory {data_path} using naming rules.")
        return None

    images[display_key] = {'lsm_path': lsm_candidate}
    return lsm_candidate
